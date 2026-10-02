"""Fixed development-only detector ablation; baseline outputs remain unchanged."""
import json, time, hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
import ecg_xqrs as improved
from train import candidates, metrics, cluster_intervals, save_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/xqrs_experiment'
OUT.mkdir(exist_ok=True)
CONFIG=json.loads((ROOT/'config.json').read_text())
SPLIT=json.loads((ROOT/'outputs/split_manifest.json').read_text())

def extract_record(record):
    with threadpool_limits(limits=1):
        archive=improved.Archive(CONFIG['dataset_zip'])
        try: return improved.record_features(archive,record,labeled_only=True)
        finally: archive.close()

def run():
    start=time.time()
    protocol=dict(detector='WFDB XQRS; independently run on each 60-second window',
        selection='Highest development OOF average precision across all nine fixed candidates',
        threshold='Development OOF balanced accuracy, 0.05 to 0.95 in 0.01 steps',
        variants=['baseline','xqrs','xqrs_relative'],models=list(candidates(42)),
        split=SPLIT, caveat='Previously examined holdout reused for comparison, not a fresh independent test.')
    save_json(OUT/'protocol.json',protocol)
    baseline=pd.read_pickle(ROOT/'outputs/features.pkl')
    provenance=hashlib.sha256((ROOT/'src/ecg_xqrs.py').read_bytes()).hexdigest()
    cache=OUT/'features.pkl'
    if cache.exists() and (OUT/'cache_sha.txt').read_text()==provenance:
        frame=pd.read_pickle(cache)
    else:
        frames=[]; records=sorted(baseline.record.unique())
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(record,f) in enumerate(zip(records,pool.map(extract_record,records))):
                f['group']=baseline.loc[baseline.record==record,'group'].iloc[0]
                frames.append(f)
                print(f'XQRS features {i+1}/34: {record}',flush=True)
        frame=pd.concat(frames,ignore_index=True)
        frame.to_pickle(cache); (OUT/'cache_sha.txt').write_text(provenance)
    # Worker serialization can change pandas string storage without changing values.
    for key in ['record','minute','label','group']:
        np.testing.assert_array_equal(baseline[key].to_numpy(),frame[key].to_numpy())
    relative=frame.copy()
    # Ratios remove absolute ECG gain and resting heart-rate scale. No labels used.
    for col in ['rr_std','rr_iqr','rmssd']:
        relative[col+'_relative']=relative[col]/relative.rr_mean.clip(lower=.01)
    for col in ['amp_std','amp_iqr','amp_rmssd']:
        relative[col+'_relative']=relative[col]/relative.amp_mean.clip(lower=1e-8)
    relative['rr_band_fraction']=relative.rr_power_fast/(relative.rr_power_slow+relative.rr_power_fast+1e-12)
    relative['amp_power_relative']=relative.amp_resp_power/(relative.amp_mean**2+1e-12)
    relative_features=['bad_rr_fraction','pnn50','rr_trend','rr_band_fraction','amp_power_relative','power_low','power_mid','power_high']+[c for c in relative if c.endswith('_relative') and c!='amp_power_relative']
    variants={'baseline':(baseline,improved.FEATURES),'xqrs':(frame,improved.FEATURES),'xqrs_relative':(relative,relative_features)}
    comparison=[]; predictions={}; estimators=candidates(CONFIG['seed'])
    for variant,(data,features) in variants.items():
        dev=data[data.record.isin(SPLIT['development_records'])].copy()
        folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=42).split(dev[features],dev.label,dev.group))
        for name,estimator in estimators.items():
            oof=np.full(len(dev),np.nan)
            for tr,va in folds:
                assert set(dev.iloc[tr].group).isdisjoint(dev.iloc[va].group)
                model=clone(estimator).fit(dev.iloc[tr][features],dev.iloc[tr].label)
                oof[va]=model.predict_proba(dev.iloc[va][features])[:,1]
            threshold=float(max(np.linspace(.05,.95,91),key=lambda t:balanced_accuracy_score(dev.label,oof>=t)))
            key=variant+'/'+name
            comparison.append(dict(candidate=key,variant=variant,model=name,**metrics(dev.label,oof,threshold)))
            predictions[key]=oof
            print('Development:',key,'AP',comparison[-1]['average_precision'],flush=True)
    winner=max(comparison,key=lambda r:r['average_precision'])
    save_json(OUT/'development_selection.json',dict(comparison=comparison,selected=winner))
    data,features=variants[winner['variant']]
    dev=data[data.record.isin(SPLIT['development_records'])]
    test=data[data.record.isin(SPLIT['holdout_records'])].copy()
    model=clone(estimators[winner['model']]).fit(dev[features],dev.label)
    test['score']=model.predict_proba(test[features])[:,1]
    result=metrics(test.label,test.score,winner['threshold'])
    metadata=dict(features=features,threshold=winner['threshold'],variant=winner['variant'],selected_model=winner['model'],split=SPLIT)
    joblib.dump(dict(pipeline=model,**metadata,training_scope='development only'),OUT/'evaluation_model.joblib')
    final=clone(estimators[winner['model']]).fit(data[features],data.label)
    joblib.dump(dict(pipeline=final,**metadata,training_scope='all eligible labeled records'),OUT/'final_model.joblib')
    test[['record','minute','label','score','group']].to_json(OUT/'holdout_predictions.json',orient='records')
    old=json.loads((ROOT/'outputs/metrics.json').read_text())['holdout']
    report=dict(selected=winner,comparison=comparison,baseline_holdout=old,holdout=result,
        intervals=cluster_intervals(test,winner['threshold'],500,42),
        delta_percentage_points={k:100*(result[k]-old[k]) for k in ['accuracy','balanced_accuracy','sensitivity','specificity']},
        caveat=protocol['caveat'],duration_seconds=time.time()-start)
    save_json(OUT/'results.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): run()
