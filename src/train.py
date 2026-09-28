"""End-to-end extraction, grouped development, locked evaluation and final fit."""
from __future__ import annotations
import argparse, hashlib, json, time, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss)
from threadpoolctl import threadpool_limits
from ecg import Archive, FEATURES, VERSION, WINDOW, groups_from_metadata, record_features, extract_window

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'

def save_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def metrics(y,p,threshold):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    return dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()),threshold=float(threshold),
        accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)) if both else None,
        sensitivity=float(tp/(tp+fn)) if tp+fn else None,specificity=float(tn/(tn+fp)) if tn+fp else None,
        precision=float(precision_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
        auroc=float(roc_auc_score(y,p)) if both else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        brier=float(brier_score_loss(y,p)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def candidates(seed):
    return {
        'logistic_regression': make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),
            LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=seed)),
        'random_forest': make_pipeline(SimpleImputer(strategy='median',add_indicator=True),
            RandomForestClassifier(n_estimators=180,min_samples_leaf=8,max_features='sqrt',
                                   class_weight='balanced_subsample',n_jobs=4,random_state=seed)),
        'hist_gradient_boosting': make_pipeline(SimpleImputer(strategy='median',add_indicator=True),
            HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=0.06,
                                           l2_regularization=2,early_stopping=False,random_state=seed))}

def cluster_intervals(frame,threshold,n_boot,seed):
    rng=np.random.default_rng(seed); groups=frame.group.unique(); vals=[]
    for _ in range(n_boot):
        selected=rng.choice(groups,len(groups),replace=True)
        sample=pd.concat([frame[frame.group==g] for g in selected],ignore_index=True)
        if sample.label.nunique()<2: continue
        vals.append(metrics(sample.label,sample.score,threshold))
    return {k:dict(low=float(np.percentile([v[k] for v in vals],2.5)),
                   high=float(np.percentile([v[k] for v in vals],97.5)),valid_replicates=len(vals))
            for k in ['balanced_accuracy','sensitivity','specificity','f1','auroc','average_precision']}

def extract(config,force=False):
    path=OUT/'features.pkl'; provenance=OUT/'feature_provenance.json'
    sha=hashlib.sha256(Path(config['dataset_zip']).read_bytes()).hexdigest()
    code_sha=hashlib.sha256((ROOT/'src'/'ecg.py').read_bytes()).hexdigest()
    expected=dict(dataset_sha256=sha,extractor_sha256=code_sha,version=VERSION)
    if path.exists() and provenance.exists() and not force:
        if json.loads(provenance.read_text())==expected:
            print('Using verified cached features.',flush=True); return pd.read_pickle(path)
    archive=Archive(config['dataset_zip']); checked=archive.verify()
    group=groups_from_metadata(archive.metadata()); frames=[]
    learning=[r for r in archive.records if r+'.apn' in archive.names and r not in config['exclude_records']]
    # Verify a full exact overlap once; its removal is pre-specified, not score-driven.
    c5,c6=archive.signal('c05'),archive.signal('c06'); n=min(len(c5),len(c6)-8000)
    assert np.array_equal(c5[:n],c6[8000:8000+n])
    save_json(OUT/'data_integrity.json',dict(verified_manifest_files=checked,duplicate_samples=n,
              duplicate_offset_seconds=80,excluded=config['exclude_records'],candidate_groups=group))
    for i,r in enumerate(learning):
        f=record_features(archive,r,labeled_only=True); f['group']=group[r]; frames.append(f)
        print(f'Features {i+1}/{len(learning)}: {r}, {len(f)} minutes',flush=True)
    archive.close(); frame=pd.concat(frames,ignore_index=True)
    assert frame.label.notna().all()
    frame['label']=frame.label.astype(int)
    frame.to_pickle(path); save_json(provenance,expected)
    # JSON preview is convenient without requiring a spreadsheet application.
    frame.head(50).to_json(OUT/'features_preview.json',orient='records',indent=2)
    return frame

def run(config,force=False):
    OUT.mkdir(exist_ok=True,parents=True); started=time.time(); frame=extract(config,force)
    seed=config['seed']; X=frame[FEATURES]; y=frame.label; groups=frame.group
    splitter=GroupShuffleSplit(n_splits=1,test_size=config['holdout_fraction_groups'],random_state=seed)
    dev_idx,test_idx=next(splitter.split(X,y,groups))
    dev=frame.iloc[dev_idx].copy(); test=frame.iloc[test_idx].copy()
    assert set(dev.group).isdisjoint(set(test.group))
    assert dev.label.nunique()==test.label.nunique()==2
    # Freeze the split BEFORE fitting; it never changes in response to scores.
    split=dict(seed=seed,development_records=sorted(dev.record.unique()),holdout_records=sorted(test.record.unique()),
               development_groups=sorted(dev.group.unique()),holdout_groups=sorted(test.group.unique()),
               grouping='Identical age/sex/height/weight connected across all records; c05/c06 linked; identities unverified')
    save_json(OUT/'split_manifest.json',split)
    print('Locked holdout:',split['holdout_records'],flush=True)
    cv=list(StratifiedGroupKFold(n_splits=config['cv_folds'],shuffle=True,random_state=seed).split(dev[FEATURES],dev.label,dev.group))
    models=candidates(seed); comparison=[]; oof_by_model={}; thresholds={}; fold_rows=[]
    for name,estimator in models.items():
        oof=np.full(len(dev),np.nan)
        for fold,(tr,va) in enumerate(cv):
            assert set(dev.iloc[tr].group).isdisjoint(set(dev.iloc[va].group))
            assert dev.iloc[tr].label.nunique()==2
            fitted=clone(estimator).fit(dev.iloc[tr][FEATURES],dev.iloc[tr].label)
            oof[va]=fitted.predict_proba(dev.iloc[va][FEATURES])[:,1]
            fold_rows.append(dict(model=name,fold=fold,train_groups=sorted(dev.iloc[tr].group.unique()),
                                 validation_groups=sorted(dev.iloc[va].group.unique())))
            print(f'{name}: development fold {fold+1}/{len(cv)} complete',flush=True)
        assert np.isfinite(oof).all()
        # Threshold selected solely from development out-of-fold predictions.
        grid=np.linspace(0.05,0.95,91)
        threshold=float(max(grid,key=lambda t:balanced_accuracy_score(dev.label,oof>=t)))
        thresholds[name]=threshold; oof_by_model[name]=oof
        comparison.append(dict(model=name,**metrics(dev.label,oof,threshold)))
    save_json(OUT/'cv_group_manifest.json',fold_rows)
    selected=max(comparison,key=lambda r:r['average_precision'])['model']
    threshold=thresholds[selected]
    print('Selected on development OOF average precision:',selected,'threshold',threshold,flush=True)
    # Holdout is first evaluated now; do not choose model/threshold using this score.
    evaluation_model=clone(models[selected]).fit(dev[FEATURES],dev.label)
    p=evaluation_model.predict_proba(test[FEATURES])[:,1]
    test['score']=p; test['prediction']=(p>=threshold).astype(int)
    holdout_metrics=metrics(test.label,p,threshold)
    intervals=cluster_intervals(test,threshold,config['bootstrap_group_replicates'],seed)
    baseline=metrics(test.label,np.zeros(len(test)),0.5)
    quality=test[test.quality_ok]
    quality_metrics=metrics(quality.label,quality.score,threshold) if len(quality) else None
    oof_frame=dev[['record','minute','label','group','quality_ok']].copy()
    for name,pred in oof_by_model.items(): oof_frame[name]=pred
    oof_frame.to_pickle(OUT/'development_oof.pkl'); test.to_pickle(OUT/'holdout_predictions.pkl')
    per_record=[]
    for r,f in test.groupby('record'):
        per_record.append(dict(record=r,**metrics(f.label,f.score,threshold)))
    save_json(OUT/'per_record_metrics.json',per_record)
    model_metadata=dict(features=FEATURES,threshold=threshold,selected_model=selected,extractor_version=VERSION,
                       sample_rate=100,window_samples=WINDOW,signal_unit='mV',calibration='Uncalibrated model scores',
                       split=split,config=config)
    joblib.dump(dict(pipeline=evaluation_model,**model_metadata,training_scope='development records only'),OUT/'evaluation_model.joblib')
    # Feature permutation is interpretive analysis after selection, not feature tuning.
    from sklearn.inspection import permutation_importance
    importance=permutation_importance(evaluation_model,test[FEATURES],test.label,scoring='average_precision',n_repeats=3,random_state=seed,n_jobs=1)
    save_json(OUT/'feature_importance.json',[dict(feature=k,ap_drop_mean=float(v),ap_drop_std=float(s))
                   for k,v,s in zip(FEATURES,importance.importances_mean,importance.importances_std)])
    # Pre-specified robustness check on evenly spaced windows per held-out record.
    archive=Archive(config['dataset_zip']); clean_rows=[]; perturbations={snr:[] for snr in config['noise_snr_db']}
    rng=np.random.default_rng(seed)
    for record,subset in test.groupby('record'):
        signal=archive.signal(record)
        positions=np.unique(np.linspace(0,len(subset)-1,min(config['noise_windows_per_record'],len(subset))).astype(int))
        for _,row in subset.iloc[positions].iterrows():
            start=int(row.start_sample); x=signal[start:start+WINDOW]
            clean_rows.append(row)
            for snr in config['noise_snr_db']:
                # Synthetic white noise is a controlled stressor, not realistic motion artifact.
                sigma=max(float(x.std()),1e-6)/10**(snr/20)
                feat,reason,_=extract_window(x+rng.normal(0,sigma,len(x)))
                perturbations[snr].append(dict(label=int(row.label),quality_ok=not bool(reason),**feat))
    clean=pd.DataFrame(clean_rows); robustness=[dict(condition='clean sampled windows',coverage=float(clean.quality_ok.mean()),**metrics(clean.label,clean.score,threshold))]
    for snr,data in perturbations.items():
        noisy=pd.DataFrame(data); scores=evaluation_model.predict_proba(noisy[FEATURES])[:,1]
        robustness.append(dict(condition=f'white noise {snr} dB SNR',coverage=float(noisy.quality_ok.mean()),**metrics(noisy.label,scores,threshold)))
    archive.close()
    # Final usable artifact is fitted on all available labeled development+holdout data.
    # Its training fit is NOT reported as an independent evaluation.
    final_model=clone(models[selected]).fit(X,y)
    joblib.dump(dict(pipeline=final_model,**model_metadata,training_scope='all 34 labeled records (c06 excluded)'),OUT/'final_model.joblib')
    report=dict(selected_model=selected,threshold=threshold,development_comparison=comparison,
                holdout=holdout_metrics,holdout_group_bootstrap_95pct=intervals,always_normal_baseline=baseline,
                quality_accepted_holdout=quality_metrics,quality_coverage=float(test.quality_ok.mean()),
                robustness=robustness,windows=len(frame),records=int(frame.record.nunique()),groups=int(frame.group.nunique()),
                development_windows=len(dev),holdout_windows=len(test),split=split,
                duration_seconds=round(time.time()-started,1),
                limitations=['Candidate groups are not verified patient identities.',
                  'Original x01-x35 labels not present; holdout drawn from labeled learning set.',
                  'Threshold/model tuned on development OOF; development metrics are selection estimates.',
                  'Model scores are not calibrated disease probabilities.',
                  'Single-minute spectral summaries are exploratory, not clinical HRV measurements.',
                  'Simple peak detector and quality rules are not clinically validated.',
                  'Predictions require the whole minute, so this is retrospective detection, not early warning.',
                  'No external clinical validation or proof of novelty.'])
    save_json(OUT/'metrics.json',report)
    import sklearn, scipy, matplotlib
    save_json(OUT/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
                    pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    from report import make_report
    make_report(ROOT,config)
    print(json.dumps(holdout_metrics,indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--force-extract',action='store_true')
    args=parser.parse_args(); config=json.loads((ROOT/'config.json').read_text())
    with threadpool_limits(limits=4): run(config,args.force_extract)
