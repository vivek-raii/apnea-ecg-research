"""Predict complete minutes of a supported 100-Hz single-channel WFDB ECG.

Only load project-generated/trusted joblib files: pickle formats execute code.
"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from ecg import Archive, FEATURES, VERSION, WINDOW, record_features, extract_window

ROOT=Path(__file__).resolve().parents[1]
def predict_frame(frame,bundle):
    if bundle['extractor_version']!=VERSION or bundle['features']!=FEATURES:
        raise ValueError('Model/extractor version or feature mismatch.')
    if frame.empty: raise ValueError('No complete minutes to predict.')
    result=frame[['record','minute','start_sample','label','quality_ok','quality_reason']].copy()
    result['score']=bundle['pipeline'].predict_proba(frame[FEATURES])[:,1]
    result['prediction']=(result.score>=bundle['threshold']).astype(int)
    result['status']=np.where(result.quality_ok,'research_prediction','review_signal_quality')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--record',help='Record ID in the configured PhysioNet ZIP, e.g. x01')
    source.add_argument('--wfdb',help='Local WFDB base path without extension; 100 Hz, single ECG channel in mV')
    p.add_argument('--model',default=str(ROOT/'outputs/final_model.joblib'))
    p.add_argument('--output',default=str(ROOT/'outputs/predictions.json'))
    args=p.parse_args(); bundle=joblib.load(args.model)
    if args.record:
        config=json.loads((ROOT/'config.json').read_text()); a=Archive(config['dataset_zip'])
        frame=record_features(a,args.record); a.close()
    else:
        import wfdb
        signal=wfdb.rdrecord(args.wfdb)
        if signal.fs!=100 or signal.n_sig!=1 or signal.units!=['mV']:
            raise ValueError('Supply one ECG channel at 100 Hz in mV. Resampling is not implicit.')
        data=[]
        for start in range(0,len(signal.p_signal)-WINDOW+1,WINDOW):
            feat,reason,_=extract_window(signal.p_signal[start:start+WINDOW,0])
            data.append(dict(record=Path(args.wfdb).name,minute=start//WINDOW,start_sample=start,label=np.nan,
                             quality_ok=not bool(reason),quality_reason=reason,**feat))
        frame=pd.DataFrame(data)
    result=predict_frame(frame,bundle); path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    result.to_json(path,orient='records',indent=2)
    print(f'Saved {len(result)} research predictions to {path}. Scores are not calibrated clinical probabilities.')
