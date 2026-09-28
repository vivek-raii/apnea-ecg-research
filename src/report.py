"""Publication-style exploratory plots and a standalone HTML results report."""
import base64, json, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from ecg import Archive, extract_window, WINDOW

def make_report(root,config):
    out=root/'outputs'; figs=out/'figures'; figs.mkdir(exist_ok=True)
    m=json.loads((out/'metrics.json').read_text()); test=pd.read_pickle(out/'holdout_predictions.pkl')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    images=[]
    def save(fig,name):
        fig.savefig(figs/name,bbox_inches='tight'); plt.close(fig)
        encoded=base64.b64encode((figs/name).read_bytes()).decode()
        images.append(f'<img alt="{name}" src="data:image/png;base64,{encoded}"/>')
    fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    ConfusionMatrixDisplay.from_predictions(test.label,test.prediction,display_labels=['N','A'],ax=axs[0],colorbar=False,cmap='Blues')
    RocCurveDisplay.from_predictions(test.label,test.score,ax=axs[1],name='Held-out groups')
    PrecisionRecallDisplay.from_predictions(test.label,test.score,ax=axs[2],name='Held-out groups')
    axs[2].axhline(test.label.mean(),ls='--',c='gray',label='Prevalence'); axs[2].legend()
    fig.suptitle('Evaluation model: held-out recording groups only'); save(fig,'holdout_evaluation.png')
    per=pd.DataFrame(json.loads((out/'per_record_metrics.json').read_text()))
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    per.set_index('record')[['sensitivity','specificity']].plot.bar(ax=ax,color=['#dc2626','#2563eb'])
    ax.set(ylim=(0,1.08),ylabel='Fraction',title='Per-record performance (undefined metrics left blank)')
    save(fig,'per_record.png')
    imp=pd.DataFrame(json.loads((out/'feature_importance.json').read_text())).sort_values('ap_drop_mean').tail(12)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    ax.barh(imp.feature,imp.ap_drop_mean,xerr=imp.ap_drop_std,color='#2563eb')
    ax.set(xlabel='Average precision drop after permutation',title='Holdout feature importance: descriptive, correlated features share credit')
    save(fig,'feature_importance.png')
    archive=Archive(config['dataset_zip']); record=m['split']['holdout_records'][0]
    subset=test[test.record==record].sort_values('minute')
    fig,axs=plt.subplots(2,1,figsize=(12,5),sharex=True,layout='constrained')
    axs[0].step(subset.minute,subset.label,where='post',color='#9333ea'); axs[0].set(ylabel='Reference A/N',ylim=(-.1,1.1))
    axs[1].plot(subset.minute,subset.score,lw=.8); axs[1].axhline(m['threshold'],color='red',ls='--')
    axs[1].set(xlabel='Minute from start',ylabel='Model score',ylim=(0,1)); fig.suptitle(f'{record}: reference and model scores')
    save(fig,'overnight_predictions.png')
    # Display peaks against raw ECG for a development example; no clinical claims.
    record=m['split']['development_records'][0]; signal=archive.signal(record)[:WINDOW]
    _,reason,peaks=extract_window(signal)
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
    ax.plot(np.arange(1000)/100,signal[:1000],lw=.8)
    show=peaks[peaks<1000]; ax.scatter(show/100,signal[show],color='red',s=18,label='Detected candidate peaks')
    ax.set(xlabel='Seconds',ylabel='ECG (mV)',title=f'{record}, minute 0: example detector output (not expert-validated)'); ax.legend()
    save(fig,'ecg_peaks_example.png'); archive.close()
    table=pd.DataFrame(m['development_comparison'])[['model','average_precision','auroc','balanced_accuracy','threshold']]
    hold=pd.DataFrame([m['holdout'],m['always_normal_baseline']],index=['Selected model','Always N'])
    rob=pd.DataFrame(m['robustness'])[['condition','n','average_precision','balanced_accuracy','sensitivity','specificity','coverage']]
    def html_table(df): return df.to_html(float_format=lambda x:f'{x:.3f}',border=0)
    page='''<!doctype html><html><head><meta charset="utf-8"><title>Apnea ECG research results</title>
<style>body{font:16px system-ui;max-width:1120px;margin:30px auto;padding:20px;color:#172033;line-height:1.5}h1{font-size:34px}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef2ff}img{width:100%;height:auto;margin:20px 0}.note{padding:15px;background:#fff7ed;border-left:4px solid #ea580c}</style></head><body>'''
    page+=f'<h1>ECG-based apnea detection</h1><p>Reproducible research prototype • {m["records"]} labeled recordings • {m["windows"]:,} complete windows</p>'
    page+='<p class="note">Research use only. Candidate recording groups are not verified patient identities. No external clinical validation. Scores are uncalibrated; this is minute-level detection, not diagnosis or early warning.</p>'
    page+=f'<p class="note"><strong>Interpretation:</strong> The selected baseline detects {m["holdout"]["sensitivity"]:.1%} of positive held-out minutes, with balanced accuracy {m["holdout"]["balanced_accuracy"]:.1%}. These results do not establish useful screening performance. The pipeline is complete, but the detector requires substantial further research. Do not retune against this holdout and continue calling it an independent test.</p>'
    page+=f'<h2>Selected model: {m["selected_model"]}</h2><p>Threshold {m["threshold"]:.2f} selected using development out-of-fold predictions. Holdout: {", ".join(m["split"]["holdout_records"])}.</p>'
    page+='<h2>Development comparison</h2><p>Selection estimates, not independent final results. No holdout scores used to select the model.</p>'+html_table(table)
    page+='<h2>Held-out evaluation</h2>'+html_table(hold[['n','accuracy','balanced_accuracy','sensitivity','specificity','precision','f1','auroc','average_precision']])+images[0]
    ci=pd.DataFrame(m['holdout_group_bootstrap_95pct']).T
    page+='<h2>Uncertainty</h2><p>95% percentile bootstrap intervals resample whole candidate groups. Very few independent groups make these intervals unstable; they do not establish population-level performance.</p>'+html_table(ci)
    page+='<h2>Individual recordings</h2>'+images[1]+images[3]
    page+=f'<h2>Quality screening</h2><p>Predefined rules retain {m["quality_coverage"]:.1%} of holdout minutes. Primary evaluation includes all complete windows; accepted-only evaluation is secondary.</p>'
    if m['quality_accepted_holdout']: page+=html_table(pd.DataFrame([m['quality_accepted_holdout']]))
    page+='<h2>Controlled noise experiment</h2><p>The same evenly spaced held-out windows are assessed under synthetic white noise. These conditions are not a substitute for real movement artifacts. Coverage is the fraction passing fixed quality rules.</p>'+html_table(rob)
    page+='<h2>Feature interpretation</h2>'+images[2]+'<h2>ECG example</h2>'+images[4]
    page+='<h2>Limitations</h2><ul>'+''.join('<li>'+x+'</li>' for x in m['limitations'])+'</ul>'
    page+='<h2>Model artifacts</h2><p>evaluation_model.joblib was fitted only on development records. final_model.joblib was refitted on all eligible labeled records for subsequent experiments. The final refit has no separate independent performance estimate.</p></body></html>'
    (out/'results_report.html').write_text(page,encoding='utf-8')
