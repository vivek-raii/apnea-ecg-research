"""Cloud research dashboard. Metrics and model are loaded from the deployed commit."""
import hashlib
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from ecg import WINDOW, extract_window
from predict import predict_frame

st.set_page_config(page_title='Apnea ECG Research', page_icon='🫀', layout='wide')
st.title('Apnea ECG Research')
st.caption('ECG-based sleep-apnea detection • Reproducible research dashboard • Auto-deployed from GitHub main')
st.warning('Experimental research model — not for diagnosis. Review the held-out metrics below. Scores are not calibrated disease probabilities.')

# Read the committed metrics on every rerun. No hardcoded performance numbers.
metrics = json.loads((ROOT / 'outputs/metrics.json').read_text())
holdout = metrics['holdout']
cols = st.columns(4)
for col, label, key in zip(cols, ['Held-out accuracy', 'Balanced accuracy', 'Sensitivity', 'Specificity'], ['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity']):
    col.metric(label, f'{100 * holdout[key]:.1f}%')
st.caption(f"{holdout['n']:,} held-out minutes • Selected model: {metrics['selected_model']} • AUROC: {holdout['auroc']:.3f}")
st.info('These metrics evaluate the development-trained model on held-out recording groups. Participant identities are unverified. The inference model was subsequently refitted on all eligible labeled records and has no independent post-refit evaluation.')

evaluation, inference, about = st.tabs(['Evaluation', 'Try inference', 'Deployment & method'])
with evaluation:
    st.subheader('Model comparison on development data')
    st.caption('These scores were used for model selection; they are not independent test performance.')
    st.dataframe(pd.DataFrame(metrics['development_comparison'])[['model', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'auroc']], hide_index=True, width='stretch')
    st.subheader('Held-out confusion matrix')
    st.dataframe(pd.DataFrame([[holdout['tn'], holdout['fp']], [holdout['fn'], holdout['tp']]], index=['Reference normal', 'Reference apnea'], columns=['Predicted normal', 'Predicted apnea']), width='stretch')
    st.subheader('Group-bootstrap 95% intervals')
    st.dataframe(pd.DataFrame(metrics['holdout_group_bootstrap_95pct']).T, width='stretch')
    st.download_button('Download complete evaluation JSON', (ROOT / 'outputs/metrics.json').read_bytes(), 'metrics.json', 'application/json')
    st.download_button('Download evaluation report', (ROOT / 'outputs/results_report.html').read_bytes(), 'results_report.html', 'text/html')

# Cache only the trusted repository model, keyed by content so updates invalidate it.
@st.cache_resource
def load_model(digest):
    return joblib.load(ROOT / 'outputs/final_model.joblib')

with inference:
    st.subheader('Predict from a single ECG channel')
    st.write('Upload a CSV with one column named ecg_mv: samples in millivolts at exactly 100 Hz. Each 6,000 samples produces one complete-minute prediction. Limit: 60 minutes / 360,000 samples.')
    st.caption('Uploads are processed in this hosted session and are not intentionally saved by this app. Use public or synthetic research signals, without personal identifiers.')
    uploaded = st.file_uploader('ECG CSV', type=['csv'])
    confirmed = st.checkbox('I confirm the signal is one ECG channel at 100 Hz in millivolts.')
    if st.button('Run model', disabled=uploaded is None or not confirmed):
        try:
            if uploaded.size > 8 * 1024 * 1024:
                raise ValueError('Maximum upload size is 8 MB.')
            frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=360001)
            if list(frame.columns) != ['ecg_mv']:
                raise ValueError('CSV must contain exactly one column: ecg_mv.')
            values = pd.to_numeric(frame.ecg_mv, errors='raise').to_numpy(dtype=float)
            if not WINDOW <= len(values) <= 360000 or not np.isfinite(values).all():
                raise ValueError('Provide 6,000–360,000 finite numeric samples.')
            with st.spinner('Extracting features and predicting…'):
                rows = []
                for start in range(0, len(values) - WINDOW + 1, WINDOW):
                    features, reason, _ = extract_window(values[start:start + WINDOW])
                    rows.append(dict(record='uploaded', minute=start // WINDOW, start_sample=start, label=np.nan, quality_ok=not bool(reason), quality_reason=reason, **features))
                bundle = load_model(hashlib.sha256((ROOT / 'outputs/final_model.joblib').read_bytes()).hexdigest())
                result = predict_frame(pd.DataFrame(rows), bundle)
            st.caption(f"Threshold: {bundle['threshold']:.2f}. Ignored trailing samples: {len(values) % WINDOW}. A prediction of 1 means apnea; 0 means normal.")
            st.line_chart(result.set_index('minute')[['score']])
            st.dataframe(result, hide_index=True, width='stretch')
            st.download_button('Download predictions', result.to_csv(index=False), 'predictions.csv', 'text/csv')
        except (ValueError, TypeError, pd.errors.ParserError) as exc:
            st.error(str(exc))

with about:
    st.subheader('Automatic updates from GitHub')
    st.write('This app reads outputs/metrics.json and outputs/final_model.joblib from the deployed main branch. After retraining, commit and push the updated metrics, models, and report together. Streamlit redeploys from main; the dashboard then shows the new values, including regressions. Pushing code alone does not retrain the model.')
    st.caption('Evaluation fingerprint: ' + hashlib.sha256((ROOT / 'outputs/metrics.json').read_bytes()).hexdigest()[:12])
    st.markdown('Source: [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/)')
    for limitation in metrics['limitations']:
        st.write('• ' + limitation)
