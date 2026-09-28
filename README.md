# ECG-based apnea detection research project

A complete local research pipeline: archive audit → minute-level ECG features → grouped model comparison → held-out evaluation → saved models → inference and localhost viewer.

The raw archive and earlier notebooks are preserved. The trained artifacts and commented notebook are under `outputs/`.

For a GitHub clone, download the original PhysioNet archive separately and update `dataset_zip` in `config.json`. The repository includes the trained models and saved notebook outputs, but excludes the local virtual environment and derived `.pkl` feature caches. Run training once to regenerate those caches before rerunning every notebook cell. Never interpret the development score as independent test performance: the selected baseline achieved only 52.6% balanced accuracy and 33.0% sensitivity on held-out recording groups.

## Start here

1. Open `outputs/results_report.html` in your browser for results and plots.
2. Open `outputs/Apnea_ECG_End_to_End.ipynb` in Jupyter or VS Code for the explained workflow and saved outputs.
3. To run predictions or repeat training, use the commands below in PowerShell. Use the included environment, not an unrelated Python kernel.

```powershell
Set-Location 'D:\Mtech Research\apnea_project'

# Predict a complete original test recording; reference minute labels are unavailable.
.\.venv\Scripts\python.exe src\predict.py --record x01 --output outputs\x01_predictions.json

# Start the local viewer. Open http://127.0.0.1:8765 and keep this terminal running.
.\.venv\Scripts\python.exe src\viewer.py

# Reproduce training (reuses verified feature cache).
.\.venv\Scripts\python.exe -u src\train.py

# Re-extract after changing feature definitions.
.\.venv\Scripts\python.exe -u src\train.py --force-extract

# Run verification.
.\.venv\Scripts\python.exe tests\test_pipeline.py
```

For the notebook, choose `D:\Mtech Research\apnea_project\.venv\Scripts\python.exe` as the Python environment in your notebook editor. Saved tables/charts are visible even before a kernel is selected. If your Jupyter installation requires a named kernel, explicitly register it yourself using `.venv\Scripts\python.exe -m ipykernel install --user --name apnea --display-name "Apnea project"`.

## External ECG inference

```powershell
.\.venv\Scripts\python.exe src\predict.py --wfdb 'D:\your_data\record_name' --output outputs\external_predictions.json
```

The external WFDB base path omits `.hea`/`.dat`. Input must have exactly one ECG channel at 100 Hz in mV with finite values. Resampling, channel selection, missing-sample interpolation and unit conversion are deliberately not performed implicitly. Predictions use complete 60-second windows and ignore a final incomplete segment. They require all samples in the minute; this is not early prediction or a causal sample-by-sample detector.

Outputs contain record, minute, score, predicted A/N, reference label if present, and signal-quality status. Scores are **uncalibrated model scores**, not clinical risk probabilities. Quality warnings remain visible; the classifier still returns an experimental score, but those windows should be reviewed.

## Project structure

```text
apnea_project/
  config.json                  frozen experiment settings and raw ZIP path
  src/ecg.py                   archive reader, grouping, window-local features
  src/train.py                 extraction, development CV, holdout, final refit
  src/report.py                result tables and Matplotlib figures
  src/predict.py               inference command
  src/viewer.py                localhost prediction viewer
  tests/test_pipeline.py       independent decoding and leakage/inference checks
  outputs/                     trained models, notebook, metrics, charts
  work/                        notebook builder and project-local kernel files
  .venv/                       isolated Python 3.12 environment
```

## Method and pre-specified decisions

- Main labels: 35 learning records from the supplied ZIP. Exclude c06, whose signal overlaps c05 after 80 seconds. Keep complete labeled windows from the remaining 34 records.
- No model input contains recording ID, age, sex, height, weight, AHI, apnea totals or labels. Metadata is used only for conservative grouping.
- Candidate groups connect records sharing age/sex/height/weight across the full archive, plus the known duplicate link. Equal demographics do not prove identity; differing demographics do not prove independence. This is **not verified patient-independent validation**.
- Fixed `GroupShuffleSplit`, seed 42, holds out 25% of candidate groups. Development uses three-fold `StratifiedGroupKFold` with the same seed. The exact records and groups are saved before fitting.
- Three fixed candidates: logistic regression, random forest, histogram gradient boosting. No test-driven hyperparameter search.
- Imputation and scaling are fitted inside each development training fold. Model selection uses pooled development out-of-fold average precision; threshold selection uses development balanced accuracy on a fixed 0.05–0.95 grid. Development metrics are selection estimates, not unbiased final performance.
- Primary evaluation includes all complete holdout windows. Accepted-only quality analysis is secondary and reports coverage. Quality thresholds are fixed before evaluation.
- Percentile confidence intervals use 500 bootstrap resamples of entire held-out candidate groups, skipping one-class resamples. With few groups, intervals are unstable.
- Controlled robustness study uses up to 15 evenly spaced windows per held-out recording, evaluated clean and with 20/10 dB synthetic white noise. This is not a clinical artifact benchmark, and was not used to retune the model.

## Features and limitations

Twenty-five ECG-only features describe signal variation, spectral content, candidate peak count, RR interval summaries, amplitude variation and quality. The simple energy-based peak detector is a transparent baseline, not a validated clinical detector. The archive's supplied `.qrs` annotations are not used as ground truth. Short-window spectral features are exploratory and should not be interpreted as clinical HRV measurements. Zero-phase filtering is confined to each complete minute; it still uses future samples within that minute.

The exact A/N label semantics follow the archive's corrected `annotations.shtml`: apnea state at the beginning of the associated minute. They do not provide precise event durations. AHI cannot be inferred just by counting positive minutes.

## Which model should I use?

- `evaluation_model.joblib`: trained only on the development partition; this is the artifact behind the held-out results.
- `final_model.joblib`: selected estimator refitted on all eligible labeled records for future research predictions. Its performance has **not** been independently tested after the refit.

Only open trusted project-generated `.joblib` and `.pkl` files; they use executable pickle serialization. The original x-record labels are not present in this ZIP, so x-record predictions do not establish test accuracy. Some x-records may share subjects with training records.

## Re-create the environment elsewhere

Use Python 3.12. Edit `config.json` to point to your copy of the original ZIP.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe src\train.py
```

If Python 3.12 is not installed, install it before these commands. The environment bundled here is specific to this machine and should not be copied as a portable dependency bundle. Exact versions and platform details are recorded in `outputs/environment.json` and `requirements-lock.txt`.

## Research contribution and next steps

This delivers a reproducible baseline and a quality/noise evaluation, not a new clinical algorithm or a publication claim. An appropriate follow-up is to evaluate a better peak detector or quality-aware method on a new, independent evaluation set, with a literature review and clinician-reviewed annotations. Additional clinical data and verified participant identities are required for stronger generalization claims. No hospital integration, deployment, or clinical effectiveness is claimed.

References: [PhysioNet Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/), [WFDB annotation format](https://physionet.org/physiotools/wag/annot-5.htm), [scikit-learn grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data). Cite the original dataset publication and comply with its attribution license when sharing derived work.
