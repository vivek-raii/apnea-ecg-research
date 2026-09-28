# Model card: ECG apnea research baseline

## Status

The end-to-end research pipeline is implemented, trained and verified. **The resulting detector does not establish useful screening performance.** It is a research baseline, not a medical product or a validated diagnostic model.

## Data and task

- Source: PhysioNet Apnea-ECG v1.0.0, supplied local ZIP.
- Inputs: one complete 60-second, single-channel, 100 Hz ECG segment in mV.
- Target: supplied minute-spaced A/N annotation, with archive-documented state at the beginning of the associated minute.
- Training/evaluation pool: 16,556 complete labeled windows, 34 recordings after excluding c06, 24 conservative candidate subject groups.
- Development: 12,224 windows, 25 recordings, 18 candidate groups.
- Holdout: 4,332 windows, 9 recordings, 6 candidate groups.
- Grouping is a heuristic based on demographic matches plus a known duplicate relationship. True patient independence is unverified.

## Model and selection

Random forest with 180 trees, minimum leaf size 8, square-root feature sampling and balanced bootstrap class weighting. Median feature imputation is fitted on the applicable training partition. Twenty-five ECG-derived features are used; no patient demographics, recording IDs, AHI or labels are model inputs.

Logistic regression, random forest and histogram gradient boosting were compared on development-only, three-fold grouped cross-validation. Random forest had the highest pooled out-of-fold average precision (0.759). Threshold 0.33 maximized development out-of-fold balanced accuracy on the pre-specified grid. The holdout was not used to select a model, threshold or hyperparameters.

## Independent held-out evaluation of the development-only artifact

| Metric | Value |
|---|---:|
| Accuracy | 53.1% |
| Balanced accuracy | 52.6% |
| Sensitivity | 33.0% |
| Specificity | 72.3% |
| Precision | 53.1% |
| F1 | 0.407 |
| AUROC | 0.623 |
| Average precision | 0.541 |
| Held-out positive prevalence | 48.8% |

Always predicting N yields 51.2% accuracy and 50% balanced accuracy on this same holdout. The detector misses 1,415 of 2,112 positive minutes. Balanced accuracy's group-bootstrap 95% interval is approximately 41.0–66.2%; AUROC's is 44.2–76.0%. Six candidate groups provide limited evidence and unstable uncertainty estimates.

Development out-of-fold balanced accuracy was 80.6% and AUROC was 0.875. The marked drop on held-out groups shows weak generalization in this experiment. Its cause has not been established; potential explanations to investigate include recording-specific features, peak-detection errors and population/recording differences. It must not be reported as an 80%-accurate independent detector.

## Quality and robustness

Fixed quality rules retain 98.0% of holdout windows. Accepted-only balanced accuracy is still 53.0%, so these rules do not resolve the generalization problem.

A 135-window held-out stress sample was assessed clean and with synthetic white noise. Some scores increased with stronger noise. This small experiment does not demonstrate that adding noise improves real-world detection; it shows that outputs and screening behavior are sensitive to the perturbation. No noise setting was used to modify or select the trained detector. Real sensor artifacts require separate study.

## Artifacts

- `evaluation_model.joblib`: fit only on development data, responsible for the reported holdout results.
- `final_model.joblib`: selected model refit on all 34 eligible labeled records. This artifact has no separate independent performance estimate.
- `x01_predictions.json`: inference demonstration only; minute-level reference labels are unavailable for x01 in this archive.

Both models emit uncalibrated scores. Neither score should be interpreted as an individual's clinical probability of disease. Loading joblib/pickle files executes serialized Python objects; load only trusted files.

## Intended use and next research step

Use for learning, reproducing the baseline, examining errors and developing a prospectively specified follow-up experiment. Do not use for diagnosis, treatment decisions, autonomous screening, sleep-apnea severity estimates, or an AIIMS population claim. The full-minute input means results are retrospective and do not demonstrate advance prediction.

Before a stronger claim: verify subject identities, validate peak detections on reviewed ECGs, obtain an independent evaluation dataset, and compare a specific improvement with a frozen protocol. This project's preprocessing, noise experiment and known duplicate removal are not claimed as algorithmic novelty.

Dataset citation: T. Penzel, G. B. Moody, R. G. Mark, A. L. Goldberger, J. H. Peter. *The Apnea-ECG Database*. Computers in Cardiology 2000;27:255–258. https://physionet.org/content/apnea-ecg/1.0.0/
