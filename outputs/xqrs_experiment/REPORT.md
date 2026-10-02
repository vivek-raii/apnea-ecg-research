# XQRS experiment — 2 October 2026

## Result

No improvement. The development selection retained the original random forest.

Held-out accuracy: 53.12%; balanced accuracy: 52.63%; sensitivity: 33.00%.

## Method

Identical records, labels, group splits and folds to the baseline. Three fixed feature variants times three existing models. Select by development OOF average precision; select threshold using development balanced accuracy. All preprocessing fitted inside folds. XQRS is run independently per minute and aligned to local filtered ECG extrema within 200 ms. Relative features remove amplitude and mean-RR scale without fitting to labels.

| Candidate | Development AP | Development balanced accuracy |
|---|---:|---:|
| baseline/logistic_regression | 0.7540 | 79.28% |
| baseline/random_forest | 0.7590 | 80.57% |
| baseline/hist_gradient_boosting | 0.7159 | 77.12% |
| xqrs/logistic_regression | 0.6616 | 73.70% |
| xqrs/random_forest | 0.7182 | 76.85% |
| xqrs/hist_gradient_boosting | 0.6585 | 74.87% |
| xqrs_relative/logistic_regression | 0.6538 | 75.00% |
| xqrs_relative/random_forest | 0.6248 | 73.44% |
| xqrs_relative/hist_gradient_boosting | 0.6175 | 72.71% |

## Interpretation

XQRS was worse on development selection; its losing variants were not evaluated on the holdout. No replacement of the production model was justified. No increase in accuracy is claimed.

Previously examined holdout reused for comparison, not a fresh independent test.

Two synthetic tests verify timing on both signal polarities, flat signals and invalid inputs. The detector plot shows three preselected development examples, not clinical validation.

## Reproduce

Run `python src/experiment_xqrs.py` with the original dataset and local feature cache. It writes the protocol before extraction and development_selection.json before holdout evaluation. Dependencies: requirements-lock.txt (including WFDB).

Detector reference: https://wfdb.readthedocs.io/en/latest/processing.html#qrs-detectors

Next experiment: pre-specify past-only temporal context and evaluate on development folds; obtain new independent data before making a generalization claim.