# Run 039 — EO mechanism diagnostic (analysis of frozen Run 038)

Run 039 **does not retrain models or refit Equalized Odds**. It analyzes saved
Run 038 predictions and EO adjustment parameters from seeds 101–110.
Both branches use 70% training / 15% calibration / 15% held-out test data.

## Calibration disparities, **before** EO (mean absolute gap across seeds)

| Metric | Without Re | With Re |
|---|---:|---:|
| TPR gap | 0.2214 | 0.0355 |
| FPR gap | 0.1605 | 0.0428 |

## EO changes on held-out test observations (mean per seed, 792 test rows)

| Diagnostic | EO alone | Re + EO |
|---|---:|---:|
| Changed predictions | 82.8 | 29.0 |
| Correct → incorrect | 57.6 | 19.5 |
| Incorrect → correct | 25.2 | 9.5 |
| Net additional errors | 32.4 | 10.0 |

## Classifier / EO-policy substitution: calculated changed predictions

| Classifier predictions | Policy learned without Re | Policy learned with Re |
|---|---:|---:|
| Unweighted LR predictions | 82.8 (observed) | 28.3 (counterfactual) |
| Weighted LR predictions | 70.4 (counterfactual) | 29.0 (observed) |

Off-diagonal counts are *analytic counterfactual calculations* from
predictions and saved policy probabilities; they are **not** separately
executed pipelines or claims about their accuracy or fairness. Diagonal
counts are checked against the original Run 038 observed counts.

## Interpretation

Reweighing substantially reduced TPR/FPR disparities on EO's calibration
set. EO then learned a different adjustment policy, made fewer changes on
the held-out test set, and introduced fewer net errors. This is evidence
consistent with the smaller EO accuracy penalty in the weighted pipeline;
it does **not** establish that reducing calibration disparities is the sole
causal mechanism. The ten seeds share observations, and realized EO
predictions depend on randomization.

## Frozen Run 038 source SHA-256

- `prediction_level_A.csv`: `487130aa31a32a05b143156d8ba5a43b123f331548985f8f6a8b979791d4d01b`
- `per_seed_results_A.csv`: `04a1bcbb49236f702f5309341c3f1dfdacdc6301592e070fe698cc9154a77a0e`

To reproduce: `python runs/run_039_eo_mechanism/scripts/analyze_run_039.py`
