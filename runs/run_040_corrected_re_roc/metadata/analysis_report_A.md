# Run 040 — Corrected Reweighing + Reject Option Classification

**Protocol:** 70/30 race-stratified train/test split, ten seeds (101–110); ROC is fitted and evaluated on the SAME test observations to isolate the original integration decisions. This is not unbiased held-out evaluation.

Four ROC-containing configurations isolate two factors: whether Reweighing weights reach logistic regression and whether ROC receives continuous prediction probabilities or hard 0/1 labels.

## Mean post-ROC metrics

| arm                                  |   accuracy |   statistical_parity_difference |   disparate_impact |   equal_opportunity_difference |   predictive_equality_difference |
|:-------------------------------------|-----------:|--------------------------------:|-------------------:|-------------------------------:|---------------------------------:|
| A: original ROC (hard scores, no Re) |   0.679356 |                        0.235996 |            1.88566 |                       0.228105 |                         0.165611 |
| B: Re weights only (hard scores)     |   0.668876 |                        0.036105 |            1.09377 |                       0.013214 |                        -0.023148 |
| C: probability scores only (no Re)   |   0.591288 |                        0.114503 |            2.69857 |                       0.107373 |                         0.085357 |
| D: Re weights and probability scores |   0.677273 |                        0.065151 |            1.1691  |                       0.038706 |                         0.005115 |

## Mean ROC adjustment diagnostics

| arm                                  |   roc_changes_from_lr05 |   threshold_only_changes_from_lr05 |   margin_only_changes_from_threshold |   roc_threshold |   roc_margin |   correct_to_incorrect_from_lr05 |   incorrect_to_correct_from_lr05 |
|:-------------------------------------|------------------------:|-----------------------------------:|-------------------------------------:|----------------:|-------------:|---------------------------------:|---------------------------------:|
| A: original ROC (hard scores, no Re) |                     0   |                                0   |                                  0   |          0.3    |       0      |                              0   |                              0   |
| B: Re weights only (hard scores)     |                     0   |                                0   |                                  0   |          0.3    |       0      |                              0   |                              0   |
| C: probability scores only (no Re)   |                   518.7 |                              518.7 |                                  0   |          0.6455 |       0      |                            329.1 |                            189.6 |
| D: Re weights and probability scores |                    88.7 |                               87.8 |                                 32.7 |          0.4828 |       0.0123 |                             37.7 |                             51   |

## Paired accuracy changes (percentage points)

| comparison                              |   mean_delta |   ci95_lower |   ci95_upper |
|:----------------------------------------|-------------:|-------------:|-------------:|
| B − A: weight fix, hard ROC scores      |      -1.048  |      -1.4062 |      -0.6897 |
| C − A: score fix, no Re                 |      -8.8068 |     -10.2467 |      -7.367  |
| D − B: score fix, with Re               |       0.8396 |       0.3551 |       1.3242 |
| D − C: weight fix, ROC scores available |       8.5985 |       7.2185 |       9.9785 |
| D − A: both fixes vs original           |      -0.2083 |      -0.6514 |       0.2347 |

## Key cautions

- A and B use hard 0/1 scores as in the original reproduction; C and D use actual `predict_proba` scores.
- ROC search uses EOD-only fairness constraints ±0.05 and may fall back if no feasible candidate is found.
- ROC calibration and evaluation use the same test observations; interpret as implementation diagnostics only.
- The ten seed estimates overlap in their underlying data. Seed intervals do not establish population confidence.
- Follow-up held-out calibration/test evaluation is needed before generalization claims.
- Larger final accuracy or fairness improvements are not guaranteed by either correction.
