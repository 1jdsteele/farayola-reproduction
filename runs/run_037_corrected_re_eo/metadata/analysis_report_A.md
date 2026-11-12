# Run 037 — Corrected Reweighing + EO

The original Run 018 and the corrected Run 037 were compared with the same split, seeds and EO calibration procedure. **EO was calibrated on the test set in both branches**, following the reproduction; these are not unbiased held-out test estimates.

| Metric | Original Re+EO (Run 018) | Corrected Re+EO (Run 037) | Paired change (037−018) | 95% t interval for change |
|---|---:|---:|---:|---:|
| Accuracy | 0.637942 | 0.657071 | +0.019129 | [+0.013302, +0.024955] |
| DI | 1.103279 | 1.098783 | -0.004496 | [-0.011017, +0.002025] |
| SPD | 0.034882 | 0.039403 | +0.004521 | [+0.003272, +0.005770] |
| EOD | -0.003885 | 0.000522 | +0.004407 | [-0.006988, +0.015802] |
| PED | 0.004535 | -0.001101 | -0.005636 | [-0.015650, +0.004378] |

## Prediction-level diagnostics

- Seeds with changes **before EO**: 10/10
- Seeds with changes **after EO**: 10/10
- Mean predictions changed before EO: 148.6 of 1584
- Mean predictions changed after EO: 304.3 of 1584
- Direct weighted pre-EO result matches frozen Run 010 metrics: 10/10 seeds

The Run 010 match is a diagnostic rather than a requirement, since Run 010 used a pandas input and Run 018/037 use AIF360 feature arrays.

## Fairness-distance diagnostics

| Metric | Mean absolute distance from ideal, original | Mean absolute distance from ideal, corrected |
|---|---:|---:|
| DI | 0.103279 | 0.098783 |
| SPD | 0.034882 | 0.039403 |
| EOD | 0.011337 | 0.003519 |
| PED | 0.010028 | 0.003445 |

**Interpretation:** Lower absolute distance from the ideal indicates better parity for each individual metric, but different fairness criteria can conflict. These estimates are descriptive across ten fixed experiment seeds, not independent population samples.

