# Run 043 — Held-out corrected Re + EGR + EO

## Protocol

5,278 COMPAS rows; 3,694 training, 792 EO calibration, 792 independent testing per seed; seeds [np.int64(101), np.int64(102), np.int64(103), np.int64(104), np.int64(105), np.int64(106), np.int64(107), np.int64(108), np.int64(109), np.int64(110)].
Reweighing modifies the per-oracle LR training weights in Fairlearn EGR only (Approach A).
EGR fairness moments/objective are not externally reweighted; the unity control reproduces legacy predictions exactly.
Original Runs 016 and 022 calibrate EO on their evaluation data and are **not** directly comparable to this held-out protocol.
EO targets equalized odds without an explicit ±0.05 threshold; descriptive gap thresholds below are not optimization feasibility.

## Primary held-out post-EO metrics

| arm | accuracy | statistical_parity_difference | disparate_impact | equal_opportunity_difference | predictive_equality_difference |
| --- | --- | --- | --- | --- | --- |
| legacy_egr | 0.665530 | 0.052469 | 1.127208 | 0.011260 | 0.010305 |
| reweighted_oracle | 0.655808 | 0.041455 | 1.102343 | 0.009464 | -0.003245 |
| unity_oracle | 0.665530 | 0.052469 | 1.127208 | 0.011260 | 0.010305 |

## Paired held-out differences (corrected − legacy)

| metric | mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| accuracy | -0.009722 | -0.024143 | 0.004698 |
| statistical_parity_difference | -0.011013 | -0.021304 | -0.000722 |
| disparate_impact | -0.024865 | -0.050832 | 0.001103 |
| equal_opportunity_difference | -0.001796 | -0.018875 | 0.015283 |
| predictive_equality_difference | -0.013550 | -0.031118 | 0.004018 |

Seed-based t intervals characterize ten correlated seed splits, not population confidence intervals.

## Mean absolute fairness gaps (held-out)

| arm | abs_DI_minus_1 | abs_equal_opportunity_difference | abs_predictive_equality_difference | abs_statistical_parity_difference |
| --- | --- | --- | --- | --- |
| legacy_egr | 0.134811 | 0.040027 | 0.034699 | 0.055672 |
| reweighted_oracle | 0.111461 | 0.038854 | 0.044480 | 0.045089 |
| unity_oracle | 0.134811 | 0.040027 | 0.034699 | 0.055672 |

## Calibration vs test descriptive fairness thresholds

| arm | split | n_seeds | post_eo_abs_EOD_le_0_05 | post_eo_abs_PED_le_0_05 | post_eo_both_abs_EOD_PED_le_0_05 |
| --- | --- | --- | --- | --- | --- |
| legacy_egr | calibration | 10 | 10 | 10 | 10 |
| legacy_egr | test | 10 | 7 | 7 | 4 |
| unity_oracle | calibration | 10 | 10 | 10 | 10 |
| unity_oracle | test | 10 | 7 | 7 | 4 |
| reweighted_oracle | calibration | 10 | 10 | 10 | 10 |
| reweighted_oracle | test | 10 | 5 | 6 | 2 |

## Pre-EO vs post-EO prediction changes

| arm | eo_changed_predictions | eo_harmful_flips | eo_helpful_flips | pre_eo_vs_legacy_changes | post_eo_vs_legacy_changes |
| --- | --- | --- | --- | --- | --- |
| legacy_egr | 35.500000 | 21.600000 | 13.900000 | 0.000000 | 0.000000 |
| reweighted_oracle | 38.000000 | 24.100000 | 13.900000 | 29.600000 | 85.100000 |
| unity_oracle | 35.500000 | 21.600000 | 13.900000 | 0.000000 | 0.000000 |

## Interpretation guardrails

- Changes of predictions demonstrate weight propagation, not guaranteed fairness improvement.
- EO may reduce disparities at the expense of accuracy; group metrics must be interpreted per seed and with absolute gaps.
- ±0.05 thresholds here are **descriptive only**, unlike the configured ROC EOD bound used in Runs 040/041.
- Calibration and test are disjoint; no final-test outcomes are supplied to EGR or EO fitting.
- The ten train/test splits overlap substantially, so nominal paired t intervals are not population-level CIs.
