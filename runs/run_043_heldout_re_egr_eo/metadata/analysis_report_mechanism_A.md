# Run 043 — EO mechanism diagnostic (read-only analysis)

No model is trained or refitted. Three saved observations per split/seed (legacy, unity, corrected) are replayed
using the pinned AIF360 0.6.1 EO seed/shuffle implementation; 60 actual EO applications exactly reproduced.
Held-out outcomes were not used to learn the EO policies. Counterfactual policy swaps do not constitute new model fits.

## Disagreement: held-out test (mean per seed of 792)
| Diagnostic | Mean |
| --- | --- |
| pre_branch_disagreement | 29.6000 |
| post_branch_disagreement | 85.1000 |
| extra_post_disagreement | 55.5000 |
| swap_classifier_under_legacy_policy | 80.3000 |
| swap_policy_on_legacy_predictions | 20.0000 |
| swap_classifier_under_weighted_policy | 83.3000 |
| swap_policy_on_weighted_predictions | 20.0000 |

The factor-swap counts are NOT additive; prediction changes and policy effects interact, and seeded shuffles select different observations.

## EO policy flip probabilities (mean of per-seed learned probabilities)
| arm | race | probability_1_to_0 | probability_0_to_1 |
| --- | --- | --- | --- |
| legacy_egr | 0 | 0.0314 | 0.1020 |
| legacy_egr | 1 | 0.0157 | 0.0000 |
| reweighted_oracle | 0 | 0.0505 | 0.0895 |
| reweighted_oracle | 1 | 0.0325 | 0.0000 |

## Pre-EO absolute calibration TPR/FPR gaps (mean of per-seed absolute differences)
| classifier | abs_TPR_gap | abs_FPR_gap |
| --- | --- | --- |
| legacy_egr | 0.0445 | 0.0758 |
| reweighted_oracle | 0.0399 | 0.0682 |

## Held-out EO changes by race (mean per seed)
| arm | race | flips | zero_to_one | one_to_zero | harmful | helpful |
| --- | --- | --- | --- | --- | --- | --- |
| legacy_egr | 0 | 33.5000 | 27.3000 | 6.2000 | 20.5000 | 13.0000 |
| legacy_egr | 1 | 2.0000 | 0.0000 | 2.0000 | 1.1000 | 0.9000 |
| reweighted_oracle | 0 | 33.8000 | 23.6000 | 10.2000 | 22.2000 | 11.6000 |
| reweighted_oracle | 1 | 4.2000 | 0.0000 | 4.2000 | 1.9000 | 2.3000 |
| unity_oracle | 0 | 33.5000 | 27.3000 | 6.2000 | 20.5000 | 13.0000 |
| unity_oracle | 1 | 2.0000 | 0.0000 | 2.0000 | 1.1000 | 0.9000 |

## Classifier predictions crossed with EO policies (held-out test)
| classifier | eo_policy | accuracy | changed_by_eo | differing_vs_original_post | SPD | DI | EOD | PED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_egr | legacy_egr | 0.6655 | 35.5000 | 0.0000 | 0.0525 | 1.1272 | 0.0113 | 0.0103 |
| legacy_egr | reweighted_oracle | 0.6660 | 38.3000 | 20.0000 | 0.0441 | 1.1087 | 0.0088 | -0.0031 |
| reweighted_oracle | legacy_egr | 0.6566 | 35.4000 | 80.3000 | 0.0503 | 1.1219 | 0.0161 | 0.0067 |
| reweighted_oracle | reweighted_oracle | 0.6558 | 38.0000 | 85.1000 | 0.0415 | 1.1023 | 0.0095 | -0.0032 |

Diagonal combinations are observed original pipelines. Off-diagonals are deterministic counterfactual replays:
apply a policy learned on one calibration prediction source to the other predictor's test predictions.

## Held-out mean absolute fairness deviations (mean across seeds)
| arm | abs_SPD | abs_DI_minus_one | abs_EOD | abs_PED |
| --- | --- | --- | --- | --- |
| legacy_egr | 0.0557 | 0.1348 | 0.0400 | 0.0347 |
| reweighted_oracle | 0.0451 | 0.1115 | 0.0389 | 0.0445 |

## Interpretation guardrails
- External Re weights enter only EGR internal learner training (Approach A), not EGR fairness moments.
- Differences in individual predictions can grow after EO even when average signed fairness metrics look similar.
- The 10 seed splits overlap; this diagnostic is descriptive and is not an independent population estimate.
- EO is randomized: this diagnostic exactly reproduces the recorded seed, not variance over new EO random seeds.
- Counterfactual cross-policy predictions are not new EO optimizations, and their effects need not add linearly.

## Input SHA-256

```json
{
  "per_seed_results": "0017a14c07f333202baf3fa9035e976b6268b26f5bf2774122ebb02cbe9f31d4",
  "prediction_change_checks": "f81f327477e3547f382b96a11f1a80b5cdd3c74bd2378f101fc99dfc940e20d8",
  "prediction_level": "b7e78b56ed93cd3e3ceae7089695e866dd2d01dd1aa94a29c0d78694925e978c"
}
```
