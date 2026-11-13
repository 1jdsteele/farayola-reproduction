# Run 041 — Held-out ROC and corrected Reweighing + ROC

**Data protocol:** COMPAS (5,278 observations), race-stratified 70% training, 15% ROC calibration, 15% final testing; seeds 101–110 (or the selected smoke-test seeds). Calibration/test halves of the original 30% outer holdout are stratified by race and label.

**Four arms:** A/B supply hard 0/1 scores to ROC, while C/D supply continuous logistic-regression probabilities. B/D use Reweighing weights in classifier training.

**Strict separation:** ROC selects the threshold and uncertainty margin exclusively using the calibration set. Test labels are used only afterward to evaluate outcomes.

## Held-out final metrics

| arm | accuracy | statistical_parity_difference | disparate_impact | equal_opportunity_difference | predictive_equality_difference |
| --- | --- | --- | --- | --- | --- |
| A: ROC / hard labels (no Re) | 0.683965 | 0.242966 | 1.928428 | 0.234747 | 0.170725 |
| B: Re + ROC / hard labels | 0.675379 | 0.041686 | 1.107255 | 0.018795 | -0.020020 |
| C: ROC / probability scores (no Re) | 0.601641 | 0.142591 | 2.787578 | 0.140950 | 0.104162 |
| D: Re + ROC / probability scores | 0.671465 | 0.062708 | 1.162171 | 0.046379 | -0.003225 |

## EOD ±0.05 feasibility

| arm | calibration_roc_feasible | test_roc_feasible | roc_warning_no_feasible_on_calibration |
| --- | --- | --- | --- |
| A: ROC / hard labels (no Re) | 0 | 0 | 10 |
| B: Re + ROC / hard labels | 7 | 8 | 3 |
| C: ROC / probability scores (no Re) | 1 | 0 | 9 |
| D: Re + ROC / probability scores | 10 | 4 | 0 |

## ROC parameters and test adjustments (means)

| arm | roc_threshold | roc_margin | test_roc_changes_from_lr05 | test_threshold_only_changes_from_lr05 | test_margin_only_changes_from_threshold | test_correct_to_incorrect_from_lr05 | test_incorrect_to_correct_from_lr05 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A: ROC / hard labels (no Re) | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| B: Re + ROC / hard labels | 0.300 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C: ROC / probability scores (no Re) | 0.605 | 0.000 | 255.400 | 255.400 | 0.000 | 160.300 | 95.100 |
| D: Re + ROC / probability scores | 0.469 | 0.011 | 53.900 | 51.400 | 14.700 | 28.500 | 25.400 |

## Mean absolute held-out fairness distance from ideal

| arm | test_absolute_eod | test_absolute_spd | test_absolute_ped | test_di_distance |
| --- | --- | --- | --- | --- |
| A: ROC / hard labels (no Re) | 0.23475 | 0.24297 | 0.17073 | 0.92843 |
| B: Re + ROC / hard labels | 0.03167 | 0.04169 | 0.03050 | 0.10725 |
| C: ROC / probability scores (no Re) | 0.14095 | 0.14259 | 0.10416 | 1.78758 |
| D: Re + ROC / probability scores | 0.06065 | 0.06271 | 0.03916 | 0.16217 |

## Paired held-out accuracy differences (percentage points)

| comparison | mean_delta | ci95_lower | ci95_upper |
| --- | --- | --- | --- |
| B − A: weights added, hard ROC scores | -0.859 | -1.477 | -0.240 |
| C − A: continuous scores, no Re | -8.232 | -10.906 | -5.559 |
| D − B: continuous scores after Re | -0.391 | -1.096 | 0.313 |
| D − C: Re weights with continuous scores | 6.982 | 4.494 | 9.471 |
| D − A: both fixes vs original | -1.250 | -2.149 | -0.351 |

## Interpretation cautions

- Feasibility on the ROC calibration data is not the same as feasibility on held-out test data.
- ROC may fall back to a candidate when no calibration candidate satisfies EOD ±0.05; examine warnings and feasibility by arm.
- Threshold-only and margin-only changes can overlap or cancel; do not add their counts to obtain total changes.
- A/B hard-score arms intentionally reproduce the released integration; C/D supply usable continuous confidence scores.
- Runs 040 and 041 use different evaluation protocols and row counts. Compare interventions *within* Run 041.
- Ten seeds reuse overlapping observations; seed-based 95% t intervals describe seed variation, not independent-population confidence.
- The 15% test samples are unseen for model fitting and ROC calibration within each seed, but are not an external-dataset validation.
- Do not choose new model settings using the final held-out test metrics.

**Source data:** `per_seed_results_A.csv`, `prediction_level_A.csv`, `analysis_paired_intervals_A.csv`.
