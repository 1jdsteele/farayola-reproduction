# Run 038 — Held-out Equalized Odds vs. corrected Reweighing + EO

**Protocol:** Within each seed, 70% of COMPAS observations train logistic regression and Reweighing; 15% separately calibrate the Equalized Odds (EO) policy; and 15% are held out for the final evaluation. These sets do not overlap. Both branches use the same splits and EO seed.

**Scientific distinction from Run 037:** Run 037 used the same 30% dataset for EO calibration and evaluation. Run 038 does not. The two experiments therefore must not be compared as paired treatment effects; the valid Run 038 comparison is *within Run 038*.

## Held-out test results, after EO

| Metric | Control EO (unweighted) | Corrected Re + EO | Paired difference (corrected − control) | 95% seed-based t interval |
|---|---:|---:|---:|---:|
| Accuracy | 0.643056 | 0.662753 | +0.019697 | [+0.009837, +0.029556] |
| DI | 1.140417 | 1.124038 | -0.016379 | [-0.095331, +0.062573] |
| SPD | 0.046208 | 0.048098 | +0.001891 | [-0.028276, +0.032057] |
| EOD | 0.004713 | 0.005229 | +0.000517 | [-0.047596, +0.048629] |
| PED | 0.015692 | 0.008878 | -0.006814 | [-0.036008, +0.022379] |

## Upstream classifier on the same held-out test observations

| Metric | Unweighted logistic regression | Weighted logistic regression | Difference (weighted − unweighted) |
|---|---:|---:|---:|
| Accuracy | 0.683965 | 0.675379 | -0.008586 |
| DI | 1.928428 | 1.107255 | -0.821173 |
| SPD | 0.242966 | 0.041686 | -0.201280 |
| EOD | 0.234747 | 0.018795 | -0.215952 |
| PED | 0.170725 | -0.020020 | -0.190745 |

## Prediction and optimization diagnostics

- All group/label combinations present; EO optimization successful: 10/10 seeds.
- Frozen Run 008 upstream model matches: 10/10 seeds.
- Reweighing weights match frozen Run 010: 10/10 seeds.
- Test-seed predictions changed before EO: 10/10; mean changed 74.6/792.
- Test-seed predictions changed after EO: 10/10; mean changed 150.0/792.
- Accuracy improvements after EO: 9/10 seeds; decreases: 1/10; ties: 0/10.

## Mean absolute fairness distance from ideal (held-out test)

| Metric | Control EO | Corrected Re + EO | Mean paired change in distance |
|---|---:|---:|---:|
| DI | 0.140417 | 0.138341 | -0.002077 |
| SPD | 0.046208 | 0.053791 | +0.007584 |
| EOD | 0.042360 | 0.045629 | +0.003270 |
| PED | 0.025518 | 0.040764 | +0.015246 |

## Calibration-versus-test diagnostic (post-EO accuracy)

| Branch | EO calibration-set accuracy (optimistic / in-sample) | Held-out test accuracy |
|---|---:|---:|
| EO | 0.633712 | 0.643056 |
| Corrected Re + EO | 0.649369 | 0.662753 |

## Interpretation cautions

- These are held-out **within the reconstructed COMPAS cohort**, not external-dataset validation.
- Different seeds share many training and evaluated observations; the ten paired seed estimates are not independent population samples. Intervals summarize seed variation rather than population uncertainty.
- EO is randomized; fitted policy parameters and realized finite-sample fairness can differ. No improvement is presumed.
- Different fairness definitions may conflict. Prefer mean absolute distance from ideal when interpreting directional fairness differences.
- Maintain the fixed protocol for this experiment; do not tune on the held-out test results. Future model selection requires nested validation.

*Source result file: `/Users/jsteele/Desktop/gradSchool/fairness_thesis/repos/farayola-reproduction/runs/run_038_heldout_re_eo/metadata/per_seed_results_A.csv`*

