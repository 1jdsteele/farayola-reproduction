# Run 042 — Reweighing + EGR (Approach A: oracle-weight integration)

## Research question

Can the instance weights produced by **AIF360 Reweighing** influence the
combined **Re + Exponentiated Gradient Reduction (EGR)** pipeline when they
are actually used in EGR's repeated, cost-sensitive logistic-regression
training calls?

This experiment **does not** implement fully weighted Fairlearn EGR; it only
changes the weights given to EGR's internal base classifiers. The fairness
moments, objective, dual updates and selection of the randomized classifier
remain EGR's original unweighted machinery. This is the scoped Approach A
decided during the source-code audit. Because Fairlearn's best-response
oracle is no longer solving precisely the original unweighted Lagrangian,
**its usual convergence and fairness guarantees should not be assumed**
for this experimental variant. In particular, the Reweighing weights were
computed against the original target labels, while EGR may pass relabeled
cost-sensitive targets to its oracles.

## What was audited

- AIF360 0.6.1 `aif360.algorithms.inprocessing.ExponentiatedGradientReduction.fit`
  passes features and labels, but not `instance_weights`, to its sklearn wrapper.
- AIF360 sklearn wrapper invokes Fairlearn reductions EGR.
- Fairlearn 0.14.0 `_Lagrangian._call_oracle()` repeatedly sends a `sample_weight`
  vector to cloned estimators, including logistic regression. This provides
  the narrow integration point.
- The original reproduction's **Run 011 EGR** and **Run 014 Re+EGR** are
  identical across all ten seeds in the frozen references.

Pinned references:

- https://raw.githubusercontent.com/Trusted-AI/AIF360/v0.6.1/aif360/sklearn/inprocessing/exponentiated_gradient_reduction.py
- https://raw.githubusercontent.com/fairlearn/fairlearn/v0.14.0/fairlearn/reductions/_exponentiated_gradient/_lagrangian.py

## Method

For every EGR internal logistic-regression oracle call:

```
w_combined[i] = w_egr_internal[i] * w_reweighing[i]
w_combined *= sum(w_egr_internal) / sum(w_combined)
```

The second step keeps the total oracle sample-weight mass unchanged, avoiding
an unintended change in the strength of logistic-regression regularization
from global weight scaling. **Relative** weights, and therefore the models
fitted by each oracle call, can change. Positive weights are aligned by the
original row order, and a complete SHA-256 feature fingerprint is checked
inside every cloned oracle's `fit()` call.

## Arms

| Arm | Meaning |
|---|---|
| `legacy_egr` | Original AIF360 EGR with standard logistic regression; must match frozen Run 011 and Run 014 |
| `unity_oracle` | Same wrapper with all external weights exactly 1; must give *identical individual predictions* to original EGR |
| `reweighted_oracle` | Wrapper multiplying EGR's internal weights by real Reweighing instance weights before each logistic-regression fit |

**The unity control is a mandatory fail-fast check.** If it diverges, do not
interpret or freeze the Run 042 results until the discrepancy is resolved.

## Protocol

- COMPAS numeric CSV: 5,278 rows; SHA-256 `6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e`
- Features and target: original reproduction (11 features, `two_year_recid`)
- Group: race 0 (unprivileged) / race 1 (privileged)
- Split: 70% train, 30% test, stratify on race
- Seeds: 101–110
- EGR: DemographicParity, eps=0.02, max_iter=20, drop_prot_attr=False
- LR base classifier: `solver='lbfgs', max_iter=1000`
- Reweighing applies only to training; evaluation remains unweighted
- Random EGR predictions: NumPy reset to experiment seed immediately before fit and predict, consistent with Runs 011 and 014
- **Test observations are not used in EGR fitting**, unlike the original-protocol EO/ROC calibration runs.

## Files

- `data/compas_farayola_numeric.csv` — pinned COMPAS data
- `provenance/` — frozen Run 011 and Run 014 per-seed results used as controls
- `scripts/weighted_oracle.py` — clone-compatible logistic regression oracle wrapper
- `scripts/check_run_042_design.py` — offline preflight and unit checks
- `scripts/run_042_re_egr_oracle_weighting_10seed.py` — model experiment
- `scripts/analyze_run_042.py` — validates saved predictions/fit traces, computes paired summaries and report
- `metadata/` — generated outputs

## Running on your Mac

From the `farayola-reproduction` repository root, with the thesis `.venv` activated:

```bash
unzip ~/Downloads/run_042_re_egr_oracle_weighting.zip -d runs/
python runs/run_042_re_egr_oracle_weighting/scripts/check_run_042_design.py
python runs/run_042_re_egr_oracle_weighting/scripts/run_042_re_egr_oracle_weighting_10seed.py --seeds 101 --tag SMOKE
python runs/run_042_re_egr_oracle_weighting/scripts/run_042_re_egr_oracle_weighting_10seed.py --tag A
python runs/run_042_re_egr_oracle_weighting/scripts/analyze_run_042.py --tag A
```

The model script requires **AIF360 0.6.1**, **Fairlearn 0.14.0** and
**scikit-learn 1.9.1** and stops if these differ. Missing TensorFlow and
inFairness are nonblocking warnings for this experiment.

## Outputs

- `per_seed_results_A.csv` — EGR test metrics and oracle diagnostics by seed and arm
- `aggregate_summary_A.csv` — means and standard deviations
- `prediction_level_A.csv` — individual held-out predictions
- `oracle_fit_traces_A.csv` — each internal LR oracle fit, original and combined sample-weight diagnostics
- `run_manifest_A.json` — environment and protocol
- `prediction_change_checks_A.csv` — validation of saved individual predictions
- `paired_deltas_A.csv` — weighted minus original EGR by seed, with t intervals
- `analysis_report_A.md` — compact professor-facing summary

## Stopping rules

Stop and share output if any dataset, version, split, reference control,
feature-alignment, weight-integrity, oracle-count or unity-control assertion
fails. **Do not weaken or delete the assertions.** The run is not ready for
freezing until its analysis and prediction-level validation pass.

The test-set results answer whether *this specific oracle integration* has
an effect on COMPAS; they do not prove fully weighted EGR is implemented nor
that any combined approach improves fairness on other data.
