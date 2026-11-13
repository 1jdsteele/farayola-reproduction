# Run 041 — Held-out Reweighing + Reject Option Classification

## Research objective

Test whether the promising **Run 040** ROC outcomes survive on observations that were **not used to choose ROC's classification threshold and reject-option margin**. Run 040 fitted ROC and evaluated it on the same 30% set; Run 041 separates calibration and testing.

This is a controlled 2×2 design that isolates **weight propagation** and **ROC score availability**. The ordinary classifier throughout is `LogisticRegression(solver='lbfgs', max_iter=1000)`.

| Arm | Reweighing weights consumed by classifier? | ROC input `.scores` |
| --- | --- | --- |
| `A_hard_unweighted` | No | Hard 0/1 classifier predictions (historical integration) |
| `B_hard_weighted` | Yes | Hard 0/1 classifier predictions |
| `C_prob_unweighted` | No | Continuous `predict_proba` scores |
| `D_prob_weighted` | Yes | Continuous `predict_proba` scores |

**Important naming:** NONE refers to *no fairness interventions*. All four arms above include ROC by design (although ROC may make zero realized changes). A/B/C/D are arms, not four entirely separate datasets.

## Exact protocol

- Dataset: reconstructed COMPAS, **5,278 observations, 11 features**, SHA-256 `6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e`.
- Target: `two_year_recid = 1` favorable in the AIF360 representation used for this reproduction (do not confuse favorable coding with a normative judgment about reoffending).
- Protected group: race 0 unprivileged, race 1 privileged.
- Seeds: **101–110**.
- Training **3,694 (70%)**, ROC calibration **792 (15%)**, final held-out test **792 (15%)**.
- Outer 70/30 split exactly uses Run 040's race stratification and seed. Inner 15/15 split uses `random_state=seed+10000` and joint race/label stratification (as in Run 038).
- Fit Reweighing **only on training**, fit logistic regression on training (with/without `sample_weight`), then fit ROC **only on calibration labels and classifier ROC-input scores**. Apply the fixed ROC policy to the final test scores. **Final test labels are not available to ROC fitting or choice of threshold/margin**.
- All arms use Run 040 ROC parameters: `metric_name='Equal opportunity difference'`, tolerance `[-0.05, +0.05]`, thresholds 0.3 to 0.8 in 100 steps, and 50 margins per threshold.
- ROC searches candidate thresholds and margins on **calibration data**, selecting the highest balanced accuracy among candidates satisfying the EOD constraint. If none are feasible, AIF360 emits a warning and selects a fallback candidate. This is recorded; fallback must not be described as feasible.
- Evaluate each arm's **pre-ROC and post-ROC metrics on both calibration (in-sample) and test (held-out)**, with the two partitions clearly separated.

**Reference reproducibility:** The saved Run 008 and 010 CSVs are included as read-only *inputs* in `provenance/`. Before running ROC, the script checks the unweighted and weighted classifiers' confusion matrices and five metrics on the original full 30% outer cohort against these frozen references. It also checks Reweighing's weight distribution.

**Methodological distinction:** Do **not** directly subtract Run 040 results from Run 041 as if they were paired model effects. They share the outer training sets but use different ROC calibration and evaluation sets. Compare arms **within Run 041**.

## Mac commands — from your thesis repository root

```bash
cd ~/Desktop/gradSchool/fairness_thesis/repos/farayola-reproduction
source .venv/bin/activate

# Extract downloaded archive once. It contains a top-level run_041_heldout_re_roc/ folder.
unzip ~/Downloads/run_041_heldout_re_roc.zip -d runs/

# No AIF360 needed; checks all 10 splits and reference files.
python runs/run_041_heldout_re_roc/scripts/check_run_041_design.py

# 1-seed smoke test: four ROC arms; saves metadata tagged SMOKE.
python runs/run_041_heldout_re_roc/scripts/run_041_heldout_re_roc_10seed.py \
  --seeds 101 --tag SMOKE

# Only after the smoke test passes, execute the full 10-seed run.
python runs/run_041_heldout_re_roc/scripts/run_041_heldout_re_roc_10seed.py \
  --tag A

# Validate saved individual predictions against per-seed metrics and report results.
python runs/run_041_heldout_re_roc/scripts/analyze_run_041.py --tag A
```

The full experiment fits **four ROC arms × ten seeds**, with 100 thresholds and 50 margins per fit. It may be computationally substantial. This package intentionally does not contain any claimed numerical outcomes for Run 041.

The runner refuses to overwrite existing results under the same tag unless `--overwrite` is given explicitly. For a separately repeated run, prefer `--tag B`; for a different smoke check, use a different tag.

**TensorFlow/inFairness warnings:** AIF360 may warn that optional modules such as AdversarialDebiasing or inFairness are absent; they are not needed for this experiment. However, other exceptions, failed reference assertions or CSV validation errors must be investigated rather than bypassed.

## Output and interpretation

Saved to `metadata/`:

- `per_seed_results_A.csv`: arm×seed results, both calibration and test metrics, fit warnings, EOD feasibility, threshold, margin, prediction flip/error counts.
- `prediction_level_A.csv`: per-observation predictions, continuous scores, hard-score input, ROC output, true labels, split membership, and row identifiers.
- `split_membership_A.csv` and `group_label_counts_A.csv`: demonstrate data separation and group/label coverage.
- `aggregate_summary_A.csv`: held-out post-ROC means.
- `paired_deltas_A.csv`: within-seed arm contrasts.
- `run_manifest_A.json`: protocol, versions, hashes, flags.
- **After analysis:** `analysis_report_A.md` and `analysis_paired_intervals_A.csv`.

### Core comparisons

| Contrast | Isolated factor |
| --- | --- |
| B − A | Using Reweighing weights while retaining hard ROC scores |
| C − A | Using continuous ROC scores without Reweighing |
| D − B | Supplying continuous scores with Reweighing already active |
| D − C | Adding effective Reweighing when ROC already receives probability scores |
| D − A | Both corrections relative to the historical ROC configuration |

### Questions the final report answers

1. Does **D** still achieve good held-out accuracy relative to B/C/A?
2. How often does each configuration satisfy **EOD ±0.05 on calibration** versus **held-out test**? Does a policy calibrated as feasible fail the test constraint?
3. Does ROC actively adjust predictions on test data? How much comes from threshold changes versus the reject-option margin?
4. Does adding Reweighing improve ROC's held-out accuracy/fairness balance?
5. Does the pattern from Run 040 survive out-of-sample calibration, or was it largely due to threshold/margin tuning on evaluation observations?

### Limitations

- This is *held-out within the reconstructed COMPAS cohort*, not an external-dataset replication.
- Ten seeds share many observations; seed-based t intervals are **descriptive across seeds**, not independent-sample population confidence intervals.
- EOD is only one fairness criterion. SPD/DI/PED can worsen even when EOD is within tolerance.
- Some arms may fail calibration feasibility. Do not interpret their selected fallback thresholds as satisfying the constraint.
- Do not tune thresholds or select a preferred arm based on the final held-out test rows; additional selection would require a nested design.
- This run is **not frozen** until its complete outputs are reviewed, checksummed, and committed.
