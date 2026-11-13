# Run 043 — Held-out corrected Reweighing + EGR + Equalized Odds

## Research question

After incorporating Reweighing's external instance weights into EGR's
per-oracle LogisticRegression training (Run 042, **Approach A**), do the small
pre-processing changes persist, amplify, or disappear when followed by
Equalized Odds post-processing on **new, held-out test observations**?

This is **Run 043**, not another execution of completed Run 042.

## Exact experiment design

- Dataset: COMPAS `compas_farayola_numeric.csv`, SHA-256
  `6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e`.
- Ten seeds: 101–110; same race-stratified **70% EGR training** split as
  Runs 011, 014, and 042 (3,694 observations).
- Divide the original 30% outer evaluation data into a **15% EO calibration**
  partition (792 observations) and a separate **15% final test** partition
  (792 observations). The split uses `random_state=seed+10000` and joint
  race/target stratification, matching the design convention in Run 038.
- EGR: `ExponentiatedGradientReduction` with `DemographicParity`, `eps=0.02`,
  `max_iter=20`, `drop_prot_attr=False`; base `LogisticRegression(solver="lbfgs", max_iter=1000)`.
- EGR is a randomized mixture. Each arm is fitted with `np.random.seed(seed)`;
  a separate outer-holdout prediction with `np.random.seed(seed)` reproduces
  the frozen Run 011/042 protocol. Calibration predictions are seeded with
  `seed+20000` and held-out test predictions with `seed+30000`. These
  prediction draws need not equal subsets of the outer-holdout draw.
- **EO policy:** `EqOddsPostprocessing(..., seed=seed)` is fitted using
  `calibration` **labels and predictions only**, then applied to test
  predictions without providing test labels to its fitting procedure.
- **EO has no configured ±0.05 feasibility tolerance**. Any threshold-based
  counts in the analysis are *descriptive*, unlike the ROC feasibility
  criterion in Runs 040–041.

### Three controlled branches

| Arm name | What is trained | Post-processing |
|---|---|---|
| `legacy_egr` | Original AIF360 EGR oracle; no Re | EO fitted on its calibration predictions |
| `unity_oracle` | Custom Run 042 EGR oracle, external weights all exactly 1 | Identical EO procedure; must be prediction-identical to `legacy_egr` |
| `reweighted_oracle` | Run 042 Approach A: Re weights multiplied into **EGR internal LR sample weights**, then renormalized to preserve each oracle call's original training-weight mass | EO fitted independently on its calibration predictions |

**Scope warning:** Approach A modifies *only the LR training oracles*. EGR's
fairness moments and objective are **not** externally reweighted. Do not label
this a fully weighted EGR implementation.

### Critical controls

1. Original Runs **011 and 014** have exactly matching original EGR metrics;
   original Runs **016 and 022** have exactly matching original post-EO
   metrics. Frozen copies of all four per-seed results are bundled.
2. Recompute original outer 30%-holdout EGR predictions using the original
   RNG protocol and **require** agreement with frozen Runs 011/014.
3. Unity-weight EGR **must** reproduce legacy EGR predictions exactly on the
   outer, calibration, and final test data; their EO outputs must also match.
4. If a sibling `runs/run_042_re_egr_oracle_weighting/metadata/prediction_level_A.csv`
   exists, require **all three** outer-holdout EGR predictions to match Run
   042's saved prediction-level outputs. If that file does not exist, this
   extra check is skipped and clearly recorded (the original Run 011/014
   controls are still required).
5. Verify that Re changes each fitted EGR oracle's training weights, that
   their total mass stays fixed, and that the oracle's training row order is
   unchanged. `weighted_oracle.py` is copied **unchanged** from Run 042.
6. Final analysis independently recomputes **all five metrics** from
   prediction-level CSVs and verifies split membership, control equality,
   prediction flips, oracle traces, and reported counts.

**Historical comparison:** Runs 016/022 calibrated EO on the same 30% used for
reporting outcomes. Run 043's test statistics are evaluated on a different,
independent partition. **Do not compare Run 043 metrics directly against
Runs 016/022 as a paired treatment effect.** Compare Run 043 arms within
this experiment.

## Mac commands

From the thesis repository root:

```bash
cd ~/Desktop/gradSchool/fairness_thesis/repos/farayola-reproduction
source .venv/bin/activate
unzip ~/Downloads/run_043_heldout_re_egr_eo.zip -d runs/
```

The ZIP has the `run_043_heldout_re_egr_eo/` top-level folder and therefore
extracts directly into `runs/run_043_heldout_re_egr_eo/`.

**1. Offline design audit** (no EGR optimization):

```bash
python runs/run_043_heldout_re_egr_eo/scripts/check_run_043_design.py
```

Expected final line:

`RUN 043 DESIGN CHECK PASSED (no EGR/EO optimization executed locally).`

**2. One-seed smoke test** (actual EGR + EO fitting):

```bash
python runs/run_043_heldout_re_egr_eo/scripts/run_043_heldout_re_egr_eo_10seed.py \
  --seeds 101 --tag SMOKE
```

If an assertion fails, **stop**: we should review the environment,
EGR reproducibility, EO optimization, or alignment before running all seeds.

**3. All ten seeds:**

```bash
python runs/run_043_heldout_re_egr_eo/scripts/run_043_heldout_re_egr_eo_10seed.py \
  --tag A
```

**4. Validation and analysis:**

```bash
python runs/run_043_heldout_re_egr_eo/scripts/analyze_run_043.py --tag A
```

The final summary is:

`runs/run_043_heldout_re_egr_eo/metadata/analysis_report_A.md`

Please share the console output and/or the report before freezing Run 043.

## Output files

- `metadata/per_seed_results_A.csv`: 3 arms × 2 partitions × 10 seeds;
  every branch has metrics before and after EO, model/EO diagnostics, flip
  counts, and outer-reference checks.
- `metadata/prediction_level_A.csv`: one row per prediction per arm per
  evaluation split, with original row IDs, race, ground truth, pre-EO label,
  and post-EO label.
- `metadata/oracle_fit_traces_A.csv`: all custom oracle LR fit traces.
- `metadata/split_membership_A.csv`: source-row membership for each seed.
- `metadata/group_label_counts_A.csv`: class/race counts per partition.
- `metadata/prediction_change_checks_A.csv`: paired per-seed changes.
- `metadata/run_manifest_A.json`: exact settings, versions, and source hashes.
- `metadata/aggregate_summary_A.csv`: mean and SD of all reported metrics,
  including **mean absolute group disparities**.
- `metadata/paired_deltas_A.csv`: seed-paired differences, with descriptive
  95% t intervals. Since the ten data splits overlap, these should **not** be
  treated as independent population-level confidence intervals.
- `metadata/descriptive_gap_thresholds_A.csv`: number of seeds with
  |EOD| and/or |PED| ≤ 0.05, for descriptive comparison only.
- `metadata/analysis_report_A.md`: concise findings and ready-to-copy tables.

## Warnings and limitations

- Optional TensorFlow and inFairness warnings from importing AIF360 do not
  affect Re, EGR, or EO.
- scikit-learn `DataConversionWarning` about 2D label arrays appears in the
  original EGR pipeline and does not necessarily indicate failure.
- Run in the same pinned thesis environment as Run 042:
  **AIF360 0.6.1, Fairlearn 0.14.0, scikit-learn 1.9.1**.
- Analyses are descriptive; no statement of improved fairness or accuracy is
  justified until the actual AIF360 experiment has been run and validated.
- The experiment does not alter or overwrite Runs 011, 014, 016, 022, or 042.
