# Farayola et al. COMPAS Fairness Reproduction

## Overview

This repository contains a controlled reproduction and implementation audit of the COMPAS fairness-mitigation experiments reported by Farayola et al. in _A fairness-focused approach to recidivism prediction: implications for accuracy, trust, and equity_.

The primary COMPAS reproduction consists of **26 non-baseline fairness configurations**, executed as **Runs 009–034**, with **10 controlled seeds per configuration**. The untreated baseline was established separately in Run 008.

The project was designed as a reproducibility study rather than a tuning exercise. The goal was not to force the reproduced numbers to match the paper, but to determine:

1. which reported results can be reproduced numerically;
2. which mitigation stages are operationally active in the released/reconstructed implementation;
3. which discrepancies are stable versus seed-dependent;
4. how post-processing behaves under different upstream score semantics; and
5. which parts of the original methodology remain under-specified.

A later diagnostic, **Run 035**, audits an overlooked `StandardScaler` step in the two released COMPAS pre-processing/post-processing notebooks. Run 035 is **not** one of the 26 primary configurations.

---

## Scientific status

The COMPAS reproduction is **substantially but non-uniformly successful**.

Many of the paper's reported rows reproduce closely, including several multi-stage configurations. However, the implementation audit also shows that some nominally integrated pipelines collapse exactly to simpler pipelines under the standardized reproduction backbone because intermediate information is not consumed downstream.

The most important high-level findings are:

- many EGR- and EO-based configurations reproduce the paper closely;
- the standardized 26-run experiment produces only **17 distinct seed-level outcome patterns**;
- all eight integrated Reweighing pathways are operationally inert under the standardized reproduction implementation;
- Adversarial Debiasing / Adversarial Learning (AL) is substantially more seed-sensitive than the non-AL methods;
- Equalized Odds (EO) strongly stabilizes AL while imposing a consistent accuracy cost;
- Reject Option Classification (ROC) behaves well after EGR but becomes unstable and accuracy-destructive after AL;
- `DIR+AL+ROC` is the largest unresolved numerical discrepancy;
- near-zero signed fairness means can partially reflect cancellation across seeds;
- endpoint reproduction and mechanistic/pipeline reproduction are not always the same thing.

These findings should be interpreted as reproducibility results, not as evidence of misconduct or necessarily as errors in the original work.

---

# 1. Dataset reconstruction

## Source

The reproduction begins from the ProPublica COMPAS two-year dataset:

`compas-scores-two-years.csv`

The reconstructed cohort follows the standard ProPublica filtering and then restricts race to African-American and Caucasian defendants.

### Cohort

- Standard filtered cohort: **6,172 rows**
- African-American/Caucasian subset: **5,278 rows**
- African-American: **3,175**
- Caucasian: **2,103**

### Target

Target:

`two_year_recid`

Counts:

- `0`: **2,795**
- `1`: **2,483**

### Protected attribute encoding

- African-American: `race = 0` = unprivileged
- Caucasian: `race = 1` = privileged

### Favorable label

The reproduction preserves the released AIF360 convention:

- `favorable_label = 1`
- `unfavorable_label = 0`

This is semantically unusual because `two_year_recid = 1` indicates recidivism, which is not intuitively a favorable outcome. It is preserved because changing the label orientation would change the experiment.

## Reconstructed feature matrix

The primary processed file is:

`data/compas_farayola_numeric.csv`

Shape:

`(5278, 12)`

This corresponds to 11 predictors plus the target.

The 11 predictors are:

1. `age`
2. `age_cat`
3. `sex`
4. `race`
5. `juv_fel_count`
6. `juv_misd_count`
7. `juv_other_count`
8. `priors_count`
9. `c_charge_degree`
10. `score_text`
11. `decile_score`

Target:

`two_year_recid`

SHA256:

`6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e`

The exact processed COMPAS CSV used by the authors was not released. Therefore this should be described as the **best-supported reconstruction** of the paper's COMPAS design matrix, not as a proven byte-identical copy of the authors' processed data.

---

# 2. Experimental design

The standardized reproduction backbone used for Runs 009–034 is:

- 5,278-row reconstructed COMPAS cohort
- 11 predictors
- target `two_year_recid`
- race as protected attribute
- African-American = unprivileged
- Caucasian = privileged
- 70/30 train/test split
- `shuffle=True`
- `stratify=race`
- reproduction seeds **101–110**
- same seed set across configurations
- no invented class-balancing procedure
- no undocumented scaling in the standardized backbone
- signed fairness differences preserved
- A/B independent-process repeatability checks for important runs
- run artifacts checksummed and frozen after completion

The reproduction seeds are **not claimed to be the authors' seeds**. The paper reports repeated runs but does not publish the exact seed list.

---

# 3. Fairness metrics

Five metrics are used throughout.

## Statistical Parity Difference (SPD)

Ideal: `0`. Signed in the reproduction.

## Disparate Impact (DI)

Ideal: `1`.

## Equal Opportunity Difference (EOD)

Ideal: `0`. Signed in the reproduction.

## Predictive Equality Difference (PED)

Operationalized as the false-positive-rate difference. Ideal: `0`. Signed in the reproduction.

## Accuracy

Higher is better.

### Important sign convention

The paper reports nonnegative fairness values in its table, while the reproduction preserves signed SPD, EOD, and PED.

Therefore `abs(mean(metric))` and `mean(abs(metric))` are not equivalent when the metric changes sign across seeds.

---

# 4. Fairness methods

## Pre-processing

- `Re` — Reweighing
- `DIR` — Disparate Impact Remover

## In-processing

- `EGR` — Exponentiated Gradient Reduction
- `AL` — Adversarial Debiasing / Adversarial Learning

## Post-processing

- `EO` — Equalized Odds post-processing
- `ROC` — Reject Option Classification

---

# 5. Primary 26 configurations

| Run | Configuration | Stages          |
| --: | ------------- | --------------- |
| 009 | DIR           | Pre             |
| 010 | Re            | Pre             |
| 011 | EGR           | In              |
| 012 | EO            | Post            |
| 013 | ROC           | Post            |
| 014 | Re+EGR        | Pre → In        |
| 015 | DIR+EGR       | Pre → In        |
| 016 | EGR+EO        | In → Post       |
| 017 | EGR+ROC       | In → Post       |
| 018 | Re+EO         | Pre → Post      |
| 019 | Re+ROC        | Pre → Post      |
| 020 | DIR+EO        | Pre → Post      |
| 021 | DIR+ROC       | Pre → Post      |
| 022 | Re+EGR+EO     | Pre → In → Post |
| 023 | Re+EGR+ROC    | Pre → In → Post |
| 024 | DIR+EGR+EO    | Pre → In → Post |
| 025 | DIR+EGR+ROC   | Pre → In → Post |
| 026 | AL            | In              |
| 027 | Re+AL         | Pre → In        |
| 028 | DIR+AL        | Pre → In        |
| 029 | AL+EO         | In → Post       |
| 030 | AL+ROC        | In → Post       |
| 031 | Re+AL+EO      | Pre → In → Post |
| 032 | Re+AL+ROC     | Pre → In → Post |
| 033 | DIR+AL+EO     | Pre → In → Post |
| 034 | DIR+AL+ROC    | Pre → In → Post |

The untreated `NONE` baseline is Run 008 and is not counted among these 26 configurations.

---

# 6. Baseline

The standardized untreated baseline (Run 008) produced approximately:

| Metric   | Paper NONE | Reproduction NONE |
| -------- | ---------: | ----------------: |
| SPD      |      0.250 |            0.2360 |
| DI       |      2.130 |            1.8857 |
| EOD      |      0.025 |            0.2281 |
| PED      |      0.180 |            0.1656 |
| Accuracy |      0.680 |            0.6794 |

Accuracy, SPD, and PED are reasonably close. The baseline EOD is a major discrepancy.

---

# 7. Paper-versus-reproduction results

The table below reports paper values and reproduced 10-seed means. Reproduced SPD, EOD, and PED retain their signs.

| Run | Model       | SPD paper → ours | DI paper → ours | EOD paper → ours | PED paper → ours | Acc paper → ours |
| --: | ----------- | ---------------- | --------------- | ---------------- | ---------------- | ---------------- |
| 009 | DIR         | .284 → .2773     | 2.092 → 2.0938  | .281 → .2771     | .211 → .2005     | .670 → .6816     |
| 010 | Re          | .097 → .0361     | 1.243 → 1.0938  | .094 → .0132     | .031 → -.0231    | .660 → .6689     |
| 011 | EGR         | .036 → -.0058    | 1.093 → .9887   | .080 → -.0253    | .088 → -.0672    | .670 → .6652     |
| 012 | EO          | .040 → .0349     | 1.125 → 1.1033  | .014 → -.0039    | .015 → .0045     | .640 → .6379     |
| 013 | ROC         | .270 → .2360     | 2.100 → 1.8857  | .287 → .2281     | .183 → .1656     | .680 → .6794     |
| 014 | Re+EGR      | .033 → -.0058    | 1.086 → .9887   | .026 → -.0253    | .032 → -.0672    | .660 → .6652     |
| 015 | DIR+EGR     | .055 → .0382     | 1.142 → 1.0968  | .038 → .0151     | .004 → -.0209    | .670 → .6688     |
| 016 | EGR+EO      | .036 → .0387     | 1.095 → 1.0923  | .003 → -.0031    | .003 → .0026     | .650 → .6516     |
| 017 | EGR+ROC     | .023 → .0362     | 1.059 → 1.0945  | .015 → .0086     | .042 → -.0200    | .670 → .6710     |
| 018 | Re+EO       | .034 → .0349     | 1.095 → 1.1033  | .001 → -.0039    | .002 → .0045     | .640 → .6379     |
| 019 | Re+ROC      | .081 → .2360     | 1.230 → 1.8857  | .091 → .2281     | .002 → .1656     | .670 → .6794     |
| 020 | DIR+EO      | .032 → .0337     | 1.097 → 1.1010  | .007 → -.0040    | .005 → .0062     | .640 → .6307     |
| 021 | DIR+ROC     | .288 → .2773     | 2.160 → 2.0938  | .300 → .2771     | .207 → .2005     | .680 → .6816     |
| 022 | Re+EGR+EO   | .034 → .0387     | 1.088 → 1.0923  | .001 → -.0031    | .0002 → .0026    | .650 → .6516     |
| 023 | Re+EGR+ROC  | .023 → .0362     | 1.059 → 1.0945  | .015 → .0086     | .042 → -.0200    | .670 → .6710     |
| 024 | DIR+EGR+EO  | .037 → .0395     | 1.097 → 1.0982  | .002 → -.0002    | .003 → -.0008    | .660 → .6580     |
| 025 | DIR+EGR+ROC | .055 → .0596     | 1.142 → 1.1585  | .038 → .0326     | .004 → .0021     | .670 → .6726     |
| 026 | AL          | .053 → .1317     | 1.174 → 1.5168  | .115 → .1152     | .085 → .0646     | .660 → .6798     |
| 027 | Re+AL       | .061 → .1317     | 1.168 → 1.5168  | .052 → .1152     | .004 → .0646     | .670 → .6798     |
| 028 | DIR+AL      | .009 → .1970     | 1.020 → 1.7332  | .028 → .1852     | .067 → .1268     | .670 → .6823     |
| 029 | AL+EO       | .036 → .0383     | 1.099 → 1.1041  | .014 → .0052     | .016 → -.0021    | .660 → .6482     |
| 030 | AL+ROC      | .079 → .0687     | 1.159 → 2.0301  | .037 → .0567     | .033 → .0261     | .680 → .6167     |
| 031 | Re+AL+EO    | .036 → .0383     | 1.099 → 1.1041  | .014 → .0052     | .016 → -.0021    | .660 → .6482     |
| 032 | Re+AL+ROC   | .079 → .0687     | 1.159 → 2.0301  | .037 → .0567     | .033 → .0261     | .680 → .6167     |
| 033 | DIR+AL+EO   | .038 → .0372     | 1.089 → 1.1049  | .001 → .0010     | .001 → .0019     | .660 → .6431     |
| 034 | DIR+AL+ROC  | .030 → .0964     | 1.056 → 2.3190  | .006 → .0865     | .032 → .0615     | .680 → .5946     |

---

# 8. Aggregate comparison statistics

Across 130 model-metric comparisons:

- Paper value inside reproduction 95% CI: **73 / 130 = 56.2%**
- Paper value inside observed ten-seed min/max range: **104 / 130 = 80.0%**
- Approximately **69.2%** are within one reproduction SD
- Approximately **85.4%** are within two reproduction SDs

These are descriptive diagnostics only, not formal equivalence tests.

---

# 9. Metric-level reproduction

| Metric   | Paper within reproduction CI | Mean absolute difference | Median absolute difference |
| -------- | ---------------------------: | -----------------------: | -------------------------: |
| SPD      |                        17/26 |                    .0324 |                      .0105 |
| DI       |                        14/26 |                    .2269 |                      .0404 |
| EOD      |                        12/26 |                    .0348 |                      .0145 |
| PED      |                        15/26 |                    .0346 |                      .0178 |
| Accuracy |                        15/26 |                    .0139 |                      .0070 |

Accuracy reproduces well overall. DI's large mean discrepancy is driven disproportionately by a small number of AL+ROC configurations.

---

# 10. Standardized-run equivalence classes

Under the standardized Runs 009–034 backbone, the 26 named configurations reduce to **17 distinct seed-level outcome patterns**.

| Group | Exactly identical standardized-run configurations |
| ----- | ------------------------------------------------- |
| A     | DIR, DIR+ROC                                      |
| B     | Re                                                |
| C     | EGR, Re+EGR                                       |
| D     | EO, Re+EO                                         |
| E     | ROC, Re+ROC                                       |
| F     | DIR+EGR                                           |
| G     | EGR+EO, Re+EGR+EO                                 |
| H     | EGR+ROC, Re+EGR+ROC                               |
| I     | DIR+EO                                            |
| J     | DIR+EGR+EO                                        |
| K     | DIR+EGR+ROC                                       |
| L     | AL, Re+AL                                         |
| M     | DIR+AL                                            |
| N     | AL+EO, Re+AL+EO                                   |
| O     | AL+ROC, Re+AL+ROC                                 |
| P     | DIR+AL+EO                                         |
| Q     | DIR+AL+ROC                                        |

Run 035 later refined the PP-specific wording because the released PP notebooks include scaling.

---

# 11. Reweighing integration collapse

Under the standardized backbone:

- `Re+EGR = EGR`
- `Re+AL = AL`
- `Re+EO = EO`
- `Re+ROC = ROC`
- `Re+EGR+EO = EGR+EO`
- `Re+EGR+ROC = EGR+ROC`
- `Re+AL+EO = AL+EO`
- `Re+AL+ROC = AL+ROC`

Across `8 × 10 × 5 = 400` paired comparisons, every Re-stage metric difference is exactly zero.

The implementation audit shows that Reweighing modifies `BinaryLabelDataset.instance_weights`, but the relevant downstream EGR, legacy AL, and PP LogisticRegression pathways do not consume those weights.

Standalone Re remains active when the weights are explicitly passed to a weight-aware Logistic Regression estimator.

---

# 12. DIR as a positive control

DIR modifies actual non-protected feature values while preserving labels and race. Those changes propagate downstream into EGR and AL.

This demonstrates that the reproduction framework detects preprocessing effects when downstream methods consume the information produced by preprocessing.

---

# 13. AL environment and reproducibility

AL required a separate environment because AIF360 0.6.1 AdversarialDebiasing uses legacy TensorFlow graph/session behavior.

Key versions:

- Python 3.11.16
- NumPy 1.26.4
- TensorFlow 2.15.0
- tensorflow-macos 2.15.0
- AIF360 0.6.1
- pandas 3.0.6
- scikit-learn 1.9.1
- SciPy 1.17.1
- BlackBoxAuditing 0.1.54

TensorFlow compatibility mode:

```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

Each fit uses a fresh graph and session.

Paper-reported AL settings used in production:

- `adversary_loss_weight = 0.2`
- `num_epochs = 50`
- `batch_size = 128`
- `debias = True`

AIF360's default hidden-unit count of 200 is inherited.

---

# 14. AL instability

AL-containing configurations are much more variable across seeds than non-AL configurations.

Approximate SD ratios:

| Metric   | AL-containing vs non-AL |
| -------- | ----------------------: |
| SPD      |                   ~3.0× |
| DI       |              **~11.2×** |
| EOD      |                   ~3.0× |
| PED      |                   ~2.6× |
| Accuracy |                   ~1.8× |

Standalone AL DI ranges approximately `0.76 → 2.26`.

This makes the unpublished author seed set a meaningful reproducibility limitation.

---

# 15. EO strongly stabilizes AL

For `AL → AL+EO`, all four fairness-distance measures improve on **10/10 paired seeds**, while accuracy decreases on **10/10 seeds**.

Approximate mean paired changes:

- `|SPD|`: −0.123
- `|DI−1|`: −0.460
- `|EOD|`: −0.141
- `|PED|`: −0.111
- Accuracy: −0.032

EO also reduces AL fairness variability by roughly **93–96%**.

---

# 16. ROC score semantics

ROC behaves differently depending on the upstream score representation.

### After ordinary Logistic Regression

Hard 0/1 prediction scores make ROC effectively inactive.

### After EGR

Non-binary scores allow ROC to operate and reproduce the paper reasonably well.

### After AL

Rich continuous scores make ROC highly active and unstable.

This score-semantics dependence is central to interpreting the ROC results.

---

# 17. AL → ROC is the main unresolved reproduction discrepancy

For `AL+ROC`:

- Paper accuracy: `.680`
- Reproduction: `.6167`
- Paper DI: `1.159`
- Reproduction: `2.0301`

For `DIR+AL+ROC`:

- Paper SPD: `.030`
- Reproduction: `.0964`
- Paper DI: `1.056`
- Reproduction: `2.3190`
- Paper EOD: `.006`
- Reproduction: `.0865`
- Paper PED: `.032`
- Reproduction: `.0615`
- Paper accuracy: `.680`
- Reproduction: `.5946`

The discrepancy remains after removing the single most extreme seed, so it is systematic across the tested seed set.

---

# 18. EO versus ROC after AL

`DIR+AL+EO` has smaller `|SPD|`, `|DI−1|`, `|EOD|`, and `|PED|` than `DIR+AL+ROC` on all ten paired seeds.

It also has higher accuracy on 9/10 seeds, with one tie.

Under conventional five-objective dominance, `DIR+AL+EO` Pareto-dominates `DIR+AL+ROC` on all ten reproduction seeds.

The equivalent EGR branch does not show this pathological relationship, which localizes the unresolved discrepancy to the AL→ROC interaction.

---

# 19. Endpoint reproduction versus mechanistic reproduction

This project distinguishes:

- **endpoint reproduction** — does the final metric vector resemble the paper?
- **mechanistic/pipeline reproduction** — do the named stages actually alter the pipeline as expected?
- **stability reproduction** — does the result remain consistent across seeds?

Examples:

- `DIR+EGR+EO`: strong endpoint and mechanistic reproduction.
- `Re+AL+EO`: strong endpoint agreement, but Re is inert.
- `DIR+AL+EO`: strong endpoint agreement, but the intermediate DIR+AL trajectory differs from the paper.

---

# 20. Signed-metric cancellation

Seed-level analysis shows sign reversals in SPD/EOD/PED.

For `DIR+AL+EO`:

- 5 EOD seeds are positive
- 5 are negative
- signed mean EOD ≈ `.001`
- mean absolute seed-level EOD ≈ `.0136`

Similarly:

- signed mean PED ≈ `.0019`
- mean absolute seed-level PED ≈ `.0096`

Near-zero signed means should therefore be interpreted alongside `mean(abs(metric))`.

---

# 21. Run 035 — StandardScaler audit

A later notebook audit found that the released PP notebooks apply `StandardScaler` before Logistic Regression.

Run 035 isolated scaling while keeping the standardized split/seeds and other choices fixed.

## Reweighing result with scaling

- Re weights changed every seed
- Re features unchanged
- Re labels unchanged
- scaled Re parent = scaled baseline LR on 10/10 seeds
- scaled Re+EO = scaled baseline+EO on 10/10
- scaled Re+ROC = scaled baseline+ROC on 10/10

Therefore Re remains operationally inert in the released PP pathway.

## ROC result with scaling

ROC remained inactive:

- scaled Re+ROC active seeds: **0/10**
- scaled DIR+ROC active seeds: **0/10**

## Magnitude of scaling effect

Scaling changes only about one LR parent prediction per seed on average and barely changes aggregate metrics.

The “17 distinct results” statement should therefore be understood as a property of standardized Runs 009–034; for the released scaled PP notebooks, the operational findings remain but the exact cross-run identities should be phrased relative to correspondingly scaled parents.

---

# 22. Analysis files

Primary analysis artifacts:

- `compas_paper_vs_reproduction.csv`
- `compas_reproduction_analysis_long.csv`
- `compas_seed_level_analysis_long.csv`
- `compas_seed_level_results_wide.csv`

Expected sizes:

- aggregate long: 130 rows
- seed-level long: 1,300 rows
- seed-level wide: 260 rows

---

# 23. Reproducibility workflow

Each primary run follows approximately:

```text
runs/
  run_0XX_<configuration>/
    data/
    scripts/
    metadata/
    provenance/
```

Important runs are executed independently as A/B processes and compared byte-for-byte.

Example:

```bash
python <run-script> --tag A
python <run-script> --tag B

diff -u metadata/per_seed_results_A.csv metadata/per_seed_results_B.csv
cmp -s metadata/per_seed_results_A.csv metadata/per_seed_results_B.csv
```

A `cmp` exit status of `0` indicates byte-identical output.

Completed runs are frozen:

```bash
chmod -R a-w runs/run_0XX_<configuration>
```

Dataset checksum:

```bash
shasum -a 256 data/compas_farayola_numeric.csv
```

Expected:

```text
6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e
```

---

# 24. Environments

## Main environment

Relevant recorded versions for non-AL runs include:

- Python 3.14.7
- pandas 3.0.5
- scikit-learn 1.9.1
- AIF360 0.6.1
- Fairlearn 0.14.0
- BlackBoxAuditing 0.1.54
- NumPy 2.5.3

Use per-run `pip_freeze.txt` files where available.

## AL environment

Use `.venv-al` for Runs 026–034.

Do not silently rerun AL in the main environment.

---

# 25. Pinned repositories

Important pinned commits:

- IF-EA: `d84cfb11f518c3ba76bcb9cfc6ec18815df005d3`
- RisCanvi: `a97905e89d71900a394a0481d3017b986229cd80`
- ProPublica compas-analysis: `bafff5da3f2e45eca6c2d5055faad269defd135a`

The public IF-EA repository is strong implementation evidence but is not automatically assumed to be the exact final journal implementation.

### Upstream Source Repositories

The following repositories were used as reference materials
for this independent reproduction:

- [Farayola IF-EA](https://github.com/mmfara/IF-EA)
- [Farayola RisCanvi](https://github.com/mmfara/RisCanvi)
- [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis)

These repositories are not included in this repository.
See the provenance documentation for the specific source
commit hashes used during reproduction.

---

# 26. Main limitations

Important unresolved issues include:

- exact processed COMPAS CSV unavailable;
- exact final feature matrix not provably identical;
- exact author seeds unpublished;
- exact class-balancing/random-sampling procedure unrecoverable;
- original dependency environment under-specified;
- paper/repository differences in AL settings;
- ROC implementation and score semantics under-specified;
- repository-style EO/ROC calibration uses the evaluation test labels;
- Pareto objective transformations/implementation remain under investigation.

No undocumented class-balancing procedure was invented.

---

# 27. Reproduction philosophy

The project follows these principles:

- preserve meaningful runs;
- never modify frozen completed runs;
- change one scientific variable at a time where practical;
- distinguish paper-faithful, repository-faithful, reconstructed, and diagnostic choices;
- never silently introduce preprocessing or hyperparameter changes;
- do not tune unexplained discrepancies away;
- treat under-specification as a reproducibility result;
- preserve scientifically informative failed attempts;
- prefer paired seed-level comparisons;
- distinguish numerical equality from mechanistic equivalence;
- document environment adaptations explicitly.

---

# 28. Main conclusions

1. **Much of the COMPAS results table is numerically reproducible.**
2. **Numerical endpoint agreement does not guarantee that every named mitigation stage contributed.**
3. **The standardized 26-run experiment produces only 17 distinct seed-level outcome patterns.**
4. **Reweighing is operationally inert in the tested integrated pathways because its weights are not consumed downstream.**
5. **DIR acts as a useful positive control because its feature changes propagate.**
6. **AL is substantially more seed-sensitive than the non-AL family.**
7. **EO consistently repairs and stabilizes AL, with a predictable accuracy cost.**
8. **ROC is highly dependent on upstream score semantics.**
9. **`DIR+AL+ROC` is the largest unresolved reproduction discrepancy.**
10. **Signed fairness averaging can conceal cross-seed direction changes.**
11. **Endpoint, mechanism, and stability should be reported separately when evaluating reproducibility.**

The aim is not to prove that either the paper or the reproduction is “right.” The aim is to make the experiment inspectable enough that agreements, disagreements, implementation collapses, and remaining uncertainties can be stated precisely.
