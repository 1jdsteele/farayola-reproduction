# Run 042 — Reweighing-integrated EGR oracle (Approach A)

**Protocol:** COMPAS 5,278 observations; 70/30 race-stratified train/test; seeds 101, 102, 103, 104, 105, 106, 107, 108, 109, 110. EGR: DemographicParity, eps=0.02, max_iter=20, drop_prot_attr=False, base LR max_iter=1000.

**Intervention:** For each EGR cost-sensitive logistic-regression oracle call, multiply its internal sample weights by AIF360 Reweighing instance weights. Renormalize to the original internal oracle weight mass. Do NOT externally weight EGR's fairness-moment constraints or mixture-selection objective.

**Implementation controls:** Original EGR reproduces frozen Run 011 and Run 014; unity external weights reproduce baseline predictions exactly for every tested seed; nontrivial external weights reach every logged logistic-regression oracle training call.

## Mean test metrics

| Configuration | accuracy | statistical_parity_difference | disparate_impact | equal_opportunity_difference | predictive_equality_difference |
| --- | --- | --- | --- | --- | --- |
| legacy_egr | 0.665215 | -0.005831 | 0.988704 | -0.025303 | -0.067176 |
| unity_oracle | 0.665215 | -0.005831 | 0.988704 | -0.025303 | -0.067176 |
| reweighted_oracle | 0.664394 | -0.000153 | 1.002215 | -0.021254 | -0.059732 |

## Paired difference: weighted oracle − original EGR

| Metric | Mean delta | 95% seed t lower | 95% seed t upper | Seeds changed |
| --- | --- | --- | --- | --- |
| accuracy | -0.000821 | -0.005153 | 0.003511 | 10 |
| statistical_parity_difference | 0.005677 | -0.000251 | 0.011605 | 10 |
| disparate_impact | 0.013511 | -0.000139 | 0.027161 | 10 |
| equal_opportunity_difference | 0.004049 | -0.008416 | 0.016514 | 10 |
| predictive_equality_difference | 0.007444 | -0.000345 | 0.015234 | 10 |

## Prediction and oracle diagnostics

- Mean changed predictions versus EGR: 58.70 / 1584.
- Seeds with changed predictions: 10/10.
- Mean oracle fits, unity: 28.00.
- Mean oracle fits, reweighted: 22.10.

## Interpretation cautions

- This is an **oracle-level integration experiment**, NOT a fully externally weighted EGR optimization. EGR's empirical error and demographic-parity moments are still evaluated without Reweighing weights.
- The altered oracle may no longer solve Fairlearn's original best-response objective exactly; do not assume the original EGR theoretical convergence or fairness guarantees apply to this variant.
- EGR uses a **Demographic Parity** constraint; eps=0.02 is an optimization parameter and should not be interpreted as guaranteeing a particular test SPD bound or statistical fairness across all metrics.
- Test observations were not used for EGR fitting (held-out 30% test). Do not tune on these test results; future robustness tests should be separate runs.
- Seed-based intervals describe variation over overlapping splits of this COMPAS cohort, not independent population sampling.
- Predictions from EGR are randomized. The historical seed protocol sets NumPy before fit/predict; dedicated EGR random-seed stability remains for future work.
