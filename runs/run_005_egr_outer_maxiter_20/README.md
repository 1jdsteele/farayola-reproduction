# Run 005 — EGR Outer max_iter = 20

## Parent

runs/run_004_egr_prediction_seed_101/

## Research question

What happens when the Exponentiated Gradient Reduction outer iteration
limit is explicitly changed from the installed AIF360/Fairlearn default
of 50 to the paper-stated maximum of 20?

## Motivation

Farayola et al. state that EGR uses a maximum of 20 iterations.

The published IF-EA COMPAS notebook does not explicitly specify max_iter
when constructing ExponentiatedGradientReduction.

In the installed AIF360 0.6.1 implementation, the default EGR max_iter is 50.

Run 005 therefore tests whether this paper/repository discrepancy affects
the reproduced COMPAS results.

## Parent controls

Run 004 established a repeatable parent configuration containing:

- LogisticRegression(max_iter=1000)
- EGR constraints="DemographicParity"
- EGR eps=0.02
- EGR drop_prot_attr=False
- np.random.seed(101) immediately before EGR prediction
- identical COMPAS input and split

The explicit prediction seed was retained so randomized EGR prediction
would not confound the max_iter comparison.

## Single scientific change from Run 004

Run 004:

ExponentiatedGradientReduction(
    base_classifier,
    constraints="DemographicParity",
    eps=0.02,
    drop_prot_attr=False
)

Installed default outer max_iter = 50.

Run 005:

ExponentiatedGradientReduction(
    base_classifier,
    constraints="DemographicParity",
    eps=0.02,
    max_iter=20,
    drop_prot_attr=False
)

No other code change was made.

The Run 004 vs Run 005 code diff is preserved in:

metadata/run004_vs_run005_code_diff.txt

## Repeatability

Run 005 was executed twice using fresh Jupyter kernels.

Executions A and B produced identical textual outputs.

## Results

### DIR + EGR — raw repository orientation

Accuracy: 0.678662
DI: 0.896137
SPD: -0.04758032186014549
EOD: -0.023545293295870184
PED: 0.010013670539986308

The first ClassificationMetric call retains the known repository
privileged/unprivileged ordering bug.

### DIR + EGR + EO

Accuracy: 0.670455
DI: 1.097347
SPD: 0.03996334688734893
EOD: 0.0026684665735319246
PED: -0.003195

### DIR + EGR + ROC

Accuracy: 0.680556
DI: 1.096814
SPD: 0.03697873628576337
EOD: 0.010195111997111761
PED: -0.018575

## Run 004 comparison

Every textual result reported by Run 005 is identical to Run 004.

Therefore changing EGR outer max_iter from the installed default of 50
to the paper-stated value of 20 produced no observable change in any
reported aggregate accuracy or fairness metric in this controlled
experiment.

This does not by itself prove that all internal fitted state or every
individual prediction is identical. The demonstrated result concerns
the outputs currently reported by the notebook.

## Reproducibility interpretation

The paper/repository max_iter discrepancy is confirmed:

- Paper: EGR maximum iterations = 20
- Repository notebook: max_iter not explicitly specified
- Installed AIF360/Fairlearn behavior: default = 50

However, under our controlled reconstructed COMPAS configuration, this
difference does not explain the remaining discrepancy between our results
and the paper's Table 1 values.

## Important fixed caveats

This run continues to preserve rather than repair:

- inferred rather than proven exact 11-feature COMPAS matrix
- repository target stratification
- favorable_label=1 for recidivism
- first-block privileged/unprivileged ordering bug
- incorrect first-block ROC heading
- postprocessing behavior
- current package environment
- all other known repository behavior

## Scientific conclusion

For this reconstruction, EGR outer max_iter=20 and max_iter=50 are
observationally equivalent with respect to all aggregate metrics currently
reported by the notebook.

The EGR max_iter discrepancy therefore appears non-explanatory for the
current paper/reproduction numerical differences, under this experimental
configuration.
