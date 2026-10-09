# Run 004 — EGR Prediction Seed Control

## Parent

runs/run_003_egr_logreg_maxiter_1000/

## Research question

Does explicitly seeding NumPy immediately before AIF360
ExponentiatedGradientReduction prediction make the observable COMPAS
pipeline repeatable across fresh executions?

## Motivation

The installed Fairlearn 0.14.0 implementation of
ExponentiatedGradient.predict() exposes a random_state parameter and uses
randomized prediction.

AIF360 0.6.1 calls the underlying prediction process without exposing that
random_state through its wrapper.

The published IF-EA COMPAS notebook specifies random_state=101 for the
train/test split, but does not explicitly stabilize EGR prediction.

Therefore earlier experimental runs contained an uncontrolled stochastic
component at EGR prediction.

## Single scientific change from Run 003

Immediately before:

predictions = inprocess_algorithm.predict(dataset_test_bld)

Run 004 adds:

np.random.seed(101)

The value 101 was chosen to align with the repository's existing
train_test_split random_state.

This is a diagnostic reproducibility control. It is NOT evidence that the
authors used this NumPy seed.

## Held constant

- reconstructed COMPAS input
- train/test split
- train_test_split random_state=101
- target stratification
- DIR
- LogisticRegression(max_iter=1000)
- EGR DemographicParity constraint
- EGR eps=0.02
- EGR drop_prot_attr=False
- EGR outer max_iter remains at installed default 50
- EO
- ROC
- protected-group definitions
- favorable_label=1
- known first-block privileged/unprivileged ordering bug
- known incorrect first-block ROC heading
- all other notebook behavior

## Repeatability test

The same source notebook was executed twice using independent fresh
Jupyter kernels.

Executions A and B produced identical textual outputs and identical
reported metrics.

## Results

### DIR + EGR — repository raw orientation

Accuracy: 0.678662
DI: 0.896137
SPD: -0.04758032186014549
EOD: -0.023545293295870184
PED: 0.010013670539986308

Because the first ClassificationMetric call uses the known reversed
privileged/unprivileged argument ordering, the comparable orientation is
approximately:

Accuracy: 0.678662
DI: 1.11590
SPD: +0.047580
EOD: +0.023545
PED: -0.010014

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

## Interpretation

The explicit NumPy seed stabilizes the reported outputs across independent
fresh executions.

Because Run 003 and Run 004 differ only in this prediction-seeding control
but produce different DIR+EGR and EO metrics, stochastic EGR prediction was
a confounding variable in comparisons involving earlier unseeded runs.

Therefore differences between Run 001 and Run 003 cannot all be attributed
solely to Logistic Regression convergence.

The Run 003 finding that LogisticRegression(max_iter=1000) removes the
LBFGS convergence warnings remains valid. What must be weakened is the
causal attribution of all associated metric changes to that convergence
change.

Interestingly, DIR+EGR+ROC has the same five aggregate metrics in Runs 003
and 004 despite the EGR prediction difference.

## Dependency caveat

Installed:

AIF360 0.6.1
Fairlearn 0.14.0

AIF360 0.6.1 declares fairlearn~=0.7 for its reductions optional
dependency, so the installed Fairlearn version is newer than AIF360's
declared reductions dependency range.

The environment was deliberately not modified because package-version
changes would introduce another experimental variable.

## Next experiment

Run 005 should branch from this frozen seeded run.

The only new scientific change should be:

EGR outer max_iter: installed default 50 -> explicit 20

LogisticRegression(max_iter=1000) and np.random.seed(101) should both
remain unchanged.
