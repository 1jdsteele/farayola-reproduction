# Run 003 — EGR Logistic Regression Convergence Diagnostic

## Purpose

Determine whether the Logistic Regression convergence warnings observed
in Run 001 materially affect the DIR + EGR + EO/ROC results.

## Parent run

This experiment branches directly from the frozen:

`run_001_repository_faithful`

The same reconstructed COMPAS input and the same notebook logic are used.

## Controlled change

Exactly one model parameter was intentionally changed.

Run 001:

    base_classifier = LogisticRegression()

Run 003:

    base_classifier = LogisticRegression(max_iter=1000)

This increases the maximum number of LBFGS optimization iterations
allowed inside each Logistic Regression fit from the scikit-learn
default of 100 to 1000.

## What is NOT changed

This run does NOT change:

- the COMPAS input dataset
- train/test split
- random_state
- target stratification
- protected-group definitions
- favorable/unfavorable labels
- DIR settings
- EGR fairness constraint
- EGR eps
- EGR's own outer-loop max_iter
- EO settings
- ROC settings
- metric definitions
- the known group-ordering bug in the first ClassificationMetric call
- the known incorrect label attached to the first DIR+EGR output

## Research question

If the EGR base learner is allowed enough Logistic Regression iterations
to converge, do:

1. the convergence warnings disappear?
2. the predictions change?
3. accuracy change?
4. SPD change?
5. DI change?
6. EOD change?
7. PED change?

## Initial execution observation

The Run 001 Logistic Regression convergence warnings no longer appear
when the base classifier uses max_iter=1000.

The scikit-learn column-vector DataConversionWarning remains, because
the original notebook still represents y as a two-dimensional column
rather than a one-dimensional Series.

## Interpretation

This is a convergence sensitivity diagnostic.

It is NOT the later paper-faithful experiment involving the paper's
reported EGR-level `max_iter = 20`.

The Logistic Regression iteration limit and EGR iteration limit are
different parameters.

## Preservation rule

Do not modify files in this directory after the run is frozen.
