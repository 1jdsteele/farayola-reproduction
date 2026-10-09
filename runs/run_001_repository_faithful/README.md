# Run 001 — Repository-Faithful COMPAS DIR+EGR+EO/ROC Reproduction

## Purpose

First successful end-to-end execution of the published IF-EA
DIR + EGR + EO/ROC COMPAS notebook.

This run is intended to preserve the published repository logic
as closely as possible while making only changes necessary for
execution.

## Upstream implementation

IF-EA repository pinned to commit:

d84cfb1

See `metadata/source_repositories.txt` for full repository URLs
and commit hashes.

## Input data

`compas_5278_raw.csv`

Exact reconstructed 5,278-person COMPAS cohort:
- 2,795 two_year_recid = 0
- 2,483 two_year_recid = 1
- 3,175 African-American
- 2,103 Caucasian

`compas_farayola_numeric.csv`

Reconstructed 11-feature numeric model input. The population is
exactly reconstructed, but the exact feature matrix is inferred
because the processed COMPAS CSV and complete 11-feature list
were not published.

## Execution-only modifications

1. Replaced the non-executable original line:

   data = pd.read_csv(COMPAS dataset)

   with:

   data = pd.read_csv("../data/compas_farayola_numeric.csv")

2. Commented out unused OptimPreproc-related imports because
   importing OptTools required the optional cvxpy dependency.
   None of those objects are used by the DIR/EGR/EO/ROC pipeline.

No fairness algorithm parameters, group definitions, favorable
label definitions, train/test split settings, thresholds, or
model logic were intentionally changed.

## Important observations

- LogisticRegression emitted convergence warnings after reaching
  its default 100-iteration limit.
- The initial DIR+EGR metrics are mislabeled in the notebook as
  "Reject Option Classification."
- The initial ClassificationMetric call reverses the positional
  privileged/unprivileged arguments.
- This run should therefore be treated as a repository-faithful
  reproduction artifact, not as a corrected implementation.

## Preservation rule

Do not modify files in this directory.

Future experiments should be placed in separate run directories.

