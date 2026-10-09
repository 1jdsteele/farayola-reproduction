# Run 006 — Candidate NONE Baseline with Race/Group Stratification

## Parent

runs/run_002_candidate_none_logreg/

## Research question

If we operationalize the paper's stated group-stratified 70:30
train/test split by stratifying on race instead of the repository's
target-stratified split, does the untreated COMPAS baseline move toward
the published NONE row, especially the unexplained Equal Opportunity
Difference?

## Important caveat

Using race as the train/test stratification variable is our direct
operationalization of the paper's description that privileged and
unprivileged groups were proportionally represented.

This experiment does NOT establish that this is the authors' exact
unpublished implementation.

## Single scientific change from Run 002

Run 002:

stratify=y

Run 006:

stratify=X["race"]

Two associated non-scientific lines were changed so the comment and
printed stratification label correctly describe the Run 006 split.

## Held constant

- reconstructed COMPAS dataset
- best-supported 11-feature model matrix
- target variable
- 70:30 split
- random_state=101
- shuffle=True
- LogisticRegression() candidate NONE classifier
- AIF360 protected-group definitions
- favorable_label=1 convention
- no fairness intervention
- metric definitions
- software environment

## Discrepancy investigated

D4 — The paper describes group/protected-attribute stratification,
whereas the repository uses target stratification.

## Repeatability

Run 006 was executed twice.

Executions A and B produced identical stdout results.

## Reporting note

The script retains the inherited heading:

RUN 002 — CANDIDATE NONE BASELINE USING LOGISTIC REGRESSION

This is a stale reporting label only. The actual Run 006 execution is
confirmed by its path, the printed "Stratified on: race" metadata, and
the preserved source diff.

## Preserved evidence

- metadata/pristine_input_hashes.txt
- metadata/run002_vs_run006_code_diff.txt
- metadata/split_composition.txt
- metadata/results_A.txt
- metadata/results_B.txt
- metadata/stderr_A.txt
- metadata/stderr_B.txt
- metadata/exit_code_A.txt
- metadata/exit_code_B.txt
- metadata/repeatability_results_diff.txt
- metadata/reporting_note.txt
- metadata/run002_results_reference.txt
- metadata/run002_vs_run006_results.txt
- environment/runtime_versions.txt
- environment/pip_freeze.txt
- metadata/source_repositories.txt

## Interpretation policy

Run 006 is preserved and frozen before interpreting whether race
stratification moves the candidate NONE baseline toward or away from
the paper's reported NONE values.
