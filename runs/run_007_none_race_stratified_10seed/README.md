# Run 007 — Race-Stratified Candidate NONE Baseline, 10-Seed Sensitivity

## Methodological parent

runs/run_006_none_race_stratified/

## Research question

How sensitive is the race-stratified candidate NONE COMPAS baseline to
random train/test splits, and can averaging across ten seeds plausibly
account for the remaining discrepancy with the paper's published NONE
row, especially Equal Opportunity Difference?

## Important limitation

The authors' exact ten random seeds have not been established.

Run 007 therefore uses the predetermined sensitivity set:

101 through 110 inclusive.

This is NOT claimed to reproduce the authors' exact ten trials.

## Held constant from Run 006

- reconstructed 5,278-person COMPAS cohort
- best-supported 11-feature model matrix
- two_year_recid target
- 70:30 split
- shuffle=True
- stratification on race
- LogisticRegression()
- no fairness intervention
- AIF360 group definitions
- favorable_label=1 convention
- fairness metric definitions
- software environment

## Experimental extension

Run 006 evaluates random_state=101.

Run 007 evaluates:

101, 102, 103, 104, 105, 106, 107, 108, 109, 110

For each seed the run preserves split composition, confusion matrix,
Logistic Regression iteration count, accuracy, DI, SPD, EOD, and PED.

Aggregate statistics include mean, sample standard deviation, minimum,
maximum, and a 95% Student-t confidence interval.

Fairness metrics are aggregated in their signed AIF360 orientation.

## Parent validation

Run 007 seed 101 is explicitly compared with frozen Run 006 to confirm
that the multi-seed implementation preserves the Run 006 procedure.

## Repeatability

The completed 10-seed experiment was executed twice.

Execution A exit code: 0
Execution B exit code: 0

Per-seed results, aggregate summaries, and stdout are identical between
the two successful executions.

## Failed preliminary execution

An initial Execution A completed the calculations but subsequently exited
with code 1 because an accidental shell/heredoc fragment had been appended
to the Python source during script construction.

That failed attempt is preserved separately.

The accidental trailing text was removed without altering the intended
experimental code, after which Execution A and Execution B both completed
successfully and identically.

## Logistic Regression convergence observation

LogisticRegression() retains its inherited default max_iter=100.

Nine seeds terminated below the iteration limit.

Seed 105 reached 100 iterations and emitted an LBFGS convergence warning.

This was intentionally NOT repaired inside Run 007 because increasing
max_iter would introduce another experimental variable.

## Interpretation policy

Run 007 is frozen before scientific interpretation of its ten-seed results.
