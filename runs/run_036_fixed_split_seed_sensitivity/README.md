# Run 036 — Fixed-Split Random-Seed Sensitivity

## Purpose

This run implements the random-seed sensitivity experiment requested by the thesis advisor.

The goal is to separate:

1. variation caused by changing the train/test split, from
2. variation caused by stochastic behavior inside fairness algorithms.

Earlier multi-seed runs generally changed both the split seed and algorithm seed together.
Run 036 instead fixes the train/test split and varies only algorithm-level randomness.

## Fixed split

- Split seed: 101
- Test size: 0.30
- Shuffle: True
- Stratification: race
- The same split must be used for every configuration and every algorithm seed.

The split protocol intentionally follows the established reproduction runs so that
Run 036 can be compared directly with Runs 007–034.

## Algorithm seeds

101 through 120 inclusive (20 runs per stochastic configuration).

## Configurations

1. NONE
   - Logistic Regression baseline
   - Negative control
   - No intentional stochastic algorithm component
   - Expected to produce identical predictions for all 20 algorithm-seed records

2. DIR + EGR + EO
   - Representative integrated EGR configuration
   - Fixed data split
   - EGR prediction randomness varies with algorithm seed
   - EO seed varies with the same algorithm seed

3. AL
   - Representative stochastic in-processing method
   - Fixed data split
   - AdversarialDebiasing seed varies with algorithm seed

4. DIR + AL + EO
   - Strong-performing AL + EO integrated configuration
   - Fixed data split
   - AL seed varies with algorithm seed
   - EO seed varies with the same algorithm seed

5. DIR + AL + ROC
   - Major unresolved reproduction discrepancy
   - Fixed data split
   - AL seed varies with algorithm seed
   - ROC has no seed and is deterministic conditional on AL scores

## Metrics

For every configuration × algorithm seed:

- Accuracy
- Disparate Impact (DI)
- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Predictive Equality Difference (PED)

Raw signed AIF360 values are retained.

## Provenance / diagnostics

Record where applicable:

- configuration
- split seed
- algorithm seed
- train/test row counts
- train/test split hashes or row-ID hashes
- prediction SHA-256
- score SHA-256
- confusion matrix: TN, FP, FN, TP
- algorithm parameters
- environment/library versions

## Environments

- `.venv`: NONE and EGR-based configurations
- `.venv-al`: AL-based configurations

The existing environments are preserved rather than modifying dependencies.

## Interpretation

Run 036 tests algorithmic seed sensitivity with the data split held constant.

It does not attempt to tune parameters to reproduce the paper.

If a paper value falls inside the observed distribution, that supports compatibility
with algorithmic randomness but does not prove randomness caused the original result.

Run 036 will also be compared with earlier runs in which the split and algorithm
seed changed together, allowing split-induced and algorithm-induced variability
to be distinguished.
