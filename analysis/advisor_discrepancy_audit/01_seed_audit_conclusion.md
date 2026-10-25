# Random Seed / RNG Audit — Conclusion

## Paper

The paper states that each experiment was repeated ten times using
different random seeds. It also states that the same random state was used
across experiments to maintain comparability.

A plausible interpretation is that one common seed was used across all
configurations within each experimental repetition, with ten repetitions
using different seeds. The exact ten author seeds are not published.

## Released COMPAS notebooks

All eight released COMPAS notebooks use:

    train_test_split(..., random_state=101, stratify=y)

No visible ten-seed experiment loop is present.

AdversarialDebiasing pathways explicitly use:

    seed=101

Equalized Odds usually uses:

    seed=101

but the two PIP EGR notebooks omit the EO seed.

The EGR constructors do not expose an explicit seed in the notebooks, and
the released notebooks contain no explicit NumPy RNG seed controlling the
stochastic EGR prediction step.

Reject Option Classification has no independent random seed in the
released configurations inspected.

Therefore the visible released notebooks do not encode the complete
ten-seed experimental protocol described by the paper and do not control
all stochastic components consistently.

## Existing standardized reproduction

Runs 008–034 used controlled reproduction seeds 101–110.

The same experimental seed was intentionally used for both:

1. train/test splitting; and
2. stochastic model/postprocessor behavior where applicable.

Examples:

- Run 008: split random_state=seed.
- Run 024: split random_state=seed; EGR prediction RNG seeded with seed;
  EO seed=seed.
- Run 026: split random_state=seed; AL seed=seed.
- Run 033: split random_state=seed; AL seed=seed; EO seed=seed.
- Run 034: split random_state=seed; AL seed=seed; ROC has no independent RNG.

Consequently, the variability measured by Runs 008–034 represents the
combined effect of changing the data split and changing algorithmic
randomness.

It does NOT isolate algorithmic seed sensitivity on a fixed data split.

## Implication for new experiment

The advisor-requested experiment should therefore use:

    split_seed = 101

for all trials and vary only:

    algorithm_seed = 101 through 120

For each repetition, the same algorithm_seed should be supplied to all
stochastic components of that configuration.

This produces a clean comparison:

Existing 10-seed experiment:
    split variation + algorithmic variation

New fixed-split experiment:
    algorithmic variation only
