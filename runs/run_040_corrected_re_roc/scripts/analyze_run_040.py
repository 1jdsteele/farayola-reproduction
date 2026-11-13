"""Analyze saved Run 040 A/B/C/D factorial results; do not rerun models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'metadata'
NAMES = [
    ('A_hard_unweighted', 'A: original ROC (hard scores, no Re)'),
    ('B_hard_weighted', 'B: Re weights only (hard scores)'),
    ('C_prob_unweighted', 'C: probability scores only (no Re)'),
    ('D_prob_weighted', 'D: Re weights and probability scores'),
]
METRICS = [
    'accuracy', 'statistical_parity_difference', 'disparate_impact',
    'equal_opportunity_difference', 'predictive_equality_difference',
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='A')
    args = parser.parse_args()
    tag = args.tag
    p = OUT / f'per_seed_results_{tag}.csv'
    assert p.exists(), f'Execute run_040_corrected_re_roc_10seed.py first: {p}'
    frame = pd.read_csv(p)
    seeds = sorted(frame.seed.unique())
    assert len(frame) == 4 * len(seeds)
    assert not frame.duplicated(['seed', 'arm']).any()
    assert set(frame.arm) == {key for key, _ in NAMES}
    if tag == 'A':
        assert seeds == list(range(101, 111)), 'Expected all ten seeds in frozen A run'
    by_arm = frame.set_index(['seed', 'arm']).sort_index()
    metrics = [f'post_roc_{m}' for m in METRICS]
    print('\nRUN 040: MEAN POST-ROC METRICS')
    report_table = frame.groupby('arm')[metrics].mean().loc[[key for key, _ in NAMES]]
    print(report_table.round(6).to_string())
    print('\nRUN 040: ROC DIAGNOSTICS')
    diagnostic_cols = [
        'roc_changes_from_lr05', 'threshold_only_changes_from_lr05',
        'margin_only_changes_from_threshold', 'roc_threshold', 'roc_margin',
        'correct_to_incorrect_from_lr05', 'incorrect_to_correct_from_lr05',
    ]
    print(frame.groupby('arm')[diagnostic_cols].mean().round(4).to_string())
    print('\nROC FEASIBILITY AND WARNINGS BY ARM (out of', len(seeds), 'seeds):')
    print(frame.groupby('arm')[['roc_feasible_on_fitted_test',
                                'roc_warning_no_feasible']].sum().to_string())

    compares = [
        ('B − A: weight fix, hard ROC scores', 'B_hard_weighted', 'A_hard_unweighted'),
        ('C − A: score fix, no Re', 'C_prob_unweighted', 'A_hard_unweighted'),
        ('D − B: score fix, with Re', 'D_prob_weighted', 'B_hard_weighted'),
        ('D − C: weight fix, ROC scores available', 'D_prob_weighted', 'C_prob_unweighted'),
        ('D − A: both fixes vs original', 'D_prob_weighted', 'A_hard_unweighted'),
    ]
    paired_rows = []
    for name, a, b in compares:
        a_sub = by_arm.xs(a, level='arm')
        b_sub = by_arm.xs(b, level='arm')
        assert a_sub.index.equals(b_sub.index)
        for metric in metrics:
            values = (a_sub[metric] - b_sub[metric]).to_numpy(dtype=float)
            assert np.isfinite(values).all()
            sample_sd = float(np.std(values, ddof=1)) if len(values) > 1 else float('nan')
            halfwidth = (float(t.ppf(.975, len(values)-1) * sample_sd / np.sqrt(len(values)))
                         if len(values) > 1 else float('nan'))
            paired_rows.append({
                'comparison': name, 'metric': metric, 'n_seeds': len(values),
                'mean_delta': float(values.mean()), 'sample_std': sample_sd,
                'min_delta': float(values.min()), 'max_delta': float(values.max()),
                'ci95_lower': float(values.mean()) - halfwidth,
                'ci95_upper': float(values.mean()) + halfwidth,
            })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / f'analysis_paired_intervals_{tag}.csv', index=False,
                  float_format='%.15g')
    print('\nPAIRED ACCURACY CHANGES (percentage points)')
    acc_table = paired[paired.metric == 'post_roc_accuracy'].copy()
    for col in ('mean_delta', 'ci95_lower', 'ci95_upper'):
        acc_table[col] *= 100
    print(acc_table[['comparison','mean_delta','ci95_lower','ci95_upper']].round(3).to_string(index=False))

    prediction = pd.read_csv(OUT / f'prediction_level_{tag}.csv')
    assert len(prediction) == 4 * len(seeds) * 1584
    group_counts = prediction.groupby(['seed','arm','race']).size()
    assert set(group_counts.index.get_level_values('race')) == {0, 1}
    assert set(prediction.lr_predicted_label.unique()) <= {0, 1}
    assert set(prediction.roc_predicted_label.unique()) <= {0, 1}
    assert set(prediction.threshold_only_label.unique()) <= {0, 1}
    # Explicitly verify every reported flip count against individual predictions.
    grouped = prediction.groupby(['seed', 'arm'])
    independently_calculated = grouped.apply(
        lambda g: int((g.lr_predicted_label != g.roc_predicted_label).sum()),
        include_groups=False,
    )
    reported = by_arm['roc_changes_from_lr05']
    assert independently_calculated.sort_index().equals(reported.astype(int).sort_index()), (
        'Prediction-level flip counts disagree with per-seed output'
    )
    # Hold each paired comparison to observed same-row test membership.
    for seed in seeds:
        sub = prediction[prediction.seed == seed]
        original_ids = None
        for arm, _ in NAMES:
            row_ids = sub[sub.arm == arm].original_row_id.to_numpy()
            if original_ids is None:
                original_ids = row_ids
            else:
                assert np.array_equal(original_ids, row_ids)
    print('\nPrediction-level checks passed: counts and rows aligned across arms.')

    lines = [
        '# Run 040 — Corrected Reweighing + Reject Option Classification',
        '',
        '**Protocol:** 70/30 race-stratified train/test split, ten seeds (101–110); '
        'ROC is fitted and evaluated on the SAME test observations to isolate '
        'the original integration decisions. This is not unbiased held-out evaluation.',
        '',
        'Four ROC-containing configurations isolate two factors: '
        'whether Reweighing weights reach logistic regression and whether ROC '
        'receives continuous prediction probabilities or hard 0/1 labels.',
        '',
        '## Mean post-ROC metrics', '',
        report_table.rename(index={a:b for a,b in NAMES},
                            columns={f'post_roc_{m}': m for m in METRICS})
        .round(6).to_markdown(), '',
        '## Mean ROC adjustment diagnostics', '',
        frame.groupby('arm')[diagnostic_cols].mean().round(4)
        .rename(index={a:b for a,b in NAMES}).to_markdown(), '',
        '## Paired accuracy changes (percentage points)', '',
        acc_table[['comparison','mean_delta','ci95_lower','ci95_upper']]
        .round(4).to_markdown(index=False), '',
        '## Key cautions', '',
        '- A and B use hard 0/1 scores as in the original reproduction; C and D use actual `predict_proba` scores.',
        '- ROC search uses EOD-only fairness constraints ±0.05 and may fall back if no feasible candidate is found.',
        '- ROC calibration and evaluation use the same test observations; interpret as implementation diagnostics only.',
        '- The ten seed estimates overlap in their underlying data. Seed intervals do not establish population confidence.',
        '- Follow-up held-out calibration/test evaluation is needed before generalization claims.',
        '- Larger final accuracy or fairness improvements are not guaranteed by either correction.',
    ]
    (OUT / f'analysis_report_{tag}.md').write_text('\n'.join(lines) + '\n')
    print(f'\nRUN 040 ANALYSIS COMPLETE: {OUT / f"analysis_report_{tag}.md"}')


if __name__ == '__main__':
    main()
