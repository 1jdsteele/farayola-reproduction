"""Summarize official Run 038 AIF360 held-out results; requires pandas and scipy.

This script does not refit anything or access the target labels to make a
prediction. Do not run it as if it generates new experimental observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
METRICS = [
    ('accuracy', 'Accuracy', None),
    ('disparate_impact', 'DI', 1.0),
    ('statistical_parity_difference', 'SPD', 0.0),
    ('equal_opportunity_difference', 'EOD', 0.0),
    ('predictive_equality_difference', 'PED', 0.0),
]
SEEDS = list(range(101, 111))


def paired_interval(v: np.ndarray) -> str:
    x = np.asarray(v, dtype=float)
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return f'[{mean-half:+.6f}, {mean+half:+.6f}]'


def report(frame: pd.DataFrame, prediction_rows: pd.DataFrame,
           membership: pd.DataFrame, source: str) -> str:
    assert len(frame) == 10 and sorted(frame.seed.to_list()) == SEEDS
    assert frame['frozen_run008_outer_pre_eo_match'].all()
    assert frame['frozen_run010_weights_match'].all()
    assert frame['lr_used_instance_weights'].all()
    assert (~frame['eo_calibration_uses_test'].astype(bool)).all()
    assert frame['control_eo_lp_success'].all()
    assert frame['weighted_eo_lp_success'].all()
    assert (frame.train_rows == 3694).all()
    assert (frame.calibration_rows == 792).all()
    assert (frame.test_rows == 792).all()
    assert (frame.outer_holdout_rows == 1584).all()
    assert set(prediction_rows.split.unique()) == {'calibration', 'test'}
    assert len(prediction_rows) == 10 * (792 + 792)
    assert len(membership) == 10 * 5278
    for seed in SEEDS:
        sub = membership.loc[membership.seed == seed]
        assert len(sub) == 5278 and sub.original_row_id.nunique() == 5278
        assert sub.groupby('split').size().to_dict() == {
            'train': 3694, 'calibration': 792, 'test': 792,
        }
        for split in ('calibration', 'test'):
            pred = prediction_rows.loc[
                (prediction_rows.seed == seed) & (prediction_rows.split == split)]
            ids = set(membership.loc[(membership.seed == seed) &
                                     (membership.split == split), 'original_row_id'])
            assert set(pred.original_row_id) == ids
        test = prediction_rows.loc[
            (prediction_rows.seed == seed) & (prediction_rows.split == 'test')]
        row = frame.loc[frame.seed == seed].iloc[0]
        assert int((test.control_pre_eo != test.weighted_pre_eo).sum()) == \
            int(row.test_pre_eo_changed_predictions)
        assert int((test.control_post_eo != test.weighted_post_eo).sum()) == \
            int(row.test_post_eo_changed_predictions)

    lines = [
        '# Run 038 — Held-out Equalized Odds vs. corrected Reweighing + EO', '',
        '**Protocol:** Within each seed, 70% of COMPAS observations train logistic '
        'regression and Reweighing; 15% separately calibrate the Equalized Odds '
        '(EO) policy; and 15% are held out for the final evaluation. '
        'These sets do not overlap. Both branches use the same splits and EO seed.', '',
        '**Scientific distinction from Run 037:** Run 037 used the same 30% '
        'dataset for EO calibration and evaluation. Run 038 does not. '
        'The two experiments therefore must not be compared as paired '
        'treatment effects; the valid Run 038 comparison is *within Run 038*.', '',
        '## Held-out test results, after EO', '',
        '| Metric | Control EO (unweighted) | Corrected Re + EO | '
        'Paired difference (corrected − control) | 95% seed-based t interval |',
        '|---|---:|---:|---:|---:|',
    ]
    for name, label, ideal in METRICS:
        control = frame[f'test_control_post_eo_{name}'].to_numpy(float)
        corrected = frame[f'test_weighted_post_eo_{name}'].to_numpy(float)
        delta = corrected - control
        lines.append(
            f'| {label} | {control.mean():.6f} | {corrected.mean():.6f} | '
            f'{delta.mean():+.6f} | {paired_interval(delta)} |'
        )

    lines += ['', '## Upstream classifier on the same held-out test observations', '',
              '| Metric | Unweighted logistic regression | Weighted logistic regression | '
              'Difference (weighted − unweighted) |',
              '|---|---:|---:|---:|']
    for name, label, _ in METRICS:
        base = frame[f'test_control_pre_eo_{name}'].to_numpy(float)
        weighted = frame[f'test_weighted_pre_eo_{name}'].to_numpy(float)
        lines.append(f'| {label} | {base.mean():.6f} | {weighted.mean():.6f} | '
                     f'{(weighted-base).mean():+.6f} |')

    n_pre = int((frame.test_pre_eo_changed_predictions > 0).sum())
    n_post = int((frame.test_post_eo_changed_predictions > 0).sum())
    delta_acc = frame['test_delta_post_eo_accuracy'].to_numpy(float)
    lines += [
        '', '## Prediction and optimization diagnostics', '',
        f'- All group/label combinations present; EO optimization successful: '
        f'{int((frame.control_eo_lp_success & frame.weighted_eo_lp_success).sum())}/10 seeds.',
        '- Frozen Run 008 upstream model matches: 10/10 seeds.',
        '- Reweighing weights match frozen Run 010: 10/10 seeds.',
        f'- Test-seed predictions changed before EO: {n_pre}/10; '
        f'mean changed {frame.test_pre_eo_changed_predictions.mean():.1f}/792.',
        f'- Test-seed predictions changed after EO: {n_post}/10; '
        f'mean changed {frame.test_post_eo_changed_predictions.mean():.1f}/792.',
        f'- Accuracy improvements after EO: {int((delta_acc > 0).sum())}/10 seeds; '
        f'decreases: {int((delta_acc < 0).sum())}/10; '
        f'ties: {int((delta_acc == 0).sum())}/10.',
        '', '## Mean absolute fairness distance from ideal (held-out test)', '',
        '| Metric | Control EO | Corrected Re + EO | Mean paired change in distance |',
        '|---|---:|---:|---:|',
    ]
    for name, label, ideal in METRICS[1:]:
        base = np.abs(frame[f'test_control_post_eo_{name}'].to_numpy(float) - ideal)
        corrected = np.abs(frame[f'test_weighted_post_eo_{name}'].to_numpy(float) - ideal)
        lines.append(f'| {label} | {base.mean():.6f} | {corrected.mean():.6f} | '
                     f'{(corrected-base).mean():+.6f} |')

    lines += [
        '', '## Calibration-versus-test diagnostic (post-EO accuracy)', '',
        '| Branch | EO calibration-set accuracy (optimistic / in-sample) | '
        'Held-out test accuracy |',
        '|---|---:|---:|',
    ]
    for branch, description in (('control', 'EO'), ('weighted', 'Corrected Re + EO')):
        cal = frame[f'calibration_{branch}_post_eo_accuracy'].mean()
        test = frame[f'test_{branch}_post_eo_accuracy'].mean()
        lines.append(f'| {description} | {cal:.6f} | {test:.6f} |')

    lines += [
        '', '## Interpretation cautions', '',
        '- These are held-out **within the reconstructed COMPAS cohort**, not '
        'external-dataset validation.',
        '- Different seeds share many training and evaluated observations; the '
        'ten paired seed estimates are not independent population samples. '
        'Intervals summarize seed variation rather than population uncertainty.',
        '- EO is randomized; fitted policy parameters and realized finite-sample '
        'fairness can differ. No improvement is presumed.',
        '- Different fairness definitions may conflict. Prefer mean absolute '
        'distance from ideal when interpreting directional fairness differences.',
        '- Maintain the fixed protocol for this experiment; do not tune on the '
        'held-out test results. Future model selection requires nested validation.',
        '', f'*Source result file: `{source}`*', '',
    ]
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='A')
    ap.add_argument('--results-dir', default=None, type=Path,
                    help='Default is ../metadata, relative to this script')
    args = ap.parse_args()
    out = args.results_dir or ROOT / 'metadata'
    path = out / f'per_seed_results_{args.tag}.csv'
    if not path.is_file():
        raise SystemExit(f'No completed Run 038 results: {path}. Run the experiment first.')
    d = pd.read_csv(path)
    prediction_rows = pd.read_csv(out / f'prediction_level_{args.tag}.csv')
    membership = pd.read_csv(out / f'split_membership_{args.tag}.csv')
    text = report(d, prediction_rows, membership, str(path))
    target = out / f'analysis_report_{args.tag}.md'
    target.write_text(text + '\n', encoding='utf-8')
    print(text)
    print('Report saved to', target)


if __name__ == '__main__':
    main()
