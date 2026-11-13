"""Validate and summarize saved Run 041 calibration/held-out ROC experiments.

No model fitting and no AIF360 imports. Avoids tabulate dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'metadata'
METRICS = (
    'accuracy', 'statistical_parity_difference', 'disparate_impact',
    'equal_opportunity_difference', 'predictive_equality_difference',
)
ARMS = (
    ('A_hard_unweighted', 'A: ROC / hard labels (no Re)'),
    ('B_hard_weighted', 'B: Re + ROC / hard labels'),
    ('C_prob_unweighted', 'C: ROC / probability scores (no Re)'),
    ('D_prob_weighted', 'D: Re + ROC / probability scores'),
)
CONTRASTS = (
    ('B − A: weights added, hard ROC scores', 'B_hard_weighted', 'A_hard_unweighted'),
    ('C − A: continuous scores, no Re', 'C_prob_unweighted', 'A_hard_unweighted'),
    ('D − B: continuous scores after Re', 'D_prob_weighted', 'B_hard_weighted'),
    ('D − C: Re weights with continuous scores', 'D_prob_weighted', 'C_prob_unweighted'),
    ('D − A: both fixes vs original', 'D_prob_weighted', 'A_hard_unweighted'),
)


def markdown(frame: pd.DataFrame, decimals: int = 4, include_index: bool = False) -> str:
    if include_index:
        frame = frame.reset_index()
    cols = list(frame.columns)
    def fmt(v):
        if pd.isna(v):
            return 'n/a'
        if isinstance(v, (float, np.floating)):
            return f'{v:.{decimals}f}'
        if isinstance(v, (bool, np.bool_)):
            return 'yes' if v else 'no'
        return str(v).replace('|', '\\|')
    return '\n'.join([
        '| ' + ' | '.join(map(str, cols)) + ' |',
        '| ' + ' | '.join('---' for _ in cols) + ' |',
        *('| ' + ' | '.join(fmt(v) for v in row) + ' |'
          for row in frame.itertuples(index=False, name=None))
    ])


def calculate_metrics(g: pd.DataFrame, label_col: str) -> dict[str, float]:
    y = g.true_label.to_numpy(dtype=int)
    p = g[label_col].to_numpy(dtype=int)
    def rates(race: int) -> tuple[float, float, float]:
        mask = g.race.to_numpy(dtype=int) == race
        yy = y[mask]
        pp = p[mask]
        assert len(yy) and sum(yy == 1) and sum(yy == 0)
        return (float((pp == 1).mean()),
                float((pp[yy == 1] == 1).mean()),
                float((pp[yy == 0] == 1).mean()))
    u = rates(0)
    priv = rates(1)
    assert priv[0] != 0
    return {
        'accuracy': float((p == y).mean()),
        'statistical_parity_difference': u[0] - priv[0],
        'disparate_impact': u[0] / priv[0],
        'equal_opportunity_difference': u[1] - priv[1],
        'predictive_equality_difference': u[2] - priv[2],
    }


def verify_prediction_level(frame: pd.DataFrame, prediction: pd.DataFrame,
                            seeds: list[int]) -> None:
    assert {'seed', 'split', 'arm', 'original_row_id', 'race', 'true_label',
            'lr_predicted_label', 'lr_positive_probability', 'roc_input_score',
            'threshold_only_label', 'roc_predicted_label'} <= set(prediction.columns)
    assert len(prediction) == len(seeds) * 2 * len(ARMS) * 792
    assert not prediction.duplicated(['seed', 'split', 'arm', 'original_row_id']).any()
    assert set(prediction.split) == {'calibration', 'test'}
    assert set(prediction.arm) == {x[0] for x in ARMS}
    for col in ('race', 'true_label', 'lr_predicted_label',
                'threshold_only_label', 'roc_predicted_label'):
        assert set(prediction[col].unique()) <= {0, 1}, col
    assert prediction.lr_positive_probability.between(0, 1).all()
    assert prediction.roc_input_score.between(0, 1).all()
    recorded = frame.set_index(['seed', 'arm'])

    for seed in seeds:
        cal_ids = None
        test_ids = None
        for arm, _ in ARMS:
            for split in ('calibration', 'test'):
                sub = prediction[(prediction.seed == seed)
                                 & (prediction.arm == arm)
                                 & (prediction.split == split)]
                assert len(sub) == 792, (seed, arm, split)
                ids = sub.original_row_id.to_numpy()
                if split == 'calibration':
                    if cal_ids is None:
                        cal_ids = ids
                    else:
                        assert np.array_equal(cal_ids, ids)
                else:
                    if test_ids is None:
                        test_ids = ids
                    else:
                        assert np.array_equal(test_ids, ids)
                assert len(np.unique(ids)) == 792
                record = recorded.loc[(seed, arm)]
                assert not bool(record['test_labels_used_for_roc_fit'])
                threshold = float(record['roc_threshold'])
                assert np.array_equal(sub.threshold_only_label.to_numpy(dtype=int),
                                      (sub.roc_input_score.to_numpy() > threshold).astype(int))
                if arm.startswith(('A_', 'B_')):
                    assert np.array_equal(sub.roc_input_score.to_numpy(dtype=int),
                                          sub.lr_predicted_label.to_numpy(dtype=int))
                else:
                    assert np.allclose(sub.roc_input_score.to_numpy(),
                                       sub.lr_positive_probability.to_numpy(),
                                       atol=1e-12, rtol=0)
                pre = sub.lr_predicted_label.to_numpy()
                post = sub.roc_predicted_label.to_numpy()
                threshold_pred = sub.threshold_only_label.to_numpy()
                y = sub.true_label.to_numpy()
                checks = {
                    'roc_changes_from_lr05': int((pre != post).sum()),
                    'threshold_only_changes_from_lr05': int((pre != threshold_pred).sum()),
                    'margin_only_changes_from_threshold': int((threshold_pred != post).sum()),
                    'correct_to_incorrect_from_lr05': int(((pre == y) & (post != y)).sum()),
                    'incorrect_to_correct_from_lr05': int(((pre != y) & (post == y)).sum()),
                }
                for key, count in checks.items():
                    expected = int(record[f'{split}_{key}'])
                    assert expected == count, (seed, arm, split, key, expected, count)
                for stage, col in (('pre_roc', 'lr_predicted_label'),
                                   ('post_roc', 'roc_predicted_label')):
                    manual = calculate_metrics(sub, col)
                    for key, value in manual.items():
                        saved = float(record[f'{split}_{stage}_{key}'])
                        assert np.isclose(saved, value, atol=1e-12, rtol=0), (
                            seed, arm, split, stage, key, saved, value
                        )
                eod = float(record[f'{split}_post_roc_equal_opportunity_difference'])
                feasible = bool(-0.05 - 1e-12 <= eod <= 0.05 + 1e-12)
                assert feasible == bool(record[f'{split}_roc_feasible'])
        assert set(cal_ids).isdisjoint(test_ids), f'Overlapping ROC calibration/test rows seed {seed}'


def t_summary(x: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    assert len(x) and np.isfinite(x).all()
    mean = float(x.mean())
    if len(x) < 2:
        return mean, float('nan'), float('nan'), float('nan')
    sd = float(x.std(ddof=1))
    hw = float(t.ppf(.975, len(x)-1) * sd / np.sqrt(len(x)))
    return mean, sd, mean - hw, mean + hw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='A')
    args = parser.parse_args()
    tag = args.tag
    assert tag.replace('_', '').replace('-', '').isalnum()
    path = OUT / f'per_seed_results_{tag}.csv'
    assert path.exists(), f'Missing {path}; execute Run 041 first'
    frame = pd.read_csv(path)
    prediction = pd.read_csv(OUT / f'prediction_level_{tag}.csv')
    seeds = sorted(frame.seed.unique().tolist())
    assert seeds and len(frame) == 4 * len(seeds)
    assert set(frame.arm) == {x[0] for x in ARMS}
    assert not frame.duplicated(['seed', 'arm']).any()
    if tag == 'A':
        assert seeds == list(range(101, 111))
    assert (frame.n_train == 3694).all()
    assert (frame.n_calibration == 792).all()
    assert (frame.n_test == 792).all()
    verify_prediction_level(frame, prediction, seeds)
    print('\nRUN 041: PREDICTION-LEVEL VALIDATION PASSED')

    names = [x[0] for x in ARMS]
    metrics = [f'test_post_roc_{m}' for m in METRICS]
    heldout = frame.groupby('arm')[metrics].mean().loc[names]
    print('\nRUN 041: HELD-OUT POST-ROC METRICS')
    print(heldout.round(6).to_string())

    feasibility = frame.groupby('arm')[['calibration_roc_feasible', 'test_roc_feasible',
                                        'roc_warning_no_feasible_on_calibration']].sum().loc[names]
    print('\nCALIBRATION / TEST EOD FEASIBILITY (out of', len(seeds), 'seeds)')
    print(feasibility.to_string())
    generalization = frame.copy()
    generalization['feasible_on_calibration_but_not_test'] = (
        generalization.calibration_roc_feasible & ~generalization.test_roc_feasible
    )
    print('\nFEASIBLE ON CALIBRATION, INFEASIBLE ON TEST (seed counts)')
    print(generalization.groupby('arm').feasible_on_calibration_but_not_test.sum().loc[names])

    diag_names = [
        'roc_threshold', 'roc_margin', 'test_roc_changes_from_lr05',
        'test_threshold_only_changes_from_lr05',
        'test_margin_only_changes_from_threshold',
        'test_correct_to_incorrect_from_lr05',
        'test_incorrect_to_correct_from_lr05',
    ]
    diagnostic = frame.groupby('arm')[diag_names].mean().loc[names]
    print('\nROC PARAMETER AND TEST PREDICTION DIAGNOSTICS')
    print(diagnostic.round(3).to_string())

    paired_rows = []
    for name, a, b in CONTRASTS:
        a_rows = frame[frame.arm == a].set_index('seed').sort_index()
        b_rows = frame[frame.arm == b].set_index('seed').sort_index()
        assert a_rows.index.equals(b_rows.index)
        for metric in METRICS:
            diffs = (a_rows[f'test_post_roc_{metric}']
                     - b_rows[f'test_post_roc_{metric}']).to_numpy(dtype=float)
            mean, sd, lower, upper = t_summary(diffs)
            paired_rows.append({
                'comparison': name, 'metric': metric, 'n_seeds': len(diffs),
                'mean_delta': mean, 'sample_std': sd,
                'ci95_lower': lower, 'ci95_upper': upper,
                'min_delta': float(diffs.min()), 'max_delta': float(diffs.max()),
            })
    paired = pd.DataFrame(paired_rows)
    paired_path = OUT / f'analysis_paired_intervals_{tag}.csv'
    paired.to_csv(paired_path, index=False, float_format='%.15g')
    acc = paired[paired.metric == 'accuracy'].copy()
    for col in ('mean_delta', 'ci95_lower', 'ci95_upper'):
        acc[col] *= 100
    print('\nPAIRED HELD-OUT ACCURACY DIFFERENCES (percentage points)')
    print(acc[['comparison', 'mean_delta', 'ci95_lower', 'ci95_upper']].round(3).to_string(index=False))

    fairness_distance = frame.assign(
        test_absolute_eod=lambda d: d.test_post_roc_equal_opportunity_difference.abs(),
        test_absolute_spd=lambda d: d.test_post_roc_statistical_parity_difference.abs(),
        test_absolute_ped=lambda d: d.test_post_roc_predictive_equality_difference.abs(),
        test_di_distance=lambda d: (d.test_post_roc_disparate_impact - 1).abs(),
    ).groupby('arm')[['test_absolute_eod', 'test_absolute_spd',
                       'test_absolute_ped', 'test_di_distance']].mean().loc[names]

    display = heldout.rename(index={a:b for a,b in ARMS},
                            columns={f'test_post_roc_{k}':k for k in METRICS})
    feas_display = feasibility.rename(index={a:b for a,b in ARMS})
    diag_display = diagnostic.rename(index={a:b for a,b in ARMS})
    distance_display = fairness_distance.rename(index={a:b for a,b in ARMS})
    lines = [
        '# Run 041 — Held-out ROC and corrected Reweighing + ROC', '',
        '**Data protocol:** COMPAS (5,278 observations), race-stratified 70% training, '
        '15% ROC calibration, 15% final testing; seeds 101–110 (or the selected smoke-test seeds). '
        'Calibration/test halves of the original 30% outer holdout are stratified by race and label.', '',
        '**Four arms:** A/B supply hard 0/1 scores to ROC, while C/D supply continuous '
        'logistic-regression probabilities. B/D use Reweighing weights in classifier training.', '',
        '**Strict separation:** ROC selects the threshold and uncertainty margin exclusively '
        'using the calibration set. Test labels are used only afterward to evaluate outcomes.', '',
        '## Held-out final metrics', '', markdown(display, 6, include_index=True), '',
        '## EOD ±0.05 feasibility', '', markdown(feas_display, 0, include_index=True), '',
        '## ROC parameters and test adjustments (means)', '',
        markdown(diag_display, 3, include_index=True), '',
        '## Mean absolute held-out fairness distance from ideal', '',
        markdown(distance_display, 5, include_index=True), '',
        '## Paired held-out accuracy differences (percentage points)', '',
        markdown(acc[['comparison', 'mean_delta', 'ci95_lower', 'ci95_upper']], 3), '',
        '## Interpretation cautions', '',
        '- Feasibility on the ROC calibration data is not the same as feasibility on held-out test data.',
        '- ROC may fall back to a candidate when no calibration candidate satisfies EOD ±0.05; examine warnings and feasibility by arm.',
        '- Threshold-only and margin-only changes can overlap or cancel; do not add their counts to obtain total changes.',
        '- A/B hard-score arms intentionally reproduce the released integration; C/D supply usable continuous confidence scores.',
        '- Runs 040 and 041 use different evaluation protocols and row counts. Compare interventions *within* Run 041.',
        '- Ten seeds reuse overlapping observations; seed-based 95% t intervals describe seed variation, not independent-population confidence.',
        '- The 15% test samples are unseen for model fitting and ROC calibration within each seed, but are not an external-dataset validation.',
        '- Do not choose new model settings using the final held-out test metrics.', '',
        f'**Source data:** `per_seed_results_{tag}.csv`, `prediction_level_{tag}.csv`, '
        f'`analysis_paired_intervals_{tag}.csv`.',
    ]
    report = OUT / f'analysis_report_{tag}.md'
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nRUN 041 ANALYSIS COMPLETE: {report}')


if __name__ == '__main__':
    main()
