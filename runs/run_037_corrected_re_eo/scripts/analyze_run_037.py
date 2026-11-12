"""Report from official AIF360 Run 037 per-seed output. Requires pandas/scipy only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'metadata'
METRICS = [
    ('accuracy', 'Accuracy', 1.),
    ('disparate_impact', 'DI', 1.),
    ('statistical_parity_difference', 'SPD', 0.),
    ('equal_opportunity_difference', 'EOD', 0.),
    ('predictive_equality_difference', 'PED', 0.),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='A')
    args = ap.parse_args()
    p = OUT / f'per_seed_results_{args.tag}.csv'
    if not p.exists():
        raise SystemExit(f'Missing official results: {p}. Execute Run 037 first.')
    d = pd.read_csv(p)
    assert len(d) == 10 and sorted(d.seed.to_list()) == list(range(101, 111))
    assert d['run008_pre_eo_match'].all() and d['run012_post_eo_match'].all()
    assert d['run018_post_eo_match'].all() and d['lr_used_instance_weights'].all()

    lines = [
        '# Run 037 — Corrected Reweighing + EO', '',
        'The original Run 018 and the corrected Run 037 were compared with the '
        'same split, seeds and EO calibration procedure. **EO was calibrated on the '
        'test set in both branches**, following the reproduction; these are not '
        'unbiased held-out test estimates.', '',
        '| Metric | Original Re+EO (Run 018) | Corrected Re+EO (Run 037) | '
        'Paired change (037−018) | 95% t interval for change |',
        '|---|---:|---:|---:|---:|',
    ]
    for key, label, ideal in METRICS:
        orig = d[f'control_post_eo_{key}'].to_numpy(float)
        new = d[key].to_numpy(float)
        delta = new - orig
        mean = np.mean(delta)
        sem = np.std(delta, ddof=1) / np.sqrt(len(delta))
        half = t.ppf(.975, df=len(delta)-1)*sem
        lines.append(f'| {label} | {np.mean(orig):.6f} | {np.mean(new):.6f} | '
                     f'{mean:+.6f} | [{mean-half:+.6f}, {mean+half:+.6f}] |')
    lines += ['', '## Prediction-level diagnostics', '',
              f'- Seeds with changes **before EO**: '
              f'{int((d.base_vs_weighted_pre_eo_changed_predictions > 0).sum())}/10',
              f'- Seeds with changes **after EO**: '
              f'{int((d.base_vs_weighted_post_eo_changed_predictions > 0).sum())}/10',
              f'- Mean predictions changed before EO: '
              f'{d.base_vs_weighted_pre_eo_changed_predictions.mean():.1f} of '
              f'{int(d.test_rows.iloc[0])}',
              f'- Mean predictions changed after EO: '
              f'{d.base_vs_weighted_post_eo_changed_predictions.mean():.1f} of '
              f'{int(d.test_rows.iloc[0])}',
              f'- Direct weighted pre-EO result matches frozen Run 010 metrics: '
              f'{int(d.run010_weighted_pre_eo_metrics_match.sum())}/10 seeds',
              '',
              'The Run 010 match is a diagnostic rather than a requirement, since '
              'Run 010 used a pandas input and Run 018/037 use AIF360 feature arrays.',
              '', '## Fairness-distance diagnostics', '',
              '| Metric | Mean absolute distance from ideal, original | '
              'Mean absolute distance from ideal, corrected |',
              '|---|---:|---:|',
    ]
    for key, label, ideal in METRICS[1:]:
        old_dist = np.mean(np.abs(d[f'control_post_eo_{key}'] - ideal))
        new_dist = np.mean(np.abs(d[key] - ideal))
        lines.append(f'| {label} | {old_dist:.6f} | {new_dist:.6f} |')
    lines += ['',
              '**Interpretation:** Lower absolute distance from the ideal indicates '
              'better parity for each individual metric, but different fairness '
              'criteria can conflict. These estimates are descriptive across ten '
              'fixed experiment seeds, not independent population samples.', '']
    out = '\n'.join(lines)
    target = OUT / f'analysis_report_{args.tag}.md'
    target.write_text(out + '\n', encoding='utf-8')
    print(out)
    print('\nReport saved to', target)

if __name__ == '__main__':
    main()
