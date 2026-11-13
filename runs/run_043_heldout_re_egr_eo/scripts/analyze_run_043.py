"""Analyze/validate Run 043. No AIF360/Fairlearn required for analysis.

All metrics are independently recomputed from per-observation predictions.
Never equate calibration performance with held-out generalization.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'metadata'
METRICS = ('accuracy','statistical_parity_difference','disparate_impact',
           'equal_opportunity_difference','predictive_equality_difference')
ARMS = ('legacy_egr','unity_oracle','reweighted_oracle')
STAGES = ('pre_eo','post_eo')


def check_close(actual, expected, what, tol=2e-10):
    if not np.isclose(actual,expected,atol=tol,rtol=0,equal_nan=False):
        raise AssertionError(f'{what}: {actual} != {expected}')


def raw_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    true = frame.true_label.to_numpy(dtype=int)
    yhat = frame[pred_col].to_numpy(dtype=int)
    group = frame.race.to_numpy(dtype=int)
    assert set(np.unique(true)).issubset({0,1})
    assert set(np.unique(yhat)).issubset({0,1})
    assert set(np.unique(group)) == {0,1}
    rates = {}
    for g in (0,1):
        m=group==g
        if not m.any() or not (m&(true==0)).any() or not (m&(true==1)).any():
            raise AssertionError('Missing protected group/true class')
        rates[g] = (
            float(np.mean(yhat[m])),
            float(np.mean(yhat[m&(true==1)])),
            float(np.mean(yhat[m&(true==0)])),
        )
    r0,r1=rates[0],rates[1]
    if r1[0] == 0:
        raise AssertionError('DI denominator is zero')
    tn=int(np.sum((true==0)&(yhat==0)))
    fp=int(np.sum((true==0)&(yhat==1)))
    fn=int(np.sum((true==1)&(yhat==0)))
    tp=int(np.sum((true==1)&(yhat==1)))
    return dict(accuracy=float(np.mean(true==yhat)),
                statistical_parity_difference=r0[0]-r1[0],
                disparate_impact=r0[0]/r1[0],
                equal_opportunity_difference=r0[1]-r1[1],
                predictive_equality_difference=r0[2]-r1[2],
                tn=tn,fp=fp,fn=fn,tp=tp)


def paired_summary(deltas: np.ndarray) -> dict:
    n=len(deltas)
    if n < 2: return dict(n=n, mean_delta=float(np.mean(deltas)), sample_std=float('nan'),
                          ci95_lower=float('nan'),ci95_upper=float('nan'))
    mean=float(np.mean(deltas))
    sd=float(np.std(deltas,ddof=1))
    h=float(student_t.ppf(.975,n-1)*sd/math.sqrt(n))
    return dict(n=n,mean_delta=mean,sample_std=sd,ci95_lower=mean-h,ci95_upper=mean+h,
                seeds_with_nonzero_change=int(np.sum(np.abs(deltas)>1e-12)))


def markdown_table(df: pd.DataFrame, col_names: list[str] | None=None) -> str:
    table=df.copy()
    if col_names is not None:
        table.columns=col_names
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col]=table[col].map(lambda x:'—' if pd.isna(x) else f'{x:.6f}')
    names=[str(x) for x in table.columns]
    lines=['| '+' | '.join(names)+' |', '| '+ ' | '.join('---' for _ in names)+' |']
    lines += ['| '+' | '.join(str(v) for v in record)+' |' for record in table.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='A')
    args=parser.parse_args()
    suf=args.tag
    inputs={name:META/f'{name}_{suf}.csv' for name in
            ('per_seed_results','prediction_level','oracle_fit_traces',
             'split_membership','group_label_counts','prediction_change_checks')}
    files={key:pd.read_csv(path) for key,path in inputs.items()}
    r=files['per_seed_results']; p=files['prediction_level']; traces=files['oracle_fit_traces']
    splits=files['split_membership']; changes=files['prediction_change_checks']
    manifest=json.loads((META/f'run_manifest_{suf}.json').read_text())
    seeds=sorted(set(r.seed.to_numpy(dtype=int)))
    if seeds != sorted(manifest['seeds']) or not set(seeds).issubset(set(range(101,111))):
        raise AssertionError('Unexpected seed list')
    if len(r)!=len(seeds)*len(ARMS)*2 or len(p)!=len(seeds)*len(ARMS)*2*792:
        raise AssertionError('Unexpected per-seed or prediction-level row count')
    assert len(splits)==len(seeds)*5278
    data=pd.read_csv(ROOT/'data/compas_farayola_numeric.csv')
    if set(p.split)!= {'calibration','test'} or set(p.arm)!=set(ARMS):
        raise AssertionError('Missing arms or partitions')
    if set(traces.arm)!= {'unity_oracle','reweighted_oracle'}:
        raise AssertionError('Unexpected oracle trace arms')
    if not r.eo_lp_success.astype(bool).all():
        raise AssertionError('EO optimization failed in at least one branch')
    if not r.outer_frozen_run011_match.astype(bool).all() or not r.unity_prediction_match.astype(bool).all():
        raise AssertionError('EGR frozen or unity model validation failed')

    diag=[]
    for seed in seeds:
        mem=splits[splits.seed==seed]
        sets={name:set(mem.loc[mem.split==name,'original_row_id'].astype(int))
              for name in ('train','calibration','test')}
        if list(map(len,sets.values())) != [3694,792,792]:
            raise AssertionError(f'Seed {seed}: wrong split sizes')
        if len(sets['train']|sets['calibration']|sets['test'])!=5278:
            raise AssertionError(f'Seed {seed}: split does not cover 5278 distinct observations')
        assert not (sets['train']&sets['calibration'] or sets['train']&sets['test'] or sets['calibration']&sets['test'])
        for split in ('calibration','test'):
            branches={}
            for arm in ARMS:
                part=p.loc[(p.seed==seed)&(p.split==split)&(p.arm==arm)].copy()
                if len(part)!=792 or len(set(part.original_row_id))!=792:
                    raise AssertionError(f'Seed {seed}/{split}/{arm}: wrong prediction count')
                if set(part.original_row_id.astype(int))!=sets[split]:
                    raise AssertionError(f'Seed {seed}/{split}/{arm}: prediction split mismatch')
                ids=part.original_row_id.to_numpy(dtype=int)
                if not np.array_equal(part.true_label.to_numpy(dtype=int),data.iloc[ids].two_year_recid.to_numpy(dtype=int)):
                    raise AssertionError('Prediction truths do not match source dataset')
                if not np.array_equal(part.race.to_numpy(dtype=int),data.iloc[ids].race.to_numpy(dtype=int)):
                    raise AssertionError('Prediction races do not match source dataset')
                row=r[(r.seed==seed)&(r.split==split)&(r.arm==arm)]
                if len(row)!=1: raise AssertionError('Missing/duplicate result row')
                row=row.iloc[0]
                branches[arm]=part.set_index('original_row_id')
                for stage in STAGES:
                    measured=raw_metrics(part, 'pre_eo_prediction' if stage=='pre_eo' else 'post_eo_prediction')
                    for metric,value in measured.items():
                        check_close(row[f'{stage}_{metric}'],value,f'{seed}/{split}/{arm}/{stage}/{metric}')
                flip=(part.pre_eo_prediction.to_numpy(dtype=int)!=part.post_eo_prediction.to_numpy(dtype=int))
                true=part.true_label.to_numpy(dtype=int)
                pre=part.pre_eo_prediction.to_numpy(dtype=int)
                post=part.post_eo_prediction.to_numpy(dtype=int)
                harmful=int(np.sum(flip&(pre==true)))
                helpful=int(np.sum(flip&(post==true)))
                if flip.sum()!=harmful+helpful: raise AssertionError('Flip accounting error')
                for key,value in (('eo_changed_predictions',int(flip.sum())),
                                  ('eo_harmful_flips',harmful),('eo_helpful_flips',helpful)):
                    if int(row[key])!=value: raise AssertionError(f'{seed}/{arm}: {key} mismatch')
            # Use stable original-row IDs, because each arm uses the same split.
            for stage in STAGES:
                name='pre_eo_prediction' if stage=='pre_eo' else 'post_eo_prediction'
                base=branches['legacy_egr'][name]
                unity=branches['unity_oracle'][name].reindex(base.index)
                weighted=branches['reweighted_oracle'][name].reindex(base.index)
                if not base.equals(unity):
                    raise AssertionError(f'Seed {seed}/{split}: unity model differs at {stage}')
                difference=int(np.sum(base.to_numpy(dtype=int)!=weighted.to_numpy(dtype=int)))
                for arm,target in (('legacy_egr',0),('unity_oracle',0),('reweighted_oracle',difference)):
                    value=int(r.loc[(r.seed==seed)&(r.split==split)&(r.arm==arm),
                                    f'{stage}_vs_legacy_changes'].iloc[0])
                    if value!=target:raise AssertionError('Cross-arm prediction comparison mismatch')
                if split=='test' and stage=='post_eo':
                    diag.append({'seed':seed,'pre_eo_weighted_vs_legacy':int(np.sum(
                        branches['legacy_egr']['pre_eo_prediction'].to_numpy(dtype=int)!=
                        branches['reweighted_oracle']['pre_eo_prediction'].reindex(base.index).to_numpy(dtype=int))),
                                 'post_eo_weighted_vs_legacy':int(np.sum(
                        branches['legacy_egr']['post_eo_prediction'].to_numpy(dtype=int)!=
                        branches['reweighted_oracle']['post_eo_prediction'].reindex(base.index).to_numpy(dtype=int)))})
        for arm in ('unity_oracle','reweighted_oracle'):
            fits=traces[(traces.seed==seed)&(traces.arm==arm)]
            row=r[(r.seed==seed)&(r.arm==arm)&(r.split=='test')].iloc[0]
            if len(fits)!=row.external_oracle_fits or len(fits)!=row.egr_oracle_calls-row.egr_dummy_oracle_calls:
                raise AssertionError('Oracle traces do not match reported oracle calls')
            if not np.allclose(fits.internal_weight_mass,fits.combined_weight_mass,rtol=0,atol=1e-8):
                raise AssertionError('Oracle weight mass mismatch')
            assert fits.n_rows.eq(3694).all()
            changed=fits.changed_from_internal.astype(bool)
            if arm=='unity_oracle' and changed.any():raise AssertionError('Unity weight changed an oracle')
            if arm=='reweighted_oracle' and not changed.all():raise AssertionError('Re weights ignored in oracle')

    diag=pd.DataFrame(diag)
    merged=diag.merge(changes,on='seed',validate='one_to_one')
    for ours,recorded in (('pre_eo_weighted_vs_legacy','test_pre_eo_weighted_vs_legacy_flips'),
                          ('post_eo_weighted_vs_legacy','test_post_eo_weighted_vs_legacy_flips')):
        if not np.array_equal(merged[ours],merged[recorded]):
            raise AssertionError('Prediction flip diagnostics mismatch')
    for name in ('test_pre_eo_unity_vs_legacy_flips','test_post_eo_unity_vs_legacy_flips'):
        if merged[name].ne(0).any():raise AssertionError('Unity flips not zero')

    # Summaries computed on individual seeds, not on pooled predictions.
    aggregates=[]
    for (arm,split),part in r.groupby(['arm','split']):
        for stage in STAGES:
            for metric in METRICS:
                x=part[f'{stage}_{metric}'].to_numpy(dtype=float)
                aggregates.append(dict(arm=arm,split=split,stage=stage,metric=metric,
                                       mean=float(x.mean()),sample_std=float(np.std(x,ddof=1)) if len(x)>1 else float('nan')))
            for metric in ('statistical_parity_difference','equal_opportunity_difference',
                           'predictive_equality_difference'):
                x=part[f'{stage}_{metric}'].to_numpy(dtype=float)
                aggregates.append(dict(arm=arm,split=split,stage=stage,metric='abs_'+metric,
                                       mean=float(np.abs(x).mean()),sample_std=float(np.abs(x).std(ddof=1)) if len(x)>1 else float('nan')))
            x=np.abs(part[f'{stage}_disparate_impact'].to_numpy(dtype=float)-1.0)
            aggregates.append(dict(arm=arm,split=split,stage=stage,metric='abs_DI_minus_1',
                                   mean=float(x.mean()),sample_std=float(x.std(ddof=1)) if len(x)>1 else float('nan')))
        for metric in ('eo_changed_predictions','eo_harmful_flips','eo_helpful_flips'):
            aggregates.append(dict(arm=arm,split=split,stage='post_eo',metric=metric,
                                   mean=float(part[metric].mean()),
                                   sample_std=float(part[metric].std(ddof=1)) if len(part)>1 else float('nan')))
    aggregate=pd.DataFrame(aggregates)
    aggregate.to_csv(META/f'aggregate_summary_{suf}.csv',index=False,float_format='%.15g')

    pairs=[]
    for split in ('calibration','test'):
        selected=r[r.split==split]
        wide=selected.set_index(['seed','arm'])
        for left,right,title in (
            ('reweighted_oracle','legacy_egr','Corrected Re+EGR vs EGR'),
            ('unity_oracle','legacy_egr','Unity vs legacy (must be zero)'),
        ):
            for stage in STAGES:
                for metric in METRICS:
                    a=wide.xs(left,level='arm')[f'{stage}_{metric}'].sort_index()
                    b=wide.xs(right,level='arm')[f'{stage}_{metric}'].sort_index()
                    pair=paired_summary((a-b).to_numpy(dtype=float))
                    pairs.append(dict(split=split,comparison=title,stage=stage,metric=metric,**pair))
                for metric in ('statistical_parity_difference','equal_opportunity_difference',
                               'predictive_equality_difference'):
                    a=wide.xs(left,level='arm')[f'{stage}_{metric}'].sort_index().abs()
                    b=wide.xs(right,level='arm')[f'{stage}_{metric}'].sort_index().abs()
                    pair=paired_summary((a-b).to_numpy(dtype=float))
                    pairs.append(dict(split=split,comparison=title,stage=stage,metric='abs_'+metric,**pair))
    paired=pd.DataFrame(pairs)
    paired.to_csv(META/f'paired_deltas_{suf}.csv',index=False,float_format='%.15g')

    test=r[r.split=='test'].copy()
    cal=r[r.split=='calibration'].copy()
    # Descriptive check, NOT a fixed bound or explicit constraint in EqOddsPostprocessing.
    diagnostic=[]
    for arm in ARMS:
        for split,part in (('calibration',cal),('test',test)):
            part=part[part.arm==arm]
            diagnostic.append(dict(arm=arm,split=split,
                    n_seeds=len(part),post_eo_abs_EOD_le_0_05=int((part.post_eo_equal_opportunity_difference.abs()<=.05).sum()),
                    post_eo_abs_PED_le_0_05=int((part.post_eo_predictive_equality_difference.abs()<=.05).sum()),
                    post_eo_both_abs_EOD_PED_le_0_05=int(((part.post_eo_equal_opportunity_difference.abs()<=.05)&
                      (part.post_eo_predictive_equality_difference.abs()<=.05)).sum())))
    diagnostic=pd.DataFrame(diagnostic)
    diagnostic.to_csv(META/f'descriptive_gap_thresholds_{suf}.csv',index=False)
    print('RUN 043 — INDEPENDENT PREDICTION / ORACLE / SPLIT VALIDATION PASSED')
    chart=aggregate[(aggregate.split=='test')&(aggregate.stage=='post_eo')&(aggregate.metric.isin(METRICS))].pivot(
        index='arm',columns='metric',values='mean')
    chart=chart[list(METRICS)]
    print('\nHELD-OUT POST-EO MEANS:\n',chart.round(6).to_string())
    important=paired[(paired.split=='test')&(paired.stage=='post_eo')&
                     (paired.comparison=='Corrected Re+EGR vs EGR')&paired.metric.isin(METRICS)]
    print('\nPAIRED HELD-OUT DELTAS (weighted minus legacy):\n',
          important[['metric','mean_delta','ci95_lower','ci95_upper']].round(6).to_string(index=False))
    print('\nDESCRIPTIVE GAP THRESHOLDS (NOT optimization feasibility):\n',
          diagnostic.to_string(index=False))
    note='Seed-based t intervals characterize ten correlated seed splits, not population confidence intervals.'
    lines=[
      '# Run 043 — Held-out corrected Re + EGR + EO',
      '',
      '## Protocol',
      '',
      '5,278 COMPAS rows; 3,694 training, 792 EO calibration, 792 independent testing per seed; seeds '+str(seeds)+'.',
      'Reweighing modifies the per-oracle LR training weights in Fairlearn EGR only (Approach A).',
      'EGR fairness moments/objective are not externally reweighted; the unity control reproduces legacy predictions exactly.',
      'Original Runs 016 and 022 calibrate EO on their evaluation data and are **not** directly comparable to this held-out protocol.',
      'EO targets equalized odds without an explicit ±0.05 threshold; descriptive gap thresholds below are not optimization feasibility.',
      '',
      '## Primary held-out post-EO metrics',
      '',
      markdown_table(chart.reset_index()),'',
      '## Paired held-out differences (corrected − legacy)', '',
      markdown_table(important[['metric','mean_delta','ci95_lower','ci95_upper']].reset_index(drop=True)),
      '',note,'',
      '## Mean absolute fairness gaps (held-out)', '',
      markdown_table(aggregate[(aggregate.split=='test')&(aggregate.stage=='post_eo')&
        aggregate.metric.str.startswith('abs_')].pivot(index='arm',columns='metric',values='mean').reset_index()),
      '',
      '## Calibration vs test descriptive fairness thresholds', '',
      markdown_table(diagnostic),'',
      '## Pre-EO vs post-EO prediction changes', '',
      markdown_table(test.groupby('arm',as_index=False)[['eo_changed_predictions','eo_harmful_flips','eo_helpful_flips',
            'pre_eo_vs_legacy_changes','post_eo_vs_legacy_changes']].mean()),'',
      '## Interpretation guardrails', '',
      '- Changes of predictions demonstrate weight propagation, not guaranteed fairness improvement.',
      '- EO may reduce disparities at the expense of accuracy; group metrics must be interpreted per seed and with absolute gaps.',
      '- ±0.05 thresholds here are **descriptive only**, unlike the configured ROC EOD bound used in Runs 040/041.',
      '- Calibration and test are disjoint; no final-test outcomes are supplied to EGR or EO fitting.',
      '- The ten train/test splits overlap substantially, so nominal paired t intervals are not population-level CIs.',
    ]
    (META/f'analysis_report_{suf}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\nRUN 043 ANALYSIS COMPLETE:',META/f'analysis_report_{suf}.md')

if __name__=='__main__': main()
