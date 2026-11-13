#!/usr/bin/env python3
"""Read-only-input mechanism diagnostic for completed Run 043 (AIF360 0.6.1 EO).

No EGR/EO refitting. Saved pre/post predictions are replayed with the exact
AIF360 0.6.1 EqOddsPostprocessing.predict seeded-shuffle/rounding algorithm.
The script refuses to produce conclusions if diagonal replays mismatch.

Install at runs/run_043_heldout_re_egr_eo/scripts/analyze_run_043_mechanism.py
and execute from repository root:
    python runs/run_043_heldout_re_egr_eo/scripts/analyze_run_043_mechanism.py --tag A

Off-diagonal applications of learned policies to the other model's predictions
are counterfactual replays, not new fitted models. Outcomes on held-out test
observations are used *only* to report accuracy/fairness, never fit an EO policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

CONTROL = 'legacy_egr'
WEIGHTED = 'reweighted_oracle'
UNITY = 'unity_oracle'
ARMS = (CONTROL, UNITY, WEIGHTED)
STAGES = ('pre_eo_prediction', 'post_eo_prediction')


def replay_aif360_eo(pre: np.ndarray, race: np.ndarray,
                     parameters: np.ndarray, seed: int) -> np.ndarray:
    """Exact AIF360 0.6.1 EqOddsPostprocessing.predict core for 0/1 labels.

    Source: Trusted-AI/AIF360 v0.6.1, eq_odds_postprocessing.py, predict().
    Order is privileged (race=1) then unprivileged (race=0), positive pool then
    negative pool, with int() truncation, and RNG reset for each call.
    """
    pre = np.asarray(pre, dtype=int)
    race = np.asarray(race, dtype=int)
    x = np.asarray(parameters, dtype=float)
    if len(x) != 4 or not np.isfinite(x).all() or (x < -1e-10).any() or (x > 1+1e-10).any():
        raise AssertionError('Invalid EO LP parameters')
    if len(pre) != len(race) or not set(np.unique(pre)).issubset({0,1}) or set(np.unique(race)) != {0,1}:
        raise AssertionError('Bad input labels/groups for EO replay')
    rng = np.random.RandomState(int(seed))  # same sequence as np.random.seed(seed)
    post = pre.copy()
    for group, keep_pos, promote_neg in ((1, x[0], x[1]), (0, x[2], x[3])):
        outer_indices = np.flatnonzero(race == group)
        within_group = pre[outer_indices]
        pp_idx = np.flatnonzero(within_group == 1)
        pn_idx = np.flatnonzero(within_group == 0)
        rng.shuffle(pp_idx)
        rng.shuffle(pn_idx)
        n_to_promote = int(len(pn_idx)*promote_neg)
        n_to_demote = int(len(pp_idx)*(1-keep_pos))
        post[outer_indices[pn_idx[:n_to_promote]]] = 1
        post[outer_indices[pp_idx[:n_to_demote]]] = 0
    return post


def basic_metrics(true: np.ndarray, race: np.ndarray, pred: np.ndarray) -> dict:
    rates = {}
    for g in (0, 1):
        idx = race == g
        pos = idx & (true == 1)
        neg = idx & (true == 0)
        if min(int(idx.sum()), int(pos.sum()), int(neg.sum())) <= 0:
            raise AssertionError('Missing race/label subgroup')
        rates[g] = (float(pred[idx].mean()), float(pred[pos].mean()), float(pred[neg].mean()))
    r0, r1 = rates[0], rates[1]
    if r1[0] <= 0:
        raise AssertionError('Undefined DI on this split')
    return dict(accuracy=float((pred == true).mean()), SPD=r0[0]-r1[0],
                DI=r0[0]/r1[0], EOD=r0[1]-r1[1], PED=r0[2]-r1[2])


def save_or_check(frame: pd.DataFrame, path: Path) -> None:
    """Idempotent and resistant to overwriting existing research artifacts."""
    if path.exists():
        previous = pd.read_csv(path)
        assert_frame_equal(previous.reset_index(drop=True), frame.reset_index(drop=True),
                           check_dtype=False, rtol=1e-10, atol=1e-10)
        print('Verified existing:', path.name)
    else:
        frame.to_csv(path, index=False, float_format='%.15g')
        print('Created:', path.name)


def md_table(frame: pd.DataFrame) -> str:
    h = list(frame.columns)
    rows = ['| '+' | '.join(h)+' |', '| '+' | '.join(['---']*len(h))+' |']
    for values in frame.itertuples(index=False, name=None):
        rows.append('| '+' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v)
                                     for v in values)+' |')
    return '\n'.join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='A')
    ap.add_argument('--run-root', type=Path, default=Path(__file__).resolve().parents[1],
                    help='For validation only; defaults to enclosing Run 043 folder')
    args = ap.parse_args()
    if not args.tag.replace('-', '').replace('_', '').isalnum():
        ap.error('Invalid tag')
    meta = args.run_root / 'metadata'
    inputs = {name:meta/f'{name}_{args.tag}.csv' for name in
              ('per_seed_results','prediction_level','prediction_change_checks')}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing Run 043 input {name}: {path}')
    result = pd.read_csv(inputs['per_seed_results'])
    pred = pd.read_csv(inputs['prediction_level'])
    changes = pd.read_csv(inputs['prediction_change_checks'])
    seeds = sorted(result.seed.unique().astype(int).tolist())
    if seeds != list(range(101,111)):
        raise AssertionError(f'Expected complete seeds 101..110, found {seeds}')
    if set(pred.arm) != set(ARMS) or set(pred.split) != {'calibration','test'}:
        raise AssertionError('Unexpected branches or splits')
    if len(pred) != 10 * 3 * 2 * 792 or len(result) != 10 * 3 * 2:
        raise AssertionError('Unexpected Run 043 row counts')
    if pred.duplicated(['seed','arm','split','original_row_id']).any():
        raise AssertionError('Duplicate prediction observation')
    if result.duplicated(['seed','arm','split']).any():
        raise AssertionError('Duplicate per-seed result')
    if not result.eo_lp_success.astype(bool).all():
        raise AssertionError('One or more EO fits failed')
    if len(changes)!=10 or set(changes.seed)!=set(seeds):
        raise AssertionError('Unexpected recorded flip diagnostics')

    # Each policy corresponds to one trained EO fit on calibration observations.
    # Per_seed_results contains the same 4 parameters in both split rows.
    policies = {}
    policy_rows = []
    for seed in seeds:
        for arm in ARMS:
            matched = result.loc[(result.seed==seed) & (result.arm==arm)]
            if set(matched.split) != {'calibration','test'}:
                raise AssertionError('Both split rows must be present')
            values = matched[[f'eo_lp_parameter_{i}' for i in range(4)]].to_numpy(dtype=float)
            if not np.allclose(values[0],values[1],atol=1e-12,rtol=0):
                raise AssertionError('EO policy changed between calibration and test')
            policies[(seed,arm)] = values[0]
            if arm==UNITY and not np.allclose(values[0],policies[(seed,CONTROL)],atol=1e-10,rtol=0):
                raise AssertionError('Unity learned EO policy differs from control')
            for race, i in ((1,0),(0,2)):
                policy_rows.append(dict(seed=seed,arm=arm,race=race,
                                        probability_1_to_0=1-values[0][i],
                                        probability_0_to_1=values[0][i+1]))
    policy = pd.DataFrame(policy_rows)

    cross_rows, group_rows, group_rate_rows, deltas = [], [], [], []
    exact_replays = 0
    for seed in seeds:
        for split in ('calibration','test'):
            branch = {arm:pred.loc[(pred.seed==seed)&(pred.arm==arm)&(pred.split==split)].copy()
                      for arm in ARMS}
            control_df = branch[CONTROL]
            ids = control_df.original_row_id.to_numpy(dtype=int)
            race = control_df.race.to_numpy(dtype=int)
            true = control_df.true_label.to_numpy(dtype=int)
            if len(ids)!=792 or len(set(ids))!=792:
                raise AssertionError(f'Unexpected {seed}/{split} row count')
            for arm in ARMS:
                df = branch[arm]
                for key, expected in (('original_row_id',ids),('race',race),('true_label',true)):
                    if not np.array_equal(df[key].to_numpy(dtype=int),expected):
                        raise AssertionError(f'{seed}/{split}: {arm} row alignment for {key}')
            for key in STAGES:
                if not np.array_equal(branch[CONTROL][key].to_numpy(dtype=int),
                                      branch[UNITY][key].to_numpy(dtype=int)):
                    raise AssertionError(f'{seed}/{split}: unity predictions differ at {key}')
            pre = {arm:branch[arm].pre_eo_prediction.to_numpy(dtype=int) for arm in ARMS}
            post = {arm:branch[arm].post_eo_prediction.to_numpy(dtype=int) for arm in ARMS}
            diag_replayed = {}
            for arm in ARMS:
                simulated = replay_aif360_eo(pre[arm],race,policies[(seed,arm)],seed)
                if not np.array_equal(simulated,post[arm]):
                    mismatch = int(np.sum(simulated != post[arm]))
                    raise AssertionError(
                        f'EO deterministic replay mismatch: seed={seed} split={split} arm={arm}, '
                        f'rows={mismatch}. Do not interpret cross-policy results; investigate version/semantics.')
                diag_replayed[arm] = simulated
                exact_replays += 1
                for group in (0,1):
                    idx = race==group
                    before=pre[arm][idx]; after=post[arm][idx]; y=true[idx]
                    flipped=before!=after
                    group_rows.append(dict(seed=seed,split=split,arm=arm,race=group,
                        n=len(y),flips=int(flipped.sum()),
                        zero_to_one=int(((before==0)&(after==1)).sum()),
                        one_to_zero=int(((before==1)&(after==0)).sum()),
                        harmful=int((flipped&(before==y)).sum()),
                        helpful=int((flipped&(after==y)).sum())))
                    pos=y==1; neg=y==0
                    group_rate_rows.append(dict(seed=seed,split=split,arm=arm,race=group,
                        pre_eo_TPR=float(before[pos].mean()),pre_eo_FPR=float(before[neg].mean()),
                        post_eo_TPR=float(after[pos].mean()),post_eo_FPR=float(after[neg].mean())))
            pre_disagree = int(np.sum(pre[WEIGHTED]!=pre[CONTROL]))
            post_disagree = int(np.sum(post[WEIGHTED]!=post[CONTROL]))
            for stage,counter in (('pre_eo',pre_disagree),('post_eo',post_disagree)):
                expected = int(result.loc[(result.seed==seed)&(result.split==split)&
                        (result.arm==WEIGHTED),f'{stage}_vs_legacy_changes'].iloc[0])
                if counter != expected:
                    raise AssertionError(f'{seed}/{split}: saved disagreement count wrong at {stage}')
            if split=='test':
                prev = changes.loc[changes.seed==seed].iloc[0]
                if pre_disagree!=int(prev.test_pre_eo_weighted_vs_legacy_flips) or \
                   post_disagree!=int(prev.test_post_eo_weighted_vs_legacy_flips):
                    raise AssertionError(f'{seed}: disagreement disagrees with independent saved checks')

            # Crossover factor design: one prediction source x one learned policy.
            # Diagonals were actually observed; off-diagonals are counterfactual replays.
            combos = {}
            for classifier in (CONTROL,WEIGHTED):
                for pol in (CONTROL,WEIGHTED):
                    trial = replay_aif360_eo(pre[classifier],race,policies[(seed,pol)],seed)
                    combos[(classifier,pol)] = trial
                    flipped = trial!=pre[classifier]
                    metric = basic_metrics(true,race,trial)
                    cross_rows.append(dict(seed=seed,split=split,classifier=classifier,eo_policy=pol,
                        observed=(classifier==pol),
                        **metric,
                        changed_by_eo=int(flipped.sum()),
                        harmful_flips=int((flipped&(pre[classifier]==true)).sum()),
                        helpful_flips=int((flipped&(trial==true)).sum()),
                        differing_vs_original_post=int(np.sum(trial!=post[CONTROL]))))
            deltas.append(dict(seed=seed,split=split,
                pre_branch_disagreement=pre_disagree,
                post_branch_disagreement=post_disagree,
                extra_post_disagreement=post_disagree-pre_disagree,
                swap_classifier_under_legacy_policy=int(np.sum(combos[(CONTROL,CONTROL)]!=combos[(WEIGHTED,CONTROL)])),
                swap_policy_on_legacy_predictions=int(np.sum(combos[(CONTROL,CONTROL)]!=combos[(CONTROL,WEIGHTED)])),
                swap_classifier_under_weighted_policy=int(np.sum(combos[(CONTROL,WEIGHTED)]!=combos[(WEIGHTED,WEIGHTED)])),
                swap_policy_on_weighted_predictions=int(np.sum(combos[(WEIGHTED,CONTROL)]!=combos[(WEIGHTED,WEIGHTED)]))))

    policy = policy.sort_values(['seed','arm','race']).reset_index(drop=True)
    group = pd.DataFrame(group_rows).sort_values(['seed','split','arm','race']).reset_index(drop=True)
    rates = pd.DataFrame(group_rate_rows).sort_values(['seed','split','arm','race']).reset_index(drop=True)
    crossed = pd.DataFrame(cross_rows).sort_values(['seed','split','classifier','eo_policy']).reset_index(drop=True)
    diag = pd.DataFrame(deltas).sort_values(['seed','split']).reset_index(drop=True)
    cal_pre = rates[(rates.split=='calibration')&(rates.arm.isin((CONTROL,WEIGHTED)))].pivot(
        index=['seed','arm'],columns='race',values=['pre_eo_TPR','pre_eo_FPR'])
    gaps = pd.DataFrame(dict(
        abs_TPR_gap=(cal_pre['pre_eo_TPR'][0]-cal_pre['pre_eo_TPR'][1]).abs(),
        abs_FPR_gap=(cal_pre['pre_eo_FPR'][0]-cal_pre['pre_eo_FPR'][1]).abs(),
    )).reset_index().rename(columns={'arm':'classifier'})

    # Per-seed (not signed-aggregate) absolute gaps for both post-EO branches.
    abs_rows=[]
    for stage in ('pre_eo','post_eo'):
        for split in ('calibration','test'):
            sel = result[(result.split==split)&(result.arm.isin((CONTROL,WEIGHTED)))]
            for _, r in sel.iterrows():
                abs_rows.append(dict(seed=int(r.seed), split=split,stage=stage, arm=r.arm,
                    abs_SPD=abs(float(r[f'{stage}_statistical_parity_difference'])),
                    abs_EOD=abs(float(r[f'{stage}_equal_opportunity_difference'])),
                    abs_PED=abs(float(r[f'{stage}_predictive_equality_difference'])),
                    abs_DI_minus_one=abs(float(r[f'{stage}_disparate_impact'])-1)))
    abs_rates=pd.DataFrame(abs_rows).sort_values(['seed','split','stage','arm']).reset_index(drop=True)

    saves = {
      'mechanism_policy_probabilities':policy,
      'mechanism_group_adjustments':group,
      'mechanism_group_rates':rates,
      'mechanism_calibration_gaps':gaps,
      'mechanism_cross_policy':crossed,
      'mechanism_prediction_disagreement':diag,
      'mechanism_absolute_gaps':abs_rates,
    }
    for name,frame in saves.items():
        save_or_check(frame,meta/f'{name}_{args.tag}.csv')

    test_diag=diag[diag.split=='test']
    cross_test=crossed[crossed.split=='test'].groupby(['classifier','eo_policy'],sort=True)[
         ['accuracy','changed_by_eo','differing_vs_original_post','SPD','DI','EOD','PED']].mean().reset_index()
    test_group=group[group.split=='test'].groupby(['arm','race'])[
         ['flips','zero_to_one','one_to_zero','harmful','helpful']].mean().reset_index()
    gap_means=gaps.groupby('classifier')[['abs_TPR_gap','abs_FPR_gap']].mean().reset_index()
    abs_test=abs_rates[(abs_rates.split=='test')&(abs_rates.stage=='post_eo')].groupby('arm')[
         ['abs_SPD','abs_DI_minus_one','abs_EOD','abs_PED']].mean().reset_index()
    pol_means=policy[policy.arm.isin((CONTROL,WEIGHTED))].groupby(['arm','race'])[
         ['probability_1_to_0','probability_0_to_1']].mean().reset_index()
    print(f'RUN 043 MECHANISM: EXACT EO REPLAY PASSED ({exact_replays} observed applications).')
    print('\nTEST disagreement and factor swaps (mean counts per seed):\n',
          test_diag.drop(columns=['seed','split']).mean().round(2).to_string())
    print('\nEO policy flip probabilities (mean across seeds):\n',pol_means.round(4).to_string(index=False))
    print('\nCalibration pre-EO mean ABS TPR/FPR gaps:\n',gap_means.round(4).to_string(index=False))
    print('\nCross-policy test outcomes (off-diagonals are counterfactual replays):\n',
          cross_test.round(4).to_string(index=False))
    print('\nTest mean ABS fairness gaps (seedwise):\n',abs_test.round(4).to_string(index=False))

    source_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in inputs.items()}
    report_lines = [
        '# Run 043 — EO mechanism diagnostic (read-only analysis)',
        '', 'No model is trained or refitted. Three saved observations per split/seed (legacy, unity, corrected) are replayed',
        f'using the pinned AIF360 0.6.1 EO seed/shuffle implementation; {exact_replays} actual EO applications exactly reproduced.',
        'Held-out outcomes were not used to learn the EO policies. Counterfactual policy swaps do not constitute new model fits.',
        '', '## Disagreement: held-out test (mean per seed of 792)',
        md_table(test_diag.drop(columns=['seed','split']).mean().to_frame('Mean').reset_index().rename(columns={'index':'Diagnostic'})),
        '', 'The factor-swap counts are NOT additive; prediction changes and policy effects interact, and seeded shuffles select different observations.',
        '', '## EO policy flip probabilities (mean of per-seed learned probabilities)',
        md_table(pol_means),
        '', '## Pre-EO absolute calibration TPR/FPR gaps (mean of per-seed absolute differences)',
        md_table(gap_means),
        '', '## Held-out EO changes by race (mean per seed)',
        md_table(test_group),
        '', '## Classifier predictions crossed with EO policies (held-out test)',
        md_table(cross_test),
        '', 'Diagonal combinations are observed original pipelines. Off-diagonals are deterministic counterfactual replays:',
        'apply a policy learned on one calibration prediction source to the other predictor\'s test predictions.',
        '', '## Held-out mean absolute fairness deviations (mean across seeds)',
        md_table(abs_test),
        '', '## Interpretation guardrails',
        '- External Re weights enter only EGR internal learner training (Approach A), not EGR fairness moments.',
        '- Differences in individual predictions can grow after EO even when average signed fairness metrics look similar.',
        '- The 10 seed splits overlap; this diagnostic is descriptive and is not an independent population estimate.',
        '- EO is randomized: this diagnostic exactly reproduces the recorded seed, not variance over new EO random seeds.',
        '- Counterfactual cross-policy predictions are not new EO optimizations, and their effects need not add linearly.',
        '', '## Input SHA-256', '', '```json', json.dumps(source_hashes,indent=2,sort_keys=True),'```',
    ]
    report=meta/f'analysis_report_mechanism_{args.tag}.md'
    written='\n'.join(report_lines)+'\n'
    if report.exists():
        if report.read_text(encoding='utf-8')!=written:
            raise AssertionError(f'Existing report differs; not overwriting: {report}')
        print('Verified existing:',report.name)
    else:
        report.write_text(written,encoding='utf-8')
        print('Created:',report.name)
    print('\nRUN 043 MECHANISM ANALYSIS COMPLETE:',report)


if __name__=='__main__':
    main()
