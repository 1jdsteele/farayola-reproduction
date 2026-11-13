"""Run 043: held-out corrected Re + EGR (Approach A) + Equalized Odds.

70/15/15 design. The training rows exactly match Run 042 (outer 70/30
race-stratified split). Equalized Odds is *fit only on calibration rows*.
The final test truth is used solely after both policies are fixed.

A: legacy EGR -> EO; B: unity-oracle EGR -> EO (exact-no-op control);
C: Reweighted EGR oracle -> EO. Also record pre-EO predictions/metrics.

Execute with pinned AIF360/Fairlearn versions in your local thesis venv.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from aif360.algorithms.inprocessing import ExponentiatedGradientReduction
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from weighted_oracle import ExternalWeightOracleLR, features_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/compas_farayola_numeric.csv'
PROV = ROOT / 'provenance'
OUT = ROOT / 'metadata'
DATA_SHA256 = '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'
SEEDS = tuple(range(101, 111))
ARMS = ('legacy_egr', 'unity_oracle', 'reweighted_oracle')
METRICS = ('accuracy', 'statistical_parity_difference', 'disparate_impact',
           'equal_opportunity_difference', 'predictive_equality_difference')
UNPRIV = [{'race': 0}]
PRIV = [{'race': 1}]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        parser.error('Tag must be alphanumeric with optional underscores/hyphens')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).issubset(SEEDS):
        parser.error('Seeds must be a nonempty, nonduplicated subset of 101..110')
    return args


def pinned_environment():
    seen = {name: version(name) for name in
            ('aif360', 'fairlearn', 'scikit-learn', 'numpy', 'pandas')}
    for name, required in (('aif360', '0.6.1'), ('fairlearn', '0.14.0'),
                           ('scikit-learn', '1.9.1')):
        if seen[name] != required:
            raise RuntimeError(f'Requires {name}=={required}, found {seen[name]}')
    import fairlearn.reductions._exponentiated_gradient._lagrangian as lag
    source = inspect.getsource(lag._Lagrangian._call_oracle)
    if 'sample_weight_name' not in source or 'redW' not in source:
        raise RuntimeError('EGR oracle interface differs from audited Run 042')
    if 'instance_weights' in inspect.getsource(ExponentiatedGradientReduction.fit):
        raise RuntimeError('AIF360 EGR wrapper has changed; audit before proceeding')
    return seen


def make_bld(X: pd.DataFrame, y: pd.Series) -> BinaryLabelDataset:
    return BinaryLabelDataset(
        df=pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1),
        label_names=['two_year_recid'], favorable_label=1, unfavorable_label=0,
        protected_attribute_names=['race'], privileged_protected_attributes=PRIV)


def preds_bld(bld: BinaryLabelDataset, labels: np.ndarray) -> BinaryLabelDataset:
    copy = bld.copy(deepcopy=True)
    values = np.asarray(labels, dtype=int).reshape(-1)
    if len(values) != len(copy.labels) or not set(np.unique(values)).issubset({0, 1}):
        raise AssertionError('Invalid predicted labels')
    copy.labels = values.astype(float).reshape(-1, 1)
    # Explicitly overwrite scores; no hidden reference to true labels is sent to EO.
    copy.scores = values.astype(float).reshape(-1, 1)
    return copy


def evaluate(bld: BinaryLabelDataset, labels: np.ndarray) -> dict:
    p = preds_bld(bld, labels)
    true = bld.labels.ravel().astype(int)
    prediction = p.labels.ravel().astype(int)
    m = ClassificationMetric(bld, p, unprivileged_groups=UNPRIV, privileged_groups=PRIV)
    tn, fp, fn, tp = map(int, confusion_matrix(true, prediction, labels=[0, 1]).ravel())
    result = dict(
        accuracy=float(accuracy_score(true, prediction)),
        statistical_parity_difference=float(m.statistical_parity_difference()),
        disparate_impact=float(m.disparate_impact()),
        equal_opportunity_difference=float(m.equal_opportunity_difference()),
        predictive_equality_difference=float(m.false_positive_rate_difference()),
        tn=tn, fp=fp, fn=fn, tp=tp)
    if not np.isfinite(list(result.values())).all():
        raise AssertionError('Undefined metrics; check race/label group counts')
    return result


def check_frozen_egr(seed: int, measured: dict, reference: pd.Series, name: str):
    for metric in METRICS:
        if not np.isclose(measured[metric], float(reference[metric]), rtol=0, atol=1e-10):
            raise AssertionError(f'Seed {seed}: {name} {metric} differs from frozen run')
    for field in ('tn', 'fp', 'fn', 'tp'):
        if measured[field] != int(reference[field]):
            raise AssertionError(f'Seed {seed}: {name} confusion-matrix mismatch {field}')


def new_egr(estimator):
    return ExponentiatedGradientReduction(
        estimator, constraints='DemographicParity', eps=0.02, max_iter=20,
        drop_prot_attr=False)


def egr_predict(model, dataset, random_state: int) -> np.ndarray:
    # EGR is a randomized mixture; reproducible per-split prediction streams.
    np.random.seed(random_state)
    labels = model.predict(dataset).labels.ravel().astype(int)
    if len(labels) != len(dataset.labels):
        raise AssertionError('Unexpected EGR prediction count')
    return labels


def fit_eo(cal_bld, cal_pred: np.ndarray, seed: int):
    eo = EqOddsPostprocessing(unprivileged_groups=UNPRIV,
                              privileged_groups=PRIV, seed=seed)
    eo.fit(cal_bld, preds_bld(cal_bld, cal_pred))
    if (eo.model_params is None or not eo.model_params.success or
        not np.isfinite(eo.model_params.x).all()):
        raise RuntimeError(f'Seed {seed}: EO linear optimization failed')
    return eo


def apply_eo(eo, bld, pre_pred: np.ndarray) -> np.ndarray:
    input_bld = preds_bld(bld, pre_pred)
    before_labels = input_bld.labels.copy()
    before_scores = input_bld.scores.copy()
    result = eo.predict(input_bld).labels.ravel().astype(int)
    if not np.array_equal(input_bld.labels, before_labels):
        raise AssertionError('EO mutated the EGR prediction input labels')
    if not np.array_equal(input_bld.scores, before_scores):
        raise AssertionError('EO mutated the EGR prediction input scores')
    if len(result) != len(pre_pred) or not set(np.unique(result)).issubset({0, 1}):
        raise AssertionError('Invalid EO output')
    return result


def verify_oracle(seed, model, traces, external, row_count):
    n = int(model.model.model_.n_oracle_calls_)
    n_dummy = int(model.model.model_.n_oracle_calls_dummy_returned_)
    if len(traces) != n - n_dummy or not traces:
        raise AssertionError(f'Seed {seed}: oracle-fit trace mismatch')
    for item in traces:
        if item['n_rows'] != row_count:
            raise AssertionError('Oracle sample count changed')
        if not np.isclose(item['combined_weight_mass'], item['internal_weight_mass'], atol=1e-8, rtol=0):
            raise AssertionError('Oracle weight normalization failed')
        if external and (not item['changed_from_internal'] or item['ratio_std'] <= 0):
            raise AssertionError('External weights did not affect EGR oracle')
        if not external and item['changed_from_internal']:
            raise AssertionError('Unity weights altered EGR oracle')
    return n, n_dummy


def verify_prior_run042_if_present(seed, original_ids, preds):
    """Optional strong check against user's already-executed local Run 042."""
    path = ROOT.parent / 'run_042_re_egr_oracle_weighting/metadata/prediction_level_A.csv'
    if not path.exists():
        return False
    ref = pd.read_csv(path)
    for arm, values in preds.items():
        part = ref.loc[(ref.seed == seed) & (ref.arm == arm)]
        if len(part) != len(original_ids):
            raise AssertionError(f'Run 042 prediction reference has unexpected row count, {arm}/{seed}')
        if not np.array_equal(part.test_original_row_id.to_numpy(), original_ids):
            raise AssertionError(f'Run 042 row-order mismatch for {arm}/{seed}')
        if not np.array_equal(part.predicted_label.to_numpy(dtype=int), values):
            raise AssertionError(f'Run 042 prediction mismatch for {arm}/{seed}')
    return True


def main():
    args = arguments()
    OUT.mkdir(exist_ok=True, parents=True)
    versions = pinned_environment()
    assert sha256(DATA) == DATA_SHA256, 'Unexpected dataset SHA256'
    data = pd.read_csv(DATA)
    assert data.shape == (5278, 12)
    assert data.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert data.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    X, y = data.drop(columns='two_year_recid'), data.two_year_recid
    refs = {run: pd.read_csv(PROV / fname).set_index('seed') for run, fname in {
        '011': 'run_011_egr_per_seed_results_A.csv',
        '014': 'run_014_re_egr_per_seed_results_A.csv',
        '016': 'run_016_egr_eo_per_seed_results_A.csv',
        '022': 'run_022_re_egr_eo_per_seed_results_A.csv',
    }.items()}
    for run, ref in refs.items():
        if sorted(ref.index.tolist()) != list(SEEDS):
            raise AssertionError(f'Frozen Run {run} missing seed(s)')
    for seed in SEEDS:
        for metric in METRICS:
            for a,b in (('011','014'),('016','022')):
                if not np.isclose(refs[a].loc[seed, metric], refs[b].loc[seed, metric], atol=1e-12, rtol=0):
                    raise AssertionError(f'Original Run {a} and Run {b} disagree: {seed} / {metric}')

    rows, pred_rows, trace_rows, split_rows, comparisons, group_rows = [], [], [], [], [], []
    for seed in args.seeds:
        train_idx, outer_idx = train_test_split(
            np.arange(len(data)), test_size=0.3, random_state=seed,
            shuffle=True, stratify=X.race)
        outer_strata = (X.iloc[outer_idx].race.astype(str).to_numpy() + '_' +
                        y.iloc[outer_idx].astype(str).to_numpy())
        cal_idx, test_idx = train_test_split(
            outer_idx, test_size=0.5, random_state=seed + 10000,
            shuffle=True, stratify=outer_strata)
        if not (len(train_idx)==3694 and len(cal_idx)==792 and len(test_idx)==792):
            raise AssertionError('Unexpected split sizes')
        if len(set(train_idx)|set(cal_idx)|set(test_idx)) != len(data):
            raise AssertionError('Split sets do not cover exactly the dataset')
        if any(set(a)&set(b) for a,b in ((train_idx, cal_idx),(train_idx,test_idx),(cal_idx,test_idx))):
            raise AssertionError('Train/calibration/test overlap')
        for part, ids in (('train',train_idx),('calibration',cal_idx),('test',test_idx)):
            counts = data.iloc[ids].groupby(['race','two_year_recid']).size()
            if any(counts.get((g,l),0) < 1 for g in (0,1) for l in (0,1)):
                raise AssertionError(f'Seed {seed}/{part}: missing group/label stratum')
            group_rows.append(dict(seed=seed, split=part, rows=len(ids),
                                   **{f'race_{g}_label_{label}':int(counts.get((g,label),0))
                                      for g in (0,1) for label in (0,1)}))
            split_rows.append(pd.DataFrame({'seed': seed, 'split':part, 'original_row_id':ids}))

        tr_bld = make_bld(X.iloc[train_idx],y.iloc[train_idx])
        outer_bld = make_bld(X.iloc[outer_idx],y.iloc[outer_idx])
        cal_bld = make_bld(X.iloc[cal_idx],y.iloc[cal_idx])
        test_bld = make_bld(X.iloc[test_idx],y.iloc[test_idx])
        re_bld = Reweighing(unprivileged_groups=UNPRIV,privileged_groups=PRIV).fit_transform(tr_bld)
        ext = np.asarray(re_bld.instance_weights,dtype=float).ravel()
        if not (np.array_equal(re_bld.features,tr_bld.features) and
                np.array_equal(re_bld.labels,tr_bld.labels) and
                np.array_equal(re_bld.protected_attributes,tr_bld.protected_attributes)):
            raise AssertionError('Reweighing changed values beyond instance weights')
        if not (np.isfinite(ext).all() and (ext>0).all() and
                np.isclose(ext.sum(),len(ext),atol=1e-8,rtol=0) and
                not np.allclose(ext,1.,atol=1e-12,rtol=0)):
            raise AssertionError('Reweighing weights invalid or unchanged')
        for key,col in (('re_weight_min','min'),('re_weight_max','max'),('re_weight_sum','sum')):
            value = getattr(ext,col)()
            if not np.isclose(value,float(refs['014'].loc[seed,key]),atol=1e-9,rtol=0):
                raise AssertionError(f'Seed {seed}: Reweighing weights differ from frozen Run 014')

        digest = features_fingerprint(pd.DataFrame(tr_bld.features,columns=tr_bld.feature_names))
        trained, outer_preds, calib_preds, test_preds, trace_by_arm = {}, {}, {}, {}, {}
        configurations = {
            'legacy_egr':(LogisticRegression(solver='lbfgs',max_iter=1000),tr_bld),
            'unity_oracle':(ExternalWeightOracleLR((1.,)*len(ext),digest),tr_bld),
            'reweighted_oracle':(ExternalWeightOracleLR(tuple(ext.tolist()),digest),re_bld),
        }
        for arm in ARMS:
            estimator,train_ds = configurations[arm]
            egr = new_egr(estimator)
            ExternalWeightOracleLR.FIT_TRACES.clear()
            np.random.seed(seed)
            egr.fit(train_ds)
            logs = [dict(t,seed=seed,arm=arm) for t in ExternalWeightOracleLR.FIT_TRACES]
            trace_by_arm[arm] = logs
            if arm != 'legacy_egr':
                verify_oracle(seed,egr,logs,arm=='reweighted_oracle',len(train_idx))
                trace_rows.extend(logs)
            trained[arm] = egr
            # Must reproduce ORIGINAL Run 011/Run 042 evaluation on the outer 30%.
            outer_preds[arm] = egr_predict(egr,outer_bld,seed)
            calib_preds[arm] = egr_predict(egr,cal_bld,seed+20000)
            test_preds[arm] = egr_predict(egr,test_bld,seed+30000)
        if not np.array_equal(outer_preds['legacy_egr'],outer_preds['unity_oracle']):
            raise AssertionError(f'Seed {seed}: unity oracle does not match frozen EGR predictions')
        if not np.array_equal(calib_preds['legacy_egr'],calib_preds['unity_oracle']):
            raise AssertionError(f'Seed {seed}: unity oracle differs on calibration predictions')
        if not np.array_equal(test_preds['legacy_egr'],test_preds['unity_oracle']):
            raise AssertionError(f'Seed {seed}: unity oracle differs on held-out predictions')
        for arm in ('legacy_egr','unity_oracle'):
            check_frozen_egr(seed,evaluate(outer_bld,outer_preds[arm]),refs['011'].loc[seed],arm)
            check_frozen_egr(seed,evaluate(outer_bld,outer_preds[arm]),refs['014'].loc[seed],arm)
        matched042 = verify_prior_run042_if_present(seed,outer_idx,outer_preds)

        post_cal, post_test, policies = {}, {}, {}
        for arm in ARMS:
            # IMPORTANT: EO sees calibration truth, *never* final test truth.
            eo = fit_eo(cal_bld,calib_preds[arm],seed)
            policies[arm] = eo
            post_test[arm] = apply_eo(eo,test_bld,test_preds[arm])
            post_cal[arm] = apply_eo(eo,cal_bld,calib_preds[arm])
        if not np.array_equal(post_test['legacy_egr'],post_test['unity_oracle']):
            raise AssertionError(f'Seed {seed}: unity EO differs from legacy EO on test')
        if not np.array_equal(post_cal['legacy_egr'],post_cal['unity_oracle']):
            raise AssertionError(f'Seed {seed}: unity EO differs from legacy EO on calibration')

        for arm in ARMS:
            for split,ids,bld,pre,post in (
                ('calibration',cal_idx,cal_bld,calib_preds[arm],post_cal[arm]),
                ('test',test_idx,test_bld,test_preds[arm],post_test[arm]),
            ):
                # Each row is one arm on one disjoint evaluation partition.
                data_slice = data.iloc[ids]
                pred_rows.append(pd.DataFrame({
                    'seed':seed,'arm':arm,'split':split,'original_row_id':ids,
                    'race':data_slice.race.to_numpy(dtype=int),
                    'true_label':data_slice.two_year_recid.to_numpy(dtype=int),
                    'pre_eo_prediction':pre,'post_eo_prediction':post,
                }))
                pre_m,post_m = evaluate(bld,pre),evaluate(bld,post)
                ref_model = trained[arm].model.model_
                changed = pre != post
                harmful = int(np.sum(changed & (pre==bld.labels.ravel().astype(int))))
                helpful = int(np.sum(changed & (post==bld.labels.ravel().astype(int))))
                if int(changed.sum()) != harmful+helpful:
                    raise AssertionError('Flip correctness accounting mismatch')
                row = dict(
                    seed=seed,arm=arm,split=split,
                    train_rows=len(train_idx),calibration_rows=len(cal_idx),test_rows=len(test_idx),
                    outer_split_stratify='race',calibration_split_stratify='race+label',
                    egr_constraint='DemographicParity',egr_eps=0.02,egr_max_iter=20,
                    egr_random_seed=seed,eo_seed=seed,
                    outer_frozen_run011_match=True,unity_prediction_match=True,
                    run042_local_predictions_match=bool(matched042),
                    egr_oracle_calls=int(ref_model.n_oracle_calls_),
                    egr_dummy_oracle_calls=int(ref_model.n_oracle_calls_dummy_returned_),
                    egr_mixture_nonzero=int(np.sum(np.asarray(ref_model.weights_)>0)),
                    external_weight_min=float(ext.min()),external_weight_max=float(ext.max()),
                    external_weight_std=float(ext.std()),external_weight_sum=float(ext.sum()),
                    external_oracle_fits=len(trace_by_arm[arm]),
                    eo_lp_success=bool(policies[arm].model_params.success),
                    eo_changed_predictions=int(changed.sum()),
                    eo_harmful_flips=harmful,eo_helpful_flips=helpful,
                    pre_eo_vs_legacy_changes=int(np.sum(pre != (calib_preds if split=='calibration' else test_preds)['legacy_egr'])),
                    post_eo_vs_legacy_changes=int(np.sum(post != (post_cal if split=='calibration' else post_test)['legacy_egr'])),
                )
                for stage,metrics in (('pre_eo',pre_m),('post_eo',post_m)):
                    row.update({f'{stage}_{k}':v for k,v in metrics.items()})
                for i,v in enumerate(policies[arm].model_params.x):
                    row[f'eo_lp_parameter_{i}'] = float(v)
                rows.append(row)
        comparisons.append(dict(
            seed=seed,
            run042_outer_prediction_match=bool(matched042),
            test_pre_eo_weighted_vs_legacy_flips=int(np.sum(test_preds['reweighted_oracle']!=test_preds['legacy_egr'])),
            test_post_eo_weighted_vs_legacy_flips=int(np.sum(post_test['reweighted_oracle']!=post_test['legacy_egr'])),
            test_pre_eo_unity_vs_legacy_flips=int(np.sum(test_preds['unity_oracle']!=test_preds['legacy_egr'])),
            test_post_eo_unity_vs_legacy_flips=int(np.sum(post_test['unity_oracle']!=post_test['legacy_egr'])),
        ))
        print(f'Seed {seed}: frozen EGR match=True, unity + EO exact=True, '
              f'Run042 local cross-check={matched042}; '
              f'weighted vs legacy test pre/post EO flips='
              f'{comparisons[-1]["test_pre_eo_weighted_vs_legacy_flips"]}/'
              f'{comparisons[-1]["test_post_eo_weighted_vs_legacy_flips"]}; '
              f'held-out EO accuracy {next(r["post_eo_accuracy"] for r in rows if r["seed"]==seed and r["split"]=="test" and r["arm"]=="legacy_egr"):.6f} -> '
              f'{next(r["post_eo_accuracy"] for r in rows if r["seed"]==seed and r["split"]=="test" and r["arm"]=="reweighted_oracle"):.6f}',flush=True)

    frame=pd.DataFrame(rows)
    prediction_frame=pd.concat(pred_rows,ignore_index=True)
    trace_frame=pd.DataFrame(trace_rows)
    frame.to_csv(OUT/f'per_seed_results_{args.tag}.csv',index=False,float_format='%.15g')
    prediction_frame.to_csv(OUT/f'prediction_level_{args.tag}.csv',index=False)
    trace_frame.to_csv(OUT/f'oracle_fit_traces_{args.tag}.csv',index=False,float_format='%.15g')
    pd.concat(split_rows,ignore_index=True).to_csv(OUT/f'split_membership_{args.tag}.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/f'group_label_counts_{args.tag}.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/f'prediction_change_checks_{args.tag}.csv',index=False)
    manifest=dict(
        experiment='run_043_heldout_re_egr_eo',tag=args.tag,seeds=args.seeds,
        design='70% train / 15% EO calibration / 15% final held-out test',
        outer_split='seed, race-stratified; identical to original Run 042 70/30 training split',
        cal_test_split='seed+10000, stratify race+label, divide outer 30% in half',
        egr_prediction_rng={'outer':'seed (frozen Run042 match)','calibration':'seed+20000',
                            'test':'seed+30000'},
        eo_policy_seed='seed',
        model='EGR(DemographicParity,eps=.02,max_iter=20,drop_prot_attr=False); LR lbfgs max_iter=1000',
        arms={
            'legacy_egr':'EGR + EO (no Reweighing)',
            'unity_oracle':'Unity-weights EGR + EO implementation control',
            'reweighted_oracle':'Re + EGR + EO (Approach A: external Re weights * per-oracle EGR LR weights, preserving mass)',
        },
        critical_scope='Approach A changes EGR classifier oracle weights only, not EGR moment/objective weighting',
        test_outcomes_used_for_model_training_or_eo_policy_selection=False,
        note='EO optimizes equalized odds; |EOD|<=.05 is an optional descriptive threshold, NOT EO fit constraint',
        original_run_016_and_022='Test-calibrated EO with leakage, only checked as historical matched controls; metrics not compared directly to this held-out run',
        original_run_011_and_014='Outer 70/30 EGR predictions and metrics verified against frozen originals',
        run042_prediction_level='Optional cross-check when the prior run is present in sibling runs/ directory',
        hashes={'dataset':sha256(DATA), **{p.name:sha256(p) for p in PROV.glob('*.csv')}},
        versions={**versions,'python':platform.python_version()},
    )
    (OUT/f'run_manifest_{args.tag}.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    means=(frame.loc[frame.split=='test'].groupby('arm')
           [[f'post_eo_{m}' for m in METRICS]+['eo_changed_predictions']].mean())
    print('\nRUN 043 COMPLETE — outputs:',OUT)
    print('\nHeld-out post-EO means:\n',means.to_string(float_format=lambda v:f'{v:.6f}'))

if __name__ == '__main__':
    main()
