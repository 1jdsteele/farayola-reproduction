"""Run 038: held-out comparison of EO versus weight-aware Reweighing + EO.

The 70/30 outer split matches Run 037. The outer 30% is split into disjoint
EO calibration and final test sets (15%/15% of the full cohort). Only the
training cohort is used for Reweighing and logistic-regression fitting.

This is a NEW evaluation protocol; its metrics are not directly paired with
Run 037's 30%-test-set, test-calibrated EO metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import aif360
import numpy as np
import pandas as pd
import scipy
import sklearn
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'compas_farayola_numeric.csv'
PROV = ROOT / 'provenance'
OUT = ROOT / 'metadata'
EXPECTED_SHA256 = '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'
SEEDS = tuple(range(101, 111))
UNPRIVILEGED = [{'race': 0}]
PRIVILEGED = [{'race': 1}]
METRICS = (
    'accuracy', 'disparate_impact', 'statistical_parity_difference',
    'equal_opportunity_difference', 'predictive_equality_difference',
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_dataset(X: pd.DataFrame, y: pd.Series) -> BinaryLabelDataset:
    return BinaryLabelDataset(
        df=pd.concat([X.reset_index(drop=True),
                      y.reset_index(drop=True)], axis=1),
        label_names=['two_year_recid'], favorable_label=1, unfavorable_label=0,
        protected_attribute_names=['race'],
        privileged_protected_attributes=PRIVILEGED,
    )


def predicted_dataset(true_dataset: BinaryLabelDataset,
                      predictions: np.ndarray) -> BinaryLabelDataset:
    copy = true_dataset.copy(deepcopy=True)
    copy.labels = np.asarray(predictions, dtype=float).reshape(-1, 1)
    return copy


def metrics_for(true_dataset: BinaryLabelDataset,
                predictions: np.ndarray) -> tuple[dict, np.ndarray]:
    pred = predicted_dataset(true_dataset, predictions)
    metric = ClassificationMetric(
        true_dataset, pred,
        unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED,
    )
    truth = true_dataset.labels.ravel().astype(int)
    guess = pred.labels.ravel().astype(int)
    cm = confusion_matrix(truth, guess, labels=[0, 1]).ravel()
    output = {
        'accuracy': float(accuracy_score(truth, guess)),
        'disparate_impact': float(metric.disparate_impact()),
        'statistical_parity_difference': float(metric.statistical_parity_difference()),
        'equal_opportunity_difference': float(metric.equal_opportunity_difference()),
        'predictive_equality_difference': float(metric.false_positive_rate_difference()),
    }
    if not all(np.isfinite(v) for v in output.values()):
        raise AssertionError('Undefined/infinite metric: inspect group/label counts')
    return output, cm


def aggregate(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for name in columns:
        x = frame[name].to_numpy(dtype=float)
        assert len(x) == len(SEEDS) and np.all(np.isfinite(x))
        sd = float(x.std(ddof=1))
        delta = float(t.ppf(.975, len(x) - 1) * sd / math.sqrt(len(x)))
        mean = float(x.mean())
        rows.append({
            'metric': name, 'n': len(x), 'mean': mean, 'sample_std': sd,
            'min': float(x.min()), 'max': float(x.max()),
            'ci95_lower': mean - delta, 'ci95_upper': mean + delta,
        })
    return pd.DataFrame(rows)


def check_frozen_outer_reference(ref: pd.DataFrame, seed: int,
                                 values: dict, cm: np.ndarray) -> None:
    """Verify training cohort, unweighted model, and preprocessing match Run 008."""
    row = ref.loc[ref.seed == seed]
    assert len(row) == 1, f'Missing frozen Run 008 reference for seed={seed}'
    row = row.iloc[0]
    for name, actual in zip(('tn', 'fp', 'fn', 'tp'), cm):
        assert int(row[name]) == int(actual), (
            f'Run 008 outer-holdout confusion mismatch, seed {seed}, {name}: '
            f'expected {int(row[name])}, observed {int(actual)}'
        )
    for name in METRICS:
        assert np.isclose(float(row[name]), values[name], rtol=0, atol=1e-12), (
            f'Run 008 outer-holdout metric mismatch, seed={seed}, {name}'
        )


def check_frozen_weight_reference(ref: pd.DataFrame, seed: int,
                                  weights: np.ndarray) -> None:
    row = ref.loc[ref.seed == seed]
    assert len(row) == 1, f'Missing frozen Run 010 reference for seed={seed}'
    row = row.iloc[0]
    for name, value in (
        ('reweigh_weight_min', weights.min()),
        ('reweigh_weight_max', weights.max()),
        ('reweigh_weight_sum', weights.sum()),
    ):
        assert np.isclose(float(row[name]), value, atol=1e-10, rtol=0), (
            f'Run 010 weight mismatch: seed {seed}, {name}'
        )


def add_stats(row: dict, prefix: str, values: dict, cm: np.ndarray) -> None:
    row.update({f'{prefix}_{name}': value for name, value in values.items()})
    row.update({f'{prefix}_{name}': int(v)
                for name, v in zip(('tn', 'fp', 'fn', 'tp'), cm)})


def group_label_integrity(partition: pd.DataFrame, label: str) -> dict:
    counts = partition.groupby(['race', 'two_year_recid']).size()
    for group in (0, 1):
        for y in (0, 1):
            if int(counts.get((group, y), 0)) < 1:
                raise ValueError(f'{label} is missing race={group}, target={y}')
    return {f'race_{group}_label_{y}': int(counts.get((group, y), 0))
            for group in (0, 1) for y in (0, 1)}


def predict_post(eo: EqOddsPostprocessing, truth: BinaryLabelDataset,
                 labels: np.ndarray) -> np.ndarray:
    pred_bld = predicted_dataset(truth, labels)
    snapshot = pred_bld.labels.copy()
    result = eo.predict(pred_bld).labels.ravel().astype(int)
    assert np.array_equal(pred_bld.labels, snapshot), 'EO mutated its predictions input'
    assert len(result) == len(labels)
    assert set(np.unique(result)).issubset({0, 1})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='A',
                        help='Output suffix. Use A for initial run, B for reproducibility rerun.')
    args = parser.parse_args()
    tag = args.tag
    if not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Tag may only contain letters, numbers, _ and -')
    assert sha256(DATA_PATH) == EXPECTED_SHA256, 'Unexpected COMPAS input data hash'
    data = pd.read_csv(DATA_PATH)
    assert data.shape == (5278, 12), 'Unexpected data dimensions'
    assert data.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert data.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    X = data.drop(columns='two_year_recid')
    y = data.two_year_recid

    ref_files = {
        '008': PROV / 'run_008_none_per_seed_results_A.csv',
        '010': PROV / 'run_010_reweighing_per_seed_results_A.csv',
    }
    refs = {k: pd.read_csv(v) for k, v in ref_files.items()}
    for key, frame in refs.items():
        assert sorted(frame.seed.tolist()) == list(SEEDS), f'Frozen {key} seeds mismatch'

    per_seed: list[dict] = []
    predictions: list[pd.DataFrame] = []
    membership: list[pd.DataFrame] = []
    group_counts: list[dict] = []

    for seed in SEEDS:
        # This outer split and its row order are identical to Run 037.
        train_idx, outer_idx = train_test_split(
            np.arange(len(data)), test_size=.3, random_state=seed,
            shuffle=True, stratify=X.race,
        )
        # The previously evaluated outer 30% is now split into *new* disjoint
        # calibration and final test cohorts. Stratification uses race+label to
        # ensure enough examples of each group/outcome for EO optimization.
        outer_strata = (
            X.iloc[outer_idx].race.astype(str).to_numpy() + '_' +
            y.iloc[outer_idx].astype(str).to_numpy()
        )
        cal_idx, test_idx = train_test_split(
            outer_idx, test_size=.5, random_state=seed + 10_000,
            shuffle=True, stratify=outer_strata,
        )
        assert len(train_idx) == 3694 and len(outer_idx) == 1584
        assert len(cal_idx) == len(test_idx) == 792
        assert set(train_idx).isdisjoint(cal_idx)
        assert set(train_idx).isdisjoint(test_idx)
        assert set(cal_idx).isdisjoint(test_idx)
        assert len(set(train_idx) | set(cal_idx) | set(test_idx)) == len(data)

        subsets = {'train': train_idx, 'calibration': cal_idx, 'test': test_idx}
        for split, row_ids in subsets.items():
            membership.append(pd.DataFrame({
                'seed': seed, 'split': split, 'original_row_id': row_ids,
            }))
            count_dict = group_label_integrity(data.iloc[row_ids], f'seed {seed}/{split}')
            group_counts.append({'seed': seed, 'split': split,
                                 'rows': len(row_ids), **count_dict})

        train_bld = make_dataset(X.iloc[train_idx], y.iloc[train_idx])
        outer_bld = make_dataset(X.iloc[outer_idx], y.iloc[outer_idx])
        cal_bld = make_dataset(X.iloc[cal_idx], y.iloc[cal_idx])
        test_bld = make_dataset(X.iloc[test_idx], y.iloc[test_idx])
        for item in (outer_bld, cal_bld, test_bld):
            assert item.feature_names == train_bld.feature_names

        # Reweighing must be fitted on training only. Neither calibration
        # labels nor final test labels are passed to this stage.
        re_train = Reweighing(
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED,
        ).fit_transform(train_bld)
        weights = np.asarray(re_train.instance_weights, dtype=float).ravel()
        assert weights.shape == (len(train_idx),)
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)
        assert not np.allclose(weights, train_bld.instance_weights.ravel(),
                               atol=1e-12, rtol=0)
        assert np.isclose(weights.sum(), len(weights), atol=1e-8, rtol=0)
        assert np.array_equal(train_bld.features, re_train.features)
        assert np.array_equal(train_bld.labels, re_train.labels)
        assert np.array_equal(train_bld.protected_attributes,
                              re_train.protected_attributes)
        check_frozen_weight_reference(refs['010'], seed, weights)

        control_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        weighted_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        control_lr.fit(re_train.features, re_train.labels.ravel())
        weighted_lr.fit(re_train.features, re_train.labels.ravel(),
                        sample_weight=weights)

        c_cal = control_lr.predict(cal_bld.features).astype(int)
        w_cal = weighted_lr.predict(cal_bld.features).astype(int)
        c_test = control_lr.predict(test_bld.features).astype(int)
        w_test = weighted_lr.predict(test_bld.features).astype(int)

        # EO is *fitted solely on calibration labels and predictions*.
        # No final test labels are passed to fit; predict uses only predicted
        # labels and protected attributes. AIF360 0.6.1 API: fit(...).predict(...).
        control_eo = EqOddsPostprocessing(
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED, seed=seed,
        )
        weighted_eo = EqOddsPostprocessing(
            unprivileged_groups=UNPRIVILEGED,
            privileged_groups=PRIVILEGED, seed=seed,
        )
        control_eo.fit(cal_bld, predicted_dataset(cal_bld, c_cal))
        weighted_eo.fit(cal_bld, predicted_dataset(cal_bld, w_cal))
        for branch, eo in (('control', control_eo), ('weighted', weighted_eo)):
            result = eo.model_params
            if result is None or not result.success or not np.all(np.isfinite(result.x)):
                raise RuntimeError(
                    f'Seed {seed}: {branch} EO optimization failed: {result}'
                )
        # Predict on disjoint test data. These predictions are evaluated only
        # after all fitting stages have finished.
        c_test_post = predict_post(control_eo, test_bld, c_test)
        w_test_post = predict_post(weighted_eo, test_bld, w_test)
        # Calibration data are also evaluated for an explicitly in-sample
        # diagnostic; never mix these estimates with the held-out test results.
        c_cal_post = predict_post(control_eo, cal_bld, c_cal)
        w_cal_post = predict_post(weighted_eo, cal_bld, w_cal)

        # Only AFTER both EO policies and final predictions are fixed, inspect
        # the outer cohort's true labels for the frozen Run 008 reference check.
        # This check is diagnostic and never used for training or policy choice.
        control_outer = control_lr.predict(outer_bld.features).astype(int)
        outer_metrics, outer_cm = metrics_for(outer_bld, control_outer)
        check_frozen_outer_reference(refs['008'], seed, outer_metrics, outer_cm)

        row = {
            'seed': seed, 'train_rows': len(train_idx),
            'calibration_rows': len(cal_idx), 'test_rows': len(test_idx),
            'outer_holdout_rows': len(outer_idx),
            'race_stratified_outer_split': True,
            'race_label_stratified_cal_test_split': True,
            'frozen_run008_outer_pre_eo_match': True,
            'frozen_run010_weights_match': True,
            'lr_used_instance_weights': True,
            'eo_calibration_uses_test': False,
            'control_lr_n_iter': int(control_lr.n_iter_[0]),
            'weighted_lr_n_iter': int(weighted_lr.n_iter_[0]),
            'coefficient_difference_l2': float(np.linalg.norm(
                control_lr.coef_ - weighted_lr.coef_)),
            'weight_min': float(weights.min()),
            'weight_max': float(weights.max()),
            'weight_std': float(weights.std()),
            'weight_sum': float(weights.sum()),
            'test_pre_eo_changed_predictions': int(np.sum(c_test != w_test)),
            'test_post_eo_changed_predictions': int(np.sum(c_test_post != w_test_post)),
            'calibration_pre_eo_changed_predictions': int(np.sum(c_cal != w_cal)),
            'calibration_post_eo_changed_predictions': int(np.sum(c_cal_post != w_cal_post)),
            'test_control_eo_changed_predictions': int(np.sum(c_test != c_test_post)),
            'test_weighted_eo_changed_predictions': int(np.sum(w_test != w_test_post)),
            'control_eo_lp_success': bool(control_eo.model_params.success),
            'weighted_eo_lp_success': bool(weighted_eo.model_params.success),
        }
        for branch, eo in (('control', control_eo), ('weighted', weighted_eo)):
            row.update({f'{branch}_eo_parameter_{i}': float(v)
                        for i, v in enumerate(eo.model_params.x)})
        for split, truth, c_pre, w_pre, c_post, w_post in (
            ('test', test_bld, c_test, w_test, c_test_post, w_test_post),
            ('calibration', cal_bld, c_cal, w_cal, c_cal_post, w_cal_post),
        ):
            for branch, stage, labels in (
                ('control', 'pre_eo', c_pre),
                ('weighted', 'pre_eo', w_pre),
                ('control', 'post_eo', c_post),
                ('weighted', 'post_eo', w_post),
            ):
                v, cm = metrics_for(truth, labels)
                add_stats(row, f'{split}_{branch}_{stage}', v, cm)
            row.update({
                f'{split}_delta_pre_eo_{key}':
                    row[f'{split}_weighted_pre_eo_{key}'] -
                    row[f'{split}_control_pre_eo_{key}']
                for key in METRICS
            })
            row.update({
                f'{split}_delta_post_eo_{key}':
                    row[f'{split}_weighted_post_eo_{key}'] -
                    row[f'{split}_control_post_eo_{key}']
                for key in METRICS
            })

        for split, ids, truth, c_pre, w_pre, c_post, w_post in (
            ('calibration', cal_idx, cal_bld, c_cal, w_cal, c_cal_post, w_cal_post),
            ('test', test_idx, test_bld, c_test, w_test, c_test_post, w_test_post),
        ):
            source = data.iloc[ids]
            assert np.array_equal(source.two_year_recid.to_numpy(),
                                  truth.labels.ravel().astype(int))
            predictions.append(pd.DataFrame({
                'seed': seed, 'split': split, 'original_row_id': ids,
                'race': source.race.to_numpy(),
                'true_label': source.two_year_recid.to_numpy(),
                'control_pre_eo': c_pre,
                'weighted_pre_eo': w_pre,
                'control_post_eo': c_post,
                'weighted_post_eo': w_post,
                'control_score': control_lr.predict_proba(truth.features)[:, 1],
                'weighted_score': weighted_lr.predict_proba(truth.features)[:, 1],
            }))
        per_seed.append(row)
        print(
            f'Seed {seed}: test pre-EO changed '
            f'{row["test_pre_eo_changed_predictions"]}/{len(test_idx)}, '
            f'post-EO changed {row["test_post_eo_changed_predictions"]}/{len(test_idx)}; '
            f'held-out accuracy delta '
            f'{row["test_delta_post_eo_accuracy"]:+.6f}', flush=True
        )

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.DataFrame(per_seed)
    deltas = tuple(f'test_delta_post_eo_{name}' for name in METRICS)
    final_columns = tuple(
        f'test_{branch}_post_eo_{name}'
        for branch in ('control', 'weighted') for name in METRICS
    )
    all_rows.to_csv(OUT / f'per_seed_results_{tag}.csv', index=False,
                    float_format='%.15g')
    aggregate(all_rows, final_columns).to_csv(
        OUT / f'aggregate_summary_{tag}.csv', index=False, float_format='%.15g')
    aggregate(all_rows, deltas).to_csv(
        OUT / f'paired_deltas_{tag}.csv', index=False, float_format='%.15g')
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / f'prediction_level_{tag}.csv', index=False, float_format='%.15g')
    pd.concat(membership, ignore_index=True).to_csv(
        OUT / f'split_membership_{tag}.csv', index=False)
    pd.DataFrame(group_counts).to_csv(OUT / f'group_label_counts_{tag}.csv', index=False)
    manifest = {
        'experiment': '038_heldout_re_eo',
        'description': 'Held-out EO vs. corrected weighted Reweighing + EO',
        'input_data_sha256': sha256(DATA_PATH),
        'reference_sha256': {k: sha256(v) for k, v in ref_files.items()},
        'script_sha256': sha256(Path(__file__).resolve()),
        'versions': {
            'python': platform.python_version(), 'aif360': getattr(aif360, '__version__', 'unreported'),
            'scikit_learn': sklearn.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__,
        },
        'seeds': list(SEEDS), 'outer_test_size': .3,
        'calibration_fraction_of_outer': .5,
        'proportions': {'train': .7, 'calibration': .15, 'test': .15},
        'outer_split_stratification': 'race',
        'calibration_test_stratification': 'race + target joint strata',
        'calibration_test_split_random_state': 'seed+10000',
        'eo_random_seed': 'seed',
        'eo_policy_fit_rows': 'calibration only',
        'eo_policy_predict_rows': 'test (and separately calibration diagnostics)',
        'test_labels_used_for_training_or_calibration': False,
        'model': "LogisticRegression(solver='lbfgs', max_iter=1000)",
        'comparability': (
            'Run 038 shares the training cohort and upstream model definition with Run 037, '
            'but EO policy learning and evaluation populations differ. '
            'Do not directly compare Run 038 and Run 037 metrics as a paired treatment effect.'
        ),
        'outcome_data_not_used_for_model_selection': True,
    }
    (OUT / f'run_manifest_{tag}.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    print('\nRUN 038 COMPLETE. Results in', OUT)
    print('\nHeld-out post-EO means:\n', aggregate(all_rows, final_columns).to_string(index=False))
    print('\nPaired changes (weighted minus control):\n',
          aggregate(all_rows, deltas).to_string(index=False))


if __name__ == '__main__':
    main()
