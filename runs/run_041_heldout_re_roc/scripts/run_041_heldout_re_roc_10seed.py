"""Run 041: held-out 2 x 2 audit of Reweighing and ROC score propagation.

Four arms (each includes ROC):
  A_hard_unweighted  : unweighted LR, hard 0/1 scores into ROC (original)
  B_hard_weighted    : Re-weighted LR, hard 0/1 scores into ROC (weight fix only)
  C_prob_unweighted  : unweighted LR, continuous probability scores into ROC (score fix only)
  D_prob_weighted    : Re-weighted LR, continuous probability scores into ROC (both fixes)

Held-out evaluation: ROC is fitted on calibration observations only and
evaluated on a disjoint final test subset. The 70% outer training set and
training seeds match Run 040, but the evaluation protocol is different.
Neither final-test true labels nor final-test fairness metrics are used
to select ROC parameters. Preserve all original reproduction runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import warnings
from pathlib import Path

import aif360
import numpy as np
import pandas as pd
import scipy
import sklearn
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'metadata'
DATA_PATH = ROOT / 'data' / 'compas_farayola_numeric.csv'
PROV = ROOT / 'provenance'
EXPECTED_SHA256 = '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'
EXPECTED_SEEDS = tuple(range(101, 111))
PRIVILEGED = [{'race': 1}]
UNPRIVILEGED = [{'race': 0}]
METRICS = (
    'accuracy', 'disparate_impact', 'statistical_parity_difference',
    'equal_opportunity_difference', 'predictive_equality_difference',
)
ARMS = (
    ('A_hard_unweighted', False, 'hard'),
    ('B_hard_weighted', True, 'hard'),
    ('C_prob_unweighted', False, 'probability'),
    ('D_prob_weighted', True, 'probability'),
)
ROC_PARAMETERS = {
    'metric_name': 'Equal opportunity difference',
    'low_class_thresh': 0.3,
    'high_class_thresh': 0.8,
    'num_class_thresh': 100,
    'num_ROC_margin': 50,
    'metric_lb': -0.05,
    'metric_ub': 0.05,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_bld(x: pd.DataFrame, y: pd.Series) -> BinaryLabelDataset:
    return BinaryLabelDataset(
        df=pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1),
        label_names=['two_year_recid'], favorable_label=1, unfavorable_label=0,
        protected_attribute_names=['race'],
        privileged_protected_attributes=PRIVILEGED,
    )


def prediction_bld(truth: BinaryLabelDataset, labels: np.ndarray,
                   scores: np.ndarray) -> BinaryLabelDataset:
    """Important: ROC uses `.scores`, NOT `.labels`.

    With hard mode, scores = label predictions to reproduce the original
    AIF360 BinaryLabelDataset(predicted dataframe) behavior. Never leave
    truth.scores in the copied object: those are *true labels*.
    """
    pred = truth.copy(deepcopy=True)
    labels = np.asarray(labels).reshape(-1, 1).astype(float)
    scores = np.asarray(scores).reshape(-1, 1).astype(float)
    assert labels.shape == scores.shape == truth.labels.shape
    assert np.all((scores >= 0) & (scores <= 1))
    pred.labels = labels
    pred.scores = scores
    return pred


def measurements(truth: BinaryLabelDataset,
                 labels: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    pred = prediction_bld(truth, labels, labels)
    metric = ClassificationMetric(truth, pred,
                                  unprivileged_groups=UNPRIVILEGED,
                                  privileged_groups=PRIVILEGED)
    actual = truth.labels.ravel().astype(int)
    predicted = pred.labels.ravel().astype(int)
    cm = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    values = {
        'accuracy': float(accuracy_score(actual, predicted)),
        'disparate_impact': float(metric.disparate_impact()),
        'statistical_parity_difference': float(metric.statistical_parity_difference()),
        'equal_opportunity_difference': float(metric.equal_opportunity_difference()),
        'predictive_equality_difference': float(metric.false_positive_rate_difference()),
    }
    assert all(np.isfinite(v) for v in values.values()), 'Undefined fairness metric'
    return values, cm


def verify_reference(ref: pd.DataFrame, seed: int,
                     values: dict, cm: np.ndarray, name: str) -> None:
    selected = ref.loc[ref.seed == seed]
    assert len(selected) == 1, f'{name}: missing/duplicate seed {seed}'
    row = selected.iloc[0]
    for name_cm, v in zip(('tn', 'fp', 'fn', 'tp'), cm):
        assert int(row[name_cm]) == int(v), (
            f'{name}, seed={seed}: {name_cm} mismatch: '
            f'expected {int(row[name_cm])}, got {int(v)}'
        )
    for m in METRICS:
        assert np.isclose(float(row[m]), float(values[m]), atol=1e-12, rtol=0), (
            f'{name}, seed={seed}: {m} mismatch: expected {row[m]}, got {values[m]}'
        )


def summarize(values: pd.DataFrame, columns: list[str],
              group: str | None = None) -> pd.DataFrame:
    samples = [(str(arm), df) for arm, df in values.groupby(group, sort=False)] if group else [('', values)]
    out = []
    for group_name, subset in samples:
        for col in columns:
            x = subset[col].to_numpy(dtype=float)
            assert x.size and np.all(np.isfinite(x)), (group_name, col)
            mean = float(x.mean())
            sd = float(x.std(ddof=1)) if len(x) > 1 else float('nan')
            margin = float(t.ppf(0.975, len(x)-1) * sd / math.sqrt(len(x))) if len(x) > 1 else float('nan')
            row = {'metric': col, 'n': int(len(x)), 'mean': mean,
                   'sample_std': sd, 'min': float(x.min()), 'max': float(x.max()),
                   'ci95_lower': mean-margin, 'ci95_upper': mean+margin}
            if group:
                row[group] = group_name
            out.append(row)
    result = pd.DataFrame(out)
    return result[[group, *[k for k in result.columns if k != group]]] if group else result


def add_measurements(row: dict, prefix: str,
                     values: dict, cm: np.ndarray) -> None:
    for metric, value in values.items():
        row[f'{prefix}_{metric}'] = value
    for name, value in zip(('tn', 'fp', 'fn', 'tp'), cm):
        row[f'{prefix}_{name}'] = int(value)



def validate_race_label_counts(frame: pd.DataFrame, name: str) -> dict[str, int]:
    counts = frame.groupby(['race', 'two_year_recid']).size()
    for race in (0, 1):
        for target in (0, 1):
            assert int(counts.get((race, target), 0)) > 0, (
                f'{name} has no examples for race={race}, label={target}'
            )
    return {f'race_{race}_label_{target}': int(counts[(race, target)])
            for race in (0, 1) for target in (0, 1)}


def roc_predict(roc: RejectOptionClassification, dataset: BinaryLabelDataset,
                labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Run an ALREADY-FITTED ROC policy; do not use ground-truth test labels.

    Input .labels is explicitly replaced with classifier predictions.
    AIF360 ROC predict consults .scores and .protected_attributes only.
    """
    pred = prediction_bld(dataset, labels, scores)
    before_labels = pred.labels.copy()
    before_scores = pred.scores.copy()
    transformed = roc.predict(pred)
    assert np.array_equal(pred.labels, before_labels), 'ROC mutated input labels'
    assert np.array_equal(pred.scores, before_scores), 'ROC mutated input scores'
    output = transformed.labels.ravel().astype(int)
    assert len(output) == len(labels)
    assert set(np.unique(output)).issubset({0, 1})
    return output


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='A',
                        help='Output suffix: SMOKE for 1 seed, A for all ten')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(EXPECTED_SEEDS),
                        help='Subset of seeds 101..110; use --tag SMOKE for a pilot')
    parser.add_argument('--overwrite', action='store_true',
                        help='Explicitly allow overwriting existing output for this tag')
    args = parser.parse_args()
    tag = args.tag
    if not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Tag must use letters, digits, _ or -')
    seeds = tuple(args.seeds)
    assert seeds and len(seeds) == len(set(seeds)) and set(seeds) <= set(EXPECTED_SEEDS)
    if tag == 'A':
        assert seeds == EXPECTED_SEEDS, 'Tag A requires the full 101–110 run'
    OUT.mkdir(parents=True, exist_ok=True)
    existing = list(OUT.glob(f'*_{tag}.csv')) + list(OUT.glob(f'*_{tag}.json'))
    if existing and not args.overwrite:
        raise FileExistsError(
            f'Run {tag} has existing outputs; choose another tag or use --overwrite: {existing}'
        )

    assert sha256(DATA_PATH) == EXPECTED_SHA256, 'Unexpected input dataset hash'
    data = pd.read_csv(DATA_PATH)
    assert data.shape == (5278, 12)
    assert data.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert data.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    x = data.drop(columns='two_year_recid')
    y = data.two_year_recid

    ref_paths = {
        '008': PROV / 'run_008_none_per_seed_results_A.csv',
        '010': PROV / 'run_010_reweighing_per_seed_results_A.csv',
    }
    refs = {key: pd.read_csv(path) for key, path in ref_paths.items()}
    for key, frame in refs.items():
        assert sorted(frame.seed.tolist()) == list(EXPECTED_SEEDS), key

    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    membership: list[pd.DataFrame] = []
    counts_rows: list[dict] = []

    for seed in seeds:
        # Identical outer 70/30 split and training ordering to Runs 038/040.
        train_idx, outer_idx = train_test_split(
            np.arange(len(data)), test_size=.30, random_state=seed,
            shuffle=True, stratify=x.race,
        )
        outer_strata = (
            x.iloc[outer_idx].race.astype(str).to_numpy() + '_' +
            y.iloc[outer_idx].astype(str).to_numpy()
        )
        cal_idx, test_idx = train_test_split(
            outer_idx, test_size=.50, random_state=seed + 10_000,
            shuffle=True, stratify=outer_strata,
        )
        assert (len(train_idx), len(cal_idx), len(test_idx)) == (3694, 792, 792)
        assert len(set(train_idx) | set(cal_idx) | set(test_idx)) == 5278
        assert set(train_idx).isdisjoint(cal_idx)
        assert set(train_idx).isdisjoint(test_idx)
        assert set(cal_idx).isdisjoint(test_idx)

        for split, ids in (('train', train_idx), ('calibration', cal_idx), ('test', test_idx)):
            membership.append(pd.DataFrame({
                'seed': seed, 'split': split, 'original_row_id': ids,
            }))
            counts_rows.append({'seed': seed, 'split': split, 'rows': len(ids),
                                **validate_race_label_counts(data.iloc[ids], f'{seed}/{split}')})

        train = create_bld(x.iloc[train_idx], y.iloc[train_idx])
        outer = create_bld(x.iloc[outer_idx], y.iloc[outer_idx])
        cal = create_bld(x.iloc[cal_idx], y.iloc[cal_idx])
        test = create_bld(x.iloc[test_idx], y.iloc[test_idx])
        assert all(part.feature_names == train.feature_names for part in (outer, cal, test))
        assert train.feature_names == x.columns.tolist()

        reweigher = Reweighing(unprivileged_groups=UNPRIVILEGED,
                               privileged_groups=PRIVILEGED)
        weighted_train = reweigher.fit_transform(train)
        weights = np.asarray(weighted_train.instance_weights, dtype=float).ravel()
        assert weights.shape == (3694,)
        assert np.isfinite(weights).all() and (weights > 0).all()
        assert not np.allclose(weights, train.instance_weights)
        assert np.isclose(weights.sum(), len(weights), atol=1e-8, rtol=0)
        assert np.array_equal(weighted_train.features, train.features)
        assert np.array_equal(weighted_train.labels, train.labels)
        assert np.array_equal(weighted_train.protected_attributes, train.protected_attributes)
        ref010 = refs['010'].loc[refs['010'].seed == seed].iloc[0]
        for label, actual in (
            ('reweigh_weight_min', weights.min()),
            ('reweigh_weight_max', weights.max()),
            ('reweigh_weight_sum', weights.sum()),
        ):
            assert np.isclose(float(ref010[label]), float(actual), atol=1e-10, rtol=0), (
                f'Frozen Run 010 weight mismatch seed={seed}: {label}'
            )

        unweighted_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        weighted_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        unweighted_lr.fit(train.features, train.labels.ravel())
        weighted_lr.fit(weighted_train.features, weighted_train.labels.ravel(),
                        sample_weight=weights)
        classifiers = {'unweighted': unweighted_lr, 'weighted': weighted_lr}

        # Diagnostic against original full 30% outer holdout, without using it
        # for any classifier/ROC training or policy selection.
        for name, lr in classifiers.items():
            outer_lr_labels = lr.predict(outer.features).astype(int)
            outer_metrics, outer_cm = measurements(outer, outer_lr_labels)
            if name == 'unweighted':
                verify_reference(refs['008'], seed, outer_metrics, outer_cm,
                                 'Frozen Run 008 NONE')
            else:
                verify_reference(refs['010'], seed, outer_metrics, outer_cm,
                                 'Frozen Run 010 Re')

        classifier_predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for name, lr in classifiers.items():
            classifier_predictions[name] = {}
            for split, dataset in (('calibration', cal), ('test', test)):
                labels = lr.predict(dataset.features).astype(int)
                prob = lr.predict_proba(dataset.features)[:, 1].astype(float)
                assert len(labels) == len(prob) == len(dataset.labels)
                assert np.isfinite(prob).all() and ((0 <= prob) & (prob <= 1)).all()
                classifier_predictions[name][split] = labels, prob

        for arm, uses_weights, score_mode in ARMS:
            classifier = 'weighted' if uses_weights else 'unweighted'
            cal_labels, cal_prob = classifier_predictions[classifier]['calibration']
            test_labels, test_prob = classifier_predictions[classifier]['test']
            cal_scores = cal_labels.astype(float) if score_mode == 'hard' else cal_prob
            test_scores = test_labels.astype(float) if score_mode == 'hard' else test_prob
            cal_prediction_bld = prediction_bld(cal, cal_labels, cal_scores)

            roc = RejectOptionClassification(
                unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED,
                **ROC_PARAMETERS,
            )
            print(f'Seed {seed} {arm}: fitting ROC on 792 calibration rows', flush=True)
            # Only calibration labels enter ROC.fit. Final test labels NEVER do.
            snapshot_labels = cal_prediction_bld.labels.copy()
            snapshot_scores = cal_prediction_bld.scores.copy()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                roc.fit(cal, cal_prediction_bld)
            assert np.array_equal(snapshot_labels, cal_prediction_bld.labels)
            assert np.array_equal(snapshot_scores, cal_prediction_bld.scores)
            warning_text = ' | '.join(str(w.message) for w in caught)
            no_feasible_warning = ('Unable to satisy fairness constraints' in warning_text
                                   or 'Unable to satisfy fairness constraints' in warning_text)
            threshold = float(roc.classification_threshold)
            margin = float(roc.ROC_margin)
            assert np.isfinite(threshold) and np.isfinite(margin)
            assert ROC_PARAMETERS['low_class_thresh'] - 1e-12 <= threshold <= ROC_PARAMETERS['high_class_thresh'] + 1e-12
            assert 0 <= margin <= min(threshold, 1-threshold) + 1e-12

            # ROC.predict does NOT use truth labels. After this, we may safely
            # evaluate both calibration (in-sample) and test (held-out).
            cal_post = roc_predict(roc, cal, cal_labels, cal_scores)
            test_post = roc_predict(roc, test, test_labels, test_scores)
            row = {
                'seed': seed, 'arm': arm, 'classifier': classifier,
                'weight_aware': uses_weights, 'roc_score_mode': score_mode,
                'n_train': len(train.labels), 'n_calibration': len(cal.labels),
                'n_test': len(test.labels), 'outer_holdout_rows': len(outer.labels),
                'run008_outer_lr_match': True, 'run010_outer_lr_match': True,
                'reweigh_weights_modified': True,
                'training_weight_min': float(weights.min()),
                'training_weight_max': float(weights.max()),
                'training_weight_sum': float(weights.sum()),
                'lr_n_iter': int(classifiers[classifier].n_iter_[0]),
                'roc_threshold': threshold, 'roc_margin': margin,
                'roc_warning_no_feasible_on_calibration': no_feasible_warning,
                'roc_warning_messages': warning_text,
                'test_labels_used_for_roc_fit': False,
            }

            for split, dataset, pre, scores, post, ids in (
                ('calibration', cal, cal_labels, cal_scores, cal_post, cal_idx),
                ('test', test, test_labels, test_scores, test_post, test_idx),
            ):
                pre_metrics, pre_cm = measurements(dataset, pre)
                post_metrics, post_cm = measurements(dataset, post)
                add_measurements(row, f'{split}_pre_roc', pre_metrics, pre_cm)
                add_measurements(row, f'{split}_post_roc', post_metrics, post_cm)
                row[f'{split}_roc_feasible'] = bool(
                    ROC_PARAMETERS['metric_lb'] - 1e-12 <= post_metrics['equal_opportunity_difference']
                    <= ROC_PARAMETERS['metric_ub'] + 1e-12
                )
                threshold_labels = (scores > threshold).astype(int)
                changed = pre != post
                row[f'{split}_roc_changes_from_lr05'] = int(changed.sum())
                row[f'{split}_threshold_only_changes_from_lr05'] = int((pre != threshold_labels).sum())
                row[f'{split}_margin_only_changes_from_threshold'] = int((threshold_labels != post).sum())
                truth = dataset.labels.ravel().astype(int)
                row[f'{split}_correct_to_incorrect_from_lr05'] = int(
                    ((pre == truth) & (post != truth)).sum()
                )
                row[f'{split}_incorrect_to_correct_from_lr05'] = int(
                    ((pre != truth) & (post == truth)).sum()
                )
                net_accuracy_change = (
                    row[f'{split}_incorrect_to_correct_from_lr05']
                    - row[f'{split}_correct_to_incorrect_from_lr05']
                ) / len(truth)
                assert np.isclose(net_accuracy_change, post_metrics['accuracy'] - pre_metrics['accuracy'],
                                  rtol=0, atol=1e-12), (
                    f'Accuracy decomposition failed seed={seed}, arm={arm}, split={split}'
                )
                predictions.append(pd.DataFrame({
                    'seed': seed, 'arm': arm, 'split': split, 'original_row_id': ids,
                    'race': dataset.protected_attributes.ravel().astype(int),
                    'true_label': truth,
                    'lr_predicted_label': pre,
                    'lr_positive_probability': (cal_prob if split == 'calibration' else test_prob),
                    'roc_input_score': scores,
                    'threshold_only_label': threshold_labels,
                    'roc_predicted_label': post,
                }))
            # Fitting to calibration can emit a no-feasible warning; this is
            # separate from whether the final test set satisfies the tolerance.
            if no_feasible_warning and row['calibration_roc_feasible']:
                raise AssertionError('ROC warned no feasible calibration candidate but policy was feasible')
            print(
                f'Seed {seed} {arm}: calibration feasible={row["calibration_roc_feasible"]}, '
                f'test EOD={row["test_post_roc_equal_opportunity_difference"]:+.5f}, '
                f'test feasible={row["test_roc_feasible"]}, '
                f'test acc={row["test_post_roc_accuracy"]:.6f}, '
                f'test changes={row["test_roc_changes_from_lr05"]}/792, '
                f'threshold={threshold:.4f}, margin={margin:.4f}', flush=True
            )
            rows.append(row)

    per_seed = pd.DataFrame(rows)
    assert len(per_seed) == len(seeds) * len(ARMS)
    assert not per_seed.duplicated(['seed', 'arm']).any()
    per_seed.to_csv(OUT / f'per_seed_results_{tag}.csv', index=False, float_format='%.15g')
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / f'prediction_level_{tag}.csv', index=False, float_format='%.15g'
    )
    pd.concat(membership, ignore_index=True).to_csv(
        OUT / f'split_membership_{tag}.csv', index=False
    )
    pd.DataFrame(counts_rows).to_csv(
        OUT / f'group_label_counts_{tag}.csv', index=False
    )
    summary_cols = [f'test_post_roc_{name}' for name in METRICS] + [
        'test_roc_changes_from_lr05', 'roc_threshold', 'roc_margin',
    ]
    summarize(per_seed, summary_cols, group='arm').to_csv(
        OUT / f'aggregate_summary_{tag}.csv', index=False, float_format='%.15g'
    )
    contrasts = [
        ('B_minus_A', 'B_hard_weighted', 'A_hard_unweighted'),
        ('C_minus_A', 'C_prob_unweighted', 'A_hard_unweighted'),
        ('D_minus_B', 'D_prob_weighted', 'B_hard_weighted'),
        ('D_minus_C', 'D_prob_weighted', 'C_prob_unweighted'),
        ('D_minus_A', 'D_prob_weighted', 'A_hard_unweighted'),
    ]
    paired_rows = []
    for name, left, right in contrasts:
        a = per_seed[per_seed.arm == left].set_index('seed').sort_index()
        b = per_seed[per_seed.arm == right].set_index('seed').sort_index()
        assert a.index.equals(b.index)
        for seed in a.index:
            item = {'seed': int(seed), 'comparison': name,
                    'intervention_arm': left, 'reference_arm': right}
            for key in METRICS:
                item[f'delta_test_post_roc_{key}'] = (
                    float(a.loc[seed, f'test_post_roc_{key}'])
                    - float(b.loc[seed, f'test_post_roc_{key}'])
                )
            item['delta_test_roc_changes'] = int(a.loc[seed, 'test_roc_changes_from_lr05']) - int(b.loc[seed, 'test_roc_changes_from_lr05'])
            paired_rows.append(item)
    pd.DataFrame(paired_rows).to_csv(
        OUT / f'paired_deltas_{tag}.csv', index=False, float_format='%.15g'
    )
    manifest = {
        'experiment': 'run_041_heldout_re_roc', 'tag': tag,
        'seeds': list(seeds), 'script_sha256': sha256(Path(__file__).resolve()),
        'input_sha256': sha256(DATA_PATH),
        'reference_sha256': {k: sha256(path) for k, path in ref_paths.items()},
        'protocol': '70% training, 15% ROC calibration, 15% final held-out test',
        'split': {'outer': '70/30 race-stratified, random_state=seed',
                  'inner': '50/50 within 30%, stratified by race+label, random_state=seed+10000'},
        'target': 'two_year_recid=1',
        'privileged': 1, 'unprivileged': 0,
        'base_model': 'LogisticRegression(solver=lbfgs, max_iter=1000)',
        'reweighing': 'fit on train only, weights consumed for B/D',
        'ROC_parameters': ROC_PARAMETERS,
        'ROC_fit_labels': 'calibration labels only',
        'ROC_predict': 'calibration diagnostics and disjoint final-test rows',
        'test_labels_used_for_roc_fit': False,
        'score_modes': 'A/B hard 0/1 ROC scores; C/D LR.predict_proba scores',
        'comparability': (
            'Same outer training cohorts as Run 040 and same inner split pattern as Run 038. '
            'Results use only half the old 30% evaluation set and independently calibrated ROC; '
            'do not treat Run 040 and Run 041 as paired performance comparisons.'
        ),
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'aif360': getattr(aif360, '__version__', 'unreported'),
        },
    }
    (OUT / f'run_manifest_{tag}.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    print('\nRUN 041 COMPLETE; outputs in', OUT, flush=True)
    print('\nHeld-out post-ROC means:')
    print(per_seed.groupby('arm')[[f'test_post_roc_{k}' for k in METRICS]].mean().round(6).to_string())
    print('\nCalibration versus held-out EOD feasibility (out of', len(seeds), 'seeds):')
    print(per_seed.groupby('arm')[['calibration_roc_feasible', 'test_roc_feasible']].sum().to_string())


if __name__ == '__main__':
    run()
