"""Run 040: 2 x 2 audit of Reweighing and ROC score propagation.

Four arms (each includes ROC):
  A_hard_unweighted  : unweighted LR, hard 0/1 scores into ROC (original)
  B_hard_weighted    : Re-weighted LR, hard 0/1 scores into ROC (weight fix only)
  C_prob_unweighted  : unweighted LR, continuous probability scores into ROC (score fix only)
  D_prob_weighted    : Re-weighted LR, continuous probability scores into ROC (both fixes)

Historical-reproduction protocol: ROC is fitted and evaluated on the same test
cohort. The experiment isolates two integration decisions, but its ROC results
are NOT unbiased held-out generalization estimates. Preserve the source runs.
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


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='A', help='A for final, SMOKE for a one-seed check')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(EXPECTED_SEEDS),
                        help='Subset of original seeds for smoke test; must be 101..110')
    args = parser.parse_args()
    tag = args.tag
    if not tag.replace('_', '').replace('-', '').isalnum():
        raise ValueError('Invalid tag: use letters, digits, underscore, hyphen')
    seeds = tuple(args.seeds)
    assert seeds and len(seeds) == len(set(seeds)) and set(seeds) <= set(EXPECTED_SEEDS)
    if tag == 'A':
        assert seeds == EXPECTED_SEEDS, 'Tag A is reserved for the full ten-seed run'
    OUT.mkdir(exist_ok=True, parents=True)
    assert sha256(DATA_PATH) == EXPECTED_SHA256, 'Input COMPAS hash mismatch'
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (5278, 12)
    assert df.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert df.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    x = df.drop(columns='two_year_recid')
    y = df.two_year_recid

    ref_paths = {
        '008': PROV / 'run_008_none_per_seed_results_A.csv',
        '010': PROV / 'run_010_reweighing_per_seed_results_A.csv',
    }
    refs = {key: pd.read_csv(path) for key, path in ref_paths.items()}
    for label, reference in refs.items():
        assert sorted(reference.seed.tolist()) == list(EXPECTED_SEEDS), label

    # Optional local references. Do not modify them or require them in the zip.
    original_root = ROOT.parent
    optional_ref_paths = {
        '013': original_root / 'run_013_individual_roc' / 'metadata' / 'per_seed_results_A.csv',
        '019': original_root / 'run_019_reweighing_roc' / 'metadata' / 'per_seed_results_A.csv',
    }
    optional_refs = {
        key: pd.read_csv(path) for key, path in optional_ref_paths.items() if path.exists()
    }
    for key, path in optional_ref_paths.items():
        if key not in optional_refs:
            print(f'NOTE: optional original Run {key} comparison not found at {path}', flush=True)

    records: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.30, random_state=seed, shuffle=True,
            stratify=x.race,
        )
        ids = x_test.index.to_numpy(copy=True)
        x_train = x_train.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        train = create_bld(x_train, y_train)
        test = create_bld(x_test, y_test)
        assert train.feature_names == test.feature_names
        assert len(test.labels) == 1584 and len(train.labels) == 3694
        assert test.feature_names == x.columns.tolist()

        rew = Reweighing(unprivileged_groups=UNPRIVILEGED,
                         privileged_groups=PRIVILEGED)
        rwtrain = rew.fit_transform(train)
        weights = np.asarray(rwtrain.instance_weights, dtype=float).ravel()
        assert weights.shape == (3694,)
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)
        assert not np.allclose(weights, train.instance_weights)
        assert np.isclose(weights.sum(), len(weights), atol=1e-8, rtol=0)
        assert np.array_equal(rwtrain.features, train.features)
        assert np.array_equal(rwtrain.labels, train.labels)
        assert np.array_equal(rwtrain.protected_attributes, train.protected_attributes)
        ref010 = refs['010'].loc[refs['010'].seed == seed].iloc[0]
        for label, actual in [
            ('reweigh_weight_min', weights.min()),
            ('reweigh_weight_max', weights.max()),
            ('reweigh_weight_sum', weights.sum()),
        ]:
            assert np.isclose(float(ref010[label]), float(actual), atol=1e-10, rtol=0), (
                f'seed {seed}, Run 010 weight mismatch: {label}'
            )

        unweighted_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        weighted_lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        unweighted_lr.fit(train.features, train.labels.ravel())
        weighted_lr.fit(rwtrain.features, rwtrain.labels.ravel(), sample_weight=weights)
        classifiers = {'unweighted': unweighted_lr, 'weighted': weighted_lr}
        raw_outputs = {}
        for name, lr in classifiers.items():
            labels = lr.predict(test.features).astype(int)
            probabilities = lr.predict_proba(test.features)[:, 1].astype(float)
            assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
            base_metrics, base_cm = measurements(test, labels)
            if name == 'unweighted':
                verify_reference(refs['008'], seed, base_metrics, base_cm, 'Run 008 NONE')
            raw_outputs[name] = (labels, probabilities, base_metrics, base_cm)

        weighted_metrics, weighted_cm = raw_outputs['weighted'][2:]
        frozen010_matches = (
            np.array_equal(weighted_cm,
                           ref010[['tn','fp','fn','tp']].to_numpy(dtype=int))
            and all(np.isclose(weighted_metrics[k], ref010[k], atol=1e-12, rtol=0)
                    for k in METRICS)
        )
        # Run 037 matched Run 010 across all 10 seeds in the user's pinned environment.
        # Fail fast if this reproduction unexpectedly diverges.
        assert frozen010_matches, (
            f'Weighted pre-ROC classifier failed frozen Run 010 comparison for seed {seed}'
        )

        for arm, is_weighted, score_mode in ARMS:
            classifier = 'weighted' if is_weighted else 'unweighted'
            labels, probabilities, before_metrics, before_cm = raw_outputs[classifier]
            scores = labels if score_mode == 'hard' else probabilities
            pred_data = prediction_bld(test, labels, scores)
            assert np.array_equal(pred_data.scores.ravel(), scores.astype(float))

            roc = RejectOptionClassification(
                unprivileged_groups=UNPRIVILEGED, privileged_groups=PRIVILEGED,
                **ROC_PARAMETERS,
            )
            pred_snapshot = pred_data.labels.copy()
            score_snapshot = pred_data.scores.copy()
            print(f'Seed {seed} {arm}: fitting ROC (100 thresholds × 50 margins)', flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                transformed = roc.fit_predict(test, pred_data)
            assert np.array_equal(pred_data.labels, pred_snapshot), 'ROC unexpectedly mutated input labels'
            assert np.array_equal(pred_data.scores, score_snapshot), 'ROC unexpectedly mutated input scores'
            result_labels = transformed.labels.ravel().astype(int)
            assert len(result_labels) == len(labels)
            assert set(np.unique(result_labels).tolist()) <= {0, 1}
            after_metrics, after_cm = measurements(test, result_labels)
            threshold = float(roc.classification_threshold)
            margin = float(roc.ROC_margin)
            assert ROC_PARAMETERS['low_class_thresh'] - 1e-12 <= threshold <= ROC_PARAMETERS['high_class_thresh'] + 1e-12
            assert 0 <= margin <= min(threshold, 1-threshold) + 1e-12
            threshold_predictions = (scores > threshold).astype(int)
            feasible = ROC_PARAMETERS['metric_lb'] <= after_metrics['equal_opportunity_difference'] <= ROC_PARAMETERS['metric_ub']
            warnings_text = ' | '.join(str(w.message) for w in caught)
            warned_no_feasible = 'Unable to satisy fairness constraints' in warnings_text
            assert not (feasible and warned_no_feasible), (
                f'Unexpected discrepancy between ROC fit feasibility and warning: {seed} {arm}'
            )

            # Original hard-score ROC was a no-op across all ten seeds in Runs 013/019.
            if arm == 'A_hard_unweighted':
                assert np.array_equal(result_labels, labels), (
                    f'Original hard-score ROC unexpectedly altered seed {seed}; '
                    'compare against Runs 013 and 019 before interpreting the rest.'
                )
                for num in ('013', '019'):
                    if num in optional_refs:
                        verify_reference(optional_refs[num], seed, after_metrics, after_cm,
                                         f'Original Run {num} (ROC hard-score control)')

            row = {
                'seed': seed, 'arm': arm, 'classifier': classifier,
                'weight_aware': is_weighted, 'roc_score_mode': score_mode,
                'n_train': len(train.labels), 'n_test': len(test.labels),
                'training_weights_modified': True,
                'training_weight_min': float(weights.min()),
                'training_weight_max': float(weights.max()),
                'training_weight_sum': float(weights.sum()),
                'run008_baseline_match': True,
                'run010_pre_roc_match': bool(frozen010_matches),
                'roc_threshold': threshold, 'roc_margin': margin,
                'roc_feasible_on_fitted_test': bool(feasible),
                'roc_warning_no_feasible': bool(warned_no_feasible),
                'roc_warning_messages': warnings_text,
                'roc_changes_from_lr05': int(np.count_nonzero(labels != result_labels)),
                'threshold_only_changes_from_lr05': int(np.count_nonzero(labels != threshold_predictions)),
                'margin_only_changes_from_threshold': int(np.count_nonzero(threshold_predictions != result_labels)),
                'correct_to_incorrect_from_lr05': int(np.sum((labels == y_test) & (result_labels != y_test))),
                'incorrect_to_correct_from_lr05': int(np.sum((labels != y_test) & (result_labels == y_test))),
            }
            add_measurements(row, 'pre_roc', before_metrics, before_cm)
            add_measurements(row, 'post_roc', after_metrics, after_cm)
            records.append(row)
            predictions.append(pd.DataFrame({
                'seed': seed, 'arm': arm, 'original_row_id': ids,
                'race': x_test.race.to_numpy(), 'true_label': y_test.to_numpy(),
                'lr_predicted_label': labels, 'lr_positive_probability': probabilities,
                'roc_input_score': scores, 'threshold_only_label': threshold_predictions,
                'roc_predicted_label': result_labels,
            }))
            print(f'Seed {seed} {arm}: ROC changes={row["roc_changes_from_lr05"]}/{len(test.labels)}, '
                  f'accuracy={after_metrics["accuracy"]:.6f}, EOD={after_metrics["equal_opportunity_difference"]:+.6f}, '
                  f'threshold={threshold:.4f}, margin={margin:.4f}, '
                  f'feasible={feasible}, warnings={warned_no_feasible}', flush=True)

    output = pd.DataFrame(records)
    assert len(output) == len(seeds) * len(ARMS)
    assert not output.duplicated(['seed', 'arm']).any()
    output.to_csv(OUT / f'per_seed_results_{tag}.csv', index=False, float_format='%.15g')
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / f'prediction_level_{tag}.csv', index=False, float_format='%.15g'
    )
    metric_columns = [f'post_roc_{k}' for k in METRICS] + [
        'roc_changes_from_lr05', 'roc_threshold', 'roc_margin',
    ]
    summarize(output, metric_columns, group='arm').to_csv(
        OUT / f'aggregate_summary_{tag}.csv', index=False, float_format='%.15g'
    )
    # Paired arm contrasts within each seed: AB isolates weights with hard scores;
    # AC isolates scores with no weights; BD isolates scores after weights;
    # CD isolates weights with continuous scores.
    comparisons = [
        ('B_minus_A', 'B_hard_weighted', 'A_hard_unweighted'),
        ('C_minus_A', 'C_prob_unweighted', 'A_hard_unweighted'),
        ('D_minus_B', 'D_prob_weighted', 'B_hard_weighted'),
        ('D_minus_C', 'D_prob_weighted', 'C_prob_unweighted'),
        ('D_minus_A', 'D_prob_weighted', 'A_hard_unweighted'),
    ]
    paired = []
    for name, left, right in comparisons:
        aa = output[output.arm == left].set_index('seed')
        bb = output[output.arm == right].set_index('seed')
        assert aa.index.equals(bb.index)
        for seed in aa.index:
            row = {'seed': int(seed), 'comparison': name,
                   'intervention_arm': left, 'reference_arm': right}
            for metric in METRICS:
                row[f'delta_{metric}'] = aa.loc[seed, f'post_roc_{metric}'] - bb.loc[seed, f'post_roc_{metric}']
            row['delta_roc_changes'] = aa.loc[seed, 'roc_changes_from_lr05'] - bb.loc[seed, 'roc_changes_from_lr05']
            paired.append(row)
    pd.DataFrame(paired).to_csv(
        OUT / f'paired_deltas_{tag}.csv', index=False, float_format='%.15g'
    )
    manifest = {
        'experiment': 'run_040_corrected_re_roc', 'tag': tag, 'seeds': list(seeds),
        'dataset_sha256': sha256(DATA_PATH),
        'reference_sha256': {key: sha256(path) for key, path in ref_paths.items()},
        'optional_reference_sources': {key: str(optional_ref_paths[key]) for key in optional_refs},
        'data_protocol': {'split': '70/30', 'stratify': 'race',
                          'ROC_calibration': 'test, same as evaluation (historical protocol; leakage)',
                          'label': 'two_year_recid=1', 'groups': {'unprivileged': 0, 'privileged': 1}},
        'LR': 'LogisticRegression(solver=lbfgs,max_iter=1000)',
        'reweighing': 'training only; sample_weight passed for weighted arms',
        'ROC_parameters': ROC_PARAMETERS,
        'ROC_scores': 'A/B hard 0/1 labels; C/D actual LR.predict_proba probability scores',
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'sklearn': sklearn.__version__,
                     'aif360': getattr(aif360, '__version__', 'unreported')},
    }
    (OUT / f'run_manifest_{tag}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('\nRUN 040 COMPLETE; results saved to', OUT, flush=True)
    print(output.groupby('arm')[['post_roc_accuracy', 'post_roc_disparate_impact',
                                 'post_roc_statistical_parity_difference',
                                 'post_roc_equal_opportunity_difference',
                                 'post_roc_predictive_equality_difference',
                                 'roc_changes_from_lr05']].mean().to_string(), flush=True)


if __name__ == '__main__':
    run()
