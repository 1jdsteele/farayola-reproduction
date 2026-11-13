"""Run 041 preflight: verify dataset, held-out split integrity, and references.

Does not import AIF360, fit models, or create result files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'compas_farayola_numeric.csv'
EXPECTED_SHA = '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'


def main() -> None:
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == EXPECTED_SHA
    frame = pd.read_csv(DATA)
    assert frame.shape == (5278, 12)
    assert frame.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert frame.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    for run, file in (
        ('008', ROOT / 'provenance/run_008_none_per_seed_results_A.csv'),
        ('010', ROOT / 'provenance/run_010_reweighing_per_seed_results_A.csv'),
    ):
        ref = pd.read_csv(file)
        assert sorted(ref.seed.tolist()) == list(range(101, 111)), f'Run {run} seeds'

    for seed in range(101, 111):
        train, outer = train_test_split(
            np.arange(len(frame)), test_size=.30, shuffle=True,
            stratify=frame.race, random_state=seed,
        )
        inner_strata = (
            frame.iloc[outer].race.astype(str).to_numpy() + '_' +
            frame.iloc[outer].two_year_recid.astype(str).to_numpy()
        )
        calibration, test = train_test_split(
            outer, test_size=.5, stratify=inner_strata,
            shuffle=True, random_state=seed + 10_000,
        )
        assert (len(train), len(calibration), len(test)) == (3694, 792, 792)
        assert len(set(train) | set(calibration) | set(test)) == len(frame)
        assert set(train).isdisjoint(calibration)
        assert set(train).isdisjoint(test)
        assert set(calibration).isdisjoint(test)
        for name, ids in (('train', train), ('calibration', calibration), ('test', test)):
            group_labels = frame.iloc[ids].groupby(['race', 'two_year_recid']).size()
            assert all(group_labels.get((race, y), 0) > 0
                       for race in (0, 1) for y in (0, 1)), (seed, name)
        print(f'Seed {seed}: train=3694, calibration=792, test=792; '
              'disjoint; all race/label groups present')

    print('\nRUN 041 DESIGN CHECK PASSED (no AIF360 / ROC fitting executed)')
    print('scikit-learn:', sklearn.__version__)


if __name__ == '__main__':
    main()
