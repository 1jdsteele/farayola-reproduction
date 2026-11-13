"""Static Run 040 input/split/logic design checks; requires no AIF360."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'compas_farayola_numeric.csv'
SHA = '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'


def main() -> None:
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == SHA
    df = pd.read_csv(DATA)
    assert df.shape == (5278, 12)
    assert df.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert df.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    for run in ('008_none', '010_reweighing'):
        ref = ROOT / 'provenance' / f'run_{run}_per_seed_results_A.csv'
        assert ref.exists(), ref
        frame = pd.read_csv(ref)
        assert sorted(frame.seed.tolist()) == list(range(101, 111))
    ids = np.arange(len(df))
    for seed in range(101, 111):
        train_idx, test_idx = train_test_split(
            ids, test_size=0.3, random_state=seed, shuffle=True, stratify=df.race,
        )
        assert len(train_idx) == 3694 and len(test_idx) == 1584
        assert set(train_idx).isdisjoint(test_idx)
        assert len(set(train_idx) | set(test_idx)) == len(df)
        counts = df.iloc[test_idx].groupby(['race', 'two_year_recid']).size()
        assert all(counts.get((race, label), 0) > 0
                   for race in (0, 1) for label in (0, 1))
        print(f'Seed {seed}: train=3694, test=1584, disjoint, all race/label groups present')
    print('\nRUN 040 DESIGN CHECK PASSED (AIF360/ROC model fitting not executed)')


if __name__ == '__main__':
    main()
