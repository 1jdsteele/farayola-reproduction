"""Dependency-light sanity check for Run 038's proposed cohort partitioning.

No AIF360 required. Checks the exact input, independent partitions,
fixed outer seeds, group/label coverage, and fraction counts.
This is NOT the AIF360 experimental run.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / 'data' / 'compas_farayola_numeric.csv'
assert hashlib.sha256(p.read_bytes()).hexdigest() == ( 
    '6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e'
)
df = pd.read_csv(p)
assert df.shape == (5278, 12)
for seed in range(101, 111):
    train, outer = train_test_split(
        np.arange(len(df)), test_size=.3, random_state=seed,
        shuffle=True, stratify=df.race,
    )
    strata = df.iloc[outer].race.astype(str).to_numpy() + '_' + (
        df.iloc[outer].two_year_recid.astype(str).to_numpy()
    )
    cal, test = train_test_split(outer, test_size=.5, random_state=seed + 10000,
                                 shuffle=True, stratify=strata)
    assert (len(train), len(cal), len(test)) == (3694, 792, 792)
    assert not (set(train) & set(cal) or set(train) & set(test) or set(cal) & set(test))
    assert set(train) | set(cal) | set(test) == set(range(len(df)))
    for name, idx in [('train', train), ('calibration', cal), ('test', test)]:
        counts = df.iloc[idx].groupby(['race', 'two_year_recid']).size()
        assert all(counts.get((group, target), 0) > 0
                   for group in (0, 1) for target in (0, 1)), (seed, name)
    print(f'Seed {seed}: train={len(train)} calibration={len(cal)} test={len(test)}; '
          'disjoint, all race/label groups present')
print('DESIGN CHECK PASSED (AIF360 training/EO not executed).')
print('scikit-learn:', sklearn.__version__)
