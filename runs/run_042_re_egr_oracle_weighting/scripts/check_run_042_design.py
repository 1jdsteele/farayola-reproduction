"""Offline preflight for Run 042. Does NOT execute EGR or require AIF360."""
from pathlib import Path
import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from weighted_oracle import ExternalWeightOracleLR, features_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/compas_farayola_numeric.csv"
EXPECTED_HASH = "6654c9ba30e95dd05567189302e033585a7c5ffb2a93fda284a7bf22e57f160e"
METRICS = ("accuracy", "disparate_impact", "statistical_parity_difference",
           "equal_opportunity_difference", "predictive_equality_difference")


def main():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == EXPECTED_HASH
    df = pd.read_csv(DATA)
    assert df.shape == (5278, 12)
    assert df.race.value_counts().to_dict() == {0: 3175, 1: 2103}
    assert df.two_year_recid.value_counts().to_dict() == {0: 2795, 1: 2483}
    X = df.drop(columns="two_year_recid")
    y = df.two_year_recid
    p11 = pd.read_csv(ROOT / "provenance/run_011_egr_per_seed_results_A.csv").set_index("seed")
    p14 = pd.read_csv(ROOT / "provenance/run_014_re_egr_per_seed_results_A.csv").set_index("seed")
    for seed in range(101, 111):
        a, b = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed,
                                shuffle=True, stratify=X["race"])
        assert len(a) == 3694 and len(b) == 1584 and len(set(a) & set(b)) == 0
        assert set(X.iloc[a]["race"]) == {0, 1} and set(X.iloc[b]["race"]) == {0, 1}
        for metric in METRICS:
            assert np.isclose(p11.loc[seed, metric], p14.loc[seed, metric], rtol=0, atol=1e-12)
        assert bool(p14.loc[seed, "run011_metric_match"])
        print(f"Seed {seed}: split disjoint (3694/1584); Run 011 vs. 014 references identical")

    # Test cloneability, exact unity identity, external-weight propagation,
    # preservation of internal weight mass, and feature-order guard.
    x = pd.DataFrame({"a": [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],
                      "b": [0.,0.,1.,1.,0.,1.,0.,1.,0.,1.]})
    y = np.array([0,0,0,0,0,1,1,1,1,1])
    internal = np.array([0.8,1.1,0.9,1.2,1.0,1.2,0.8,1.1,0.9,1.0])
    ext = (0.8,1.2,0.8,1.2,0.8,1.2,0.8,1.2,0.8,1.2)
    digest = features_fingerprint(x)
    base = LogisticRegression(max_iter=1000, solver="lbfgs").fit(x, y, sample_weight=internal)
    unity = clone(ExternalWeightOracleLR((1.0,)*10, digest)).fit(x, y, sample_weight=internal)
    assert np.array_equal(unity.predict(x), base.predict(x))
    assert np.allclose(unity.model_.coef_, base.coef_, rtol=0, atol=1e-12)
    assert not unity.FIT_TRACES[-1]["changed_from_internal"]
    weighted = clone(ExternalWeightOracleLR(ext, digest)).fit(x, y, sample_weight=internal)
    assert weighted.FIT_TRACES[-1]["changed_from_internal"]
    assert weighted.FIT_TRACES[-1]["ratio_std"] > 0
    assert np.isclose(weighted.FIT_TRACES[-1]["combined_weight_mass"],internal.sum())
    try:
        clone(ExternalWeightOracleLR(ext, digest)).fit(x.iloc[::-1].reset_index(drop=True),
                                                         y, sample_weight=internal)
    except AssertionError as error:
        assert "reordered/changed" in str(error)
    else:
        raise AssertionError("Feature-order integrity check failed to reject reordered rows")
    print("\nRUN 042 DESIGN CHECK PASSED (AIF360 EGR model fitting not executed).")


if __name__ == "__main__":
    main()
