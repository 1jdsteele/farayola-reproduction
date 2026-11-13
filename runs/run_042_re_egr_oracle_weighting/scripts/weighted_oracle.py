"""Experimental, clone-compatible EGR oracle with external Reweighing weights.

Approach A ONLY: multiply Fairlearn's per-oracle training sample_weight by
AIF360 Reweighing weights. Fairlearn's constraints, objective and mixture
selection are NOT externally reweighted. It is not a fully weighted EGR.

Version target: AIF360 0.6.1 / Fairlearn 0.14.0 / sklearn 1.9.1.
"""

from __future__ import annotations

import hashlib
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def features_fingerprint(X: pd.DataFrame) -> str:
    """Detect any change to numeric feature rows, their order or columns."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fairlearn oracle must pass a pandas DataFrame")
    values = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(repr(tuple(str(c) for c in X.columns)).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


class ExternalWeightOracleLR(ClassifierMixin, BaseEstimator):
    """Fit LogisticRegression using EGR oracle weights * external weights.

    Cloneable constructor parameters are immutable. Each Fairlearn oracle fit
    receives EGR's sample_weight; this wrapper multiplies by the original
    training-observation Reweighing vector and renormalizes to the EGR weight
    total to avoid changing effective LR regularization merely due to scaling.

    Row-order is verified by a full SHA-256 feature fingerprint at each fit.
    A class-level per-process trace captures ALL fitted clones, including
    clones not selected into the final randomized mixture.
    """

    FIT_TRACES: ClassVar[list[dict]] = []

    def __init__(
        self,
        external_weights: tuple[float, ...],
        feature_sha256: str,
        *,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        preserve_weight_mass: bool = True,
    ):
        self.external_weights = external_weights
        self.feature_sha256 = feature_sha256
        self.max_iter = max_iter
        self.solver = solver
        self.preserve_weight_mass = preserve_weight_mass

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            raise ValueError("Missing EGR internal sample_weight; refusing fit")
        if features_fingerprint(X) != self.feature_sha256:
            raise AssertionError("EGR oracle reordered/changed training features")
        inner = np.asarray(sample_weight, dtype=float).reshape(-1)
        ext = np.asarray(self.external_weights, dtype=float).reshape(-1)
        if len(inner) != len(ext) or len(inner) != len(X):
            raise AssertionError("Internal and Reweighing weights not aligned")
        if not (np.all(np.isfinite(inner)) and np.all(inner >= 0) and inner.sum() > 0):
            raise AssertionError("EGR produced invalid weights")
        if not (np.all(np.isfinite(ext)) and np.all(ext > 0)):
            raise AssertionError("External Reweighing weights invalid")

        # Exact identity operation on the unity-control branch.
        if np.all(ext == 1.0):
            combined = inner.copy()
        else:
            combined = inner * ext
            if combined.sum() <= 0:
                raise AssertionError("Invalid zero combined oracle weight mass")
            if self.preserve_weight_mass:
                combined = combined * (inner.sum() / combined.sum())
        if not np.all(np.isfinite(combined)):
            raise AssertionError("Invalid combined oracle weights")

        positive = inner > 0
        ratios = combined[positive] / inner[positive]
        if ratios.size == 0:
            raise AssertionError("No positive EGR oracle weights")

        self.model_ = LogisticRegression(max_iter=self.max_iter, solver=self.solver)
        self.model_.fit(X, y, sample_weight=combined)
        self.classes_ = self.model_.classes_
        # This is a deliberate side-channel for audit only. Prediction uses
        # fitted model_, never FIT_TRACES.
        type(self).FIT_TRACES.append({
            "oracle_fit": len(type(self).FIT_TRACES) + 1,
            "n_rows": len(X),
            "internal_weight_mass": float(inner.sum()),
            "combined_weight_mass": float(combined.sum()),
            "internal_weight_min": float(inner.min()),
            "internal_weight_max": float(inner.max()),
            "combined_weight_min": float(combined.min()),
            "combined_weight_max": float(combined.max()),
            "external_min": float(ext.min()),
            "external_max": float(ext.max()),
            "external_std": float(ext.std()),
            "ratio_std": float(ratios.std()),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "changed_from_internal": bool(not np.array_equal(combined, inner)),
            "n_iter": int(np.max(self.model_.n_iter_)),
        })
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
