"""Optional per-region numeric leaves: constant-mean or small OLS, the kind
chosen by train-only k-fold CV against the target -- mirroring the research
code, where OLS-beating-mean cost a real amount on the benchmarks
(concrete ~0.81 -> ~0.83 at the model level).

Leaves predict from features only (never the target), so region assignment
by rule matching and the leaf numbers never leak y. The extractor attaches
the chosen leaf (``kind = "mean" | "ols"``) to each ``Region`` when
``with_leaves=True``.
"""
from __future__ import annotations

import numpy as np

from .lits import Leaf


def _ols_coef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([X, np.ones(len(y))])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol          # (d+1,): feature coefs then intercept


def _fold_assignment(n: int, folds: int, seed: int = 42) -> list[int]:
    """Deterministic shuffle-then-round-robin fold IDs, 0..k-1 per index."""
    perm = np.random.RandomState(seed).permutation(n)
    fold_of = [0] * n
    for pos, i in enumerate(perm):
        fold_of[int(i)] = pos % folds
    return fold_of


def _cv_error(X: np.ndarray, y: np.ndarray, fold_of: list[int], kind: str) -> float:
    se = 0.0
    for f in set(fold_of):
        te = np.asarray([i for i in range(len(y)) if fold_of[i] == f])
        tr = np.setdiff1d(np.arange(len(y)), te)
        if kind == "mean":
            pred = np.full(len(te), y[tr].mean() if len(tr) else y.mean())
        else:
            if len(tr) <= X.shape[1]:
                return float("inf")          # under-determined fold -> not better
            coef = _ols_coef(X[tr], y[tr])
            pred = X[te] @ coef[:-1] + coef[-1]
        se += float(((y[te] - pred) ** 2).sum())
    return se


def pick_leaf_kind(X: np.ndarray, y: np.ndarray, folds: int = 5,
                   min_n: int = 8) -> str:
    """'mean' or 'ols' by train-only k-fold squared error.

    Regions too small to make an honest comparison (< ``min_n`` rows)
    default to the cheap constant mean:

    >>> import numpy as np
    >>> pick_leaf_kind(np.zeros((3, 2)), np.array([1., 2., 3.]))
    'mean'
    """
    if len(y) < min_n:
        return "mean"
    fold_of = _fold_assignment(len(y), folds)
    return "ols" if _cv_error(X, y, fold_of, "ols") < \
        _cv_error(X, y, fold_of, "mean") else "mean"


def fit_leaf(X: np.ndarray, y: np.ndarray, kind: str, names=()) -> Leaf:
    """Fit the leaf of the given kind on the region's rows.

    >>> import numpy as np
    >>> fit_leaf(np.zeros((5, 2)), np.array([1., 2., 3., 4., 2.]), "mean", ["a", "b"])
    Leaf(kind='mean', value=2.4, coef=None, names=['a', 'b'])
    >>> fit_leaf(np.zeros((5, 2)), np.array([1., 2., 3., 4., 2.]), "ols", ["a", "b"]).kind
    'ols'
    """
    if kind == "mean":
        return Leaf("mean", float(y.mean()), None, list(names))
    d = X.shape[1]
    coef = _ols_coef(X, y) if len(y) > d else np.zeros(d + 1)
    return Leaf("ols", float("nan"), coef, list(names))
