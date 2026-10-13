"""Shared threshold grid: training-data quantiles, original units.

All stage-2 learners (RIPPER-style beam, GA) search over the discrete space
of ``(feature, op, grid_position)`` so thresholds are data-stable, discrete
edits (beam/GA mutations move a few positions), and always interpretable
training-data quantiles. Positions are internal; the public ``Region``
literals carry the original-scale ``threshold`` value (via ``threshold()``).
"""
from __future__ import annotations

import numpy as np


class FeatureBins:
    """Discretize each feature into a grid of training-data quantiles.

    ``q[f]`` is a strictly increasing array of grid thresholds (deduped
    quantiles of training column ``f``). A literal ``(f, op, pos)``
    (internal form) evaluates to ``X[:, f] op q[f][pos]``; the public
    (name, op, threshold) literal uses ``threshold(f, pos)``.

    >>> import numpy as np
    >>> X = np.array([[1., 10.], [2., 20.], [3., 30.], [4., 40.]])
    >>> b = FeatureBins(X, n_quantiles=4)
    >>> b.n_positions(0), b.n_positions(1)
    (4, 4)
    >>> b.pos_of(0, 2.5)
    2
    >>> b.threshold(0, 2)
    3.0
    >>> np.asarray(b.literal_mask(np.array([[3.], [1.]]), 0, "<=", b.pos_of(0, 2.0)))
    array([False,  True])
    """

    def __init__(self, X_train: np.ndarray, n_quantiles: int = 40):
        self.n_quantiles = n_quantiles
        quantiles = np.linspace(0.0, 1.0, n_quantiles)
        q_all = np.quantile(X_train, quantiles, axis=0)
        self.q: list[np.ndarray] = [np.unique(q_all[:, j])
                                    for j in range(X_train.shape[1])]

    def n_positions(self, f: int) -> int:
        return len(self.q[f])

    def threshold(self, f: int, pos: int) -> float:
        return float(self.q[f][pos])

    def pos_of(self, f: int, value: float) -> int:
        """Nearest grid position for a raw value (clipped into range)."""
        return int(np.clip(np.searchsorted(self.q[f], value), 0,
                           len(self.q[f]) - 1))

    def literal_mask(self, X: np.ndarray, f: int, op: str, pos: int) -> np.ndarray:
        thr = self.q[f][pos]
        if op == "<=":
            return X[:, f] <= thr
        if op == ">":
            return X[:, f] > thr
        raise ValueError(f"unknown op {op!r}; use '<=' or '>'")
