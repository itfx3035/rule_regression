"""Stage 1, alternative learner (target-agnostic): DBSCAN in a
standardized feature space ("zx").

This is the *unsupervised* region-discovery line. Unlike ``stage1="cart"``
(a deterministic no-reuse CART on (x, y)), the space uses only the features:
the target never enters the geometry. That is the honest trade: the regions
here are what the *inputs* look like, so the model can be fit before the
target is known (or when y is only a surrogate); the cost is that region
boundaries are not necessarily y-informative.

Semantics (port of the research ``stage1.py`` / ``common.py`` pipeline,
zx space, library conventions):

* rows are standardized with the train mean/std (z-score fit on the rows
  the space is fit from);
* ``eps`` candidates are multipliers (1.0 / 1.5 / 2.0) over the kNN
  heuristic: the median, over rows, of the distance to the
  ``(min_samples - 1)``-th nearest neighbour (the distance at which a row
  just becomes a core point); the extractor CVs those candidates;
* DBSCAN (sklearn convention: core points drive expansion, border points
  are not re-expanded);
* clusters smaller than ``min_cluster_size`` are demoted to *noise*;
* regions are ordered largest-first and renumbered 0..k-1;
* a *noise row is simply not in any region* -- it matches no rule and
  falls back to the global train mean at inference.

Coverage is a **measured** property here, not a guarantee: CART leaves
partition the row space (100% by construction); DBSCAN clusters can leave
gaps, small-cluster demotions, and precision-gated salvages all
contribute to unmatched rows. ``match()`` returns -1 for them and
``predict()`` uses ``fallback_value`` (the train mean).
"""
from __future__ import annotations

import numpy as np

#: Multipliers over the kNN-heuristic eps, forming the CV candidate grid.
#: (Per-dataset widening of this grid is a research-side lever, not a
#: library knob; the default grid is the research "default/zx" grid.)
EPS_RATIOS: tuple[float, ...] = (1.0, 1.5, 2.0)


def fit_z(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardization parameters (mean, std; zero std -> 1)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _pairwise(X: np.ndarray) -> np.ndarray:
    return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))


def eps_candidates(Z: np.ndarray, min_samples: int,
                   ratios: tuple[float, ...] = EPS_RATIOS) -> list[float]:
    """kNN-heuristic eps, rounded, times each ratio; sorted ascending,
    deduped. The heuristic is *scale discovery only* -- it never sees
    held-out rows when the extractor calls this (fit rows only)."""
    if min_samples > Z.shape[0]:
        raise ValueError(f"min_samples={min_samples} > n={Z.shape[0]}")
    s = np.sort(_pairwise(Z), axis=1)
    kth = s[:, min_samples - 1]        # s[:, 0] is the self-distance (0)
    e0 = float(np.median(kth))
    out = {round(e0 * r, 2) for r in ratios}
    return sorted(out)


def dbscan_labels(Z: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN (density-reachability / sklearn convention).

    Only core points drive cluster expansion; a non-core point becomes a
    border point when it enters a core's epsilon-neighbourhood, but its own
    neighbourhood is not re-expanded. Labels: 0..C-1 for clusters, -1 noise.
    Deterministic (row-order traversal).
    """
    D = _pairwise(Z)
    nbrs = D <= eps
    core = nbrs.sum(axis=1) >= min_samples
    labels = -2 * np.ones(Z.shape[0], dtype=np.int64)
    cid = 0
    for i in range(Z.shape[0]):
        if labels[i] != -2:
            continue
        if not core[i]:
            labels[i] = -1
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            for k in np.flatnonzero(nbrs[j]):
                if labels[k] == -2:
                    labels[k] = cid
                    stack.append(int(k))
                elif labels[k] == -1:
                    labels[k] = cid          # border point; do not re-expand
        cid += 1
    return labels


def cluster(Z: np.ndarray, eps: float, min_samples: int,
            min_cluster_size: int) -> tuple[list[int], np.ndarray]:
    """DBSCAN on the geometry. Returns (ids largest-first, labels) with
    clusters smaller than ``min_cluster_size`` demoted to noise so stage 2
    only regions well-populated clusters."""
    labels = dbscan_labels(Z, eps, min_samples)
    ids = [c for c in np.unique(labels)
           if c >= 0 and (labels == c).sum() >= min_cluster_size]
    ids.sort(key=lambda c: -int((labels == c).sum()))
    for c in np.unique(labels):
        if c >= 0 and c not in ids:
            labels[labels == c] = -1
    return ids, labels


def centroid_assign(Zte: np.ndarray, Ztr: np.ndarray, labels: np.ndarray,
                    ids: list[int], eps: float) -> np.ndarray:
    """Pure stage-1 test assignment: nearest train-cluster centroid, gated by
    ``eps`` (a row must be within ``eps`` of the centroid). Rows outside
    every cluster return -1 (noise -> fallback). This is the stage-1 proxy
    used by the eps CV; it is *not* what ``predict()`` does (that is rule
    matching), but it is a cheap, leakage-free estimate of how well the
    regions partition new rows.
    """
    n = Zte.shape[0]
    if not ids:
        return np.full(n, -1, dtype=int)
    Cc = np.stack([Ztr[labels == c].mean(axis=0) for c in ids])
    D = ((Zte[:, None, :] - Cc[None, :, :]) ** 2).sum(axis=2)
    nn = D.argmin(axis=1)
    return np.where(D.min(axis=1) <= eps ** 2, np.asarray(ids)[nn], -1)
