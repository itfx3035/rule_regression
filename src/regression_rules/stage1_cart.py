"""Stage 1, default: deterministic no-reuse CART on (X, y).

A **region** = a leaf of a depth-limited CART regressed on the target. Each
feature appears at most once *down a root-to-leaf path*, so every region is
an exact flat interval box: for every feature, a (possibly empty) ``lo`` and/or
``hi`` bound -- the same form stage 2's beam re-derives as a disjunction of
conjunctions, but found from supervised structure instead of unsupervised
point geometry.

This module is deliberately small: ``build_tree`` grows a deterministic tree
and ``region_mask`` evaluates one region's box. Hyperparameter choice, the
region budget, CV and public-region assembly all live in ``extractor.py``,
so the same building block serves every stage-2 path in the library.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import FeatureBins


@dataclass
class CartRegion:
    region_id: int
    splits: tuple        # ((f, "<=", thr), (f, ">", thr), ...) one per feature
    n: int               # train rows in the leaf


@dataclass
class FitResult:
    regions: list            # [CartRegion] in region_id order (largest first)
    depth: int
    min_leaf: int


# ---------------------------------------------------------------------------
# grow: deterministic depth-limited CART, no feature reuse per path
# ---------------------------------------------------------------------------

def _sse(y: np.ndarray) -> float:
    return float(((y - y.mean()) ** 2).sum())


def build_tree(X: np.ndarray, y: np.ndarray, *, depth: int, min_leaf: int,
               bins: FeatureBins) -> FitResult:
    """BFS grow; a feature may split at most once per path (no-reuse), so
    leaves are flat boxes with one bound per feature at most. Region ids are
    deterministic: largest leaf first, ties broken by the split list.
    `bins` supplies the per-feature quantile candidate thresholds."""
    d = X.shape[1]
    nodes: list[tuple] = [(np.arange(len(y)), 0, (), frozenset())]
    leaves: list[tuple[int, tuple]] = []
    ni = 0
    while ni < len(nodes):
        rows, dd, splits, used = nodes[ni]
        ni += 1
        n = len(rows)
        if dd >= depth or n <= 2 * min_leaf:
            leaves.append((n, splits))
            continue
        ys = y[rows]
        base = _sse(ys)
        best = None
        for f in range(d):
            if f in used:
                continue
            q = bins.q[f]
            v = X[rows, f]
            for p in range(len(q)):
                thr = q[p]
                m = v <= thr
                n1 = int(m.sum())
                if n1 < min_leaf or (n - n1) < min_leaf:
                    continue
                drop = base - _sse(ys[m]) - _sse(ys[~m])
                if best is None or drop > best[0] + 1e-12:
                    best = (drop, f, thr)
        if best is None:
            leaves.append((n, splits))
            continue
        _, f, thr = best
        m = X[rows, f] <= thr
        rows_l, rows_r = rows[m], rows[~m]
        if len(rows_l) < 5 or len(rows_r) < 5:      # sanity net
            leaves.append((n, splits))
            continue
        nodes.append((rows_l, dd + 1, splits + ((f, "<=", thr),), used | {f}))
        nodes.append((rows_r, dd + 1, splits + ((f, ">", thr),), used | {f}))
    leaves.sort(key=lambda t: (-t[0],
                               [(str(a), str(b), str(c)) for a, b, c in t[1]]))
    regions = [CartRegion(region_id=rid, splits=tuple(s), n=n)
               for rid, (n, s) in enumerate(leaves)]
    return FitResult(regions=regions, depth=depth, min_leaf=min_leaf)


def region_mask(r: CartRegion, X: np.ndarray) -> bool:
    """Boolean mask of ``X`` matched by the region's flat interval box."""
    m = np.ones(X.shape[0], dtype=bool)
    for f, op, thr in r.splits:
        if op == "<=":
            m &= X[:, f] <= thr
        else:
            m &= X[:, f] > thr
    return m
