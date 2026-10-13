"""Stage 2, main line: RIPPER-style beam search for a region's defining rules.

Per region: P = train rows inside, N = all other train rows. Grow a covering
rule set (disjunction of conjunctions) over the shared literal space
(top-``k`` defining features x the quantile grid):

repeat (at most ``max_rules`` times):
  1. **grow**   beam search, scored by train F1 against the *working set*
               (cluster points not yet covered), literals restricted to the
               cluster's top-k defining features (MI + ANOVA rank);
  2. **prune**  free-slim reduced pruning (drop a literal only if the rule's
               working-set and negative coverage stay exactly unchanged,
               never producing the empty rule);
  3. **accept** keep the rule only if it covers >= 1 working point and
               strictly raises the union's train F1 (scored vs the full P).

Then the *union* is gated on train precision vs the base rate; a failing
union is salvaged by dropping the least-contributing rule until it clears.

Internal literals are ``(f, op, pos)`` over the shared grid; the extractor
translates them to public (name, op, threshold) literals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import FeatureBins


# --------------------------------------------------------------------------
# scoring against one-vs-rest membership
# --------------------------------------------------------------------------

@dataclass
class Score:
    coverage: float      # recall of the region by the rule
    precision: float
    f1: float
    base_rate: float     # precision floor: region fraction of scored rows
    n_cluster: int
    n_match: int
    n_tp: int
    n_fp: int
    n_fn: int


def score_mask(mask: np.ndarray, y: np.ndarray) -> Score:
    """Score a rule (or rule set) mask against one-vs-rest membership."""
    y = np.asarray(y, dtype=bool)
    if mask.shape != y.shape:
        raise ValueError(f"mask {mask.shape} vs y {y.shape}")
    tp = int(np.count_nonzero(mask & y))
    fp = int(np.count_nonzero(mask & ~y))
    fn = int(np.count_nonzero(~mask & y))
    n_cluster = tp + fn
    base_rate = n_cluster / len(y) if len(y) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    cov = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * cov / (prec + cov) if (prec + cov) else 0.0
    return Score(cov, prec, f1, base_rate=base_rate, n_cluster=n_cluster,
                 n_match=tp + fp, n_tp=tp, n_fp=fp, n_fn=fn)


def _rule_mask(bins: FeatureBins, X: np.ndarray, rule) -> np.ndarray:
    m = np.ones(X.shape[0], dtype=bool)
    for f, op, pos in rule:
        m &= bins.literal_mask(X, f, op, pos)
    return m


def _ruleset_mask(bins: FeatureBins, X: np.ndarray, rules) -> np.ndarray:
    m = np.zeros(X.shape[0], dtype=bool)
    for r in rules:
        if len(r):
            m |= _rule_mask(bins, X, r)
    return m


@dataclass
class FeatureImportance:
    feature: int
    score: float
    mi: float
    anova_f: float


def defining_features(bins: FeatureBins, X: np.ndarray, y: np.ndarray,
                      k: int = 3) -> list[FeatureImportance]:
    """Top-``k`` features by combined MI + ANOVA-F rank against membership."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not (0 < n1 < len(y)) or min(n1, n0) < 2:
        raise ValueError(f"degenerate membership: {n1} in / {n0} out")
    d = X.shape[1]
    mi = np.zeros(d)
    fa = np.zeros(d)
    for f in range(d):
        x = X[:, f]
        edges = bins.q[f]
        b = np.clip(np.searchsorted(edges, x, side="left"), 0, len(edges) - 1)
        joint = np.zeros((len(edges), 2))
        np.add.at(joint, (b, 0), ~y)
        np.add.at(joint, (b, 1), y)
        px = joint.sum(axis=1) / len(y)
        py = joint.sum(axis=0) / len(y)
        pij = joint / len(y)
        cells = pij > 0
        expected = np.where(cells, px[:, None] * py[None, :], 1.0)
        mi[f] = float(np.sum(pij[cells] * np.log(pij[cells] / expected[cells])))
        a, c = x[y], x[~y]
        pooled = ((n1 - 1) * a.var(ddof=1) + (n0 - 1) * c.var(ddof=1)) / (n1 + n0 - 2)
        if pooled > 0:
            fa[f] = (n1 * n0 / (n1 + n0 - 1)) * (a.mean() - c.mean()) ** 2 / pooled

    def _norm(v: np.ndarray) -> np.ndarray:
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 0 else np.zeros_like(v)

    s = 0.5 * _norm(mi) + 0.5 * _norm(fa)
    order = np.lexsort((np.arange(d), -s))
    return [FeatureImportance(int(f), float(s[f]), float(mi[f]), float(fa[f]))
            for f in order[:k]]


# --------------------------------------------------------------------------
# beam search over the region's one-vs-rest membership
# --------------------------------------------------------------------------

def _literal_masks(bins: FeatureBins, X: np.ndarray,
                   features: list[int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for f in features:
        qn = len(bins.q[f])
        M = np.empty((qn, 2), dtype=object)
        for p in range(qn):
            thr = bins.q[f][p]
            M[p, 0] = X[:, f] <= thr
            M[p, 1] = X[:, f] > thr
        masks[f] = M
    return masks


def _bounds(rule) -> tuple[dict, dict]:
    lo: dict[int, int] = {}
    hi: dict[int, int] = {}
    for f, op, p in rule:
        if op == ">":
            if f not in lo or p > lo[f]:
                lo[f] = p
        else:
            if f not in hi or p < hi[f]:
                hi[f] = p
    return lo, hi


def _union_f1(union: np.ndarray, pos: np.ndarray, n_pos: int) -> float:
    tp = int(np.count_nonzero(union & pos))
    if tp == 0:
        return 0.0
    fp = int(union.sum()) - tp
    cov = tp / n_pos
    prec = tp / (tp + fp)
    return 2 * prec * cov / (prec + cov)


def _grow(bins: FeatureBins, X: np.ndarray, pos: np.ndarray,
          lit: dict, features: list[int], beam_width: int,
          max_len: int):
    n_pos = int(pos.sum())
    universe = np.ones(X.shape[0], dtype=bool)
    best = (0.0, [], universe)
    beam = [best]
    for depth in range(1, max_len + 1):
        seen: dict[tuple, tuple] = {}
        for f1p, rule, m in beam:
            if len(rule) >= max_len:
                continue
            lo, hi = _bounds(rule)
            for f in features:
                qn = len(bins.q[f])
                for p in range(qn):
                    if f not in lo or p >= lo[f]:
                        if f not in hi or p < hi[f]:
                            cm2 = m & lit[f][p, 0]
                            tp = int(np.count_nonzero(cm2 & pos))
                            if tp:
                                nfp = int(cm2.sum()) - tp
                                f1 = 2 * (tp / (tp + nfp)) * (tp / n_pos) / \
                                     (tp / (tp + nfp) + tp / n_pos)
                                if f1 > 0:
                                    key = (len(rule), f, p, "<=")
                                    cur = seen.get(key)
                                    if cur is None or f1 > cur[0]:
                                        seen[key] = (f1, rule + [(f, "<=", p)], cm2)
                    if f not in hi or p > hi[f]:
                        if f not in lo or p > lo[f]:
                            cm2 = m & lit[f][p, 1]
                            tp = int(np.count_nonzero(cm2 & pos))
                            if tp:
                                nfp = int(cm2.sum()) - tp
                                f1 = 2 * (tp / (tp + nfp)) * (tp / n_pos) / \
                                     (tp / (tp + nfp) + tp / n_pos)
                                if f1 > 0:
                                    key = (len(rule), f, p, ">")
                                    cur = seen.get(key)
                                    if cur is None or f1 > cur[0]:
                                        seen[key] = (f1, rule + [(f, ">", p)], cm2)
        if not seen:
            break
        cands = list(seen.values())
        cands.sort(key=lambda t: (-t[0], len(t[1])))
        beam = cands[:beam_width]
        for f1, rule, m in cands:
            if f1 > best[0]:
                best = (f1, rule, m)
        if best[0] >= 1.0 - 1e-9:
            break
    return best[1], best[0], best[2]


def _reduce_prune(bins: FeatureBins, X: np.ndarray, rule, W, N):
    rule = list(rule)
    while len(rule) > 1:
        cm = _rule_mask(bins, X, rule)
        t0 = int(np.count_nonzero(cm & W))
        n0 = int(np.count_nonzero(cm & N))
        hit = None
        for i in range(len(rule)):
            trial = rule[:i] + rule[i + 1:]
            tm = _rule_mask(bins, X, trial)
            if int(np.count_nonzero(tm & W)) == t0 and int(np.count_nonzero(tm & N)) == n0:
                hit = i
                break
        if hit is None:
            break
        del rule[hit]
    return rule


def gate_and_salvage(bins: FeatureBins, X: np.ndarray, rules: list,
                     per_rule: list, pos: np.ndarray, n_pos: int,
                     base_rate: float, min_precision_margin: float):
    """Union precision gate + least-contribution salvage. Returns
    (rules, per_rule, union_score, failed)."""
    def _reunion(remaining):
        return _ruleset_mask(bins, X, remaining)

    union_m = _reunion(rules)
    u = score_mask(union_m, pos)
    if u.precision < base_rate + min_precision_margin:
        while len(rules) > 1:
            worst, worst_gain = None, None
            for i in range(len(rules)):
                trial = rules[:i] + rules[i + 1:]
                gain = _union_f1(union_m, pos, n_pos) - _union_f1(_reunion(trial), pos, n_pos)
                if worst_gain is None or gain < worst_gain:
                    worst, worst_gain = i, gain
            rules.pop(worst)
            per_rule.pop(worst)
            union_m = _reunion(rules)
            u = score_mask(union_m, pos)
            if u.precision >= base_rate + min_precision_margin:
                break
    if u.precision < base_rate + min_precision_margin:
        return rules, per_rule, u, True
    return rules, per_rule, u, False


@dataclass
class FitResult:
    """One region's extracted rule set (internal literal form).

    ``pareto`` (GA mode only) holds the (union F1, total literals) front
    tracked over generations; the GA returns the max-F1 point of it."""
    rules: list = field(default_factory=list)          # (f, op, pos) tuples
    per_rule: list = field(default_factory=list)       # Score per rule (train)
    union: Score | None = None
    n_points_train: int = 0
    base_rate: float = 0.0
    reason: str | None = None
    pareto: list | None = None  # [(f1, n_literals)] -- GA only

    @property
    def ok(self) -> bool:
        return self.reason is None and bool(self.rules)


def fit_region_rules(bins: FeatureBins, X: np.ndarray, pos: np.ndarray, *,
                     k: int = 3, max_rules: int = 3, max_len: int = 4,
                     beam_width: int = 20, min_precision_margin: float = 0.05,
                     min_points: int = 5) -> FitResult:
    """Fit one covering rule set for a region (mask ``pos``)."""
    X = np.asarray(X, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n = X.shape[0]
    n_pos = int(pos.sum())
    res = FitResult(n_points_train=n_pos, base_rate=n_pos / n if n else 0.0)
    if n_pos < min_points:
        res.reason = f"too few train points ({n_pos})"
        return res
    if min(n_pos, n - n_pos) < 2:
        res.reason = "degenerate one-vs-rest (no negatives)"
        return res
    try:
        feats = [f.feature for f in defining_features(bins, X, pos, k=k)]
    except ValueError as e:
        res.reason = f"feature preselection: {e}"
        return res
    lit = _literal_masks(bins, X, feats)
    W = pos.copy()
    union_m = np.zeros(n, dtype=bool)
    for _ in range(max_rules):
        rule, f1, _ = _grow(bins, X, W, lit, feats, beam_width, max_len)
        rule = _reduce_prune(bins, X, rule, W, ~pos)
        m = _rule_mask(bins, X, rule)
        if int(np.count_nonzero(m & W)) == 0 \
                or _union_f1(union_m | m, pos, n_pos) <= _union_f1(union_m, pos, n_pos):
            break
        res.rules.append(rule)
        res.per_rule.append(score_mask(m, pos))
        union_m |= m
        W &= ~m
        if not W.any():
            break
    if not res.rules:
        res.reason = "grow found no improving rule"
        return res
    res.rules, res.per_rule, u, failed = gate_and_salvage(
        bins, X, res.rules, res.per_rule, pos, n_pos,
        res.base_rate, min_precision_margin)
    if failed:
        res.reason = (f"union failed the precision gate "
                      f"(best union prec {u.precision:.3f})")
        res.union = u
        return res
    res.union = u
    return res
