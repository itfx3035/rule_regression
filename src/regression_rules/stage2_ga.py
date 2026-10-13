"""Stage 2, alternative learner: a genetic algorithm over the SAME
literal space and objective as the beam in ``stage2_beam.py``.

Fairness contract (identical to the beam):
* same per-region literal space -- top-``k`` defining features
  (:meth:`stage2_beam.defining_features`) x the shared ``FeatureBins``
  quantile grid;
* same objective -- the union-of-rules F1 on train (``_union_f1``) over the
  region's one-vs-rest membership;
* same acceptance contract -- the shared :func:`stage2_beam.gate_and_salvage`
  union precision gate + least-contribution salvage, and the same
  degenerate-case outcomes.

What the GA adds over the beam:
* searches a *population of rules* in parallel (elitism + tournament +
  per-slot crossover/mutation); the rule set is the union of the current
  elite individuals -- the same "set of covering rules" idea as the beam's
  current covering, but discovered by population search instead of
  sequential grow/prune;
* reports a **Pareto front** of (union-F1, total literals) so the
  accuracy/simplicity trade becomes visible -- something a greedy covering
  beam can't do on its own. The returned rules are the max-F1 Pareto point.

Encoding / operators:
* a literal slot is ``None`` or ``(feature, op_idx, pos)`` -- op_idx 0 is
  "<=", 1 is ">";
* an individual is a fixed list of ``max_len`` slots = one rule
  (conjunction of its non-None slots); an all-None individual is the empty
  rule, which matches the universe -- per the beam's "empty rule scores 0"
  convention it gets solo fitness 0 and never wins;
* fitness: solo F1 of the individual's mask against the region positives;
* the RULE SET reported is the union of the top-``max_rules`` current
  population individuals (elite union);
* duplicate individuals (identical literal sets) are dropped from the elite
  union before any metric is computed -- OR is idempotent, so removing a
  rule's twin leaves the union mask/F1 unchanged while making the returned
  text and the pareto (F1, lits) counts honest (a duplicated rule was
  being billed twice before this fix).

Deterministic given ``rng`` (all draws from np.random.Generator).
"""
from __future__ import annotations

import numpy as np

from .grid import FeatureBins
from .stage2_beam import (_literal_masks, _union_f1, defining_features,
                          gate_and_salvage, score_mask)
from .stage2_beam import FitResult


def _slots_mask(slots, lit: dict, n: int) -> np.ndarray:
    """Conjunction mask of a slot tuple. All-None -> universe."""
    m = None
    for s in slots:
        if s is None:
            continue
        f, o, p = s
        m = lit[f][p, o] if m is None else m & lit[f][p, o]
    return np.ones(n, dtype=bool) if m is None else m


def _random_lit(feats: list[int], bins: FeatureBins, rng):
    f = feats[int(rng.integers(len(feats)))]
    o = int(rng.integers(2))
    p = int(rng.integers(bins.n_positions(f)))
    return (f, o, p)


def _random_slots(feats, bins, rng, max_len: int):
    out = []
    for _ in range(max_len):
        out.append(None if rng.random() < 0.3 else _random_lit(feats, bins, rng))
    return tuple(out)


def _tournament(pop: list, rng, k: int = 3) -> int:
    i = int(rng.integers(len(pop)))
    for _ in range(k - 1):
        j = int(rng.integers(len(pop)))
        if pop[j].fit > pop[i].fit:
            i = j
    return i


# --------------------------------------------------------------------------
# the GA learner
# --------------------------------------------------------------------------

def fit_region_rules_ga(bins: FeatureBins, X: np.ndarray, pos: np.ndarray, *,
                        k: int = 3, max_rules: int = 3, max_len: int = 4,
                        pop_size: int = 80, n_gen: int = 60,
                        elite_frac: float = 0.2,
                        crossover_prob: float = 0.7, mutation_prob: float = 0.2,
                        reset_prob: float = 0.05,
                        min_precision_margin: float = 0.05,
                        min_points: int = 5,
                        rng: np.random.Generator | None = None) -> FitResult:
    """GA over whole rule sets for one region. Signature mirrors
    ``fit_region_rules`` (k/max_rules/max_len/min_* shared; pop_size/n_gen/
    elite_frac/crossover_prob/mutation_prob/reset_prob are GA-specific).
    Returns a ``FitResult`` with the ``pareto`` front populated.
    """
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n, d = X.shape                      # noqa: F841  (d part of signature parity)
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
    base_rate = n_pos / n
    lit = _literal_masks(bins, X, feats)   # masks[f][pos, 0]=<= , 1=>

    class Ind:
        __slots__ = ("slots", "mask", "fit", "nlit")

        def __init__(self, slots_):
            self.slots = slots_
            self.nlit = sum(1 for sl in slots_ if sl is not None)
            self.mask = _slots_mask(slots_, lit, n)
            self.fit = 0.0 if self.nlit == 0 else float(
                _union_f1(self.mask, pos, n_pos))

    def set_stats(set_members):
        union = np.zeros(n, dtype=bool)
        for i in set_members:
            union |= i.mask
        return float(_union_f1(union, pos, n_pos)), \
               sum(i.nlit for i in set_members)

    # --- init: random population + one single-literal seed per top feature ---
    pop = []
    for i in range(pop_size):
        if i < len(feats):
            f, o, p = _random_lit([feats[i]], bins, rng)
            slots = [(f, o, p)] + [None] * (max_len - 1)
        else:
            slots = _random_slots(feats, bins, rng, max_len)
        pop.append(Ind(tuple(slots)))
    pop.sort(key=lambda i: -i.fit)

    pareto: list[tuple] = []

    def pareto_add(point):
        f1, nl = point
        if any(q[0] == f1 and q[1] == nl for q in pareto):
            return                                # identical point: skip
        if any(q[0] >= f1 and q[1] <= nl
               and (q[0] > f1 or q[1] < nl) for q in pareto):
            return                                # dominated by an existing point
        kept = [q for q in pareto
                if not (f1 >= q[0] and nl <= q[1]
                        and (f1 > q[0] or nl < q[1]))]
        kept.append((f1, nl))
        pareto[:] = kept

    def top_set(p: list[Ind]) -> list[Ind]:
        return sorted(p, key=lambda i: (-i.fit, i.nlit))[:max_rules]

    best = (0.0, None, 0)                        # (setfit, members, nlit)

    def _uniq(lst: list[Ind]) -> list[Ind]:
        """Drop duplicate slot-sets. OR is idempotent, so the union mask/F1
        are unchanged by removing a rule's twin; only the literal & rule count
        drop -- which makes the returned text AND the pareto (F1, lits)
        counts honest (a duplicated rule was being billed twice)."""
        seen: set = set()
        out = []
        for m in lst:
            lits = tuple(sorted(tuple(sl) for sl in m.slots if sl is not None))
            if lits in seen:
                continue
            seen.add(lits)
            out.append(m)
        return out

    ts = _uniq(top_set(pop))
    sf, sl = set_stats(ts)
    pareto_add((sf, sl))
    if sf > best[0]:
        best = (sf, list(ts), sl)

    n_elite = max(3, int(round(elite_frac * pop_size)))
    for _g in range(n_gen):
        # breed
        children = []
        elites = pop[:n_elite]
        while len(elites) + len(children) < pop_size:
            if rng.random() < reset_prob:
                children.append(Ind(_random_slots(feats, bins, rng, max_len)))
                continue
            pa, pb = pop[_tournament(pop, rng)], pop[_tournament(pop, rng)]
            slots = []
            for j in range(max_len):
                if rng.random() < crossover_prob:
                    s = (pa.slots[j] if rng.random() < 0.5 else pb.slots[j])
                else:
                    s = pa.slots[j]
                if s is not None and rng.random() < mutation_prob:
                    if rng.random() < 1 / 3:
                        s = None
                    else:
                        s = _random_lit(feats, bins, rng)
                slots.append(s)
            children.append(Ind(tuple(slots)))
        pop = elites + children
        pop.sort(key=lambda i: -i.fit)

        ts = _uniq(top_set(pop))
        sf, sl = set_stats(ts)
        pareto_add((sf, sl))
        if sf > best[0]:
            best = (sf, list(ts), sl)

    if best[1] is None or best[0] <= 0.0:
        res.reason = "GA found no improving rule set"
        res.pareto = [(round(a, 3), int(b)) for a, b in pareto]
        return res

    members = [m for m in best[1] if m.nlit > 0]   # drop any empty (universe)
    if not members:
        res.reason = "GA best set degenerated to empty"
        res.pareto = [(round(a, 3), int(b)) for a, b in pareto]
        return res

    per_rule = []
    for m in members:
        s = score_mask(m.mask, pos)
        s.w_cov = int(np.count_nonzero(m.mask & pos))   # dynamic (parity with research Score)
        per_rule.append(s)
    # order rules by solo coverage desc (biggest first, like the beam)
    order = sorted(range(len(members)),
                   key=lambda i: (-per_rule[i].coverage, i))
    members = [members[i] for i in order]
    per_rule = [per_rule[i] for i in order]
    for s in per_rule:
        s.pf1 = float(best[0])

    rules = [tuple((f, "<=" if o == 0 else ">", p)
                   for s in m.slots if s is not None for (f, o, p) in [s])
             for m in members]
    rules, per_rule, u, failed = gate_and_salvage(
        bins, X, rules, per_rule, pos, n_pos, base_rate, min_precision_margin)
    res.pareto = [(round(a, 3), int(b)) for a, b in pareto]
    if failed:
        res.reason = (f"GA union failed the precision gate "
                      f"(best union prec {u.precision:.3f} < "
                      f"base rate {base_rate:.3f} + margin "
                      f"{min_precision_margin:.2f}) [best set F1 {best[0]:.3f}]")
        return res
    res.rules = rules
    res.per_rule = per_rule
    res.union = u
    return res
