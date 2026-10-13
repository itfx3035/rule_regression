"""Shared rule vocabulary: literals, rule sets, Regions, masks and text.

A **literal** is ``(feature_index, op, threshold)`` with ``op`` in ``"<="`` /
``">"`` and ``threshold`` a float on the feature's *original* scale. A **rule**
(conjunction) is a tuple of literals; a **rule set** (disjunction) is a tuple
of rules. A **Region** is the library's extraction product: a rule set plus
its train statistics and an optional leaf (mean or OLS-on-features).

Text format (what ``RegressionRulesExtractor.to_text()`` emits and
``from_text`` round-trips):

    regression_rules v0.1.0 stage1=cart stage2=beam n_regions=8 fallback=21.4
    region 0 [n=95] rules=1 literals=2 train_f1=0.963:
      rule 0: cement <= 120.0 and water > 145.0
      leaf: 22.35
    region 1 [n=205] rules=2 literals=3 train_f1=0.877:
      rule 0: cement > 120.0
      rule 1: fly_ash <= 8.0
      leaf: -2.35 + 0.014*cement + -0.21*water
    fallback: 21.4

One line per rule; a region's *disjunction* is structural (all its `rule k`
lines). The disjunction operator is implicit, so feature names must not
contain the tokens `` and `` (spaces included) or the ops ``<=`` / ``>`` --
enforced at ``fit``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

Literal = tuple[int, str, float]      # (feature index, "<="|">", threshold)
Rule = tuple[Literal, ...]            # conjunction
Ruleset = tuple[Rule, ...]            # disjunction

VERSION = "0.1.0"
HEADER = "regression_rules"


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

def literal_mask(X: np.ndarray, f: int, op: str, thr: float) -> np.ndarray:
    """Boolean mask: rows where ``X[:, f] op thr``.

    >>> import numpy as np
    >>> X = np.array([[1., 5.], [2., 6.], [3., 7.]])
    >>> literal_mask(X, 0, "<=", 2.0)
    array([ True,  True, False])
    >>> literal_mask(X, 1, ">", 5.0)
    array([False,  True,  True])
    """
    v = X[:, f]
    if op == "<=":
        return v <= thr
    if op == ">":
        return v > thr
    raise ValueError(f"unknown op {op!r}; use '<=' or '>'")


def rule_mask(X: np.ndarray, rule: Sequence[Literal]) -> np.ndarray:
    """Boolean mask of rows matched by a conjunction.

    >>> X = np.array([[1., 5.], [2., 6.], [3., 7.]])
    >>> rule_mask(X, [(0, "<=", 2.0), (1, ">", 5.0)])
    array([False,  True, False])
    """
    m = np.ones(X.shape[0], dtype=bool)
    for f, op, thr in rule:
        m &= literal_mask(X, f, op, thr)
    return m


def rule_set_mask(X: np.ndarray, rules: Sequence[Sequence[Literal]]) -> np.ndarray:
    """Boolean mask of rows matched by any rule of a disjunction.

    >>> X = np.array([[1., 5.], [2., 6.], [3., 7.]])
    >>> rule_set_mask(X, [[(0, "<=", 1.0)], [(1, ">", 6.0)]])
    array([ True, False,  True])
    """
    m = np.zeros(X.shape[0], dtype=bool)
    for r in rules:
        if len(r):
            m |= rule_mask(X, r)
    return m


def first_match_assign(X: np.ndarray, rulesets: Sequence[Ruleset]) -> np.ndarray:
    """Assign each row to the first region (in order) whose disjunction
    matches; -1 if none matches. Regions should be ordered largest-first.

    ``rulesets[i]`` is region ``i``'s disjunction (a list of rules, each a
    list of ``(feature, op, threshold)`` literals).

    >>> X = np.array([[1., 5.], [2., 6.], [3., 7.]])
    >>> first_match_assign(X, [[[(0, "<=", 2.0)]], [[(1, ">", 6.0)]]])
    array([0, 0, 1])
    >>> first_match_assign(X, [[[(0, ">", 100.0)]]])
    array([-1, -1, -1])
    """
    n = X.shape[0]
    which = np.full(n, -1, dtype=int)
    assigned = np.zeros(n, dtype=bool)
    for c, rs in enumerate(rulesets):
        if not rs:
            continue
        m = rule_set_mask(X, rs) & ~assigned
        if m.any():
            which[m] = c
            assigned |= m
    return which


# --------------------------------------------------------------------------
# text: format
# --------------------------------------------------------------------------

def fmt_num(v: float) -> str:
    """Render a float as its shortest exact round-trip string (``repr``).

    >>> fmt_num(21.4)
    '21.4'
    """
    # repr is Python's shortest representation that round-trips exactly to the
    # same float -- the text format's contract is that from_text(to_text(x))
    # reproduces x, so thresholds must survive a float(...) parse unchanged.
    # (a 6-sig-fig formatter silently truncates dense quantile edges, e.g.
    #  2.333333333333343 -> '2.33333' -> a different threshold.)
    return repr(float(v))


def format_rule(names: Sequence[str], rule: Sequence[Literal]) -> str:
    """Render a conjunction as ``f1 op t1 and f2 op t2`` (original units).

    >>> format_rule(["cement", "water"], [(0, "<=", 120.0), (1, ">", 145.0)])
    'cement <= 120.0 and water > 145.0'
    """
    if not rule:
        return "(all)"
    return " and ".join(f"{names[int(f)]} {op} {fmt_num(thr)}"
                        for f, op, thr in rule)


# --------------------------------------------------------------------------
# text: parse (inverse of formatting)
# --------------------------------------------------------------------------

def _condition_literal(cond: str, known_names: Sequence[str]) -> Literal:
    toks = cond.split()
    # operator is the last token that is an op (a name can contain 'and')
    for i in range(len(toks) - 2, 0, -1):
        if toks[i] in ("<=", ">"):
            name = " ".join(toks[:i]).strip()
            thr = float(toks[-1])
            if name in known_names:
                return (list(known_names).index(name), toks[i], thr)
    raise ValueError(f"unparseable condition {cond!r}")


def parse_rule(text: str, known_names: Sequence[str]) -> Rule:
    """Parse one rule line's body (a conjunction) into a Rule.

    >>> parse_rule("cement <= 120.0 and water > 145.0", ["cement", "water"])
    ((0, '<=', 120.0), (1, '>', 145.0))
    >>> parse_rule("(all)", ["a"])
    ()
    """
    text = text.strip()
    if text in ("", "(all)"):
        return ()
    if text.startswith("if "):
        text = text[3:].strip()
    return tuple(_condition_literal(c, known_names) for c in text.split(" and "))


# --------------------------------------------------------------------------
# Leaf: an optional per-region numeric
# --------------------------------------------------------------------------

@dataclass
class Leaf:
    kind: str = "none"                     # "none" | "mean" | "ols"
    value: float = float("nan")            # mean leaf
    coef: np.ndarray | None = None         # ols: (d+1,), intercept last
    names: list[str] = field(default_factory=list)

    def is_fitted(self) -> bool:
        return self.kind in ("mean", "ols")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict each row of ``X``. mean: constant; ols: ``X @ coef[:-1] + coef[-1]``.

        >>> Leaf("mean", 2.5).predict(np.zeros((2, 3)))
        array([2.5, 2.5])
        >>> Leaf("ols", float("nan"), np.array([0.5, -1.0, 3.0]), ["a", "b"]).predict(np.array([[0., 1.], [2., 2.]]))
        array([2., 2.])
        """
        if self.kind == "mean":
            return np.full(len(X), self.value)
        if self.kind == "ols":
            return X @ self.coef[:-1] + self.coef[-1]
        raise ValueError("leaf not fitted")

    def to_text(self) -> str:
        """Render the leaf as model text (numbers via ``repr`` so they round-trip).

        >>> Leaf("mean", 22.35).to_text()
        '22.35'
        >>> Leaf("ols", float("nan"), np.array([0.5, -1.0, 3.0]), ["a", "b"]).to_text()
        '3.0 + 0.5*a + -1.0*b'
        """
        if self.kind == "mean":
            return fmt_num(self.value)
        if self.kind == "ols":
            terms = " + ".join(f"{fmt_num(self.coef[j])}*{self.names[j]}"
                               for j in range(len(self.names)))
            return f"{fmt_num(self.coef[-1])} + {terms}"
        return ""

    @classmethod
    def from_text(cls, text: str, names: Sequence[str]) -> "Leaf":
        """Parse a leaf block back into a ``Leaf`` (inverse of :meth:`to_text`).

        >>> leaf = Leaf.from_text("3.0 + 0.5*a + -1.0*b", ["a", "b"])
        >>> leaf.coef.tolist()
        [0.5, -1.0, 3.0]
        >>> Leaf.from_text("22.35", []).value
        22.35
        """
        text = text.strip()
        try:
            return cls("mean", float(text), None, list(names))
        except ValueError:
            pass
        parts = [p.strip() for p in text.split(" + ")]
        intercept = float(parts[0])
        coef = np.zeros(len(names) + 1)
        coef[-1] = intercept
        for p in parts[1:]:
            c, name = p.split("*", 1)
            name = name.strip()
            if name not in names:
                raise ValueError(f"unknown feature {name!r} in leaf {text!r}")
            coef[list(names).index(name)] = float(c)
        return cls("ols", float("nan"), coef, list(names))


# --------------------------------------------------------------------------
# Region: the extraction product
# --------------------------------------------------------------------------

@dataclass
class Region:
    region_id: int
    ruleset: Ruleset = ()
    n_train: int = 0
    train_coverage: float = 0.0
    train_f1: float = 0.0
    leaf: Leaf | None = None
    pareto: list | None = None  # [(train_f1_union, n_literals)] -- stage2='ga'

    @property
    def n_rules(self) -> int:
        return len(self.ruleset)

    @property
    def n_literals(self) -> int:
        return sum(len(r) for r in self.ruleset)

    def to_text(self, names: Sequence[str]) -> str:
        """Render the region as model text (one line per rule + the leaf).

        >>> from regression_rules.lits import Leaf
        >>> r = Region(
        ...     region_id=0, ruleset=[[(0, "<=", 120.0)]], n_train=95,
        ...     train_f1=0.963, leaf=Leaf("mean", 22.35, None, [])
        ... )
        >>> print(r.to_text(["cement"]))
        region 0 [n=95] rules=1 literals=1 train_f1=0.963:
          rule 0: cement <= 120.0
          leaf: 22.35
        """
        L = [f"region {self.region_id} [n={self.n_train}] "
             f"rules={self.n_rules} literals={self.n_literals} "
             f"train_f1={self.train_f1:.3f}:"]
        for i, r in enumerate(self.ruleset):
            L.append(f"  rule {i}: {format_rule(names, r)}")
        if self.leaf is not None and self.leaf.is_fitted():
            L.append(f"  leaf: {self.leaf.to_text()}")
        return "\n".join(L)

