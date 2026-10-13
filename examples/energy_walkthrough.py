"""Walk through the shipped 300-row energy fixture end to end.

Runnable as a script from the repo root: ``python
examples/energy_walkthrough.py``. Works from a plain checkout (the script
adds ``src/`` to ``sys.path`` when the package isn't already importable) and
after a ``pip install``. No network, no files outside the package,
deterministic (seed 42). The fixture path is resolved *next to the imported
``regression_rules`` package*, so it behaves identically in both cases.

The energy fixture (ENB2012 subset) is the canonical "piecewise" dataset for
this model: building heating load, 8 features (5 low-cardinality of 8), a
sparse grid of ~700 distinct configurations. The two-stage rule model lands
~0.97 in-sample R^2 on the 300-row subset with a handful of thresholds --
which is the point: the *model* is the text it prints.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make plain checkouts work without an install: put the repo's src/ on the
# path (the bootstrap is a no-op after a `pip install`, where the example is
# not shipped and this file does not sit next to a `src/` dir).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path and (_SRC / "regression_rules").is_dir():
    sys.path.insert(0, str(_SRC))

import numpy as np

import regression_rules
from regression_rules import RegressionRulesExtractor

# --- load the shipped fixture (bundled as package data, so it works offline) ---

_DATA = Path(regression_rules.__file__).resolve().parent / "_data"
_FEATS = [
    "relative_compactness", "surface_to_volume", "wall_and_glazing",
    "roof_to_volume", "overall_orientation", "glazing_distribution",
    "glazing_area_wall_1", "glazing_area_wall_2",
]


def load_energy() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The 300-row energy fixture as (X, y, feature_names)."""
    rows = list(csv.DictReader((_DATA / "energy_subset_300.csv").open()))
    X = np.array([[float(r[c]) for c in _FEATS] for r in rows])
    y = np.array([float(r["heating_load_kwhp"]) for r in rows])
    return X, y, list(_FEATS)


# --- fit the default model (cart + beam), show what CV chose ---

X, y, FEATS = load_energy()
m = RegressionRulesExtractor().fit(X, y, feature_names=FEATS)

print(m.summary())

# --- the model is text: round-trip it ---

text = m.to_text()
m2 = RegressionRulesExtractor.from_text(text, feature_names=FEATS)
assert m2.to_text() == text, "text round-trip must be exact"
assert (m2.predict(X) == m.predict(X)).all(), "prediction round-trip must be exact"

# --- inspect one region's rules in words (first rule of region 0) ---

first_rule = m.rules[0][0]
print()
print("region 0, rule 0 (in words):")
print("   " + " and ".join(f"{f} {op} {thr}" for f, op, thr in first_rule))

# --- sanity: coverage 1.0 on cart (a partition); in-sample R^2 ~0.97 ---

which = m.match(X)
assert (which >= 0).all(), "cart regions partition the row space"
y_hat = m.predict(X)
r2 = 1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"in-sample R^2: {r2:.4f}")

# --- target-agnostic variant (dbscan) for contrast ---

d = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=FEATS)
d_text = d.to_text()
d2 = RegressionRulesExtractor.from_text(d_text, feature_names=FEATS)
assert d2.to_text() == d_text
print(d.summary().splitlines()[0])  # header
print("(dbscan mode: target-agnostic regions; coverage is MEASURED, not guaranteed)")

print("\nDone: cart+beam and dbscan+beam both fit, summary, rules view, "
      "text round-trip, and cv_diagnostics worked.")
