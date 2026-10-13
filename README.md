# regression_rules

Two-stage, **text-readable** region + rule extraction for regression data. The
two stages are independent knobs:

- **Stage 1 — region discovery** (`stage1`, CV'd internally):
  - `"cart"` (default): deterministic no-reuse CART on the target `(x, y)`.
    Regions are flat interval boxes; they partition the row space, so coverage
    is 100% by construction and every region is *exactly* a box.
  - `"dbscan"`: **target-agnostic** — DBSCAN on standardized features (the
    target never enters the geometry). An `eps` grid (kNN heuristic × ratios)
    is CV'd on a stage-1 proxy (cluster-mean leaves + centroid assignment,
    fold RMSE). Coverage is a **measured, not guaranteed** property: rows
    between clusters match no rule, `match()` returns `-1`, and `predict()`
    returns the train mean (`fallback_value`). The train noise fraction is
    reported as `noise_fraction_train`.
- **Stage 2 — rule writing** (`stage2`, same literal space and objective for
  both learners, so the modes are directly comparable):
  - `"beam"` (default): RIPPER-style covering search.
  - `"ga"`: population search over the beam's literal space, plus a per-region
    Pareto front of (union-F1, literal count) at `Region.pareto`.

Optional per-region leaves (constant or small OLS) complete the model into a
regressor; the leaf kind is chosen per region by train-only CV.

## What's in 0.1.0

- All four `stage1 × stage2` combinations (cart/dbscan × beam/ga).
- Deterministic for a fixed `seed` — verified cross-process (a second Python
  process reproduces the same model text and the same predictions).
- `to_text()` / `from_text()` round-trip: the whole model is plain text;
  every threshold is reproduced bit-for-bit (`repr`-based formatting).
- `cv_diagnostics` (the full CV'd candidate grids) and `summary()` report.
- Strict input/name validation with actionable `ValueError`s.
- A bundled 300-row energy fixture + `examples/energy_walkthrough.py`.
- 40-test suite + 16 doctests, green in CI (Python 3.10–3.13) with a
  wheel-content check.
- MIT licensed, single runtime dependency (`numpy`).

## Installation

```bash
pip install .                 # from this repo, into your venv
pip install .[test]           # + pytest for the test suite
```

Or, once published: `pip install regression-rules` from PyPI.

## 60-second example

The shipped 300-row energy fixture (building heating load, 8 features) is
the only thing you need to try it:

```python
import csv, numpy as np
from pathlib import Path
from regression_rules import RegressionRulesExtractor, __file__ as _pkg_file

path = Path(_pkg_file).parent / "_data" / "energy_subset_300.csv"
rows = list(csv.DictReader(open(path)))
feats = ["relative_compactness", "surface_to_volume", "wall_and_glazing",
         "roof_to_volume", "overall_orientation", "glazing_distribution",
         "glazing_area_wall_1", "glazing_area_wall_2"]
X = np.array([[float(r[c]) for c in feats] for r in rows])
y = np.array([float(r["heating_load_kwhp"]) for r in rows])

m = RegressionRulesExtractor().fit(X, y, feature_names=feats)   # cart + beam
print(m.summary())          # one-screen: what was extracted + what CV chose
print(m.to_text())          # the whole model, in plain text
print(m.predict(X)[:5])     # per-region leaf predictions

# save / load the model as plain text:
txt = m.to_text()
Path("model.txt").write_text(txt)
m2 = RegressionRulesExtractor.from_text(txt, feature_names=feats)
assert m2.to_text() == txt
```

The model is *fully* the text produced by `to_text()` — rules plus leaf
numbers — and `from_text()` rebuilds it exactly (numbers go through `repr`, so
every threshold round-trips bit-for-bit).

## Public API

`RegressionRulesExtractor` is a single class:

```python
m = RegressionRulesExtractor(
        *, stage1="cart"|"dbscan",        # default "cart"
           stage2="beam"|"ga",            # default "beam"
           n_quantiles=40, max_regions=8,
           depths=(2, 3, 4), min_leaves=(15, 30),
           max_depth=None, cv_folds=5,
           stage2_beam_width=20, max_rules=3,
           min_precision_margin=0.05,
           leaf_folds=5, with_leaves=True,
           ga_pop_size=80, ga_n_gen=60,
           dbscan_min_samples=5, dbscan_min_cluster_size=15,
           dbscan_eps_ratios=(1.0, 1.5, 2.0),
           seed=42)

m.fit(X, y, feature_names=[...], target_name="y")   # X: (n, d), y: (n,)
m.regions                    # list[Region]: ruleset, n_train, leaf, pareto (ga)
m.rules                      # named (name, op, threshold) literals, per region
m.stage1_params              # what stage 1 CV'd and chose (depth/min_leaf | eps)
m.noise_fraction_train       # dbscan only; None for cart
m.fallback_value             # train y-mean, used for no-match / gaps
m.match(X_new)               # int array: first-match region id per row (-1 none)
m.predict(X_new)             # leaf prediction (needs with_leaves=True)
m.summary()                  # one-screen str report
m.cv_diagnostics             # see below
m.to_text()                  # the model, as text
RegressionRulesExtractor.from_text(text, feature_names=[...])
```

All hyperparameters (`depths`/`min_leaves`/`min_leaf`, `dbscan` `eps`, per-region
leaf kind) are selected by **internal k-fold CV on the rows `fit()` receives** —
the train-only discipline: `fit` never holds any of its data out, and every test
prediction must use rows `fit()` has not seen, exactly like sklearn.

### `cv_diagnostics`

A plain dict, set at the end of `fit()`, that surfaces every CV decision:

```python
{
  "stage1": {            # which candidate won, and the full grid
     "objective": "cv_r2 (mean of per-fold held-out R^2 on this data)",
     "folds": 5,
     "candidates": [{"depth": 2, "min_leaf": 15, "cv_r2": 0.846}, ...],
     "selected": {"depth": 3, "min_leaf": 30},
     "selected_cv_r2": 0.856,
     "tie_break": "smaller (depth, min_leaf) found first"},
  "leaves": {            # what each region's leaf kind was
     "objective": "per-region 5-fold CV: constant mean vs OLS on that region's rows",
     "policy": "CV only when a region has >= 8 training rows; ...",
     "choices": {"ols": 4, "mean": 1}},
  "stage2": {"note": "no CV: deterministic seeded search ...",
             "mode": "beam"},
}
```

(For `stage1="dbscan"` the `stage1` block has `candidates = [{eps, cv_rmse}, ...]`
and `selected = {"eps": ...}` instead.)

### The text format

```
regression_rules v0.1.0 stage1=cart stage2=beam n_regions=8
    seed=42 fallback=21.399999999999999
region 0 [n=95] rules=1 literals=2 train_f1=0.963:
  rule 0: cement <= 120.0 and water > 145.0
  leaf: 22.35
region 1 [n=205] rules=2 literals=3 train_f1=0.877:
  rule 0: cement > 120.0
  rule 1: fly_ash <= 8.0
  leaf: -2.35 + 0.014*cement + -0.21*water
fallback: 21.399999999999999
```

- One line per rule; a region's *disjunction* is its set of `rule k` lines.
- Numbers use `repr`, the shortest string that round-trips every threshold
  float exactly.
- `from_text()` revalidates the model (region count vs header, name contract,
  header provenance: `stage1`/`stage2`/`seed` come from the header).
- **Name contract** (enforced at `fit`, at `from_text`, and in the text): each
  `feature_name` must be unique, non-empty, and contain neither the token
  ` and ` nor any `<`/`>` character — those are the exact strings that would
  mis-parse the rules.

## Layout

```
src/regression_rules/
    __init__.py            # public API
    extractor.py           # RegressionRulesExtractor (fit / match / predict / text)
    lits.py                # Region, Leaf, masks, rule text formats and parsers
    stage1_cart.py         # deterministic no-reuse CART (regions)
    stage1_dbscan.py       # standardization, DBSCAN, eps heuristic (target-agnostic)
    stage2_beam.py         # RIPPER-style beam (rule sets per region)
    stage2_ga.py           # GA over the beam's literal space (dedup + Pareto front)
    leaves.py              # CV leaf choice (mean / small OLS)
    grid.py                # shared quantile grid
    _data/                 # shipped 300-row energy fixture
examples/
    energy_walkthrough.py  # runnable end-to-end demo on the energy fixture
tests/
    test_package.py        # 40 tests + doctests in src/
.github/workflows/ci.yml   # CI: pytest (py3.10-3.13) + example + build+wheel check
pyproject.toml             # packaging (name: regression-rules, v0.1.0, MIT)
LICENSE                    # MIT
.gitignore                 # Python / pytest / build artifacts
```

## Tests

```
pytest            # from the repo root — runs the 40 tests and the 16 doctests
```

Coverage: every `stage1 × stage2` combination fit on the 300-row energy
fixture and synthetic blobs; the `ga` learner's Pareto front and its
duplicate-rule dedup invariant; the `dbscan` path for determinism, text
round-trip, composition with `stage2="ga"`, the measured-coverage contract,
and the min-cluster-size error path; `to_text()` / `from_text()` round-trips
reproducing rulesets, region ids, predictions and the `stage1`/
`stage2`/`seed` header bytes; **the same fit in a separate Python process
producing the identical model text** (cross-process determinism guard); and
every constructor / input / name violation raising a `ValueError` that names
the offending value and the knob to change.
