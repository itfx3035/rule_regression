# Changelog — `regression-rules`

All notable behavior guarantees of the released version live here.

## [0.1.0]

Initial release. Two-stage, text-readable region + rule extraction for
regression: stage 1 discovers regions, stage 2 expresses each region as a
disjunction of threshold conjunctions over the original features, and an
optional per-region leaf (constant or small OLS) completes the model into a
regressor. NumPy only; sklearn is never a dependency of the library.

### Stage 1 — region discovery

- **`stage1="cart"` (default).** Deterministic no-reuse CART on the target:
  each feature may split at most once per path, so every region is an exact
  flat interval box. A `(depth, min_leaf)` grid is chosen by internal k-fold
  CV on R²; regions are ordered largest-first. CART regions partition the
  row space, so coverage is 100% by construction.
- **`stage1="dbscan"`.** Target-agnostic: DBSCAN on standardized features
  (the target never enters the geometry). An `eps` candidate grid (kNN
  heuristic × `dbscan_eps_ratios`) is CV'd on a stage-1 proxy — cluster-mean
  leaves + centroid assignment scored by fold RMSE. Coverage here is
  *measured*, not guaranteed: rows between clusters (or demoted to noise)
  match no region, `match()` returns `-1`, and `predict()` returns the train
  mean (`fallback_value`). `noise_fraction_train` reports the train rows that
  are noise on the stage-1 grid (0 when a perfect partition is found).

### Stage 2 — rule writing

Same literal space and objective for both learners, so the modes are directly
comparable:

- **`stage2="beam"` (default).** RIPPER-style covering search over
  training-data quantile thresholds.
- **`stage2="ga"`.** Population search over the beam's literal space
  (`ga_pop_size`/`ga_n_gen`), plus a per-region Pareto front of
  (union-F1, literal count) at `Region.pareto` (beam regions carry
  `pareto=None`). Duplicate-rule dedup invariant: no exact-duplicate rule can
  end up in an output ruleset (OR is idempotent, so dedup is accuracy-neutral;
  it keeps the emitted text and the Pareto literal counts honest).

### Leaves

With `with_leaves=True` (default) each region gets a constant-mean or a small
OLS-on-features leaf, the kind chosen per region by train-only k-fold CV;
regions too small for an honest comparison (< `leaf_folds`-fold minimum,
8 rows) default to the constant mean. The leaf predicts from features only,
so region assignment and leaf numbers never touch `y` at predict time.
`with_leaves=False` gives a pure classifier-style `match()` model
(`predict()` then raises a self-explanatory error).

### Guarantees

- **Determinism.** Everything is deterministic for a fixed `seed`: the same
  fit — even in a *separate Python process* — produces the identical model
  text and identical predictions (locked by a cross-process test in the
  suite).
- **Text round-trip.** `to_text()` / `from_text()` round-trip exactly: the
  whole model is plain text and every threshold float is reproduced
  bit-for-bit (shortest-`repr` formatting). Feature names appear by name in
  the text, so `from_text()` requires `feature_names`; the header carries
  provenance (`stage1`/`stage2`/`n_regions`/`seed`/`fallback`), and
  `from_text()` revalidates the model (region count vs header, name
  contract, unparseable lines raise).
- **Name contract.** Feature names must be unique, non-empty, and contain
  neither the token ` and ` nor any `<` / `>` character — the exact strings
  that would mis-parse or re-tokenize rule text. Enforced at `fit()`, at
  `from_text()`, and checked against the name contract when parsing.
- **Constant target.** A zero-variance `y` fits a constant model without
  error (R² is undefined for it, so the stage-1 CV is skipped and said so in
  the diagnostics); `predict()` returns the constant everywhere and the
  model text round-trips.
- **Train-only discipline.** `fit()` never holds any of its rows out: all
  internal CV (stage-1 grid, leaf kind) runs on the rows `fit()` receives,
  and the final model is fitted on all of them. Score with rows `fit()` has
  not seen, exactly like sklearn.

### Public API

- `RegressionRulesExtractor(stage1, stage2, ..., seed=42).fit(X, y,
  feature_names=[...], target_name="y")` → `self` (sklearn-style chaining).
- Fitted interface: `regions` (`list[Region]`: `ruleset`, `n_train`,
  `train_coverage`, `train_f1`, `leaf`, `pareto`), `rules` (named
  `(feature, op, threshold)` literal view), `stage1_params`,
  `noise_fraction_train` (dbscan), `fallback_value`,
  `feature_names_` / `target_name` / `n_features_` / `n_samples_`,
  `match(X)` (first-match region id, -1 = no match), `predict(X)`
  (per-region leaf), `summary()` (one-screen str report),
  `cv_diagnostics` (the CV'd candidate grids + selections), `to_text()`,
  `from_text(text, feature_names=[...])`.
- `cv_diagnostics` (a plain dict, set at the end of `fit()`, never
  serialized into the model text) surfaces the internal CV:
  - `stage1` — all `(depth, min_leaf)` candidates with their k-fold `cv_r2`
    and the selected one (dbscan: all `eps` candidates with fold RMSE);
  - `leaves` — the per-region CV policy and which regions fell to the
    constant-mean default;
  - `stage2` — a note that stage 2 is **not** CV'd (deterministic seeded
    search with an acceptance gate on training rows) plus the mode.
- The text format itself is documented in `README.md` ("The text format").

### Validation / error surface

First-class `ValueError`s, each naming the offending value and the knob to
change, for: non-numeric or non-2-D `X`; row-count mismatch with `y`;
non-finite values; empty `X`; `feature_names` count / uniqueness / empty /
name-contract violations; `target_name` colliding with a feature name;
constructor invariants (`n_quantiles>=2`, `cv_folds>=2`, non-empty integral
`depths`/`min_leaves`, `max_regions>=1`, `max_rules>=1`,
`0<=min_precision_margin<1`, `ga_pop_size>=2`, `ga_n_gen>=1`,
`dbscan_min_samples>=2`, `dbscan_min_cluster_size>=2*dbscan_min_samples`,
positive `dbscan_eps_ratios`); and an infeasible CV grid (e.g. no
`(depth, min_leaf)` fits within `max_regions`).

### Tests

40-test pytest suite plus 16 doctests in the public API (`pytest` runs both;
`pyproject` sets `testpaths`/`pythonpath` and `addopts = "--doctest-modules"`).
The suite covers round-trips, cross-process determinism, every
`stage1 × stage2` combination on the shipped 300-row energy fixture and
synthetic blobs, the GA dedup invariant and Pareto front, the dbscan
measured-coverage contract and noise path, `cv_diagnostics` contents,
`summary()` content, the `rules` view, the constant-target guarantee, the
text-format round-trip including the header, and the full validation surface.
CI (`.github/workflows/ci.yml`) runs the suite on Python 3.10–3.13, the
`examples/energy_walkthrough.py` walk-through, and a `python -m build`
artifact check (wheel contains the package, `_data` fixture, and `LICENSE`).

### Bundled data

`src/regression_rules/_data/energy_subset_300.csv` — 300 rows of the UCI
Energy Efficiency (ENB2012) building-heating-load data (8 numeric features),
shipped as package data so the example runs out of the box.
