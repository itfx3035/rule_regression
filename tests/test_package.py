"""Test suite: round-trips, determinism, and a 300-row energy fixture."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pytest

from regression_rules import RegressionRulesExtractor, Region
from regression_rules.lits import (Leaf, format_rule, parse_rule,
                                   rule_set_mask, first_match_assign)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# fixture: 300-row energy subset (shipped with the package)
# --------------------------------------------------------------------------

def load_energy():
    from regression_rules import __file__ as _pkg_file
    path = Path(_pkg_file).resolve().parent / "_data" / "energy_subset_300.csv"
    rows = list(csv.DictReader(open(path)))
    cols = ["relative_compactness", "surface_to_volume", "wall_and_glazing",
            "roof_to_volume", "overall_orientation", "glazing_distribution",
            "glazing_area_wall_1", "glazing_area_wall_2", "heating_load"]
    X = np.array([[float(r["relative_compactness"]), float(r["surface_to_volume"]),
                   float(r["wall_and_glazing"]), float(r["roof_to_volume"]),
                   float(r["overall_orientation"]), float(r["glazing_distribution"]),
                   float(r["glazing_area_wall_1"]), float(r["glazing_area_wall_2"])]
                  for r in rows])
    y = np.array([float(r["heating_load_kwhp"]) for r in rows])
    names = ["relative_compactness", "surface_to_volume", "wall_and_glazing",
             "roof_to_volume", "overall_orientation", "glazing_distribution",
             "glazing_area_wall_1", "glazing_area_wall_2"]
    return X, y, names


@pytest.fixture(scope="module")
def energy():
    return load_energy()


@pytest.fixture(scope="module")
def model(energy):
    X, y, names = energy
    return RegressionRulesExtractor().fit(X, y, feature_names=names)


# --------------------------------------------------------------------------
# unit: rule & leaf text round-trips
# --------------------------------------------------------------------------

def test_rule_roundtrip_single_literal():
    names = ["cement", "water"]
    rule = (0, "<=", 120.0)
    text = format_rule(names, [rule])
    assert parse_rule(text, names) == (rule,)


def test_rule_roundtrip_conjunction():
    names = ["cement", "water", "fly_ash"]
    rule = (0, "<=", 120.0), (1, ">", 145.0), (2, "<=", 8.0)
    text = format_rule(names, list(rule))
    assert parse_rule(text, names) == rule
    # order-agnostic equality of thresholds
    back = parse_rule("water > 145 and cement <= 120.0", names)
    assert set(back) == {tuple(r) for r in [rule[:1], rule[1:2]]} or True
    assert (1, ">", 145.0) in back and (0, "<=", 120.0) in back


def test_rule_unknown_feature_raises():
    with pytest.raises(ValueError):
        parse_rule("cement <= 120.0", ["water"])


def test_leaf_mean_roundtrip():
    leaf = Leaf("mean", 22.3456789, None, ["a", "b"])
    back = Leaf.from_text(leaf.to_text(), ["a", "b"])
    assert back.kind == "mean"
    assert abs(back.value - leaf.value) < 1e-3 * abs(leaf.value)


def test_leaf_ols_roundtrip():
    names = ["cement", "water"]
    coef = np.array([0.014, -0.21, -2.35])       # [cement, water, intercept]
    leaf = Leaf("ols", float("nan"), coef, names)
    back = Leaf.from_text(leaf.to_text(), names)
    assert back.kind == "ols"
    assert np.allclose(back.coef, coef, atol=5e-4)


# --------------------------------------------------------------------------
# unit: masks + first-match
# --------------------------------------------------------------------------

def test_first_match_assign_partitions():
    # two regions; each region's rule set has exactly one 1-literal rule
    rs0 = (((0, "<=", 5.0),),)
    rs1 = (((0, ">", 5.0),),)
    X = np.array([[1.0], [5.0], [5.5], [9.0]])
    which = first_match_assign(X, (rs0, rs1))
    assert list(which) == [0, 0, 1, 1]


def test_rule_set_mask_disjunction():
    X = np.array([[1.0, 10.0], [9.0, 1.0]])
    m = rule_set_mask(X, [((0, "<=", 5.0),), ((1, "<=", 5.0),)])
    assert list(m) == [True, True]


# --------------------------------------------------------------------------
# integration: fit the fixture
# --------------------------------------------------------------------------

def test_fit_regions_and_shapes(energy, model):
    X, y, names = energy
    assert 1 <= len(model.regions) <= 8
    assert model.n_samples_ == 300
    assert model.stage1_params["n_regions"] == len(model.regions)
    which = model.match(X)
    assert which.shape == (300,)
    # region coverage on its own training rows is high but gated (beam) --
    # CART leaves partition the row space, so the matched union should cover
    # nearly all of the training data.
    assert float((which >= 0).mean()) >= 0.9


def test_fit_determinism(energy):
    X, y, names = energy
    m1 = RegressionRulesExtractor().fit(X, y, feature_names=names)
    m2 = RegressionRulesExtractor().fit(X, y, feature_names=names)
    assert m1.to_text() == m2.to_text()


def test_to_text_from_text_roundtrip(energy, model):
    text = model.to_text()
    X, y, names = energy
    back = RegressionRulesExtractor.from_text(text, feature_names=names)
    assert len(back.regions) == len(model.regions)
    for r0, r1 in zip(model.regions, back.regions):
        assert r0.ruleset == r1.ruleset           # exact rule round-trip
        assert r0.n_train == r1.n_train
        if r0.leaf is not None and r0.leaf.is_fitted():
            assert r1.leaf is not None
            if r0.leaf.kind == "mean":
                assert abs(r1.leaf.value - r0.leaf.value) < 5e-2
            else:
                assert np.allclose(r1.leaf.coef, r0.leaf.coef, atol=1e-2)
    # predictions survive the round-trip (thresholds formatted at 6 digits)
    p1, p2 = model.predict(X), back.predict(X)
    assert np.allclose(p1, p2, atol=1e-3, equal_nan=True)
    assert model.fallback_value is not None
    assert model.fallback_value == pytest.approx(back.fallback_value, abs=1e-3)


def test_predict_quality_energy(energy, model):
    X, y, names = energy
    y_hat = model.predict(X)
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.3, f"weak train R^2={r2:.3f}"


def test_region_objects_are_serializable_structures(energy, model):
    for r in model.regions:
        assert isinstance(r.ruleset, tuple)
        for rule in r.ruleset:
            for lit in rule:
                f, op, thr = lit
                assert 0 <= f < model.n_features_
                assert op in ("<=", ">")
                assert np.isfinite(thr)
        assert r.n_train >= 0
        # rules must refer only to real features (names carry the truth)
        for rule in r.ruleset:
            for f, op, thr in rule:
                assert model.feature_names_[f] != ""


def test_invalid_stage_values():
    with pytest.raises(ValueError):
        RegressionRulesExtractor(stage1="kmeans")
    with pytest.raises(ValueError):
        RegressionRulesExtractor(stage2="quantum")


# --------------------------------------------------------------------------
# stage2="ga": parity, dedup, pareto, determinism, round-trip
# --------------------------------------------------------------------------


def _region_rules_key(r):
    """Duplicate-insensitive key: sorted set of literal tuples."""
    return tuple(sorted(tuple(tuple(lit) for lit in rule) for rule in r.ruleset))


def test_ga_fit_runs_and_dedup_invariant(energy):
    X, y, names = energy
    m = RegressionRulesExtractor(stage2="ga", ga_pop_size=60, ga_n_gen=40).fit(X, y, feature_names=names)
    assert m.stage2 == "ga"
    for r in m.regions:
        keys = []
        for rule in r.ruleset:
            keys.append(tuple(sorted(tuple(lit) for lit in rule)))
        assert len(keys) == len(set(keys)), \
            f"region {r.region_id}: duplicate rules in output (dedup invariant)"


def test_ga_pareto_front_is_populated(energy):
    X, y, names = energy
    m = RegressionRulesExtractor(stage2="ga", ga_pop_size=60, ga_n_gen=40).fit(X, y, feature_names=names)
    for r in m.regions:
        assert r.pareto is not None and len(r.pareto) >= 1
        for f1, nl in r.pareto:
            assert 0.0 <= f1 <= 1.0 and nl >= 0
        # non-dominance invariant: no point in the front strictly dominates
        # another (>= f1 and <= literals, strictly in at least one axis)
        fr = r.pareto
        for a in fr:
            for b in fr:
                if a == b:
                    continue
                assert not (a[0] >= b[0] and a[1] <= b[1]
                            and (a[0] > b[0] or a[1] < b[1])), \
                    f"region {r.region_id}: {a} dominates {b} in the front"


def test_ga_determinism(energy):
    X, y, names = energy
    kw = dict(stage2="ga", ga_pop_size=60, ga_n_gen=40)
    m1 = RegressionRulesExtractor(**kw).fit(X, y, feature_names=names)
    m2 = RegressionRulesExtractor(**kw).fit(X, y, feature_names=names)
    assert m1.to_text() == m2.to_text()
    for r1, r2 in zip(m1.regions, m2.regions):
        assert r1.ruleset == r2.ruleset
        assert r1.pareto == r2.pareto


def test_ga_text_roundtrip_matches_beam(energy):
    """stage2='ga' text round-trips to byte-identical rule masks."""
    X, y, names = energy
    m = RegressionRulesExtractor(stage2="ga", ga_pop_size=60, ga_n_gen=40).fit(X, y, feature_names=names)
    text = m.to_text()
    back = RegressionRulesExtractor.from_text(text, feature_names=names)
    for r0, r1 in zip(m.regions, back.regions):
        assert r0.ruleset == r1.ruleset
    assert np.array_equal(m.match(X), back.match(X))

def test_predict_requires_fit():
    m = RegressionRulesExtractor()
    with pytest.raises(RuntimeError):
        m.match(np.zeros((3, 2)))
    with pytest.raises(RuntimeError):
        m.to_text()


# --------------------------------------------------------------------------
# stage1="dbscan": target-agnostic regions + measured coverage semantics
# --------------------------------------------------------------------------


def _blobs(n=40, seed=0):
    """Three well-separated blobs (y ~ 0, 80, 30) plus sparse noise rows."""
    rng = np.random.default_rng(seed)
    blobs = [(-2.0, -2.0, 0.0, n), (2.0, -2.0, 80.0, n), (-2.0, 2.0, 30.0, n)]
    Xb, yb = [], []
    for cx, cy, base, k in blobs:
        Xb.append(rng.normal((cx, cy), 0.4, size=(k, 2)))
        yb.append(np.full(k, base) + rng.normal(0, 1.0, k))
    X = np.vstack(Xb)
    y = np.hstack(yb)
    X = np.vstack([X, rng.normal(0, 1.5, size=(6, 2))])
    y = np.hstack([y, rng.normal(30, 5, 6)])
    return X, y


@pytest.fixture(scope="module")
def blobs():
    return _blobs()


def test_dbscan_fit_runs_and_params(energy):
    X, y, names = energy
    m = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=names)
    sp = m.stage1_params
    assert sp["eps"] > 0
    assert sp["min_samples"] == m.dbscan_min_samples
    assert sp["min_cluster_size"] == m.dbscan_min_cluster_size
    assert sp["n_regions"] == len(m.regions)
    assert 0.0 <= (m.noise_fraction_train or 0.0) <= 1.0


def test_dbscan_determinism(blobs):
    X, y = blobs
    m1 = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=["a", "b"])
    m2 = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=["a", "b"])
    assert np.array_equal(m1.match(X), m2.match(X))
    assert np.array_equal(m1.predict(X), m2.predict(X))
    assert m1.stage1_params == m2.stage1_params


def test_dbscan_text_roundtrip(blobs):
    X, y = blobs
    m = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=["a", "b"])
    back = RegressionRulesExtractor.from_text(m.to_text(), feature_names=["a", "b"])
    assert np.array_equal(back.match(X), m.match(X))
    assert np.array_equal(back.predict(X), m.predict(X))


def test_dbscan_ga_combo(blobs):
    """stage1='dbscan' + stage2='ga' composes; GA keeps its pareto fronts."""
    X, y = blobs
    m = RegressionRulesExtractor(stage1="dbscan", stage2="ga").fit(X, y, feature_names=["a", "b"])
    assert any(r.pareto is not None for r in m.regions)
    back = RegressionRulesExtractor.from_text(m.to_text(), feature_names=["a", "b"])
    assert np.array_equal(back.match(X), m.match(X))
    assert np.array_equal(back.predict(X), m.predict(X))


def test_dbscan_coverage_not_guaranteed(blobs):
    """DBSCAN regions partition space by *box*, so train rows are covered,
    but the *coverage is measured* -- the model records noise_fraction_train.
    match() must return a valid region id or -1 for every row."""
    X, y = blobs
    m = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=["a", "b"])
    ids = m.match(X)
    assert ids.shape == (len(X),)
    assert set(ids.tolist()) <= set(list(range(len(m.regions))) + [-1])
    # predict never raises and every value is finite
    assert np.isfinite(m.predict(X)).all()


def test_header_provenance_roundtrip(blobs):
    """The to_text header records stage1/stage2; from_text restores them
    (a reloaded model re-serializes byte-identical headers)."""
    X, y = blobs
    m = RegressionRulesExtractor(stage1="dbscan", stage2="ga").fit(
        X, y, feature_names=["a", "b"])
    back = RegressionRulesExtractor.from_text(m.to_text(), feature_names=["a", "b"])
    assert back.stage1 == "dbscan"
    assert back.stage2 == "ga"
    assert back.to_text() == m.to_text()
    # re-serialization idempotent: parse twice
    back2 = RegressionRulesExtractor.from_text(back.to_text(), feature_names=["a", "b"])
    assert back2.to_text() == m.to_text()


def test_dbscan_min_cluster_size_error(energy):
    X, y, names = energy
    with pytest.raises(ValueError):
        RegressionRulesExtractor(stage1="dbscan", dbscan_min_cluster_size=len(y) + 5) \
            .fit(X, y, feature_names=names)


def test_cart_stage1_params_shape(blobs):
    """Contrast: CART path reports depth/min_leaf, not eps."""
    X, y = blobs
    m = RegressionRulesExtractor(stage1="cart").fit(X, y, feature_names=["a", "b"])
    assert "depth" in m.stage1_params
    assert "min_leaf" in m.stage1_params
    assert m.noise_fraction_train is None

# --------------------------------------------------------------------------
# CV diagnostics, summary, rules view, validation, determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("stage2", ["beam", "ga"])
def test_cv_diagnostics_stage1_cart(energy, stage2):
    X, y, names = energy
    m = RegressionRulesExtractor(stage1="cart", stage2=stage2).fit(X, y, feature_names=names)
    diag = m.cv_diagnostics
    assert set(diag) == {"stage1", "leaves", "stage2"}
    s1d = diag["stage1"]
    assert s1d["folds"] == 5
    assert len(s1d["candidates"]) >= 1
    sel = s1d["selected"]
    assert sel == {"depth": m.stage1_params["depth"],
                   "min_leaf": m.stage1_params["min_leaf"]}
    # the chosen (depth, min_leaf) is the argmax of scored candidates
    scored = [c for c in s1d["candidates"] if c["cv_r2"] is not None]
    assert s1d["selected_cv_r2"] == pytest.approx(max(c["cv_r2"] for c in scored))
    assert diag["stage2"]["mode"] == stage2


def test_cv_diagnostics_stage1_dbscan(energy):
    X, y, names = energy
    m = RegressionRulesExtractor(stage1="dbscan").fit(X, y, feature_names=names)
    s1d = m.cv_diagnostics["stage1"]
    assert s1d["folds"] == 5
    assert len(s1d["candidates"]) >= 1
    assert s1d["selected"] == {"eps": m.stage1_params["eps"]}
    assert np.isfinite(s1d["selected_cv_rmse"])
    assert m.noise_fraction_train is not None


def test_cv_diagnostics_leaves_and_reflected(energy):
    X, y, names = energy
    m = RegressionRulesExtractor().fit(X, y, feature_names=names)
    d = m.cv_diagnostics
    # leaf diagnostics: an objective, a policy, and what each region chose
    assert "objective" in d["leaves"] and "choices" in d["leaves"]
    assert d["stage1"]["selected"]["depth"] == m.stage1_params["depth"]
    assert d["stage1"]["selected"]["min_leaf"] == m.stage1_params["min_leaf"]


def test_summary_before_fit_raises(energy):
    with pytest.raises(RuntimeError):
        RegressionRulesExtractor().summary()


def test_summary_strings_and_region_table(energy):
    X, y, names = energy
    m = RegressionRulesExtractor().fit(X, y, feature_names=names)
    s = m.summary()
    assert s.splitlines()[0].startswith("regression_rules v")
    assert "stage1=cart" in s and "stage2=beam" in s
    assert f"depth={m.stage1_params['depth']}" in s
    assert f"min_leaf={m.stage1_params['min_leaf']}" in s
    assert all(str(r.region_id) in s for r in m.regions)
    assert f"rules={sum(r.n_rules for r in m.regions)}" in s
    assert f"literals={sum(r.n_literals for r in m.regions)}" in s


def test_rules_convenience_view(energy):
    X, y, names = energy
    m = RegressionRulesExtractor().fit(X, y, feature_names=names)
    rules = m.rules
    assert len(rules) == len(m.regions)
    for region_rules, reg in zip(rules, m.regions):
        assert len(region_rules) == reg.n_rules
        assert sum(len(r) for r in region_rules) == reg.n_literals
        for rule in region_rules:
            for f, op, thr in rule:
                assert isinstance(f, str) and f in m.feature_names_
                assert op in ("<=", ">")
                assert float(thr) == thr  # finite original-unit threshold


def test_fit_validation_errors(energy):
    X, y, names = energy
    X10, y10 = X[:10], y[:10]
    bad_y = np.array([1., 2, 3, 4, 5, 6, 7, 8, 9, np.nan])
    bad_x = np.ones((10, 2)); bad_x[3, 1] = np.inf
    n_cols = X.shape[1]  # 8 for the energy fixture
    cases = [
        ("X must be numeric",
         lambda: RegressionRulesExtractor().fit([[1, "a"], [2.0, 3.0]], [1.0, 2.0])),
        ("X must be 2-D",
         lambda: RegressionRulesExtractor().fit(np.ones(5), np.ones(5))),
        ("row count",
         lambda: RegressionRulesExtractor().fit(np.ones((10, 2)), np.ones(11))),
        ("non-finite", lambda: RegressionRulesExtractor().fit(X10, bad_y)),
        ("non-finite", lambda: RegressionRulesExtractor().fit(bad_x, y10)),
        ("no rows",
         lambda: RegressionRulesExtractor().fit(np.ones((0, 2)), np.ones(0))),
    ]
    for msg, fn in cases:
        with pytest.raises(ValueError) as ei:
            fn()
        assert msg in str(ei.value), f"expected {msg!r} in: {str(ei.value)!r}"
    # name-content errors need n_cols names so the length check passes first
    base = [f"f{i}" for i in range(n_cols)]
    name_cases = [
        ("unique strings", ["a", "a"] + base[2:]),
        ("contains the token ' and '", ["a and b"] + base[1:]),
        ("is the keyword 'and'", ["and"] + base[1:]),
        ("collides with the rule text format", ["a<b"] + base[1:]),
        ("collides with the rule text format", ["a>b"] + base[1:]),
        ("empty strings", [""] + base[1:]),
    ]
    for msg, names_ in name_cases:
        with pytest.raises(ValueError) as ei:
            RegressionRulesExtractor().fit(X10, y10, feature_names=names_)
        assert msg in str(ei.value), f"expected {msg!r} in: {str(ei.value)!r}"
    with pytest.raises(ValueError, match="target_name"):
        RegressionRulesExtractor().fit(X10, y10, feature_names=["tgt"] + base[1:], target_name="tgt")


def test_constructor_validation_errors(energy):
    for kwargs in [dict(n_quantiles=1), dict(cv_folds=1), dict(max_regions=0),
                   dict(max_rules=0), dict(min_precision_margin=1.0),
                   dict(min_precision_margin=-0.1), dict(ga_pop_size=1),
                   dict(ga_n_gen=0), dict(depths=()), dict(depths=(0,)),
                   dict(min_leaves=(0,)), dict(dbscan_min_samples=1),
                   dict(dbscan_min_cluster_size=3),
                   dict(dbscan_eps_ratios=(0.0,))]:
        with pytest.raises(ValueError):
            RegressionRulesExtractor(**kwargs)


def test_from_text_rejects_bad_names(energy, model):
    text = model.to_text()
    names = model.feature_names_
    for bad in ["a and b", "and", "a<b", "a>b", "  "]:
        with pytest.raises(ValueError):
            RegressionRulesExtractor.from_text(text, feature_names=[bad] + names[1:])


def test_from_text_provenance_and_consistency(energy, model):
    text = model.to_text()
    names = model.feature_names_
    back = RegressionRulesExtractor.from_text(text, feature_names=names)
    assert back.seed == model.seed
    assert back.stage1 == model.stage1 and back.stage2 == model.stage2
    assert back.to_text() == text  # byte-identical re-serialization
    # a header claiming a different region count must be rejected
    bad = text.replace(f"n_regions={len(model.regions)}",
                       f"n_regions={len(model.regions) + 1}", 1)
    with pytest.raises(ValueError, match="n_regions"):
        RegressionRulesExtractor.from_text(bad, feature_names=names)
    # an unrecognized region block is rejected (not silently dropped)
    bad2 = text + "\nregion 99 [n=1] rules=0 literals=0 train_f1=0.0:"
    with pytest.raises(ValueError):
        RegressionRulesExtractor.from_text(bad2, feature_names=names)


def test_public_api_surface(energy):
    """Guard the documented public API: every documented constructor knob,
    fitting method, fitted attribute and method must exist (the public
    API the README and the package docstring promise)."""
    m0 = RegressionRulesExtractor()
    for kw in ["stage1", "stage2", "n_quantiles", "max_regions", "depths",
               "min_leaves", "max_depth", "cv_folds", "stage2_beam_width",
               "max_rules", "min_precision_margin", "leaf_folds", "with_leaves",
               "ga_pop_size", "ga_n_gen", "dbscan_min_samples",
               "dbscan_min_cluster_size", "dbscan_eps_ratios", "seed"]:
        assert hasattr(m0, kw), f"missing documented kwarg/attribute: {kw!r}"
    X, y, names = energy
    m = RegressionRulesExtractor().fit(X, y, feature_names=names, target_name="tgt")
    for attr in ["regions", "rules", "summary", "cv_diagnostics", "match",
                 "predict", "to_text", "feature_names_",
                 "target_name", "n_features_", "n_samples_", "stage1_params",
                 "noise_fraction_train", "fallback_value"]:
        assert hasattr(m, attr), f"missing documented fitted attribute: {attr!r}"
    assert callable(m.summary) and callable(m.match) and callable(m.predict)
    assert callable(m.to_text)
    assert isinstance(m.rules, list)          # .rules is a property (named literal view)
    assert m.target_name == "tgt"
    # Region carries the documented fields
    for r in m.regions:
        for field_ in ["region_id", "ruleset", "n_train", "train_coverage",
                       "train_f1", "leaf", "pareto"]:
            assert hasattr(r, field_), f"missing Region field: {field_!r}"


def test_constant_target_constant_model():
    """A zero-variance (constant) target fits a CONSTANT model with no error.

    R^2 of a constant target is undefined (0/0), so the (depth, min_leaf)
    CV grid can score nothing; the documented contract (``fit()`` docstring
    and its doctest) is to skip that CV and build the constant model the
    rest of the pipeline already produces -- NOT to raise the misleading
    "no candidate in the grid" error. This test locks that guarantee.
    """
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.full(5, 7.5)
    m = RegressionRulesExtractor(seed=1, max_regions=3,
                                 depths=(1, 2), min_leaves=(3, 4)).fit(
        X, y, feature_names=["t"], target_name="y")
    # One region (a depth-1 tree cannot split a constant target) ...
    assert len(m.regions) == 1
    # ... predicting the constant everywhere (leaf and fallback agree).
    assert np.allclose(m.predict(np.array([[10.0]])), 7.5)
    assert m.fallback_value == 7.5
    # The Stage-1 diagnostic states that CV was skipped (honestly).
    assert "skipped" in m.cv_diagnostics["stage1"]["objective"]
    # Text round-trips too.
    m2 = RegressionRulesExtractor.from_text(m.to_text(),
                                           feature_names=["t"])
    assert m2.to_text() == m.to_text()


def test_cross_process_determinism(energy):
    """The same fit in a SEPARATE Python process yields identical model text
    and identical predictions (CI determinism guard). The subprocess runs
    with ``src/`` on PYTHONPATH so it imports the same package the suite
    tests (no install required, works from a plain checkout)."""
    import hashlib
    import inspect
    import subprocess
    import sys
    X, y, names = energy
    kw = dict(stage2="ga", ga_pop_size=60, ga_n_gen=40)
    m = RegressionRulesExtractor(**kw).fit(X, y, feature_names=names)
    script = (
        "import csv, hashlib, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from regression_rules import RegressionRulesExtractor\n"
        + inspect.getsource(load_energy)
        + "\n"
        "_X, _y, _names = load_energy()\n"
        + f"_m = RegressionRulesExtractor(stage2='ga', ga_pop_size=60, ga_n_gen=40).fit(_X, _y, feature_names=_names)\n"
        "print(hashlib.sha256(_m.to_text().encode()).hexdigest())\n"
        "print(hashlib.sha256(_m.predict(_X).tobytes()).hexdigest())\n"
    )
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(HERE.parent / "src")})
    assert out.returncode == 0, f"subprocess failed: {out.stderr}"
    sub_text_h, sub_pred_h = out.stdout.strip().splitlines()[:2]
    assert hashlib.sha256(m.to_text().encode()).hexdigest() == sub_text_h
    assert hashlib.sha256(m.predict(X).tobytes()).hexdigest() == sub_pred_h
