"""RegressionRulesExtractor: the public API of regression_rules.

Two stages, two knobs each. Stage 1 finds regions; stage 2 expresses each
region as a human-readable disjunction of threshold rules.

Stage 1 (region discovery, CV-chosen within the mode):
  * ``stage1="cart"``   (default) deterministic no-reuse CART on (x, y); a
    ``(depth, min_leaf)`` grid is CV'd on R^2. Regions are flat interval
    boxes; they partition the row space, so coverage is 100% by
    construction.
  * ``stage1="dbscan"`` target-agnostic: DBSCAN in a standardized
    feature space (the y never enters the geometry). An ``eps`` candidate
    grid (kNN-heuristic eps x ratios) is CV'd on a stage-1 proxy: mean leaf
    + centroid assignment, scored by fold RMSE (the stage-1 quality the
    stage-2 learner cannot fully repair). Coverage is a *measured* property
    here -- rows that fall between clusters (or were demoted to noise)
    match no region: ``match() = -1`` and ``predict()`` returns the train
    mean (``fallback_value``).

Stage 2 (rule writing, same literal space and objective for both):
  * ``stage2="beam"`` (default) RIPPER-style covering search;
  * ``stage2="ga"``    population search over the beam's literal space, plus
    a per-region Pareto front (``Region.pareto``).

    X, y = ...                      # training rows (fit never holds out)
    m = RegressionRulesExtractor(stage1="cart", stage2="beam")
        .fit(X, y, feature_names=[...])
    m.regions                     # list[Region] -- the extraction product
    m.rules                       # convenience: named (name, op, thr) tuples
    m.summary()                   # one-screen human-readable report
    m.cv_diagnostics              # what the internal CV chose + candidate grids
    m.to_text()                   # plain-text model, round-trippable
    m.match(X_new)                # first-match region id per row (-1 = none)
    m.predict(X_new)              # per-region leaf (mean or OLS), with_leaves

All hyperparameters are chosen by internal k-fold CV on the rows fit()
receives (train-only discipline; no leakage); everything is deterministic
for a fixed seed; rules and leaves are in original-data units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

import numpy as np

from .grid import FeatureBins
from .lits import (HEADER, Leaf, Region, VERSION, first_match_assign, fmt_num,
                    parse_rule)
from .stage1_cart import build_tree
from .stage1_dbscan import (EPS_RATIOS, centroid_assign, cluster, eps_candidates,
                            fit_z, standardize)
from .stage2_beam import fit_region_rules
from .stage2_ga import fit_region_rules_ga
from .leaves import fit_leaf, pick_leaf_kind


def _check_names(names):
    """Feature-name contract for the text format: rejected names are exactly
    the ones that would break parsing of the model text.

    ``parse_rule`` splits a rule body on the token `` and `` to find the
    literals, and within each condition it takes the *last* operator token
    (a ``<=`` or ``>`` substring) as the comparison, the final token as the
    threshold, and everything before the op as the feature name. So a name
    must (a) not contain `` and `` as a token (it would be split into a
    spurious literal), and (b) not contain any ``<``/``>`` character
    (it would be mis-tokenized as an operator). Enforced at ``fit`` and on
    ``from_text`` re-fits so a corrupted or hand-edited model file fails
    loudly instead of mis-parsing.
    """
    for nm in names:
        if not nm.strip():
            raise ValueError("feature_names must not contain empty strings")
        if " and " in nm:
            raise ValueError(
                f"feature name {nm!r} contains the token ' and ': rules are "
                "joined with it, so it would split into a spurious literal "
                "and break to_text()/from_text()")
        if nm.strip().lower() == "and":
            raise ValueError(
                f"feature name {nm!r} is the keyword 'and' used to join rule "
                "conditions; rename it")
        if "<" in nm or ">" in nm:
            raise ValueError(
                f"feature name {nm!r} contains a '<' or '>' character: it "
                "collides with the rule text format (ops are '<=' and '>') "
                "and would break to_text()/from_text() round-trips")


def _ga_region_rng(seed: int, region_id: int):
    """Deterministic, per-region generator derived from the model seed.
    Deriving from (seed, region_id) keeps regions independent of the order
    and count of regions (stable across fits), unlike a shared rng."""
    return np.random.default_rng(np.random.SeedSequence([seed, region_id]))


def _fold_assignment(n: int, folds: int, seed: int) -> list[int]:
    perm = np.random.RandomState(seed).permutation(n)
    fold_of = [0] * n
    for pos, i in enumerate(perm):
        fold_of[int(i)] = pos % folds
    return fold_of


def _r2(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    ss = float(((y_true - y_hat) ** 2).sum())
    var = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss / var if var > 0 else float("nan")


def _region_box_mask(X: np.ndarray, box: tuple) -> np.ndarray:
    """Mask of one flat interval box (each feature at most twice: lo + hi)."""
    m = np.ones(X.shape[0], dtype=bool)
    for f, op, t in box:
        m &= (X[:, f] <= t if op == "<=" else X[:, f] > t)
    return m


class RegressionRulesExtractor:
    """Two-stage interpretable rule model. See the module docstring and README.

    Minimal deterministic example (a small piecewise target):

    >>> import numpy as np
    >>> m = RegressionRulesExtractor(
    ...     seed=42, max_regions=2, depths=(1, 2), min_leaves=(8, 12)
    ... ).fit(np.linspace(0., 1., 100).reshape(-1, 1),
    ...       np.where(np.linspace(0., 1., 100) < 0.5, 5.0, 50.0),
    ...       feature_names=["t"])
    >>> (m.n_features_, m.n_samples_, m.target_name)
    (1, 100, 'y')
    >>> len(m.regions)
    2
    >>> m.match(np.array([[0.1], [0.9]])).tolist()
    [1, 0]
    >>> m.to_text().splitlines()[0].startswith("regression_rules v")
    True
    >>> m.to_text().splitlines()[0].count("n_regions=2") == 1
    True
    >>> from_texted = RegressionRulesExtractor.from_text(m.to_text(),
    ...         feature_names=["t"])
    >>> from_texted.to_text() == m.to_text()   # exact text round-trip
    True
    """
    def __init__(self, *,
                 stage1: str = "cart",
                 stage2: str = "beam",
                 n_quantiles: int = 40,
                 max_regions: int = 8,
                 depths: tuple[int, ...] = (2, 3, 4),
                 min_leaves: tuple[int, ...] = (15, 30),
                 max_depth: int | None = None,
                 cv_folds: int = 5,
                 stage2_beam_width: int = 20,
                 max_rules: int = 3,
                 min_precision_margin: float = 0.05,
                 leaf_folds: int = 5,
                 with_leaves: bool = True,
                 ga_pop_size: int = 80,
                 ga_n_gen: int = 60,
                 dbscan_min_samples: int = 5,
                 dbscan_min_cluster_size: int = 15,
                 dbscan_eps_ratios: tuple[float, ...] = EPS_RATIOS,
                 seed: int = 42):
        if stage1 not in ("cart", "dbscan"):
            raise ValueError(f"unknown stage1 {stage1!r}; use 'cart' or 'dbscan'")
        if stage2 not in ("beam", "ga"):
            raise ValueError(f"unknown stage2 {stage2!r}; use 'beam' or 'ga'")
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles!r}")
        if cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2 (need a held-out fold), got {cv_folds!r}")
        if not depths or any(int(v) < 1 for v in depths):
            raise ValueError(f"depths must be a non-empty tuple of ints >= 1, got {depths!r}")
        if min_leaves is not None and (not min_leaves or any(int(v) < 2 for v in min_leaves)):
            raise ValueError(f"min_leaves must be positive ints >= 2, got {min_leaves!r}")
        if max_regions < 1:
            raise ValueError(f"max_regions must be >= 1, got {max_regions!r}")
        if max_rules < 1:
            raise ValueError(f"max_rules must be >= 1, got {max_rules!r}")
        if not (0.0 <= min_precision_margin < 1.0):
            raise ValueError(
                f"min_precision_margin must be in [0, 1), got {min_precision_margin!r}")
        if ga_pop_size < 2:
            raise ValueError(f"ga_pop_size must be >= 2, got {ga_pop_size!r}")
        if ga_n_gen < 1:
            raise ValueError(f"ga_n_gen must be >= 1, got {ga_n_gen!r}")
        if dbscan_min_samples < 2:
            raise ValueError(f"dbscan_min_samples must be >= 2, got {dbscan_min_samples!r}")
        if dbscan_min_cluster_size < 2 * max(1, dbscan_min_samples):
            raise ValueError(
                f"dbscan_min_cluster_size={dbscan_min_cluster_size} must be at"
                f" least 2*dbscan_min_samples={2 * max(1, dbscan_min_samples)};"
                " otherwise DBSCAN can only form single-row core chains")
        if not dbscan_eps_ratios or any(not (r > 0) for r in dbscan_eps_ratios):
            raise ValueError(
                f"dbscan_eps_ratios must be positive, got {dbscan_eps_ratios!r}")
        self.stage1 = stage1
        self.stage2 = stage2
        self.n_quantiles = n_quantiles
        self.max_regions = max_regions
        self.depths = tuple(depths)
        self.min_leaves = tuple(min_leaves) if min_leaves else (15, 30)
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.stage2_beam_width = stage2_beam_width
        self.max_rules = max_rules
        self.min_precision_margin = min_precision_margin
        self.leaf_folds = leaf_folds
        self.with_leaves = with_leaves
        self.ga_pop_size = ga_pop_size
        self.ga_n_gen = ga_n_gen
        self.dbscan_min_samples = int(dbscan_min_samples)
        self.dbscan_min_cluster_size = int(dbscan_min_cluster_size)
        self.dbscan_eps_ratios = tuple(float(r) for r in dbscan_eps_ratios)
        self.seed = seed

        # fitted state
        self.regions: list[Region] = []
        self.feature_names_: list[str] = []
        self.target_name: str = "y"
        self.fallback_value: float | None = None
        self.n_features_: int = 0
        self.n_samples_: int = 0
        self.stage1_params: dict = {}
        self.noise_fraction_train: float | None = None  # dbscan only
        self.cv_diagnostics: dict = {}
        self._leaf_kinds: dict = {}
        self._fit: bool = False

    # ------------------------------------------------------------- surface

    @property
    def rules(self) -> list:
        """One item per region: its ruleset as a list of rules, each rule a
        list of ``(feature_name, op, threshold)`` in original units -- the
        plan's ``rules_`` convenience view of ``regions``."""
        if not self._fit:
            raise RuntimeError("call fit() first")
        out = []
        for r in self.regions:
            out.append([[(self.feature_names_[int(f)], op, float(thr))
                         for f, op, thr in rule] for rule in r.ruleset])
        return out

    # ------------------------------------------------------------------ fit

    def fit(self, X, y, *, feature_names=None, target_name="y"):
        """Fit the two-stage model on ``(X, y)``.

        This data is treated as the **train set**: ``fit`` never holds any of
        it out for its own validation, and every internal CV (stage-1
        parameters, leaf choice) is fit on, and scored on, subsets of exactly
        these rows. Call ``predict`` / ``match`` on new rows to evaluate.

        Parameters
        ----------
        X : (n, d) array-like of real numbers
        y : (n,) array-like of real numbers
            Must be finite. A zero-variance ``y`` fits a constant model
            (R^2 undefined) without error -- the library still extracts the
            region text.
        feature_names : (d,) sequence of str, optional
            Descriptive names in the emitted rules and leaves. Must be unique,
            non-empty, and free of the text-format tokens (`` and `` and the
            ``<``/``>`` characters); defaults to ``x0...x{d-1}``.
        target_name : str
            Name of the target for reporting / text context only.

        Returns ``self`` (sklearn-style for chaining).

        A constant target produces a constant model, not an error (R^2 of a
        zero-variance target is undefined, so CV is skipped for it):

        >>> import numpy as np
        >>> c = RegressionRulesExtractor(seed=1, max_regions=3,
        ...                              depths=(1, 2), min_leaves=(3, 4)).fit(
        ...     np.array([[0.], [1.], [2.], [3.], [4.]]),
        ...     np.full(5, 7.5), feature_names=["t"], target_name="y")
        >>> len(c.regions)
        1
        >>> c.predict(np.array([[10.0]])).tolist()
        [7.5]

        Raises
        ------
        ValueError
            Non-2-D or non-numeric ``X``, non-finite values, row-count
            mismatch with ``y``, duplicate or text-incompatible
            ``feature_names``, or a documented infeasibility of the selected
            modes (e.g. an empty ``(depth, min_leaf)`` CV grid, an ``eps``
            grid that clusters no folds, an exceeded region budget). Every
            message names the offending value and the constructor knob to
            change.
        """
        try:
            X = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"X must be numeric; could not convert to float: {e}") from e
        try:
            y = np.asarray(y, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"y must be numeric; could not convert to float: {e}") from e
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        n, d = X.shape
        y = y.reshape(-1) if y.ndim > 1 else y
        if y.ndim != 1 or y.shape[0] != n:
            raise ValueError(f"y must be 1-D with the same row count as X "
                             f"(X has {n} rows, y has shape {y.shape})")
        if n == 0:
            raise ValueError("X has no rows to fit on")
        n_bad_y = int(np.count_nonzero(~np.isfinite(y)))
        if n_bad_y:
            raise ValueError(
                f"y has {n_bad_y} non-finite value(s); clean or impute before "
                "fitting (the library does not impute)")
        n_bad_x = int(np.count_nonzero(~np.isfinite(X)))
        if n_bad_x:
            raise ValueError(
                f"X has {n_bad_x} non-finite value(s); clean or impute before "
                "fitting (the library does not impute)")
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(d)]
        feature_names = [str(f) for f in feature_names]
        if len(feature_names) != d or len(set(feature_names)) != d:
            raise ValueError(f"feature_names needs {d} unique strings, "
                             f"got {feature_names!r}")
        _check_names(feature_names)
        target_name = str(target_name)
        if target_name in feature_names:
            raise ValueError(f"target_name {target_name!r} collides with a feature name")

        bins = FeatureBins(X, self.n_quantiles)

        # --- stage 1: region discovery, CV'd within the mode ---
        stage1_regions = self._fit_stage1(X, y, bins)

        # --- stage 2 + leaves, one region per stage-1 region ---
        self.regions = []
        for rid, (box, pos) in enumerate(stage1_regions):
            if self.stage2 == "ga":
                fr = fit_region_rules_ga(
                    bins, X, pos,
                    k=3, max_rules=self.max_rules, max_len=4,
                    pop_size=self.ga_pop_size, n_gen=self.ga_n_gen,
                    min_precision_margin=self.min_precision_margin,
                    min_points=5,
                    rng=_ga_region_rng(self.seed, rid))
            else:
                fr = fit_region_rules(
                    bins, X, pos,
                    k=3, max_rules=self.max_rules, max_len=4,
                    beam_width=self.stage2_beam_width,
                    min_precision_margin=self.min_precision_margin,
                    min_points=5)
            ruleset = tuple(
                tuple((int(f), op, float(bins.q[f][pos_])) for f, op, pos_ in rule)
                for rule in fr.rules)
            # degradation: if stage 2 found nothing, the region's tight box
            # is its exact, honest rule (one flat conjunction covering every
            # region member by construction)
            if not ruleset:
                ruleset = (tuple(box),)
            # train F1/coverage of the union, against one-vs-rest (train)
            f1, cov = (0.0, 0.0)
            if fr.union is not None:
                f1, cov = fr.union.f1, fr.union.coverage
            leaf = None
            if self.with_leaves:
                kind = self._leaf_kind_for_region(X, y, pos)
                leaf = fit_leaf(X[pos], y[pos], kind, feature_names)
            self.regions.append(Region(
                region_id=rid,
                ruleset=ruleset,
                n_train=int(pos.sum()),
                train_coverage=cov,
                train_f1=f1,
                leaf=leaf,
                pareto=fr.pareto))
        self.feature_names_ = list(feature_names)
        self.target_name = target_name
        self.n_features_ = d
        self.n_samples_ = n
        self.fallback_value = float(y.mean())
        self._finish_cv_diagnostics()
        self._fit = True
        return self

    def _finish_cv_diagnostics(self):
        """Assemble ``self.cv_diagnostics`` (fit-only, never serialized):
        what was CV'd, the scored candidate grid, and what won. The
        stage-1 block is filled by ``_pick_params`` / ``_pick_dbscan_eps``;
        the leaf block here."""
        if self.with_leaves:
            self.cv_diagnostics["leaves"] = {
                "objective": ("per-region 5-fold (leaf_folds) CV comparing a "
                              "constant mean leaf vs OLS on the region's "
                              "training rows; mean wins unless OLS's mean "
                              "held-out MSE is strictly smaller"),
                "policy": ("CV only when a region has >= 8 training rows "
                            "(smaller regions default to the constant mean); "
                            "regions are fitted on all training rows, never "
                            "held-out folds"),
                "choices": dict(self._leaf_kinds),
            }
        self.cv_diagnostics["stage2"] = {
            "note": ("stage 2 (beam / ga) has no CV: it is a deterministic, "
                     "seeded search whose acceptance gate (union precision "
                     "vs base rate, or the Pareto F1 front in GA) is "
                     "evaluated on training rows and never re-selected"),
            "mode": self.stage2,
        }

    # ------------------------------------------------------------- report

    def summary(self) -> str:
        """One-screen human-readable report: what was extracted, what CV
        chose, and what each region says. Purely informational -- the
        machine-readable versions are ``regions`` and ``cv_diagnostics``.
        """
        if not self._fit:
            raise RuntimeError("call fit() first")
        L = [f"regression_rules v{VERSION}  stage1={self.stage1}  "
             f"stage2={self.stage2}  seed={self.seed}",
             f"train: {self.n_samples_} rows x {self.n_features_} features   "
             f"target: {self.target_name}"]
        if self.stage1 == "cart":
            L.append(f"stage 1:  no-reuse CART, depth={self.stage1_params.get('depth')} "
                    f"min_leaf={self.stage1_params.get('min_leaf')}  "
                    f"({len(self.cv_diagnostics.get('stage1', {}).get('candidates', []))} CV candidates)")
        else:
            L.append(f"stage 1:  DBSCAN eps={self.stage1_params.get('eps')}  "
                    f"min_samples={self.stage1_params.get('min_samples')}  "
                    f"min_cluster_size={self.stage1_params.get('min_cluster_size')}"
                    + (f"  noise={self.noise_fraction_train:.1%}" if self.noise_fraction_train is not None else ""))
        L.append(f"stage 2:  {'GA' if self.stage2 == 'ga' else 'RIPPER beam'} covering loop "
                 f"(max_rules={self.max_rules}, margin={self.min_precision_margin})")
        if self.with_leaves:
            L.append(f"leaves:   'mean' | 'ols' chosen by per-region {self.leaf_folds}-fold CV  "
                     f"choices={self.cv_diagnostics.get('leaves', {}).get('choices', {})}")
        L.append(f"regions:  {len(self.regions)}   fallback={self.fallback_value}")
        L.append("")
        L.append(f"{'region':>6} {'n':>5} {'rules':>5} {'lits':>5} "
                 f"{'train_cov':>9} {'train_f1':>9}  leaf")
        for r in self.regions:
            leaf_txt = (r.leaf.kind if r.leaf is not None else "-")
            L.append(f"{r.region_id:>6} {r.n_train:>5} {r.n_rules:>5} "
                     f"{r.n_literals:>5} {r.train_coverage:>9.3f} "
                     f"{r.train_f1:>9.3f}  {leaf_txt}")
        L.append(f"TOTAL  rules={sum(r.n_rules for r in self.regions)}  "
                 f"literals={sum(r.n_literals for r in self.regions)}")
        return "\n".join(L)

    def _leaf_kind_for_region(self, X, y, pos) -> str:
        """Pick the per-region leaf kind with train-only CV, recording what
        was chosen. Uses CV only when the region has enough rows to make the
        comparison honest; small regions default to the constant mean (no CV
        spent, no diagnostic needed)."""
        n_pos = int(np.asarray(pos).sum())
        if not self.with_leaves or n_pos < 8:
            return "mean"
        kind = pick_leaf_kind(X[pos], y[pos], folds=self.leaf_folds)
        self._leaf_kinds.setdefault(kind, 0)
        self._leaf_kinds[kind] += 1
        return kind

    # ------------------------------------------------------------ stage 1

    def _fit_stage1(self, X, y, bins):
        """Fit stage 1 in the selected mode. Returns ``(box, pos)`` tuples,
        ordered largest-first (which fixes the region ids). ``pos`` is the
        *true* stage-1 membership over train rows: on the CART path the leaf
        cell (CART leaves partition box space, so ``pos`` is exactly the box
        mask); on the DBSCAN path the cluster itself -- a subset of its
        box, because a box can contain rows that are not in the cluster.
        ``box`` is a flat interval box in original units that contains every
        member of that region: exact tight box for CART; for DBSCAN, the
        feature-wise bounding box of the cluster, an honest fallback rule
        if stage 2 produces nothing."""
        if self.stage1 == "cart":
            depth, min_leaf = self._pick_params(X, y, bins)
            depth = depth if self.max_depth is None else min(depth, self.max_depth)
            tree = build_tree(X, y, depth=depth, min_leaf=min_leaf, bins=bins)
            if len(tree.regions) > self.max_regions:
                raise ValueError(
                    f"stage 1 produced {len(tree.regions)} regions; the budget "
                    f"is max_regions={self.max_regions}. Raise max_regions or "
                    f"shrink depths/min_leaves.")
            boxes = [tuple(r.splits) for r in tree.regions]
            self.stage1_params = {"depth": depth, "min_leaf": min_leaf,
                                  "n_regions": len(tree.regions)}
            self.noise_fraction_train = None
            return [(b, _region_box_mask(X, b)) for b in boxes]
        else:
            n = len(y)
            if n < self.dbscan_min_cluster_size:
                raise ValueError(
                    f"stage1='dbscan': min_cluster_size="
                    f"{self.dbscan_min_cluster_size} exceeds n={n}")
            mean, std = fit_z(X)
            Z = standardize(X, mean, std)
            eps = self._pick_dbscan_eps(X, y, mean, std, Z)
            ids, lab = cluster(Z, eps, self.dbscan_min_samples,
                               self.dbscan_min_cluster_size)
            if not ids:
                raise ValueError(
                    f"stage1='dbscan' found no clusters (eps={eps}, "
                    f"min_samples={self.dbscan_min_samples}); try a larger "
                    f"dbscan_min_samples or wider dbscan_eps_ratios")
            n_clusters_raw = len(ids)
            if len(ids) > self.max_regions:
                for c in ids[self.max_regions:]:
                    lab[lab == c] = -1
                ids = ids[:self.max_regions]
            d = X.shape[1]
            out = []
            for c in ids:                          # ids sorted largest-first
                mtr = lab == c
                lows = X[mtr].min(axis=0)
                highs = X[mtr].max(axis=0)
                box = []
                for f in range(d):
                    if highs[f] > lows[f]:
                        box.append((f, ">", float(lows[f])))
                        box.append((f, "<=", float(highs[f])))
                out.append((tuple(box), mtr))
            self.stage1_params = {"eps": eps,
                                  "min_samples": self.dbscan_min_samples,
                                  "min_cluster_size": self.dbscan_min_cluster_size,
                                  "n_clusters": n_clusters_raw,
                                  "n_regions": len(ids)}
            self.noise_fraction_train = float((lab == -1).mean())
            return out

    def _pick_dbscan_eps(self, X, y, mean, std, Z) -> float:
        """Internal k-fold CV over the candidate ``eps`` grid, on the
        stage-1 proxy (cluster-mean leaves + centroid assignment, fold
        RMSE -- the same objective the research ``compare_learners.py``
        CV'd). An ``eps`` whose fold produced no clusters is skipped.
        Smaller ``eps`` wins ties, so the grid's order (ascending) gives
        simpler, denser structures a deterministic preference on equal
        quality. The full grid and the winner are recorded in
        ``self.cv_diagnostics['stage1']``."""
        n = len(y)
        k = max(2, self.cv_folds)
        fold_of = _fold_assignment(n, k, self.seed)
        folds_idx = {f: [i for i in range(n) if fold_of[i] == f] for f in range(k)}
        cand = eps_candidates(Z, self.dbscan_min_samples,
                             self.dbscan_eps_ratios)
        best, besps = None, None
        candidates = []
        for eps in cand:
            errs = []
            ok = True
            for f in range(k):
                va = np.asarray(folds_idx[f], dtype=int)
                tr = np.setdiff1d(np.arange(n), va)
                e = self._dbscan_fold_rmse(X, y, tr, va, float(eps))
                if e is None:
                    ok = False
                    break
                errs.append(e)
            r = float(np.mean(errs)) if ok else None
            candidates.append({"eps": float(eps), "cv_rmse": r})
            if not ok:
                continue
            if best is None or r < best - 1e-12:
                best, besps = r, float(eps)
        if besps is None:
            raise ValueError(
                "no dbscan eps candidate produced clusters on every CV fold; "
                "raise dbscan_min_samples or widen dbscan_eps_ratios")
        self.cv_diagnostics["stage1"] = {
            "objective": ("cv_rmse (mean of per-fold held-out RMSE of the "
                          "stage-1 proxy: cluster-mean leaves + centroid "
                          "assignment)"),
            "folds": k,
            "candidates": candidates,
            "selected": {"eps": besps},
            "selected_cv_rmse": float(best),
            "tie_break": "smaller eps found first",
        }
        return besps

    def _dbscan_fold_rmse(self, X, y, tr, va, eps):
        """Fold RMSE of the stage-1 proxy: DBSCAN on this fold's train rows
        (standardization fit on those rows only); leaves = cluster means;
        validation rows assigned to the nearest centroid within eps
        (misses -> train global mean, so gaps are punished). Returns None
        when the fold found no clusters."""
        Xtr, ytr, Xva = X[tr], y[tr], X[va]
        mean, std = fit_z(Xtr)
        Ztr = standardize(Xtr, mean, std)
        Zva = standardize(Xva, mean, std)
        ids, lab = cluster(Ztr, eps, self.dbscan_min_samples,
                           self.dbscan_min_cluster_size)
        if not ids:
            return None
        gmean = float(ytr.mean())
        leaf_vals = np.array([float(ytr[lab == c].mean()) for c in ids])
        assign = centroid_assign(Zva, Ztr, lab, ids, eps)
        y_hat = np.full(len(va), gmean)
        hit = assign >= 0
        if hit.any():
            id2pos = {c: i for i, c in enumerate(ids)}
            y_hat[hit] = leaf_vals[[id2pos[int(c)] for c in assign[hit]]]
        return float(np.sqrt(np.mean((y[va] - y_hat) ** 2)))

    # ------------------------------------------------------------ internals

    def _pick_params(self, X, y, bins):
        """Internal k-fold CV of (depth, min_leaf); ties -> the smaller
        (depth, min_leaf) found first, so simpler trees win ties. Returns
        ``(depth, min_leaf)`` and records the full CV grid in
        ``self.cv_diagnostics['stage1']`` (candidates with their mean
        k-fold R^2, and which candidate won)."""
        n = len(y)
        k = max(2, self.cv_folds)
        fold_of = _fold_assignment(n, k, self.seed)
        folds_idx = {f: [i for i in range(n) if fold_of[i] == f] for f in range(k)}

        # Zero-variance target: R^2 is 0/0 == undefined on every fold, so the
        # CV grid can score nothing, and fit() would raise the misleading
        # "no candidate in the grid" error. The documented contract is a
        # constant model without error, and the rest of the pipeline already
        # builds one (a tree that can't split; a constant-mean leaf). Pick the
        # simplest candidate deterministically and skip the CV, saying so in
        # the diagnostics.
        if float(y.var()) == 0.0:
            d0 = self.depths[0] if (self.max_depth is None
                                    or self.depths[0] <= self.max_depth) \
                else min(self.depths)
            self.cv_diagnostics["stage1"] = {
                "objective": "cv_r2 (skipped: y is constant; R^2 undefined)",
                "folds": k,
                "candidates": [],
                "selected": {"depth": int(d0), "min_leaf": int(self.min_leaves[0])},
                "selected_cv_r2": None,
                "tie_break": "zero-variance y: CV skipped, simplest candidate "
                             "(depths[0], min_leaves[0]) chosen",
            }
            return (int(d0), int(self.min_leaves[0]))

        def cv_r2(depth: int, min_leaf: int) -> float | None:
            if n < 5 * max(1, min_leaf):
                return None
            errs = []
            for f in range(k):
                te = np.asarray(folds_idx[f], dtype=int)
                tr = np.setdiff1d(np.arange(n), te)
                tree = build_tree(X[tr], y[tr], depth=depth, min_leaf=min_leaf,
                                  bins=bins)
                if len(tree.regions) > self.max_regions:
                    return None
                which = np.full(len(te), -1, dtype=int)
                for r in tree.regions:
                    m = _region_box_mask(X[te], r.splits) & (which == -1)
                    which[m] = r.region_id
                y_hat = np.zeros(len(te))
                for r in tree.regions:
                    mm = which == r.region_id
                    if not mm.any():
                        continue
                    tr_m = _region_box_mask(X[tr], r.splits)
                    vals = y[tr][tr_m]
                    y_hat[mm] = float(vals.mean()) if len(vals) else y[tr].mean()
                errs.append(_r2(y[te], y_hat))
            if any(np.isnan(e) for e in errs):
                return None
            return float(np.mean(errs))

        candidates = []
        best = None
        for depth in self.depths:
            if self.max_depth is not None and depth > self.max_depth:
                continue
            for min_leaf in self.min_leaves:
                s = cv_r2(depth, min_leaf)
                candidates.append({"depth": depth, "min_leaf": min_leaf,
                                   "cv_r2": s})
                if s is not None and (best is None or s > best[0] + 1e-12):
                    best = (s, depth, min_leaf)
        if best is None:
            raise ValueError(
                "no (depth, min_leaf) in the CV grid fits within "
                f"max_regions={self.max_regions}; raise max_regions or shrink "
                "depths/min_leaves.")
        self.cv_diagnostics["stage1"] = {
            "objective": "cv_r2 (mean of per-fold held-out R^2 on this data)",
            "folds": k,
            "candidates": candidates,
            "selected": {"depth": best[1], "min_leaf": best[2]},
            "selected_cv_r2": float(best[0]),
            "tie_break": "smaller (depth, min_leaf) found first",
        }
        return (best[1], best[2])

    # --------------------------------------------------------------- match

    def match(self, X) -> np.ndarray:
        """First-match region id per row; -1 for no match."""
        X = np.asarray(X, dtype=np.float64)
        if not self._fit:
            raise RuntimeError("call fit() first")
        rulesets = []
        for r in self.regions:
            rs = [tuple((int(f), op, float(thr)) for f, op, thr in rule)
                  for rule in r.ruleset]
            rulesets.append(tuple(rs))
        return first_match_assign(X, rulesets)

    # ------------------------------------------------------------- predict

    def predict(self, X) -> np.ndarray:
        if not self.with_leaves:
            raise ValueError(
                "this model was fit with with_leaves=False: it has no numeric "
                "leaves to predict with. Use match() for region ids, or refit "
                "with with_leaves=True.")
        X = np.asarray(X, dtype=np.float64)
        which = self.match(X)
        y_hat = np.full(len(X), self.fallback_value)
        for r in self.regions:
            mm = which == r.region_id
            if mm.any():
                if r.leaf is not None and r.leaf.is_fitted():
                    y_hat[mm] = r.leaf.predict(X[mm])
        return y_hat

    # ---------------------------------------------------------------- text

    def to_text(self) -> str:
        if not self._fit:
            raise RuntimeError("call fit() first")
        L = [f"{HEADER} v{VERSION} stage1={self.stage1} stage2={self.stage2} "
             f"n_regions={len(self.regions)} seed={self.seed} "
             f"fallback={fmt_num(self.fallback_value)}"]
        for r in self.regions:
            L.append(r.to_text(self.feature_names_))
        L.append(f"fallback: {fmt_num(self.fallback_value)}")
        return "\n".join(L)

    @classmethod
    def from_text(cls, text: str, *, feature_names: list[str],
                  seed: int | None = None, **kwargs) -> "RegressionRulesExtractor":
        """Load a model from its ``to_text()`` file. ``feature_names`` is
        required because the text references features by name (it cannot be
        stored unambiguously without it). ``seed`` is taken from the header
        (the model's provenance); the ``seed`` parameter here only applies
        when the header predates seed recording. Returns a fitted model:
        ``predict``/``match``/``to_text`` work immediately."""
        lines = text.strip().splitlines()
        if not lines or not lines[0].startswith(HEADER + " "):
            raise ValueError(f"not a regression_rules model file: {lines[0]!r}")
        feature_names = [str(f) for f in feature_names]
        _check_names(feature_names)
        header: dict[str, str] = {}
        for tok in lines[0].split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                header[k] = v
        regions: list[Region] = []
        fb = None
        i = 1
        while i < len(lines):
            ln = lines[i]
            if ln.startswith("region "):
                block = [ln]
                i += 1
                while i < len(lines) and not lines[i].startswith("region ") \
                        and not lines[i].startswith("fallback:"):
                    block.append(lines[i])
                    i += 1
                m = re.match(
                    r"region (\d+) \[n=(\d+)\] rules=(\d+) literals=(\d+) "
                    r"train_f1=([0-9.]+)(?: train_cov=([0-9.]+))?:", ln)
                if not m:
                    raise ValueError(f"unparseable region header: {ln!r}")
                ruleset = []
                leaf_txt = None
                for bl in block[1:]:
                    bl = bl.strip()
                    if bl.startswith("rule "):
                        body = bl.split(":", 1)[1]
                        ruleset.append(parse_rule(body, feature_names))
                    elif bl.startswith("leaf:"):
                        leaf_txt = bl.split(":", 1)[1].strip()
                leaf = Leaf.from_text(leaf_txt, feature_names) \
                    if leaf_txt is not None else None
                regions.append(Region(
                    region_id=int(m.group(1)),
                    ruleset=tuple(ruleset),
                    n_train=int(m.group(2)),
                    train_f1=float(m.group(5)),
                    train_coverage=(float(m.group(6)) if m.group(6) is not None
                                    else 0.0),
                    leaf=leaf))
            elif ln.startswith("fallback:"):
                fb = float(ln.split(":", 1)[1].strip())
                i += 1
            else:
                blank = not ln.strip()
                if not blank:
                    raise ValueError(f"unrecognized model line: {ln!r}")
                i += 1
        if "n_regions" in header and int(header["n_regions"]) != len(regions):
            raise ValueError(
                f"header says n_regions={header['n_regions']} but the model "
                f"has {len(regions)} region blocks; the file is inconsistent")
        obj_kwargs = {k: v for k, v in kwargs.items()
                     if k not in ("stage1", "stage2", "seed")}
        obj = cls(**obj_kwargs)
        # header tokens are the model's provenance and win any caller value
        if "stage1" in header:
            obj.stage1 = header["stage1"]
        if "stage2" in header:
            obj.stage2 = header["stage2"]
        if "seed" in header:
            obj.seed = int(header["seed"])
        elif seed:
            obj.seed = int(seed)
        obj.regions = regions
        obj.feature_names_ = list(feature_names)
        obj.target_name = "y"
        obj.n_features_ = len(feature_names)
        obj.n_samples_ = max([r.n_train for r in regions], default=0)
        obj.fallback_value = float(fb) if fb is not None else 0.0
        obj.cv_diagnostics = {
            "note": ("model loaded from text: the internal CV ran during the "
                     "original fit and its diagnostics are not part of the "
                     "text format"),
            "stage2": {"mode": obj.stage2},
        }
        obj._fit = True
        return obj
