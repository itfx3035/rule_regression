"""regression_rules: two-stage, text-readable region + rule extraction for
regression data, with optional per-region leaves.

Public API:

    RegressionRulesExtractor(stage1="cart"|"dbscan",
                             stage2="beam"|"ga", ...)
        .fit(X, y, feature_names=[...])
        .regions                 # list[Region] (rule sets + leaves + stats)
        .rules                   # named (name, op, threshold) literals
        .summary()               # one-screen human-readable report (str)
        .cv_diagnostics          # per-component: what CV chose + candidate grid
        .match(X_new)            # int array, first-match region id (-1 = none)
        .predict(X_new)          # leaf prediction (requires with_leaves=True)
        .to_text()
        .from_text(text, feature_names=...)
"""
from .lits import Leaf, Region
from .extractor import RegressionRulesExtractor

__version__ = "0.1.0"
__all__ = ["RegressionRulesExtractor", "Region", "Leaf", "__version__"]
