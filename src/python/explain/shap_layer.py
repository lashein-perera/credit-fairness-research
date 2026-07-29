"""Module 5 — Explainability Layer.

Produces the leakage-vs-SHAP cross-plot: features that are both influential and
leaky sit in the top-right danger quadrant. Best single figure in the dissertation.
"""


def global_importance(model, X):
    """Mean |SHAP| per feature. Use TreeExplainer for gradient boosting."""
    raise NotImplementedError("Week 6")


def explain_applicant(model, x_row):
    """Per-applicant explanation — what a regulator or rejected applicant would need."""
    raise NotImplementedError("Week 6")


def stability_across_folds(model_factory, X, y, n_splits: int = 5) -> float:
    """Rank correlation of SHAP importance between folds — an explainability quality metric."""
    raise NotImplementedError("Week 6")


def leakage_vs_shap_plot(leakage_scores, shap_importance, output_path):
    """Scatter: x = leakage, y = SHAP importance, one point per feature.

    Annotate the top-right quadrant by name. Produce before and after mitigation
    to show the quadrant emptying.
    """
    raise NotImplementedError("Week 6")
