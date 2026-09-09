"""Module 5 — SHAP Explainability Layer.

Two jobs. The first is ordinary: attribute each model's predictions to features
via Shapley values (Lundberg & Lee, 2017), giving the global importance ranking
a lender would put in front of a regulator.

The second is the one the research needs. Module 1 measures which features
carry the protected attribute; this module measures which features the credit
model actually relies on. Crossing the two answers RQ3 — whether the proxies
are load-bearing. A feature that leaks heavily but contributes nothing can be
dropped for free; a feature that is both leaky and influential is where the
accuracy-fairness trade-off actually lives, and is what Module 4 must treat
rather than delete.

    leakage high, SHAP low     removable
    leakage low,  SHAP high    safe workhorse
    leakage high, SHAP high    THE PROBLEM QUADRANT

EXPLAINER CHOICE
----------------
    gbm       TreeExplainer      exact, fast, tree-structured
    logistic  LinearExplainer    exact given a background distribution
    mlp       PermutationExplainer  model-agnostic sampling, far slower

FEATURE ALIGNMENT
-----------------
Module 2's preprocessor ordinal-encodes categoricals rather than one-hot
encoding them, so every transformed column corresponds to exactly one input
feature. SHAP values therefore map back to original feature names without
aggregation, and join directly against Module 1's leakage table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42

# model family -> explainer family
EXPLAINERS = {"gbm": "tree", "logistic": "linear", "mlp": "permutation"}


# --------------------------------------------------------------------------
# alignment between the transformed matrix and the original feature names
# --------------------------------------------------------------------------
def _feature_names(prep) -> list[str]:
    """Original column names, in transformed-column order."""
    names = list(prep.get_feature_names_out())
    return [n.split("__", 1)[1] if n.split("__", 1)[0] in ("num", "cat") else n
            for n in names]


def _as_matrix(values, n_features: int) -> np.ndarray:
    """Reduce any SHAP return shape to (n_rows, n_features) for the positive class.

    Explainers disagree on layout: a list per class, a 3-D array with a class
    axis, or a plain 2-D array for a single output. All three are normalised
    here so the rest of the module sees one shape.
    """
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, -1]          # positive class
    if values.ndim != 2:
        raise ValueError(f"unexpected SHAP shape {values.shape}")
    if values.shape[1] != n_features:
        raise ValueError(
            f"SHAP width {values.shape[1]} != {n_features} features")
    return values


# --------------------------------------------------------------------------
# core computation
# --------------------------------------------------------------------------
def shap_matrix(model, X: pd.DataFrame, model_kind: str = "gbm",
                max_rows: int = 2_000, background: int = 100,
                random_state: int = RANDOM_STATE, verbose: bool = False):
    """SHAP values for a fitted Module 2 pipeline.

    Returns (values, feature_names, X_sample). `model` is the full pipeline;
    the preprocessor is applied here and the explainer is attached to the
    bare classifier, because SHAP must see the numeric space the model was
    actually fitted on.
    """
    import shap

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = _feature_names(prep)

    rng = np.random.default_rng(random_state)
    if len(X) > max_rows:
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X_sample = X.iloc[np.sort(idx)]
    else:
        X_sample = X
    Z = prep.transform(X_sample)
    Z = np.asarray(Z.todense()) if hasattr(Z, "todense") else np.asarray(Z)

    kind = EXPLAINERS.get(model_kind, "permutation")
    n_bg = min(background, len(Z))
    bg = shap.utils.sample(Z, n_bg, random_state=random_state)

    if verbose:
        print(f"    explainer [{kind}]  rows {Z.shape[0]:,}  "
              f"features {Z.shape[1]}  background {n_bg}")

    if kind == "tree":
        # np.nan is meaningful to HistGradientBoosting; TreeExplainer reads
        # the fitted tree structure, so no background sample is needed.
        explainer = shap.TreeExplainer(clf)
        values = explainer.shap_values(Z, check_additivity=False)
    elif kind == "linear":
        explainer = shap.LinearExplainer(clf, bg)
        values = explainer.shap_values(Z)
    else:
        explainer = shap.PermutationExplainer(
            lambda z: clf.predict_proba(z)[:, 1], bg, seed=random_state)
        values = explainer(Z).values

    return _as_matrix(values, len(names)), names, X_sample


def global_importance(model, X: pd.DataFrame, model_kind: str = "gbm",
                      max_rows: int = 2_000, background: int = 100,
                      verbose: bool = False) -> pd.DataFrame:
    """Mean |SHAP| per feature — the global importance ranking.

    Columns: feature, shap_importance, shap_std, rank. Every feature in X is
    present, including those the model ignores, so the frame joins cleanly
    against a full leakage ranking.
    """
    values, names, _ = shap_matrix(model, X, model_kind=model_kind,
                                   max_rows=max_rows, background=background,
                                   verbose=verbose)
    df = pd.DataFrame({
        "feature": names,
        "shap_importance": np.abs(values).mean(axis=0),
        "shap_std": np.abs(values).std(axis=0),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    return df


def explain_applicant(model, x_row):
    """Per-applicant explanation — what a regulator or rejected applicant would need."""
    raise NotImplementedError("Week 6")


def stability_across_folds(model_factory, X, y, n_splits: int = 5) -> float:
    """Rank correlation of SHAP importance between folds — an explainability quality metric."""
    raise NotImplementedError("Week 6")


# --------------------------------------------------------------------------
# RQ3 — leakage against model reliance
# --------------------------------------------------------------------------
def leakage_vs_shap(leakage_scores: pd.DataFrame,
                    shap_importance: pd.DataFrame,
                    thresholds: tuple[float, float] | None = None) -> pd.DataFrame:
    """Join Module 1's per-feature leakage against Module 5's SHAP importance.

    `leakage_scores` is a frame from rank_leaky_features (feature,
    leakage_drop, ...); `shap_importance` is a frame from global_importance.
    The join is inner, so it is the caller's responsibility to supply a
    leakage ranking covering every feature — a truncated top-k ranking
    silently drops the low-leakage, high-importance region that makes the
    comparison informative.

    Each feature is placed in a quadrant by a split on both axes, at the
    median by default.

    `thresholds` overrides those cut-points. This matters for before/after
    comparison: medians recomputed on mitigated data move down with the data,
    so a median split leaves roughly a fixed share of features in the top-right
    quadrant however well the mitigation worked. Freezing the cut-points at the
    unmitigated values is what makes a fall in the count meaningful.
    """
    left = leakage_scores.rename(columns={"rank": "leakage_rank"})
    right = shap_importance.rename(columns={"rank": "shap_rank"})
    keep = ["feature", "leakage_drop", "leakage_std", "leakage_rank"]
    df = left[[c for c in keep if c in left.columns]].merge(
        right[["feature", "shap_importance", "shap_std", "shap_rank"]],
        on="feature", how="inner")

    lx, ly = (thresholds if thresholds is not None
              else (df["leakage_drop"].median(), df["shap_importance"].median()))
    df.attrs["thresholds"] = (float(lx), float(ly))
    high_leak = df["leakage_drop"] > lx
    high_shap = df["shap_importance"] > ly
    df["quadrant"] = np.select(
        [high_leak & high_shap, high_leak & ~high_shap, ~high_leak & high_shap],
        ["leaky_and_relied_on", "leaky_removable", "safe_workhorse"],
        default="low_low")

    return df.sort_values("shap_importance", ascending=False).reset_index(drop=True)


def rank_correlation(cross: pd.DataFrame) -> dict:
    """Spearman and Pearson between leakage and SHAP importance.

    Spearman is the headline figure: both quantities are heavily
    right-skewed, so a rank statistic is the defensible one.
    """
    from scipy.stats import pearsonr, spearmanr

    rho, p_rho = spearmanr(cross["leakage_drop"], cross["shap_importance"])
    r, p_r = pearsonr(cross["leakage_drop"], cross["shap_importance"])
    return {
        "n_features": int(len(cross)),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
    }


def _linthresh(values: np.ndarray) -> float:
    """Linear-region width for a symlog axis, from the data's own scale.

    The median non-zero magnitude. For the leakage axis this lands on the
    permutation-noise floor, so the linear region absorbs exactly those
    features whose score is not distinguishable from zero, and the log region
    is spent on the ones that are.
    """
    nz = np.abs(values[values != 0])
    return float(max(np.median(nz), 1e-7)) if len(nz) else 1e-6


def _draw_cross(ax, cross: pd.DataFrame, thresholds: tuple[float, float],
                n_annotate: int = 8, log: bool = True, highlight: list | None = None,
                xlim=None, ylim=None, linthresh=None):
    """Draw one cross-plot panel. Shared by the single and side-by-side figures."""
    lx, ly = thresholds
    danger = cross["quadrant"] == "leaky_and_relied_on"

    ax.scatter(cross.loc[~danger, "leakage_drop"],
               cross.loc[~danger, "shap_importance"],
               s=30, color="#4C72B0", alpha=0.7, edgecolor="none",
               label="other features")
    ax.scatter(cross.loc[danger, "leakage_drop"],
               cross.loc[danger, "shap_importance"],
               s=46, color="#C44E52", alpha=0.85, edgecolor="none",
               label="leaky and relied upon")

    ax.axvline(lx, ls="--", c="k", lw=1, alpha=0.5)
    ax.axhline(ly, ls="--", c="k", lw=1, alpha=0.5)

    xv, yv = cross["leakage_drop"].values, cross["shap_importance"].values
    if log:
        lt_x, lt_y = linthresh if linthresh else (_linthresh(xv), _linthresh(yv))
        ax.set_xscale("symlog", linthresh=lt_x)
        ax.set_yscale("symlog", linthresh=lt_y)
        # Explicit limits: symlog margins are applied in decades and would
        # otherwise pad the axis out by orders of magnitude. The generous top
        # pad is headroom for the annotations.
        ax.set_xlim(xlim if xlim else (min(xv.min() * 3, -lt_x), xv.max() * 5))
        ax.set_ylim(ylim if ylim else (0, yv.max() * 4))
    else:
        ax.margins(x=0.08, y=0.12)

    # Named features are pinned across panels when `highlight` is given, so the
    # reader can follow the same feature from before to after. Otherwise the
    # strongest members of the danger quadrant are labelled, by the product of
    # the two normalised scores.
    if highlight is not None:
        d = cross[cross["feature"].isin(highlight)].copy()
    else:
        d = cross[danger].copy()
        if not d.empty:
            d["joint"] = ((d["leakage_drop"] / max(cross["leakage_drop"].max(), 1e-12)) *
                          (d["shap_importance"] / max(cross["shap_importance"].max(), 1e-12)))
            d = d.nlargest(n_annotate, "joint")

    for i, r in enumerate(d.itertuples()):
        right = ax.transLimits.transform((r.leakage_drop, 0))[0] > 0.7
        dy = 6 if i % 2 == 0 else -13          # stagger to limit collisions
        ax.annotate(r.feature, (r.leakage_drop, r.shap_importance),
                    textcoords="offset points",
                    xytext=(-7 if right else 7, dy),
                    ha="right" if right else "left",
                    fontsize=8, color="#8B2E31")

    ax.grid(alpha=0.3)
    return int(danger.sum())


def leakage_vs_shap_plot(leakage_scores, shap_importance, output_path,
                         dataset: str = "", attribute: str = "",
                         model_kind: str = "", n_annotate: int = 8,
                         log: bool = True, thresholds=None):
    """Scatter: x = leakage, y = SHAP importance, one point per feature.

    Features in the top-right quadrant — leaky AND relied upon — are coloured
    separately and the strongest are annotated by name. That quadrant is the
    subject of RQ3, and emptying it is what Module 4 is trying to do.

    Both quantities span several orders of magnitude, so symlog axes are the
    default: on linear axes the great majority of features collapse onto the
    origin and the quadrant structure is invisible. Permutation importance is
    genuinely signed, hence symlog rather than log on x; mean |SHAP| cannot be
    negative, so the y-axis is clipped at zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cross = (leakage_scores
             if "shap_importance" in getattr(leakage_scores, "columns", [])
             else leakage_vs_shap(leakage_scores, shap_importance, thresholds))
    stats = rank_correlation(cross)
    thr = thresholds or cross.attrs.get(
        "thresholds", (cross["leakage_drop"].median(),
                       cross["shap_importance"].median()))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    _draw_cross(ax, cross, thr, n_annotate=n_annotate, log=log)

    ax.set_xlabel("Leakage contribution (drop in probe AUC when feature is permuted)")
    ax.set_ylabel(f"SHAP importance (mean |SHAP|){f' — {model_kind}' if model_kind else ''}")

    title = "Attribute leakage vs model reliance"
    if dataset or attribute:
        title += f" — {dataset} / {attribute}"
    ax.set_title(
        f"{title}\nSpearman $\\rho$ = {stats['spearman_rho']:.3f} "
        f"(p = {stats['spearman_p']:.3g}), n = {stats['n_features']} features",
        fontsize=12)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
# before / after mitigation
# --------------------------------------------------------------------------
def quadrant_transitions(before: pd.DataFrame, after: pd.DataFrame,
                         leakage_threshold: float | None = None,
                         n_unique_before: dict | None = None,
                         n_unique_after: dict | None = None) -> pd.DataFrame:
    """Per-feature movement between two cross-plots.

    Both frames must have been classified against the SAME thresholds, or the
    transition is an artefact of the boundary moving rather than the feature.

    MANUFACTURED PROXIES
    Setting `leakage_threshold` additionally flags features whose leakage rose
    from below that threshold to above it — features that carried no
    recoverable protected-attribute signal before mitigation and do afterwards.
    A repair that stratifies on a PREDICTED attribute can write stratum
    identity into a feature that had no distribution worth repairing, turning a
    near-constant column into a proxy. Passing the distinct-value counts
    records the evidence for that mechanism alongside the flag: a near-constant
    feature emerging as leaky while its cardinality explodes is the signature.
    """
    b = before[["feature", "leakage_drop", "shap_importance", "quadrant"]]
    a = after[["feature", "leakage_drop", "shap_importance", "quadrant"]]
    df = b.merge(a, on="feature", suffixes=("_before", "_after"))

    was = df["quadrant_before"] == "leaky_and_relied_on"
    now = df["quadrant_after"] == "leaky_and_relied_on"
    df["transition"] = np.select(
        [was & ~now, was & now, ~was & now],
        ["left_danger", "stayed_danger", "entered_danger"],
        default="never_danger")

    df["leakage_change"] = df["leakage_drop_after"] - df["leakage_drop_before"]
    df["shap_change"] = df["shap_importance_after"] - df["shap_importance_before"]

    if leakage_threshold is not None:
        df["leakage_emerged"] = ((df["leakage_drop_before"] <= leakage_threshold) &
                                 (df["leakage_drop_after"] > leakage_threshold))
    if n_unique_before is not None:
        df["n_unique_before"] = df["feature"].map(n_unique_before)
    if n_unique_after is not None:
        df["n_unique_after"] = df["feature"].map(n_unique_after)

    return df.sort_values("shap_importance_before", ascending=False).reset_index(drop=True)


def before_after_plot(panels: list, output_path, thresholds: tuple[float, float],
                      dataset: str = "", attribute: str = "",
                      model_kind: str = "", highlight: list | None = None,
                      n_annotate: int = 6, log: bool = True, ncols: int = 3):
    """Side-by-side cross-plots on shared axes and shared quadrant boundaries.

    `panels` is a list of (label, cross_frame). Every panel is drawn against
    the same frozen thresholds and the same limits, so the point cloud shifting
    left between panels is the result rather than an artefact of rescaling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allx = np.concatenate([c["leakage_drop"].values for _, c in panels])
    ally = np.concatenate([c["shap_importance"].values for _, c in panels])
    lt = (_linthresh(allx), _linthresh(ally))
    xlim = (min(allx.min() * 3, -lt[0]), allx.max() * 5)
    ylim = (0, ally.max() * 4)

    ncols = min(ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 6.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(panels):]:
        ax.set_visible(False)

    for ax, (label, cross) in zip(axes, panels):
        n = _draw_cross(ax, cross, thresholds, n_annotate=n_annotate, log=log,
                        highlight=highlight, xlim=xlim, ylim=ylim, linthresh=lt)
        stats = rank_correlation(cross)
        ax.set_title(f"{label}\n{n} features leaky and relied upon   "
                     f"($\\rho$ = {stats['spearman_rho']:.3f})", fontsize=11)
        ax.set_xlabel("Leakage contribution (drop in probe AUC)")

    for r in range(nrows):
        axes[r * ncols].set_ylabel(
            f"SHAP importance (mean |SHAP|){f' — {model_kind}' if model_kind else ''}")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)

    sup = "Leakage vs model reliance, before and after mitigation"
    if dataset or attribute:
        sup += f" — {dataset} / {attribute}"
    fig.suptitle(sup, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def default_thresholds(cross: pd.DataFrame, k: float = 2.0) -> tuple[float, float]:
    """Quadrant cut-points: median SHAP, and leakage at the permutation noise floor.

    The SHAP axis splits at its median, which is a real value.

    The leakage axis cannot. Its distribution is zero-inflated — most features
    carry no protected-attribute signal whatsoever, so their permutation drop
    is zero or slightly negative — and the median consequently sits at or below
    zero. Splitting there classifies positive noise as leakage, and, worse,
    does so more often after mitigation than before: mitigation collapses the
    large leakage values into the noise band, where roughly half of them fall
    on the positive side by chance. The quadrant count then RISES while leakage
    genuinely falls.

    The cut-point is therefore raised to k standard deviations of the
    permutation estimate itself, below which a drop is not distinguishable from
    the noise of the permutation.
    """
    floor = k * float(cross["leakage_std"].median()) if "leakage_std" in cross else 0.0
    lx = max(float(cross["leakage_drop"].median()), floor)
    return lx, float(cross["shap_importance"].median())


def residual_leakage(cross: pd.DataFrame, shap_threshold: float) -> dict:
    """Leakage still carried by the features the model actually relies on.

    Free of any cut-point on the leakage axis, so unlike a quadrant count it
    does not inherit the instability of that boundary. This is the measure to
    quote when asking whether mitigation reduced the problem rather than merely
    moved features across a line.
    """
    hi = cross[cross["shap_importance"] > shap_threshold]
    pos = hi["leakage_drop"].clip(lower=0)
    return {
        "n_relied_on": int(len(hi)),
        "max_leakage_relied_on": float(hi["leakage_drop"].max()) if len(hi) else float("nan"),
        "mean_leakage_relied_on": float(hi["leakage_drop"].mean()) if len(hi) else float("nan"),
        "total_leakage_relied_on": float(pos.sum()),
    }


def leakage_vs_shap_quadrant(cross: pd.DataFrame,
                             thresholds: tuple[float, float]) -> pd.Series:
    """Reclassify an existing cross frame against different cut-points."""
    high_leak = cross["leakage_drop"] > thresholds[0]
    high_shap = cross["shap_importance"] > thresholds[1]
    return pd.Series(np.select(
        [high_leak & high_shap, high_leak & ~high_shap, ~high_leak & high_shap],
        ["leaky_and_relied_on", "leaky_removable", "safe_workhorse"],
        default="low_low"), index=cross.index)
