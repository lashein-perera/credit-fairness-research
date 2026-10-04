"""Module 6 — Evaluation Harness.

Executes the experimental grid and returns one standardised row per
configuration, so that every method, dataset and condition is directly
comparable.

    dataset | method | condition | fold | AUC | KS | Brier |
    DP_diff | DI_ratio | EO_diff | EOpp_diff | probe_AUC_after | fell_back | runtime

CONDITIONS
    explicit  protected attribute available during fitting and mitigation
    latent    attribute withheld — the deployment condition

The proposed method occupies a third position: it uses the attribute during
FITTING to locate leaky features, but never at inference. It is therefore
evaluated under 'latent' alongside the others, since what is being compared is
what each method can achieve when a lender cannot supply the attribute at
scoring time.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from evaluation.metrics import (demographic_parity_difference,
                                disparate_impact_ratio,
                                equal_opportunity_difference,
                                equalized_odds_difference, auc_roc,
                                brier_score, ks_statistic)
from mitigation.bank import build as build_mitigator
from mitigation.proxy_aware import ProxyAwareMitigator
from models.baselines import build_preprocessor

RANDOM_STATE = 42

METHODS = ["none", "reweighing", "disparate_impact_remover",
           "adversarial_debiasing", "reject_option", "proxy_aware"]


def _probe(X: pd.DataFrame, a: pd.Series) -> float:
    """Residual leakage: how recoverable is the attribute from these features."""
    a_str = pd.Series(a).astype(str).reset_index(drop=True)
    if a_str.nunique() < 2:
        return float("nan")
    ab = (a_str == sorted(a_str.unique())[0]).astype(int).values
    Z = build_preprocessor(X, dense=True).fit_transform(X)
    try:
        Ztr, Zte, atr, ate = train_test_split(
            Z, ab, test_size=0.3, stratify=ab, random_state=RANDOM_STATE)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(Ztr, atr)
        return float(roc_auc_score(ate, clf.predict_proba(Zte)[:, 1]))
    except Exception:
        return float("nan")


# ITERATION 2: the configuration chosen on the validation split, registered so
# it can be evaluated under the same folds and subsample as every other method.
PROXY_AWARE_V2 = dict(skip_dominant_share=0.99, max_accuracy_loss=0.02)


def _make(method: str, model_kind: str):
    if method == "proxy_aware":
        return ProxyAwareMitigator(model_kind=model_kind)
    if method == "proxy_aware_v2":
        return ProxyAwareMitigator(model_kind=model_kind, **PROXY_AWARE_V2)
    if method.startswith("proxy_aware_b"):
        # trade-off sweep: the chosen guard, varying only the accuracy budget.
        # 'none' removes the budget rather than setting it to zero.
        tok = method[len("proxy_aware_b"):]
        kw = dict(PROXY_AWARE_V2)
        kw["max_accuracy_loss"] = 1e9 if tok == "none" else float(tok)
        return ProxyAwareMitigator(model_kind=model_kind, **kw)
    return build_mitigator(method, model_kind=model_kind)


def resolve_threshold(proba, y_train, threshold) -> float:
    """Choose a decision threshold.

    A fixed 0.5 is degenerate on imbalanced credit data: with an 8% default
    rate almost no applicant exceeds it, every applicant is approved, and
    every group-fairness metric reads as perfect parity while telling you
    nothing. Defaulting to the base rate makes the predicted-positive rate
    match the observed default rate, which is both realistic and produces
    metrics that discriminate between methods.
    """
    if threshold is not None:
        return float(threshold)
    rate = float(np.mean(y_train))
    return float(np.quantile(proba, 1.0 - rate))


def run_fold(X, y, A, train_idx, test_idx, method: str, condition: str,
             model_kind: str = "gbm", threshold: float | None = None,
             measure_leakage: bool = True) -> dict:
    """Evaluate one method on one fold under one condition."""
    t0 = time.time()

    Xtr, Xte = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    ytr, yte = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
    Atr, Ate = A.iloc[train_idx].reset_index(drop=True), A.iloc[test_idx].reset_index(drop=True)

    # The proposed method uses the attribute at FIT time by design; the
    # established methods are denied it under the latent condition.
    #
    # ITERATION 2 adds 'training_only': every method receives the attribute at
    # fitting and none receives it at inference. That is exactly the access the
    # proposed method has always had, so 'latent' compared a method holding the
    # attribute against four holding nothing. Under this condition reweighing
    # and adversarial debiasing are fully functional, since neither consults the
    # attribute when scoring; reject-option is inert because it adjusts
    # decisions at inference, and disparate impact remover scores raw features
    # with a model fitted on repaired ones.
    if method.startswith("proxy_aware"):
        fit_A = Atr
    elif condition in ("explicit", "training_only"):
        fit_A = Atr
    else:
        fit_A = None

    m = _make(method, model_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr, fit_A)
        # attribute at inference: only the explicit condition may use it,
        # and never for the proposed method
        infer_A = (Ate if (condition == "explicit"
                           and not method.startswith("proxy_aware")) else None)
        proba = m.predict_proba(Xte, infer_A)

    thr = resolve_threshold(proba, ytr, threshold)
    pred = (np.asarray(proba) >= thr).astype(int)

    leak = np.nan
    if measure_leakage:
        Xte_t = (m.transform(Xte) if hasattr(m, "transform")
                 and method.startswith("proxy_aware") else Xte)
        leak = _probe(Xte_t, Ate)

    return {
        "method": method,
        "condition": condition,
        "auc": auc_roc(yte, proba),
        "ks": ks_statistic(yte, proba),
        "brier": brier_score(yte, proba),
        "dp_diff": demographic_parity_difference(pred, Ate),
        "di_ratio": disparate_impact_ratio(pred, Ate),
        "eo_diff": equalized_odds_difference(yte, pred, Ate),
        "eopp_diff": equal_opportunity_difference(yte, pred, Ate),
        "probe_auc_after": leak,
        "threshold": round(thr, 4),
        "fell_back": bool(getattr(m, "fell_back_", False)),
        "runtime_s": round(time.time() - t0, 2),
    }


def run_grid(X, y, A, attribute: str, dataset: str,
             methods=None, conditions=("explicit", "latent"),
             n_splits: int = 5, n_repeats: int = 2,
             model_kind: str = "gbm", threshold: float | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Run the full experimental grid with repeated stratified cross-validation."""
    methods = methods or METHODS
    y = pd.Series(y).reset_index(drop=True)
    A = pd.Series(A).reset_index(drop=True)
    X = X.reset_index(drop=True)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=RANDOM_STATE)
    folds = list(cv.split(X, y))
    total = len(folds) * len(methods) * len(conditions)

    if verbose:
        print(f"grid: {len(methods)} methods x {len(conditions)} conditions "
              f"x {len(folds)} folds = {total} runs")

    rows, done = [], 0
    for fold_i, (tr, te) in enumerate(folds):
        for method in methods:
            for cond in conditions:
                # the proposed method is attribute-free at inference, so the
                # explicit/latent distinction does not apply to it
                if method.startswith("proxy_aware") and cond == "explicit":
                    continue
                r = run_fold(X, y, A, tr, te, method, cond,
                             model_kind=model_kind, threshold=threshold)
                r.update({"dataset": dataset, "attribute": attribute,
                          "fold": fold_i})
                rows.append(r)
                done += 1
                if verbose:
                    print(f"  [{done:>3}/{total}] {method:26s} {cond:8s} "
                          f"AUC {r['auc']:.4f}  EO {r['eo_diff']:.4f}  "
                          f"leak {r['probe_auc_after']:.3f}")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation per method and condition."""
    metrics = ["auc", "ks", "brier", "dp_diff", "di_ratio",
               "eo_diff", "eopp_diff", "probe_auc_after"]
    g = df.groupby(["dataset", "attribute", "method", "condition"])
    out = g[metrics].agg(["mean", "std"]).round(4)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def compare(df: pd.DataFrame, metric: str = "eo_diff",
            reference: str = "proxy_aware",
            condition: str = "latent") -> pd.DataFrame:
    """Paired Wilcoxon comparison of every method against the reference."""
    from evaluation.metrics import wilcoxon_compare

    sub = df[df.condition == condition]
    ref = sub[sub.method == reference].sort_values("fold")[metric].values
    rows = []
    for m in sub.method.unique():
        if m == reference:
            continue
        other = sub[sub.method == m].sort_values("fold")[metric].values
        n = min(len(ref), len(other))
        if n < 3:
            continue
        try:
            res = wilcoxon_compare(ref[:n], other[:n])
        except Exception:
            continue
        rows.append({"metric": metric, "reference": reference, "against": m,
                     "ref_median": float(np.median(ref[:n])),
                     "other_median": float(np.median(other[:n])),
                     **{k: res[k] for k in
                        ("p_value", "effect_size", "effect_magnitude",
                         "significant")}})
    return pd.DataFrame(rows)
