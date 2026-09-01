"""Fairness, accuracy and leakage metrics.

MEASUREMENT DESIGN — the point a panel will probe
--------------------------------------------------
Standard fairness metrics require the protected attribute. This research
concerns the setting where a lender does not have it. The two are reconciled by
separating the roles the attribute plays:

    TRAINING / MITIGATION : attribute withheld  (the deployment condition)
    EVALUATION            : attribute supplied  (the audit condition)

The attribute is ground truth held by the researcher, exactly as a laboratory
holds a reference value the instrument under test cannot see. A lender scoring
an unbanked applicant has no gender field; a regulator or researcher auditing
that lender's decisions afterwards does.

FAVOURABLE OUTCOME
------------------
In these datasets the target encodes DEFAULT (1 = defaulted). The favourable
decision for an applicant is therefore a prediction of NON-default, i.e. loan
approval. All group-fairness metrics below are computed on the approval rate,
not the default-prediction rate. Getting this backwards inverts every result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


# ==========================================================================
# ACCURACY
# ==========================================================================
def auc_roc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: maximum separation between the two score
    distributions. Standard in credit scoring practice."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def brier_score(y_true, y_score) -> float:
    """Calibration. Lower is better."""
    return float(brier_score_loss(y_true, y_score))


# ==========================================================================
# GROUP FAIRNESS
# ==========================================================================
def _approval_rates(y_pred, a) -> dict:
    """P(approved | group) for each group. Approved == predicted non-default."""
    a = pd.Series(a).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    approved = (y_pred == 0).astype(int)
    return {g: float(approved[a == g].mean()) for g in a.dropna().unique()}


def demographic_parity_difference(y_pred, a) -> float:
    """Largest gap in approval rate between groups. 0 = parity.

    Kamiran & Calders (2012); Feldman et al. (2015).
    """
    r = _approval_rates(y_pred, a)
    return float(max(r.values()) - min(r.values()))


def disparate_impact_ratio(y_pred, a) -> float:
    """Ratio of lowest to highest approval rate. 1 = parity.

    Below 0.80 fails the four-fifths rule used in US fair-lending practice —
    the threshold that makes this metric legally legible.
    """
    r = _approval_rates(y_pred, a)
    hi = max(r.values())
    return float(min(r.values()) / hi) if hi > 0 else 0.0


def equalized_odds_difference(y_true, y_pred, a) -> float:
    """Larger of the TPR gap and the FPR gap across groups. 0 = parity.

    Hardt, Price & Srebro (2016). Stricter than demographic parity because it
    conditions on the true outcome: it permits different approval rates when
    genuine risk differs, but not different ERROR rates.
    """
    a = pd.Series(a).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    tprs, fprs = [], []
    for g in a.dropna().unique():
        m = (a == g)
        yt, yp = y_true[m], y_pred[m]
        pos, neg = (yt == 1), (yt == 0)
        if pos.sum() > 0:
            tprs.append(float((yp[pos] == 1).mean()))
        if neg.sum() > 0:
            fprs.append(float((yp[neg] == 1).mean()))

    gap_tpr = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0
    gap_fpr = max(fprs) - min(fprs) if len(fprs) > 1 else 0.0
    return float(max(gap_tpr, gap_fpr))


def equal_opportunity_difference(y_true, y_pred, a) -> float:
    """TPR gap only — equal detection of genuine defaulters across groups."""
    a = pd.Series(a).reset_index(drop=True)
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    tprs = []
    for g in a.dropna().unique():
        m = (a == g) & (y_true == 1)
        if m.sum() > 0:
            tprs.append(float((y_pred[m] == 1).mean()))
    return float(max(tprs) - min(tprs)) if len(tprs) > 1 else 0.0


# ==========================================================================
# LEAKAGE — the metric distinctive to this research
# ==========================================================================
def residual_leakage(X_transformed, a, probe: str = "gbm") -> float:
    """Probe AUC measured AFTER mitigation.

    Group-fairness metrics assess OUTCOMES. This assesses INFORMATION. A model
    can equalise approval rates while the protected attribute remains fully
    recoverable from its features — the disparity is suppressed at the decision
    boundary but the encoding survives, and will resurface under distribution
    shift or when the model is retrained.

    Reporting outcome fairness and residual leakage together is the measurement
    contribution of this research.
    """
    from leakage.probe import probe_auc
    return probe_auc(X_transformed, a, kind=probe)["auc_mean"]


# ==========================================================================
# COMPARISON — is method A fairer than method B?
# ==========================================================================
def wilcoxon_compare(scores_a, scores_b, alternative: str = "two-sided") -> dict:
    """Paired Wilcoxon signed-rank test over matched CV folds.

    Non-parametric, so it makes no normality assumption. Paired because both
    methods are evaluated on identical folds, which removes fold difficulty as
    a confound.

    Reports rank-biserial correlation as effect size. A significant p-value
    with a negligible effect is a real and reportable outcome: statistical
    significance is not practical significance.
    """
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    if len(a) != len(b):
        raise ValueError("paired test requires equal-length inputs")

    diff = a - b
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "effect_size": 0.0,
                "effect_magnitude": "negligible", "median_difference": 0.0,
                "n_pairs": len(a), "significant": False}

    stat, p = stats.wilcoxon(a, b, alternative=alternative)
    n = len(nonzero)
    # rank-biserial correlation: scipy returns the smaller signed-rank sum,
    # and the two sums total n(n+1)/2
    total = n * (n + 1) / 2
    r = float(1 - (2 * stat) / total)

    mag = abs(r)
    size = ("negligible" if mag < 0.1 else "small" if mag < 0.3
            else "medium" if mag < 0.5 else "large")

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "effect_size": r,
        "effect_magnitude": size,
        "median_difference": float(np.median(diff)),
        "n_pairs": int(len(a)),
        "significant": bool(p < 0.05),
    }


def pareto_dominates(acc_a, fair_a, acc_b, fair_b,
                     fairness_lower_is_better: bool = True) -> bool:
    """Does configuration A dominate B on both accuracy and fairness?

    Necessary because fairness alone is trivially gameable — a random
    classifier is perfectly fair. A defensible claim of improvement requires
    no loss on the other axis.
    """
    fa = -fair_a if fairness_lower_is_better else fair_a
    fb = -fair_b if fairness_lower_is_better else fair_b
    return (acc_a >= acc_b and fa >= fb) and (acc_a > acc_b or fa > fb)


def fairness_accuracy_tradeoff(baseline_acc, baseline_fair,
                               method_acc, method_fair) -> dict:
    """Fairness gained per unit of accuracy surrendered.

    The single number that answers 'is it worth it'. A method that halves the
    disparity for a 0.005 AUC cost is a different proposition from one that
    halves it for 0.05.
    """
    d_acc = baseline_acc - method_acc          # positive = accuracy lost
    d_fair = baseline_fair - method_fair       # positive = disparity reduced
    ratio = float(d_fair / d_acc) if abs(d_acc) > 1e-9 else float("inf")
    return {
        "accuracy_cost": float(d_acc),
        "fairness_gain": float(d_fair),
        "gain_per_unit_cost": ratio,
        "free_improvement": bool(d_acc <= 0 and d_fair > 0),
    }


# ==========================================================================
# one-call evaluation
# ==========================================================================
def evaluate(y_true, y_score, a, threshold: float = 0.5,
             X_transformed=None) -> dict:
    """Every metric for one configuration, as a single result row."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    out = {
        "auc": auc_roc(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "brier": brier_score(y_true, y_score),
        "demographic_parity_difference": demographic_parity_difference(y_pred, a),
        "disparate_impact_ratio": disparate_impact_ratio(y_pred, a),
        "equalized_odds_difference": equalized_odds_difference(y_true, y_pred, a),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, a),
    }
    out["passes_four_fifths_rule"] = bool(out["disparate_impact_ratio"] >= 0.80)
    if X_transformed is not None:
        out["residual_leakage"] = residual_leakage(X_transformed, a)
    return out
