"""Accuracy, fairness and leakage metrics. Cite the originator of each in Chapter 3."""


# ---- Accuracy ----
def auc_roc(y_true, y_score): raise NotImplementedError("Week 2")
def ks_statistic(y_true, y_score): raise NotImplementedError("Week 2")
def brier_score(y_true, y_score): raise NotImplementedError("Week 2")


# ---- Fairness ----
def demographic_parity_difference(y_pred, a): raise NotImplementedError("Week 4")
def equalized_odds_difference(y_true, y_pred, a): raise NotImplementedError("Week 4")
def disparate_impact_ratio(y_pred, a): raise NotImplementedError("Week 4")


# ---- Leakage ----
def normalised_mutual_information(feature, a): raise NotImplementedError("Week 3")
