"""Module 1 — Proxy Leakage Detector.

If a probe classifier can predict a protected attribute from the remaining
features, those features carry the protected information, and any credit model
built on them can discriminate through them.

Operationalises the predictability test of Feldman et al. (2015).
"""
import numpy as np
import pandas as pd


def probe_auc(X: pd.DataFrame, a: pd.Series, model: str = "logistic") -> float:
    """Cross-validated AUC for predicting protected attribute `a` from `X`.

    ~0.50 no leakage | >0.65 meaningful | >0.80 severe
    """
    raise NotImplementedError("Week 3")


def rank_leaky_features(X: pd.DataFrame, a: pd.Series, top_k: int = 15) -> pd.DataFrame:
    """Rank features by leakage contribution (mutual information + permutation importance)."""
    raise NotImplementedError("Week 3")


def shuffle_test(X: pd.DataFrame, a: pd.Series) -> float:
    """Sanity check: with `a` shuffled, probe AUC must collapse to ~0.5.

    If it does not, there is a bug and the leakage result cannot be defended.
    """
    raise NotImplementedError("Week 3")
