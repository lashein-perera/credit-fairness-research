"""Module 4 — Proxy-Aware Mitigator. THE NOVEL CONTRIBUTION.

Design requirement: must work WITHOUT the protected attribute at inference time.
A deployed lender scoring an unbanked applicant has no ethnicity field; a method
that needs one at scoring time solves nothing.

Algorithm
---------
    leakage = probe_AUC(X -> A)
    while leakage > tau and accuracy_loss < delta_max:
        rank features by leakage contribution
        select top-k untreated leaky features
        apply targeted decorrelation to those features ONLY
        retrain credit model, measure AUC
        recompute leakage
    return fitted transformer + credit model

The transformer is fitted once and applied at inference — no protected
attribute required downstream.
"""
from .bank import Mitigator
from .. import config


class ProxyAwareMitigator(Mitigator):
    """Iterative, targeted proxy suppression.

    Parameters
    ----------
    top_k : features treated per iteration
    tau : target probe AUC (stop when leakage falls below this)
    max_accuracy_loss : tolerance in AUC before stopping
    strategy : 'residualisation' | 'selective_repair' | 'suppression'
    """

    def __init__(
        self,
        top_k: int = config.DEFAULT_TOP_K,
        tau: float = config.DEFAULT_TAU,
        max_accuracy_loss: float = config.DEFAULT_MAX_ACCURACY_LOSS,
        strategy: str = "residualisation",
    ):
        self.top_k = top_k
        self.tau = tau
        self.max_accuracy_loss = max_accuracy_loss
        self.strategy = strategy
        self.history_ = []

    def fit(self, X, y, A=None):
        """A is used during FITTING/AUDIT only, never at inference."""
        raise NotImplementedError("Week 5")

    def transform(self, X):
        """Apply the fitted decorrelation. No protected attribute needed."""
        raise NotImplementedError("Week 5")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 5")


def residualise(feature, predicted_attribute):
    """Regress a leaky feature on the predicted protected attribute, keep the residual."""
    raise NotImplementedError("Week 5")


def selective_repair(feature, attribute, repair_strength: float = 1.0):
    """Feldman-style distribution repair applied to selected features only."""
    raise NotImplementedError("Week 5")
