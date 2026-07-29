"""Module 3 — Mitigation Bank.

Four established methods behind one interface, so the evaluation harness and
the ablation study stay trivial.

    Reweighing                 pre-processing   Kamiran & Calders (2012)
    Disparate Impact Remover   pre-processing   Feldman et al. (2015)
    Adversarial Debiasing      in-processing    Zhang et al. (2018)
    Reject-Option Classification post-processing
"""
from abc import ABC, abstractmethod


class Mitigator(ABC):
    """Common interface. Every method implements fit/predict identically."""

    @abstractmethod
    def fit(self, X, y, A=None):
        ...

    @abstractmethod
    def predict_proba(self, X, A=None):
        ...


class NoMitigation(Mitigator):
    """Control condition."""

    def fit(self, X, y, A=None):
        raise NotImplementedError("Week 4")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 4")


class Reweighing(Mitigator):
    def fit(self, X, y, A=None):
        raise NotImplementedError("Week 4")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 4")


class DisparateImpactRemover(Mitigator):
    def fit(self, X, y, A=None):
        raise NotImplementedError("Week 4")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 4")


class AdversarialDebiasing(Mitigator):
    def fit(self, X, y, A=None):
        raise NotImplementedError("Week 4")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 4")


class RejectOptionClassification(Mitigator):
    def fit(self, X, y, A=None):
        raise NotImplementedError("Week 4")

    def predict_proba(self, X, A=None):
        raise NotImplementedError("Week 4")
