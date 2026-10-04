"""Module 3 — Mitigation Bank.

Four established bias-mitigation methods behind one common interface, plus an
unmitigated control. The uniform interface is a methodological decision rather
than a convenience: it ensures the evaluation harness treats the proposed
method and the established methods identically, removing a source of
experimental bias.

Implemented directly from the source papers rather than called from a library,
so that behaviour is transparent and reproducible without external fairness
dependencies.

    NoMitigation                 control
    Reweighing                   pre-processing    Kamiran & Calders (2012)
    DisparateImpactRemover       pre-processing    Feldman et al. (2015)
    AdversarialDebiasing         in-processing     Zhang et al. (2018)
    RejectOptionClassification   post-processing   Kamiran et al. (2012)

THE LATENT CONDITION
--------------------
Every method below requires the protected attribute A. That dependency is the
object of study. Under the latent condition A is not supplied, and each method
must fall back to whatever a real lender could do without it — which for most
means behaving as the unmitigated control. Recording that fallback honestly IS
the experiment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import clone

from models.baselines import build_model

RANDOM_STATE = 42


class Mitigator(ABC):
    """Common interface. `A` is None under the latent condition."""

    name = "base"
    requires_attribute = True

    def __init__(self, model_kind: str = "gbm"):
        self.model_kind = model_kind
        self.model_ = None
        self.fell_back_ = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y, A=None):
        ...

    def predict_proba(self, X: pd.DataFrame, A=None) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def _fit_plain(self, X, y, sample_weight=None):
        self.model_ = build_model(self.model_kind, X)
        if sample_weight is not None:
            self.model_.fit(X, y, clf__sample_weight=sample_weight)
        else:
            self.model_.fit(X, y)
        return self


# ==========================================================================
class NoMitigation(Mitigator):
    """Control condition."""

    name = "none"
    requires_attribute = False

    def fit(self, X, y, A=None):
        return self._fit_plain(X, y)


# ==========================================================================
class Reweighing(Mitigator):
    """Kamiran & Calders (2012).

    Assigns each training instance a weight equal to the ratio of the expected
    joint probability of its (group, label) pair under independence to the
    observed joint probability. Instances in under-represented combinations
    receive higher weight, correcting the association between group and label
    without altering any feature value.
    """

    name = "reweighing"

    def fit(self, X, y, A=None):
        if A is None:
            self.fell_back_ = True
            return self._fit_plain(X, y)

        a = pd.Series(A).reset_index(drop=True).astype(str)
        y = pd.Series(y).reset_index(drop=True).astype(int)
        n = len(y)

        w = np.ones(n, dtype=float)
        for g in a.unique():
            for lab in (0, 1):
                m = (a == g) & (y == lab)
                n_gl = int(m.sum())
                if n_gl == 0:
                    continue
                expected = (a == g).sum() * (y == lab).sum() / n
                w[m.values] = expected / n_gl

        self.weights_ = w
        return self._fit_plain(X, y, sample_weight=w)


# ==========================================================================
class DisparateImpactRemover(Mitigator):
    """Feldman et al. (2015).

    Repairs numeric features by mapping each group's distribution toward the
    pooled median distribution, quantile by quantile. A repair level of 1.0
    makes the feature distribution identical across groups, removing the
    group's predictability from that feature while preserving within-group
    ordering — so rank information relevant to risk survives.
    """

    name = "disparate_impact_remover"

    def __init__(self, model_kind: str = "gbm", repair_level: float = 1.0):
        super().__init__(model_kind)
        self.repair_level = float(np.clip(repair_level, 0.0, 1.0))

    @staticmethod
    def _repair_column(col: pd.Series, a: pd.Series, level: float) -> pd.Series:
        out = col.copy().astype(float)
        groups = a.unique()
        if len(groups) < 2:
            return out

        grid = np.linspace(0, 1, 101)
        # median quantile function across groups = the repair target
        per_group_q = {}
        for g in groups:
            v = col[a == g].dropna()
            if len(v) == 0:
                return out
            per_group_q[g] = np.quantile(v, grid)
        target = np.median(np.vstack(list(per_group_q.values())), axis=0)

        for g in groups:
            m = (a == g) & col.notna()
            if m.sum() == 0:
                continue
            v = col[m].astype(float)
            # rank within group -> position on the grid -> target value
            ranks = v.rank(pct=True).clip(1e-6, 1 - 1e-6)
            repaired = np.interp(ranks, grid, target)
            out.loc[m] = (1 - level) * v + level * repaired
        return out

    def fit(self, X, y, A=None):
        if A is None:
            self.fell_back_ = True
            return self._fit_plain(X, y)

        a = pd.Series(A).reset_index(drop=True).astype(str)
        Xr = X.reset_index(drop=True).copy()
        num = Xr.select_dtypes(include=[np.number]).columns

        for c in num:
            Xr[c] = self._repair_column(Xr[c], a, self.repair_level)

        self.repaired_columns_ = list(num)
        self._fit_plain(Xr, y)
        self._a_for_transform = a
        return self

    def predict_proba(self, X, A=None):
        # repair is fitted per-group; at inference without A the raw features
        # are used, which is the honest deployment behaviour
        return self.model_.predict_proba(X)[:, 1]


# ==========================================================================
class AdversarialDebiasing(Mitigator):
    """Zhang, Lemoine & Mitchell (2018), simplified.

    A logistic predictor is trained on the outcome while an adversary attempts
    to recover the protected attribute from the predictor's output. The
    predictor's gradient has the adversary's gradient projected out and
    subtracted, so updates that would help the adversary are suppressed.

    Implemented directly rather than through a deep-learning framework: the
    mechanism is the projection step, which is preserved exactly here.

    GRADIENT SCALING
    The predictor update subtracts the adversary's gradient after projecting
    its component out of the predictor's own gradient. Subtracting it at a
    fixed weight, with no regard to the relative magnitude of the two
    gradients, lets the adversary term dominate whenever it is the larger:
    the predictor then never fits the outcome. Measured on Home Credit, that
    cost roughly 0.10 AUC, far more than the model family (0.006) or the step
    count (0.002), and left the method scoring worse WITH the protected
    attribute than without it.

    `scale_adversary` rescales the adversary term to the norm of the
    predictor gradient, so neither side dominates by accident. This plays the
    role Zhang et al.'s annealing schedule plays in the original. Set it to
    False to reproduce the frozen iteration-1 and iteration-2 results.
    """

    name = "adversarial_debiasing"

    def __init__(self, model_kind: str = "gbm", epochs: int = 120,
                 lr: float = 0.05, adversary_weight: float = 1.0,
                 scale_adversary: bool = True):
        super().__init__(model_kind)
        self.epochs = epochs
        self.lr = lr
        self.adversary_weight = adversary_weight
        self.scale_adversary = scale_adversary

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, A=None):
        if A is None:
            self.fell_back_ = True
            return self._fit_plain(X, y)

        from models.baselines import build_preprocessor
        self.prep_ = build_preprocessor(X, dense=True)
        Z = np.asarray(self.prep_.fit_transform(X), dtype=float)
        Z = np.hstack([Z, np.ones((len(Z), 1))])          # bias term

        y = np.asarray(pd.Series(y).astype(int))
        a_ser = pd.Series(A).astype(str)
        a = (a_ser == sorted(a_ser.unique())[0]).astype(int).values

        rng = np.random.default_rng(RANDOM_STATE)
        w = rng.normal(0, 0.01, Z.shape[1])       # predictor
        u = rng.normal(0, 0.01, 2)                # adversary: [slope, bias]

        for _ in range(self.epochs):
            p = self._sigmoid(Z @ w)
            # adversary predicts A from the predictor's logit
            logit = Z @ w
            q = self._sigmoid(u[0] * logit + u[1])

            # adversary update (ascends its own objective)
            g_u = np.array([np.mean((q - a) * logit), np.mean(q - a)])
            u -= self.lr * g_u

            # predictor gradients
            g_pred = Z.T @ (p - y) / len(y)
            g_adv = Z.T @ ((q - a) * u[0]) / len(y)

            # project out the adversary direction, then subtract it
            norm = np.linalg.norm(g_adv) + 1e-12
            proj = (g_pred @ g_adv) / (norm ** 2) * g_adv
            # rescale so the adversary term cannot swamp the predictor's own
            # gradient; without this the predictor never fits the outcome
            scale = (np.linalg.norm(g_pred) / norm) if self.scale_adversary else 1.0
            g = g_pred - proj - self.adversary_weight * scale * g_adv

            w -= self.lr * g

        self.w_ = w
        return self

    def predict_proba(self, X, A=None):
        if self.fell_back_:
            return self.model_.predict_proba(X)[:, 1]
        Z = np.asarray(self.prep_.transform(X), dtype=float)
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return self._sigmoid(Z @ self.w_)


# ==========================================================================
class RejectOptionClassification(Mitigator):
    """Kamiran, Karim & Zhang (2012).

    Leaves the model untouched and intervenes only on decisions falling within
    a band of uncertainty around the threshold. Inside that band, instances
    from the disadvantaged group are assigned the favourable outcome and those
    from the advantaged group the unfavourable one, on the reasoning that
    decisions the model is least certain about are where bias does most work
    and where correction costs least accuracy.

    THRESHOLD ALIGNMENT
    The critical region is a band around the classifier's DECISION BOUNDARY.
    The original implementation centred it on a fixed 0.5. On Home Credit,
    where the default rate is about 8%, roughly 1% of applicants score above
    0.5, while the evaluation decides at the base-rate quantile (about 0.18).
    The band therefore sat entirely among applicants already rejected, moved
    them from one rejected score to another, and changed no decision: every
    metric came out identical to no mitigation. Group advantage was also read
    off approval rates at 0.5, where both groups are about 99% approved.

    `align_threshold` centres the band on the base-rate quantile of the
    training scores, the same rule harness.resolve_threshold uses to decide.
    The band grid and the search are unchanged. Set it to False to reproduce
    the frozen results.
    """

    name = "reject_option"

    def __init__(self, model_kind: str = "gbm", band: float | None = None,
                 threshold: float = 0.5, align_threshold: bool = True):
        super().__init__(model_kind)
        self.band = band              # None => selected by search during fit
        self.threshold = threshold
        self.align_threshold = align_threshold
        self.threshold_ = threshold

    def _dpd(self, proba, a):
        approved = pd.Series(proba < self.threshold_).astype(int)
        a = pd.Series(a).reset_index(drop=True)
        r = [approved[(a == g).values].mean() for g in a.unique()]
        return max(r) - min(r)

    def fit(self, X, y, A=None):
        self._fit_plain(X, y)
        if A is None:
            self.fell_back_ = True
            return self

        a = pd.Series(A).reset_index(drop=True).astype(str)
        proba = self.model_.predict_proba(X)[:, 1]

        self.threshold_ = (float(np.quantile(proba, 1.0 - float(np.mean(y))))
                           if self.align_threshold else self.threshold)

        rates = {g: float((proba[(a == g).values] < self.threshold_).mean())
                 for g in a.unique()}
        self.disadvantaged_ = min(rates, key=rates.get)
        self.advantaged_ = max(rates, key=rates.get)

        if self.band is not None:
            self.band_ = self.band
            return self

        # Select the band width that minimises disparity on training data.
        # A fixed band overcorrects: pushing every uncertain case in both
        # directions can invert the disparity rather than remove it.
        best, best_dpd = 0.0, self._dpd(proba, a)
        for cand in np.linspace(0.01, 0.45, 45):
            self.band_ = cand
            d = self._dpd(self._apply_band(proba, a), a)
            if d < best_dpd:
                best, best_dpd = cand, d
        self.band_ = best
        return self

    def _apply_band(self, proba, a):
        out = proba.copy()
        in_band = np.abs(proba - self.threshold_) <= self.band_
        fav = in_band & (a == self.disadvantaged_).values
        unf = in_band & (a == self.advantaged_).values
        out[fav] = self.threshold_ - self.band_ - 1e-3
        out[unf] = self.threshold_ + self.band_ + 1e-3
        return out

    def predict_proba(self, X, A=None):
        proba = self.model_.predict_proba(X)[:, 1]
        if self.fell_back_ or A is None or self.band_ <= 0:
            return proba
        a = pd.Series(A).reset_index(drop=True).astype(str)
        return self._apply_band(proba, a)


# ==========================================================================
BANK = {
    "none": NoMitigation,
    "reweighing": Reweighing,
    "disparate_impact_remover": DisparateImpactRemover,
    "adversarial_debiasing": AdversarialDebiasing,
    "reject_option": RejectOptionClassification,
}


def build(name: str, model_kind: str = "gbm", **kw) -> Mitigator:
    if name not in BANK:
        raise ValueError(f"unknown method '{name}'. options: {list(BANK)}")
    return BANK[name](model_kind=model_kind, **kw)
