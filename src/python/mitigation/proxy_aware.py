"""Module 4 — Proxy-Aware Mitigator. THE RESEARCH CONTRIBUTION.

Every method in the mitigation bank requires the protected attribute at
fitting time, and most require it again at inference. In alternative-data
lending to thin-file borrowers the attribute is not collected, so those
methods fall back to no mitigation at all.

This component is designed for that condition. Two properties distinguish it:

    TARGETED   Mitigation is applied only to the features that measurably
               carry the protected attribute, rather than globally across
               the feature space. Features that leak nothing are left
               untouched, so their predictive signal survives intact.

    ATTRIBUTE-FREE AT INFERENCE
               The protected attribute is used during FITTING to locate the
               leaky features and learn the transformation. Once fitted, the
               transformer is a fixed function of the features alone. A
               lender scoring an applicant with no demographic record can
               apply it.

ALGORITHM
---------
    leakage <- probe_AUC(X -> A)
    while leakage > tau and accuracy_loss < delta:
        rank features by leakage contribution
        select top-k untreated features
        apply targeted decorrelation to those features only
        retrain, measure accuracy, recompute leakage
    return fitted transformer + model
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mitigation.bank import Mitigator, Reweighing
from models.baselines import build_model, build_preprocessor

RANDOM_STATE = 42


def _numeric_view(col: pd.Series) -> tuple[pd.Series, dict | None]:
    """Return a numeric version of any column.

    Categorical features are the strongest proxies in practice — occupation
    encodes gender through occupational segregation — so they must be
    treatable, not skipped. Categories are ordinal-coded by frequency, which
    preserves an ordering the transformation can then operate on.
    """
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_numeric(col, errors="coerce"), None
    codes = col.astype(str)
    order = codes.value_counts().index.tolist()
    mapping = {c: i for i, c in enumerate(order)}
    return codes.map(mapping).astype(float), mapping


def degenerate_features(X: pd.DataFrame, max_dominant_share: float | None = None,
                        min_unique: int | None = None) -> list[str]:
    """Features the conditional repair must not touch. ITERATION 2.

    Iteration 1 treated every selected feature regardless of its distribution,
    and that is where the method manufactured proxies. Percentile-mapping a
    feature within strata of a PREDICTED attribute presupposes the feature has
    a distribution worth repairing. A near-constant column does not: almost all
    of its mass sits on one value, so the percentile a row receives is decided
    by which stratum it fell in rather than by its own value, and the mapping
    writes stratum identity — the predicted attribute — into the column.

    FLAG_CONT_MOBILE is the canonical case: 99.8% constant, two distinct values
    and no measurable leakage before treatment, 30-41 distinct values and the
    largest residual leakage of any feature after it.

    Thresholds are calibrated so that no known proxy is excluded: the lowest
    dominant-value share among features that genuinely carry the attribute is
    NAME_EDUCATION_TYPE at 0.71, well clear of a 0.99 cut.
    """
    out = []
    for c in X.columns:
        v = X[c].dropna()
        if v.empty:
            out.append(c)
            continue
        if min_unique is not None and v.nunique() < min_unique:
            out.append(c)
            continue
        if (max_dominant_share is not None
                and v.value_counts(normalize=True).iloc[0] > max_dominant_share):
            out.append(c)
    return out


# ==========================================================================
# decorrelation strategies
# ==========================================================================
class _ConditionalRepair:
    """Repair each feature's distribution across strata of the PREDICTED attribute.

    This is the strategy that resolves the central difficulty. Distribution
    repair (Feldman et al., 2015) equalises a feature across groups and so
    removes the group's recoverability from it — but it needs group membership
    at transform time, which a lender does not have.

    The resolution is to stratify on an attribute PREDICTION learned from the
    features themselves. Because that predictor is part of the fitted
    transformer, it is available at inference without any protected attribute.
    Within each stratum the feature is mapped, by its percentile in the stored
    training distribution for that stratum, onto a pooled target distribution.
    Equalising the conditional distributions removes the stratum's — and hence
    the attribute's — signal from the feature.

    Percentiles are computed against stored training values rather than the
    incoming batch, so a single applicant can be transformed in isolation.
    """

    def __init__(self, n_bins: int = 4, repair_level: float = 1.0):
        self.n_bins = n_bins
        self.repair_level = repair_level
        self.grid_ = np.linspace(0, 1, 101)
        self.attr_model_ = None
        self.prep_ = None
        self.bin_edges_ = None
        self.sorted_by_bin_ = {}     # (feature, bin) -> sorted training values
        self.targets_ = {}           # feature -> pooled target quantiles
        self.maps_ = {}
        self.medians_ = {}

    def _predict_attr(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.prep_.transform(X[self.other_cols_])
        return self.attr_model_.predict_proba(Z)[:, 1]

    def fit(self, X: pd.DataFrame, a: pd.Series, features: list[str]):
        others = [c for c in X.columns if c not in features] or list(X.columns)
        self.other_cols_ = others
        self.prep_ = build_preprocessor(X[others], dense=True)
        Z = self.prep_.fit_transform(X[others])

        a_str = pd.Series(a).astype(str).reset_index(drop=True)
        a_bin = (a_str == sorted(a_str.unique())[0]).astype(int)
        self.attr_model_ = LogisticRegression(max_iter=1000,
                                              random_state=RANDOM_STATE)
        self.attr_model_.fit(Z, a_bin)

        a_hat = self.attr_model_.predict_proba(Z)[:, 1]
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.bin_edges_ = np.quantile(a_hat, qs) if len(qs) else np.array([])
        bins = np.digitize(a_hat, self.bin_edges_)

        for f in features:
            col, mp = _numeric_view(X[f])
            self.maps_[f] = mp
            med = float(col.median()) if col.notna().any() else 0.0
            self.medians_[f] = med
            col = col.fillna(med).reset_index(drop=True)

            per_bin = []
            for b in range(self.n_bins):
                v = col[bins == b]
                if len(v) < 5:
                    continue
                sv = np.sort(v.values)
                self.sorted_by_bin_[(f, b)] = sv
                per_bin.append(np.quantile(sv, self.grid_))
            if per_bin:
                self.targets_[f] = np.median(np.vstack(per_bin), axis=0)
        return self

    def transform(self, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = X.copy()
        a_hat = self._predict_attr(X)
        bins = np.digitize(a_hat, self.bin_edges_)

        for f in features:
            if f not in self.targets_:
                continue
            mp = self.maps_.get(f)
            col = (pd.to_numeric(X[f], errors="coerce") if mp is None
                   else X[f].astype(str).map(mp).astype(float))
            col = col.fillna(self.medians_[f]).reset_index(drop=True).values

            new = np.empty(len(col), dtype=float)
            for i, (v, b) in enumerate(zip(col, bins)):
                sv = self.sorted_by_bin_.get((f, int(b)))
                if sv is None or len(sv) == 0:
                    new[i] = v
                    continue
                pct = np.searchsorted(sv, v, side="left") / max(len(sv) - 1, 1)
                new[i] = np.interp(np.clip(pct, 0, 1), self.grid_,
                                   self.targets_[f])
            # ITERATION 2: partial repair. A full remap discards the feature's
            # own values entirely and replaces them with positions in a pooled
            # target distribution, which is what lets stratum information in.
            # Blending back the original limits how far any single pass can
            # move a value, and the effect still compounds across iterations.
            lam = self.repair_level
            out[f] = new if lam >= 1.0 else (1.0 - lam) * col + lam * new
        return out


class _Residualiser:
    """Regress a feature on the PREDICTED protected attribute, keep the residual.

    The prediction is produced from the other features, so at inference no
    protected attribute is required — the predictor is part of the fitted
    transformer.
    """

    def __init__(self):
        self.attr_model_ = None
        self.residual_models_ = {}
        self.prep_ = None
        self.medians_ = {}

    def fit(self, X: pd.DataFrame, a: pd.Series, features: list[str]):
        # model that predicts the protected attribute from all OTHER features
        others = [c for c in X.columns if c not in features]
        if not others:
            others = list(X.columns)
        self.prep_ = build_preprocessor(X[others], dense=True)
        Z = self.prep_.fit_transform(X[others])

        a_bin = (pd.Series(a).astype(str) ==
                 sorted(pd.Series(a).astype(str).unique())[0]).astype(int)
        self.attr_model_ = LogisticRegression(max_iter=1000,
                                              random_state=RANDOM_STATE)
        self.attr_model_.fit(Z, a_bin)
        self.other_cols_ = others

        a_hat = self.attr_model_.predict_proba(Z)[:, 1].reshape(-1, 1)
        self.maps_ = {}
        for f in features:
            col, mp = _numeric_view(X[f])
            self.maps_[f] = mp
            med = float(col.median()) if col.notna().any() else 0.0
            self.medians_[f] = med
            v = col.fillna(med).values.reshape(-1, 1)
            lr = LinearRegression().fit(a_hat, v)
            self.residual_models_[f] = lr
        return self

    def transform(self, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        out = X.copy()
        Z = self.prep_.transform(X[self.other_cols_])
        a_hat = self.attr_model_.predict_proba(Z)[:, 1].reshape(-1, 1)
        for f in features:
            if f not in self.residual_models_:
                continue
            mp = self.maps_.get(f)
            if mp is None:
                col = pd.to_numeric(X[f], errors="coerce")
            else:
                col = X[f].astype(str).map(mp).astype(float)
            col = col.fillna(self.medians_[f])
            pred = self.residual_models_[f].predict(a_hat).ravel()
            out[f] = col.values - pred
        return out


class _Suppressor:
    """Replace the feature with a constant. Deliberately crude comparison."""

    def __init__(self):
        self.fill_ = {}

    def fit(self, X, a, features):
        for f in features:
            col, _ = _numeric_view(X[f])
            self.fill_[f] = float(col.median()) if col.notna().any() else 0.0
        return self

    def transform(self, X, features):
        out = X.copy()
        for f in features:
            if f in self.fill_:
                out[f] = self.fill_[f]
        return out


class _SelectiveRepair:
    """Feldman-style distribution repair applied to the selected features only."""

    def __init__(self):
        self.targets_ = {}
        self.grid_ = np.linspace(0, 1, 101)

    def fit(self, X, a, features):
        a = pd.Series(a).astype(str).reset_index(drop=True)
        self.maps_ = {}
        for f in features:
            col, mp = _numeric_view(X[f])
            self.maps_[f] = mp
            col = col.reset_index(drop=True)
            qs = []
            for g in a.unique():
                v = col[a == g].dropna()
                if len(v) == 0:
                    continue
                qs.append(np.quantile(v, self.grid_))
            if qs:
                self.targets_[f] = np.median(np.vstack(qs), axis=0)
        return self

    def transform(self, X, features):
        out = X.copy()
        for f in features:
            if f not in self.targets_:
                continue
            mp = self.maps_.get(f)
            col = (pd.to_numeric(X[f], errors="coerce") if mp is None
                   else X[f].astype(str).map(mp).astype(float))
            ranks = col.rank(pct=True).clip(1e-6, 1 - 1e-6).fillna(0.5)
            out[f] = np.interp(ranks, self.grid_, self.targets_[f])
        return out


STRATEGIES = {
    "conditional_repair": _ConditionalRepair,
    "residualisation": _Residualiser,
    "selective_repair": _SelectiveRepair,
    "suppression": _Suppressor,
}


# ==========================================================================
class ProxyAwareMitigator(Mitigator):
    """Iterative, targeted proxy suppression.

    Parameters
    ----------
    top_k             features treated per iteration
    tau               target probe AUC; iteration stops below this
    max_accuracy_loss tolerated fall in model AUC before stopping
    strategy          conditional_repair | residualisation | selective_repair | suppression
    max_iter          hard cap on iterations
    random_features   if True, select features at random instead of by
                      leakage rank. THE ABLATION CONTROL — if this performs
                      as well as targeting, the method's claimed mechanism
                      is false.

    ITERATION 2 additions. All default to iteration-1 behaviour, so the frozen
    results remain reproducible from this module.

    skip_dominant_share  exclude features whose most common value covers more
                         than this share of rows. None disables the guard.
    skip_min_unique      exclude features with fewer than this many distinct
                         values. None disables the guard.
    repair_level         1.0 is the full remap of iteration 1; below 1.0 the
                         repaired value is blended with the original.
    """

    name = "proxy_aware"
    requires_attribute = False        # not required at inference

    def __init__(self, model_kind: str = "gbm", top_k: int = 15,
                 tau: float = 0.55, max_accuracy_loss: float = 0.08,
                 strategy: str = "conditional_repair", max_iter: int = 10,
                 random_features: bool = False,
                 skip_dominant_share: float | None = None,
                 skip_min_unique: int | None = None,
                 repair_level: float = 1.0):
        super().__init__(model_kind)
        self.top_k = top_k
        self.tau = tau
        self.max_accuracy_loss = max_accuracy_loss
        self.strategy = strategy
        self.max_iter = max_iter
        self.random_features = random_features
        self.skip_dominant_share = skip_dominant_share
        self.skip_min_unique = skip_min_unique
        self.repair_level = repair_level
        self.skipped_ = []
        self.history_ = []
        self.treated_ = []
        self.transformers_ = []

    # ---------------- internals ----------------
    def _probe_auc(self, X: pd.DataFrame, a: pd.Series) -> float:
        """Cheap single-split probe. Full CV is used only for reporting."""
        a_bin = (pd.Series(a).astype(str) ==
                 sorted(pd.Series(a).astype(str).unique())[0]).astype(int).values
        prep = build_preprocessor(X, dense=True)
        Z = prep.fit_transform(X)
        Ztr, Zte, atr, ate = train_test_split(
            Z, a_bin, test_size=0.3, stratify=a_bin, random_state=RANDOM_STATE)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(Ztr, atr)
        return float(roc_auc_score(ate, clf.predict_proba(Zte)[:, 1]))

    def _model_auc(self, X, y) -> float:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
        m = build_model(self.model_kind, Xtr)
        m.fit(Xtr, ytr)
        return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))

    def _rank_features(self, X, a, exclude) -> list[str]:
        """Rank untreated numeric features by leakage contribution.

        Degenerate features are excluded before ranking rather than after, so
        that a skipped feature does not consume one of the top_k slots.
        """
        blocked = set(exclude) | set(self.skipped_)
        candidates = [c for c in X.columns if c not in blocked]
        if not candidates:
            return []

        if self.random_features:
            rng = np.random.default_rng(RANDOM_STATE + len(exclude))
            return list(rng.permutation(candidates))[:self.top_k]

        a_bin = (pd.Series(a).astype(str) ==
                 sorted(pd.Series(a).astype(str).unique())[0]).astype(int).values
        scores = {}
        for c in candidates:
            v, _ = _numeric_view(X[c])
            v = v.fillna(v.median() if v.notna().any() else 0.0).values.reshape(-1, 1)
            try:
                clf = LogisticRegression(max_iter=400, random_state=RANDOM_STATE)
                clf.fit(v, a_bin)
                scores[c] = abs(roc_auc_score(a_bin, clf.predict_proba(v)[:, 1]) - 0.5)
            except Exception:
                scores[c] = 0.0
        ordered = sorted(scores, key=scores.get, reverse=True)
        return ordered[:self.top_k]

    # ---------------- interface ----------------
    def fit(self, X: pd.DataFrame, y, A=None):
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        if A is None:
            # No attribute at fitting time either — nothing to locate.
            self.fell_back_ = True
            return self._fit_plain(X, y)

        a = pd.Series(A).reset_index(drop=True)
        self.skipped_ = degenerate_features(X, self.skip_dominant_share,
                                            self.skip_min_unique)
        baseline_auc = self._model_auc(X, y)
        leakage = self._probe_auc(X, a)

        self.history_.append({"iteration": 0, "leakage": leakage,
                              "model_auc": baseline_auc, "treated": []})

        Xc = X.copy()
        for it in range(1, self.max_iter + 1):
            if leakage <= self.tau:
                break

            picks = self._rank_features(Xc, a, exclude=self.treated_)
            if not picks and not self.treated_:
                break

            # Re-treat everything selected so far, not only the new features.
            # Each round refits the attribute predictor on the current
            # (already partly repaired) data, so the strata it defines are
            # progressively less informative and the repair compounds. A
            # single pass leaves substantial leakage; iteration is what
            # drives it toward chance.
            self.treated_ = list(dict.fromkeys(self.treated_ + picks))

            tf = (STRATEGIES[self.strategy](repair_level=self.repair_level)
                  if self.strategy == "conditional_repair"
                  else STRATEGIES[self.strategy]())
            tf.fit(Xc, a, self.treated_)
            Xn = tf.transform(Xc, self.treated_)

            auc = self._model_auc(Xn, y)
            if baseline_auc - auc > self.max_accuracy_loss:
                self.history_.append({"iteration": it, "leakage": leakage,
                                      "model_auc": auc,
                                      "treated": list(self.treated_),
                                      "rejected": True})
                self.treated_ = [t for t in self.treated_ if t not in picks]
                break

            Xc = Xn
            self.transformers_.append((tf, list(self.treated_)))
            leakage = self._probe_auc(Xc, a)
            self.history_.append({"iteration": it, "leakage": leakage,
                                  "model_auc": auc,
                                  "treated": list(self.treated_)})

        self.final_leakage_ = leakage
        self.model_ = build_model(self.model_kind, Xc)
        self.model_.fit(Xc, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation. No protected attribute required."""
        Xc = X.copy()
        for tf, feats in self.transformers_:
            Xc = tf.transform(Xc, feats)
        return Xc

    def predict_proba(self, X: pd.DataFrame, A=None) -> np.ndarray:
        if self.fell_back_:
            return self.model_.predict_proba(X)[:, 1]
        return self.model_.predict_proba(self.transform(X))[:, 1]

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)


# ==========================================================================
class ProxyAwareReweighed(ProxyAwareMitigator):
    """ITERATION 3: proxy-aware feature repair followed by reweighing.

    The two components address different defects. Repair is the only method in
    the bank that reduces proxy leakage, since it is the only one that changes
    the features; reweighing is the only one that improved gender equalised
    odds under equal access, since it rebalances (group, outcome) combinations
    in the training data. Repair alone regressed gender equalised odds.

    The hypothesis is that they compose: repair the features exactly as the
    parent class does, then train the final model on the repaired features
    with Kamiran-Calders weights. The weighting is delegated to the bank's own
    Reweighing class rather than re-implemented, so "reweighing" here is
    precisely the established method, applied to different inputs.

    The attribute is used at fitting only. Repair needs it to locate and treat
    leaky features; reweighing needs it to compute the weights. Neither the
    transformation nor the weighted model consults it at inference.
    """

    name = "proxy_aware_reweighed"

    def fit(self, X: pd.DataFrame, y, A=None):
        super().fit(X, y, A)
        if A is None or self.fell_back_:
            return self

        X = X.reset_index(drop=True)
        Xr = self.transform(X)
        rw = Reweighing(model_kind=self.model_kind)
        rw.fit(Xr, pd.Series(y).reset_index(drop=True),
               pd.Series(A).reset_index(drop=True))
        self.model_ = rw.model_
        self.weights_ = rw.weights_
        return self

