"""Module 1 — Proxy Leakage Detector.

Central idea: if a classifier can predict a protected attribute from the
remaining features, then those features encode protected information, and any
credit model built on them can differentiate along that attribute without ever
receiving it.

Operationalises the predictability test of Feldman et al. (2015).

Interpretation of probe AUC
---------------------------
    ~0.50   no leakage
    >0.65   meaningful leakage
    >0.80   severe leakage

Balanced accuracy is reported alongside AUC because the primary dataset is
imbalanced with respect to gender (~66/34), and a majority-class predictor
would otherwise appear informative.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------
def _column_types(X: pd.DataFrame):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return num, cat


def _build_preprocessor(X: pd.DataFrame, kind: str) -> ColumnTransformer:
    num, cat = _column_types(X)

    if kind == "logistic":
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)),
            ("scale", StandardScaler()),
        ])
    else:  # gradient boosting tolerates NaN natively
        num_pipe = "passthrough"
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1)),
        ])

    return ColumnTransformer(
        [("num", num_pipe, num), ("cat", cat_pipe, cat)],
        remainder="drop",
    )


def _build_probe(X: pd.DataFrame, kind: str) -> Pipeline:
    if kind == "logistic":
        model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif kind == "gbm":
        model = HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE, early_stopping=True
        )
    else:
        raise ValueError(f"unknown probe: {kind}")
    return Pipeline([("prep", _build_preprocessor(X, kind)), ("clf", model)])


def _encode_target(a: pd.Series):
    """Return (y, positive_label, is_binary)."""
    vals = pd.Series(a).dropna().unique()
    if len(vals) == 2:
        pos = sorted(vals)[0]
        return (pd.Series(a) == pos).astype(int).values, pos, True
    codes, uniques = pd.factorize(pd.Series(a))
    return codes, list(uniques), False


# --------------------------------------------------------------------------
# main measurement
# --------------------------------------------------------------------------
def probe_auc(X: pd.DataFrame, a: pd.Series, kind: str = "logistic",
              n_splits: int = 5, verbose: bool = False) -> dict:
    """Cross-validated ability to predict protected attribute `a` from `X`."""
    y, pos, is_binary = _encode_target(a)
    mask = ~pd.isna(pd.Series(a)).values
    X, y = X.loc[mask].reset_index(drop=True), y[mask]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs, baccs = [], []

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        pipe = _build_probe(X, kind)
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])
        pred = pipe.predict(X.iloc[te])

        if is_binary:
            auc = roc_auc_score(y[te], proba[:, 1])
        else:
            auc = roc_auc_score(y[te], proba, multi_class="ovr",
                                average="macro")
        aucs.append(auc)
        baccs.append(balanced_accuracy_score(y[te], pred))
        if verbose:
            print(f"      fold {fold}: AUC {auc:.4f}  bal-acc {baccs[-1]:.4f}")

    return {
        "probe": kind,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "balanced_acc_mean": float(np.mean(baccs)),
        "balanced_acc_std": float(np.std(baccs)),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_label": str(pos),
    }


def shuffle_test(X: pd.DataFrame, a: pd.Series, kind: str = "logistic",
                 n_splits: int = 5, random_state: int = RANDOM_STATE) -> dict:
    """Control: with the attribute permuted, probe AUC must collapse to ~0.5.

    If it does not, the measurement is faulty rather than the finding real.
    """
    rng = np.random.default_rng(random_state)
    shuffled = pd.Series(rng.permutation(pd.Series(a).values), index=X.index)
    out = probe_auc(X, shuffled, kind=kind, n_splits=n_splits)
    out["control"] = True
    return out


def rank_leaky_features(X: pd.DataFrame, a: pd.Series, kind: str = "gbm",
                        top_k: int = 20, n_repeats: int = 5,
                        max_rows: int = 20_000) -> pd.DataFrame:
    """Rank features by how much each contributes to attribute leakage.

    Permutation importance on a held-out split: shuffle one feature, measure how
    far probe AUC falls. A large drop means that feature carried the signal.
    """
    y, _, is_binary = _encode_target(a)
    mask = ~pd.isna(pd.Series(a)).values
    X, y = X.loc[mask].reset_index(drop=True), y[mask]

    if len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=RANDOM_STATE
        )
        X = X.reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )
    pipe = _build_probe(X_tr, kind)
    pipe.fit(X_tr, y_tr)

    scoring = "roc_auc" if is_binary else "roc_auc_ovr"
    result = permutation_importance(
        pipe, X_te, y_te, scoring=scoring, n_repeats=n_repeats,
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": X_te.columns,
        "leakage_drop": result.importances_mean,
        "leakage_std": result.importances_std,
    }).sort_values("leakage_drop", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    return df.head(top_k)


def classify_leakage(auc: float) -> str:
    if auc < 0.55:
        return "none"
    if auc < 0.65:
        return "mild"
    if auc < 0.80:
        return "meaningful"
    return "severe"
