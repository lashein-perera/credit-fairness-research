"""Module 2 — Baseline credit-scoring models.

Three model families, all trained through a shared preprocessing pipeline so
that differences between configurations are attributable to the mitigation
applied rather than to preprocessing.

    logistic  interpretable reference, consistent with regulated practice
    gbm       performance reference
    mlp       required by adversarial debiasing
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

RANDOM_STATE = 42


def column_types(X: pd.DataFrame):
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return num, cat


def build_preprocessor(X: pd.DataFrame, dense: bool = True) -> ColumnTransformer:
    """Shared preprocessing.

    dense=True  impute + scale, for models that cannot handle NaN
    dense=False leave numerics untouched, for gradient boosting
    """
    num, cat = column_types(X)

    if dense:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        num_pipe = "passthrough"

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ])
    if dense:
        cat_pipe.steps.append(("scale", StandardScaler()))

    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)],
                             remainder="drop")


def build_model(kind: str, X: pd.DataFrame) -> Pipeline:
    """Return a fitted-ready pipeline for the named model family."""
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
        prep = build_preprocessor(X, dense=True)
    elif kind == "gbm":
        clf = HistGradientBoostingClassifier(
            max_iter=200, random_state=RANDOM_STATE, early_stopping=True)
        prep = build_preprocessor(X, dense=False)
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                            random_state=RANDOM_STATE, early_stopping=True)
        prep = build_preprocessor(X, dense=True)
    else:
        raise ValueError(f"unknown model: {kind}")

    return Pipeline([("prep", prep), ("clf", clf)])


def to_matrix(X: pd.DataFrame, dense: bool = True) -> np.ndarray:
    """Preprocess a frame to a plain array.

    Used by mitigation methods that operate on the numeric feature space
    directly rather than through a pipeline.
    """
    prep = build_preprocessor(X, dense=dense)
    return prep.fit_transform(X)


MODELS = ["logistic", "gbm", "mlp"]
