"""Preprocessing pipeline.

Design rule enforced here: protected attributes are separated from the feature
matrix before any model sees the data. See tests/test_protected_attribute_isolation.py
"""
import pandas as pd


def separate_protected(df: pd.DataFrame, protected_cols: list[str]):
    """Split a raw frame into (features, protected_attributes)."""
    present = [c for c in protected_cols if c in df.columns]
    A = df[present].copy()
    X = df.drop(columns=present)
    return X, A


def build_preprocessor(X: pd.DataFrame):
    """Impute, encode and scale. Document every choice — imputation affects fairness."""
    raise NotImplementedError("Week 2")


def stratified_split(X, y, test_size: float = 0.2, random_state: int = 42):
    raise NotImplementedError("Week 2")
