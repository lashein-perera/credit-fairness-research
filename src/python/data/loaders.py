"""Dataset loaders.

Each loader returns (X, y, A):
    X : features WITHOUT any protected attribute
    y : binary default target
    A : DataFrame of protected attributes, for EVALUATION ONLY
"""
import pandas as pd
from .. import config


def load_home_credit(subsample: int | None = None):
    """Home Credit Default Risk — primary thin-file dataset."""
    raise NotImplementedError("Week 2")


def load_german_credit():
    """UCI German Credit (Statlog) — benchmark with explicit protected attributes."""
    raise NotImplementedError("Week 2")


def load_uci_default():
    """UCI Default of Credit Card Clients — secondary benchmark."""
    raise NotImplementedError("Week 2")


LOADERS = {
    "home_credit": load_home_credit,
    "german_credit": load_german_credit,
    "uci_default": load_uci_default,
}
