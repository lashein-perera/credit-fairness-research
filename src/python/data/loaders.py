"""Dataset loaders.

Each loader returns (X, y, A):
    X : feature matrix WITHOUT any protected attribute
    y : binary default target
    A : DataFrame of protected attributes — EVALUATION ONLY, never trained on

The separation enforced here is the central design rule of this research.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

# Columns that must never appear in X, per dataset.
# Includes the protected attributes themselves plus near-duplicates that would
# trivially reveal them (e.g. a city-level version of the same region rating).
PROTECTED = {
    "home_credit": {
        "CODE_GENDER": ["CODE_GENDER"],
        "REGION_RATING_CLIENT": ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "REGION_POPULATION_RELATIVE"],
    },
    "german_credit": {
        "sex": ["sex", "personal_status_sex"],
        "age_group": ["age_group", "age"],
    },
    "uci_default": {
        "SEX": ["SEX"],
        "MARRIAGE": ["MARRIAGE"],
    },
}

# Never useful as features
DROP_ALWAYS = {
    "home_credit": ["SK_ID_CURR", "TARGET"],
    "german_credit": ["target"],
    "uci_default": ["ID", "default payment next month"],
}


def _split(df: pd.DataFrame, dataset: str, target_col: str):
    """Split a raw frame into (X, y, A) with protected attributes isolated."""
    prot_cols = sorted({c for cols in PROTECTED[dataset].values() for c in cols})
    present = [c for c in prot_cols if c in df.columns]

    A = df[present].copy()
    y = df[target_col].astype(int).copy()

    drop = set(present) | set(DROP_ALWAYS.get(dataset, []))
    X = df.drop(columns=[c for c in drop if c in df.columns]).copy()

    return X, y, A


def load_home_credit(subsample: int | None = 50_000, random_state: int = 42):
    """Home Credit Default Risk — primary thin-file dataset."""
    path = DATA_DIR / "home_credit" / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(path)

    # CODE_GENDER contains a handful of 'XNA' rows — remove them so the probe
    # target is cleanly binary.
    df = df[df["CODE_GENDER"].isin(["M", "F"])].copy()

    if subsample is not None and subsample < len(df):
        frac = subsample / len(df)
        parts = [
            grp.sample(n=max(1, int(round(len(grp) * frac))),
                       random_state=random_state)
            for _, grp in df.groupby("TARGET", sort=False)
        ]
        df = pd.concat(parts).sample(frac=1, random_state=random_state)
        df = df.reset_index(drop=True)

    return _split(df, "home_credit", "TARGET")


def load_german_credit():
    """UCI German Credit (Statlog).

    Sex is not a standalone column — it is encoded jointly with personal status
    in attribute 9 (A91-A94), where A92 denotes female and A91/A93/A94 male.
    """
    path = DATA_DIR / "german_credit" / "german.data"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    cols = [f"attr_{i}" for i in range(1, 21)] + ["target"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)

    df["personal_status_sex"] = df["attr_9"]
    df["sex"] = (df["attr_9"] == "A92").map({True: "F", False: "M"})
    df["age"] = df["attr_13"]
    df["age_group"] = np.where(df["attr_13"] >= 25, "older", "younger")
    df = df.drop(columns=["attr_9", "attr_13"])

    # target: 1 = good, 2 = bad -> recode so 1 = default
    df["target"] = (df["target"] == 2).astype(int)

    return _split(df, "german_credit", "target")


def load_uci_default():
    """UCI Default of Credit Card Clients. Header sits on the second row."""
    folder = DATA_DIR / "uci_default"
    files = list(folder.glob("*.xls*")) if folder.exists() else []
    if not files:
        raise FileNotFoundError(f"No .xls/.xlsx in {folder}")

    df = pd.read_excel(files[0], header=1)
    return _split(df, "uci_default", "default payment next month")


LOADERS = {
    "home_credit": load_home_credit,
    "german_credit": load_german_credit,
    "uci_default": load_uci_default,
}
