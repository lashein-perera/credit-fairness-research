"""Rebuild german_with_lender_score.csv, the upload file browser_checks.py uses.

    python docs/test_evidence/make_german_upload.py
    python docs/test_evidence/make_german_upload.py --out some/other/path.csv

German Credit read through the project's own loader (src/python/data/loaders.py),
so the protected attributes are split off exactly as in the experiments. The file
holds 1,000 rows x 22 columns, in this order:

    outcome        1 = defaulted (the loader's target)
    sex            the protected attribute (M / F)
    attr_1 ...     the 18 credit features the loader keeps (attr_9, the joint
                   personal-status/sex code, and attr_13, age, are not among them)
    age            kept so the upload test can leave it out
    lender_score   a stand-in for a lender's own risk score: the probability of
                   default from a logistic regression (models.baselines,
                   build_model("logistic")) fitted on the 18 features and the
                   outcome, rounded to 4 decimals

The default output path is under data/, which git ignores: datasets are never
committed to this repository. German Credit must first be downloaded as
described in data/README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from models.baselines import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "german_with_lender_score.csv"),
                    help="where to write the CSV (default: data/german_with_lender_score.csv)")
    out = Path(ap.parse_args().out)

    X, y, A = LOADERS["german_credit"]()
    model = build_model("logistic", X)
    model.fit(X, y)
    df = pd.concat([y.rename("outcome"), A[["sex"]], X, A[["age"]]], axis=1)
    df["lender_score"] = model.predict_proba(X)[:, 1].round(4)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {df.shape[0]:,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
