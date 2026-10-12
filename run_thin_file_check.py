"""Thin-file check: proxy leakage without the external-source scores.

    python run_thin_file_check.py

EXT_SOURCE_1, EXT_SOURCE_2 and EXT_SOURCE_3 are Home Credit's normalised
scores from external data sources, which a genuinely thin-file applicant may
not have. This re-runs the Module 1 leakage audit on the same 50,000-row
sample as run_leakage_experiment.py

    (a) with all 116 features
    (b) with the three EXT_SOURCE columns removed

using the standard gradient-boosting probe (5-fold) and the shuffled-attribute
control, and ranks the features of (b) by leakage contribution. Leakage audit
only: nothing is mitigated.

Output: results/thin_file_check.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from leakage.probe import (  # noqa: E402
    classify_leakage, probe_auc, rank_leaky_features, shuffle_test,
)

ATTRIBUTES = ["CODE_GENDER", "REGION_RATING_CLIENT"]
EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
TOP = 5
OUT = ROOT / "results" / "thin_file_check.csv"


def main(subsample: int = 50_000):
    t0 = time.time()
    X, _, A = LOADERS["home_credit"](subsample=subsample)
    missing = [c for c in EXT if c not in X.columns]
    if missing:
        raise SystemExit(f"columns not found: {missing}")
    sets = {"all features": X,
            "without EXT_SOURCE_1-3": X.drop(columns=EXT)}
    print(f"loaded {len(X):,} rows x {X.shape[1]} features in {time.time() - t0:.0f}s")

    rows = []
    for attr in ATTRIBUTES:
        a = A[attr]
        for name, Xs in sets.items():
            t = time.time()
            act = probe_auc(Xs, a, kind="gbm")
            ctl = shuffle_test(Xs, a, kind="gbm")
            row = {"dataset": "home_credit", "attribute": attr, "feature_set": name,
                   "n_features": Xs.shape[1], "n_samples": int(act["n_samples"]),
                   "probe": "gbm", "auc_mean": act["auc_mean"], "auc_std": act["auc_std"],
                   "balanced_acc_mean": act["balanced_acc_mean"],
                   "control_auc_mean": ctl["auc_mean"], "control_auc_std": ctl["auc_std"],
                   "verdict": classify_leakage(act["auc_mean"])}
            if name != "all features":
                ranked = rank_leaky_features(Xs, a, kind="gbm", top_k=TOP)
                for i, r in enumerate(ranked.itertuples(index=False), start=1):
                    row[f"top{i}_feature"] = r.feature
                    row[f"top{i}_leakage_drop"] = r.leakage_drop
            rows.append(row)
            print(f"{attr:22s} {name:24s} AUC {act['auc_mean']:.4f} ± {act['auc_std']:.4f}"
                  f"   control {ctl['auc_mean']:.4f}   ({time.time() - t:.0f}s)", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
