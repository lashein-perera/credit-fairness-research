"""Re-run the corrected adversarial baseline — iteration 2.

    python run_adversarial_fixed.py

The shipped adversary term was subtracted at a fixed weight with no regard to
the relative magnitude of the predictor's own gradient, so it dominated the
update and the predictor never fit the outcome. Scaling it to the predictor
gradient norm is the single change; no hyperparameter search was run.

Only adversarial_debiasing is re-run. Every other method is unaffected by the
change, so the frozen numbers for them stand.

Outputs results/iteration2/adversarial_fixed_*.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import run_grid, summarise  # noqa: E402

OUT = ROOT / "results" / "iteration2"


def main(splits=5, repeats=5):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t_all = [], time.time()

    jobs = [("home_credit", "CODE_GENDER", 8000),
            ("home_credit", "REGION_RATING_CLIENT", 8000),
            ("german_credit", "sex", None),
            ("uci_default", "SEX", None)]

    for dataset, attribute, subsample in jobs:
        t0 = time.time()
        print("=" * 74)
        print(f"CORRECTED ADVERSARIAL — {dataset} / {attribute}")
        print("=" * 74)
        X, y, A = (LOADERS[dataset](subsample=subsample)
                   if dataset == "home_credit" else LOADERS[dataset]())
        if attribute not in A.columns:
            print(f"  SKIP: not in {list(A.columns)}")
            continue
        # benchmarks used 5x2 folds; home_credit tables used 5x5
        r = repeats if dataset == "home_credit" else 2
        df = run_grid(X, y, A[attribute], attribute=attribute, dataset=dataset,
                      methods=["adversarial_debiasing"],
                      conditions=("explicit", "training_only", "latent"),
                      n_splits=splits, n_repeats=r, verbose=False)
        s = summarise(df)
        s.insert(0, "dataset_", dataset)
        rows.append(s)
        print(s[["method", "condition", "auc_mean", "dp_diff_mean",
                 "di_ratio_mean", "eo_diff_mean",
                 "probe_auc_after_mean"]].round(4).to_string(index=False))
        print(f"  ({time.time()-t0:.0f}s)\n")
        pd.concat(rows, ignore_index=True).to_csv(
            OUT / "adversarial_fixed_summary.csv", index=False)

    print(f"total {time.time()-t_all:.0f}s")
    print(f"wrote {OUT/'adversarial_fixed_summary.csv'}")


if __name__ == "__main__":
    main()
