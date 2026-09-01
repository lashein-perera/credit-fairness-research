"""Run the main comparison experiment.

    python run_comparison.py                          # gender, 8k subsample, quick
    python run_comparison.py --attribute REGION_RATING_CLIENT
    python run_comparison.py --subsample 50000 --repeats 5   # full-scale

Outputs
    results/main_comparison.csv        one row per method x condition x fold
    results/main_comparison_summary.csv  mean +/- sd per configuration
    results/wilcoxon_tests.csv         paired tests against the proposed method
    results/figures/comparison_*.png   fairness and leakage by method

RUNTIME WARNING
    The proposed method probes and refits internally, so it is far slower than
    the baselines. Start with the default 8,000-row subsample and 2 repeats to
    confirm the pipeline runs, then scale up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS, PROTECTED  # noqa: E402
from evaluation.harness import compare, run_grid, summarise  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def plot(summary: pd.DataFrame, attribute: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = summary[summary.attribute == attribute]
    if d.empty:
        return

    labels = [f"{r.method}\n({r.condition})" for r in d.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(labels, d["eo_diff_mean"], xerr=d["eo_diff_std"],
                 color="#4C72B0")
    axes[0].set_xlabel("Equalised odds difference (lower is fairer)")
    axes[0].set_title(f"Fairness by method — {attribute}")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(labels, d["probe_auc_after_mean"],
                 xerr=d["probe_auc_after_std"], color="#C44E52")
    axes[1].axvline(0.5, ls="--", c="k", lw=1, label="chance")
    axes[1].set_xlabel("Residual leakage (probe AUC after mitigation)")
    axes[1].set_title(f"Residual leakage — {attribute}")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"comparison_{attribute}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="home_credit", choices=list(LOADERS))
    ap.add_argument("--attribute", default="CODE_GENDER")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--model", default="gbm", choices=["logistic", "gbm", "mlp"])
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed threshold; default uses the base rate")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print(f"MAIN COMPARISON — {args.dataset} / {args.attribute}")
    print("=" * 70)

    loader = LOADERS[args.dataset]
    if args.dataset == "home_credit":
        X, y, A = loader(subsample=args.subsample)
    else:
        X, y, A = loader()

    if args.attribute not in A.columns:
        raise SystemExit(f"attribute '{args.attribute}' not in {list(A.columns)}")
    a = A[args.attribute]

    print(f"  X: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  default rate: {y.mean():.2%}")
    print(f"  {args.attribute}: {dict(a.value_counts())}\n")

    df = run_grid(X, y, a, attribute=args.attribute, dataset=args.dataset,
                  n_splits=args.splits, n_repeats=args.repeats,
                  model_kind=args.model, threshold=args.threshold)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "main_comparison.csv", index=False)

    summary = summarise(df)
    summary.to_csv(RESULTS / "main_comparison_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (mean across folds)")
    print("=" * 70)
    cols = ["method", "condition", "auc_mean", "eo_diff_mean",
            "dp_diff_mean", "di_ratio_mean", "probe_auc_after_mean"]
    print(summary[cols].to_string(index=False))

    tests = []
    for metric in ("eo_diff", "dp_diff", "probe_auc_after", "auc"):
        t = compare(df, metric=metric)
        if not t.empty:
            tests.append(t)
    if tests:
        allt = pd.concat(tests)
        allt.to_csv(RESULTS / "wilcoxon_tests.csv", index=False)
        print("\n" + "=" * 70)
        print("PAIRED WILCOXON — proposed method vs each baseline (latent)")
        print("=" * 70)
        print(allt.to_string(index=False))

    plot(summary, args.attribute)

    print(f"\nwrote {RESULTS/'main_comparison.csv'}")
    print(f"wrote {RESULTS/'main_comparison_summary.csv'}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
