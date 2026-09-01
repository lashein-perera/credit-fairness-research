"""Run the proxy leakage experiment — Module 1.

    python run_leakage_experiment.py                 # Home Credit, 50k subsample
    python run_leakage_experiment.py --full          # all 307,511 rows (slow)
    python run_leakage_experiment.py --dataset german_credit

Outputs
    results/leakage_table.csv          probe AUC per dataset x attribute x probe
    results/leaky_features_ranked.csv  features ranked by leakage contribution
    results/figures/leakage_*.png      bar chart of top leaky features
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
from leakage.probe import (  # noqa: E402
    classify_leakage, probe_auc, rank_leaky_features, shuffle_test,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def run(dataset: str, subsample: int | None, probes: list[str], top_k: int):
    print("=" * 70)
    print(f"PROXY LEAKAGE DETECTION — {dataset}")
    print("=" * 70)

    t0 = time.time()
    loader = LOADERS[dataset]
    X, y, A = loader(subsample=subsample) if dataset == "home_credit" else loader()

    print(f"\nloaded in {time.time()-t0:.1f}s")
    print(f"  X : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  y : default rate {y.mean():.2%}")
    print(f"  A : {list(A.columns)}")

    # confirm the design rule holds
    leaked = [c for c in A.columns if c in X.columns]
    if leaked:
        raise SystemExit(f"FATAL: protected attribute(s) {leaked} present in X")
    print("  protected attributes absent from X — isolation holds")

    rows, ranked_all = [], []

    for attr in PROTECTED[dataset]:
        if attr not in A.columns:
            continue
        a = A[attr]
        print(f"\n{'-'*70}\nATTRIBUTE: {attr}")
        print(f"  distribution: {dict(a.value_counts())}")

        for kind in probes:
            t = time.time()
            r = probe_auc(X, a, kind=kind)
            verdict = classify_leakage(r["auc_mean"])
            print(f"\n  probe [{kind}]  ({time.time()-t:.1f}s)")
            print(f"    AUC          {r['auc_mean']:.4f} +/- {r['auc_std']:.4f}")
            print(f"    balanced acc {r['balanced_acc_mean']:.4f} "
                  f"+/- {r['balanced_acc_std']:.4f}")
            print(f"    verdict      {verdict.upper()} LEAKAGE")

            rows.append({"dataset": dataset, "attribute": attr, "condition": "actual",
                         **r, "verdict": verdict})

        # control
        t = time.time()
        ctrl = shuffle_test(X, a, kind=probes[-1])
        ok = ctrl["auc_mean"] < 0.55
        print(f"\n  SHUFFLE CONTROL [{probes[-1]}]  ({time.time()-t:.1f}s)")
        print(f"    AUC          {ctrl['auc_mean']:.4f} +/- {ctrl['auc_std']:.4f}")
        print(f"    {'PASS — measurement is sound' if ok else 'FAIL — investigate before trusting the result'}")
        rows.append({"dataset": dataset, "attribute": attr, "condition": "shuffled",
                     **ctrl, "verdict": classify_leakage(ctrl["auc_mean"])})

        # feature ranking
        print(f"\n  ranking features by leakage contribution...")
        t = time.time()
        ranked = rank_leaky_features(X, a, top_k=top_k)
        ranked.insert(0, "attribute", attr)
        ranked.insert(0, "dataset", dataset)
        ranked_all.append(ranked)
        print(f"  ({time.time()-t:.1f}s)\n")
        print(ranked[["rank", "feature", "leakage_drop"]].head(10).to_string(index=False))

        _plot(ranked, dataset, attr)

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    lt = RESULTS / "leakage_table.csv"
    pd.DataFrame(rows).to_csv(lt, index=False)
    print(f"\nwrote {lt}")

    if ranked_all:
        lf = RESULTS / "leaky_features_ranked.csv"
        pd.concat(ranked_all).to_csv(lf, index=False)
        print(f"wrote {lf}")

    print(f"\ntotal {time.time()-t0:.1f}s")


def _plot(ranked: pd.DataFrame, dataset: str, attr: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    d = ranked.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["feature"], d["leakage_drop"], color="#4C72B0")
    ax.set_xlabel("Drop in probe AUC when feature is permuted")
    ax.set_title(f"Features leaking {attr} — {dataset}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"leakage_{dataset}_{attr}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  figure -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="home_credit", choices=list(LOADERS))
    ap.add_argument("--full", action="store_true", help="use all rows")
    ap.add_argument("--subsample", type=int, default=50_000)
    ap.add_argument("--probes", nargs="+", default=["logistic", "gbm"])
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    run(args.dataset, None if args.full else args.subsample,
        args.probes, args.top_k)
