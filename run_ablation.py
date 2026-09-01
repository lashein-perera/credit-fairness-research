"""Ablation study — isolate the contribution of each design decision.

    python run_ablation.py                                    # gender
    python run_ablation.py --attribute REGION_RATING_CLIENT   # region

Four ablations, each removing one component:

    full            targeted selection, iterative, conditional repair
    random          features chosen AT RANDOM instead of by leakage rank
    single_pass     one iteration instead of iterating to threshold
    suppression     features deleted instead of transformed

THE DECISIVE ONE IS 'random'.
If treating randomly chosen features works as well as treating the features
identified as leaky, then the leakage ranking serves no purpose and the
mechanism claimed for this method is false. The experiment is constructed so
as to be capable of refuting the contribution — which is what makes a positive
result meaningful.

Outputs
    results/ablation.csv           one row per configuration per fold
    results/ablation_summary.csv   mean +/- sd
    results/ablation_tests.csv     paired Wilcoxon, full vs each ablation
    results/figures/ablation_*.png
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import _probe, resolve_threshold  # noqa: E402
from evaluation.metrics import (demographic_parity_difference,  # noqa: E402
                                disparate_impact_ratio,
                                equalized_odds_difference, auc_roc,
                                wilcoxon_compare)
from mitigation.proxy_aware import ProxyAwareMitigator  # noqa: E402
from sklearn.model_selection import RepeatedStratifiedKFold  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# name -> constructor kwargs
ABLATIONS = {
    "full":         dict(),
    "random":       dict(random_features=True),
    "single_pass":  dict(max_iter=1),
    "suppression":  dict(strategy="suppression"),
}


def run(dataset, attribute, subsample, splits, repeats, model_kind):
    loader = LOADERS[dataset]
    X, y, A = (loader(subsample=subsample) if dataset == "home_credit"
               else loader())
    if attribute not in A.columns:
        raise SystemExit(f"attribute '{attribute}' not in {list(A.columns)}")
    a = A[attribute]

    print("=" * 70)
    print(f"ABLATION STUDY — {dataset} / {attribute}")
    print("=" * 70)
    print(f"  X: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  {attribute}: {dict(a.value_counts())}\n")

    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats,
                                 random_state=42)
    folds = list(cv.split(X, y))
    total = len(folds) * len(ABLATIONS)
    print(f"grid: {len(ABLATIONS)} configurations x {len(folds)} folds "
          f"= {total} runs\n")

    rows, done = [], 0
    for fi, (tr, te) in enumerate(folds):
        Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
        ytr, yte = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
        atr, ate = a.iloc[tr].reset_index(drop=True), a.iloc[te].reset_index(drop=True)

        for name, kw in ABLATIONS.items():
            t0 = time.time()
            m = ProxyAwareMitigator(model_kind=model_kind, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xtr, ytr, atr)
                proba = m.predict_proba(Xte)
                leak = _probe(m.transform(Xte), ate)

            thr = resolve_threshold(proba, ytr, None)
            pred = (np.asarray(proba) >= thr).astype(int)

            rows.append({
                "dataset": dataset, "attribute": attribute, "fold": fi,
                "ablation": name,
                "auc": auc_roc(yte, proba),
                "dp_diff": demographic_parity_difference(pred, ate),
                "di_ratio": disparate_impact_ratio(pred, ate),
                "eo_diff": equalized_odds_difference(yte, pred, ate),
                "probe_auc_after": leak,
                "n_treated": len(m.treated_),
                "runtime_s": round(time.time() - t0, 1),
            })
            done += 1
            print(f"  [{done:>3}/{total}] {name:12s} AUC {rows[-1]['auc']:.4f}  "
                  f"EO {rows[-1]['eo_diff']:.4f}  leak {leak:.3f}  "
                  f"treated {len(m.treated_)}")

    return pd.DataFrame(rows)


def plot(summary, attribute):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    d = summary.set_index("ablation")
    order = [k for k in ABLATIONS if k in d.index]
    ax.bar(order, d.loc[order, "probe_auc_after_mean"],
           yerr=d.loc[order, "probe_auc_after_std"], color="#4C72B0")
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    ax.set_ylabel("Residual leakage (probe AUC)")
    ax.set_title(f"Ablation — {attribute}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"ablation_{attribute}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="home_credit", choices=list(LOADERS))
    ap.add_argument("--attribute", default="CODE_GENDER")
    ap.add_argument("--subsample", type=int, default=8000)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--model", default="gbm")
    args = ap.parse_args()

    t0 = time.time()
    df = run(args.dataset, args.attribute, args.subsample,
             args.splits, args.repeats, args.model)

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "ablation.csv", index=False)

    metrics = ["auc", "dp_diff", "di_ratio", "eo_diff",
               "probe_auc_after", "n_treated"]
    summary = df.groupby("ablation")[metrics].agg(["mean", "std"]).round(4)
    summary.columns = [f"{x}_{y}" for x, y in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS / "ablation_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary[["ablation", "auc_mean", "eo_diff_mean", "dp_diff_mean",
                   "probe_auc_after_mean", "n_treated_mean"]].to_string(index=False))

    # full vs each ablation
    tests = []
    for metric in ("probe_auc_after", "eo_diff", "dp_diff", "auc"):
        ref = df[df.ablation == "full"].sort_values("fold")[metric].values
        for name in ABLATIONS:
            if name == "full":
                continue
            other = df[df.ablation == name].sort_values("fold")[metric].values
            n = min(len(ref), len(other))
            if n < 3:
                continue
            try:
                r = wilcoxon_compare(ref[:n], other[:n])
            except Exception:
                continue
            tests.append({"metric": metric, "full_median": float(np.median(ref[:n])),
                          "ablation": name,
                          "ablation_median": float(np.median(other[:n])),
                          **{k: r[k] for k in ("p_value", "effect_size",
                                               "effect_magnitude", "significant")}})
    if tests:
        t = pd.DataFrame(tests)
        t.to_csv(RESULTS / "ablation_tests.csv", index=False)
        print("\n" + "=" * 70)
        print("PAIRED WILCOXON — full method vs each ablation")
        print("=" * 70)
        print(t.to_string(index=False))

    plot(summary, args.attribute)
    print(f"\nwrote {RESULTS/'ablation.csv'}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
