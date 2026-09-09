"""Run the SHAP explainability layer — Module 5.

    python run_explainability.py                            # gender, 20k subsample
    python run_explainability.py --attribute REGION_RATING_CLIENT
    python run_explainability.py --model logistic --linear

Outputs
    results/shap_importance.csv          mean |SHAP| per feature, per model
    results/leaky_features_full.csv      leakage contribution for EVERY feature
    results/leakage_vs_shap.csv          the joined table behind the cross-plot
    results/leakage_vs_shap_stats.csv    rank correlation per configuration
    results/figures/leakage_vs_shap_*.png   the RQ3 cross-plot

NOTE ON THE LEAKAGE AXIS
    results/leaky_features_ranked.csv holds only the top 20 features, which
    would truncate the cross-plot to its leaky corner. This script recomputes
    the ranking over the full feature set and writes it separately, leaving
    Module 1's own output untouched.
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
from explain.shap_layer import (  # noqa: E402
    global_importance, leakage_vs_shap, leakage_vs_shap_plot, rank_correlation,
)
from leakage.probe import rank_leaky_features  # noqa: E402
from models.baselines import build_model  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def _append(path: Path, df: pd.DataFrame, keys: list[str]):
    """Write, replacing any rows for the same configuration."""
    if path.exists():
        old = pd.read_csv(path)
        if all(k in old.columns for k in keys):
            mask = pd.Series(True, index=old.index)
            for k in keys:
                mask &= old[k].astype(str) == str(df[k].iloc[0])
            old = old[~mask]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def run(dataset: str, attribute: str, subsample: int | None, model_kind: str,
        max_rows: int, background: int, n_annotate: int, log: bool = True):
    print("=" * 70)
    print(f"SHAP EXPLAINABILITY — {dataset} / {attribute} / {model_kind}")
    print("=" * 70)

    t0 = time.time()
    loader = LOADERS[dataset]
    X, y, A = loader(subsample=subsample) if dataset == "home_credit" else loader()

    if attribute not in A.columns:
        raise SystemExit(f"attribute '{attribute}' not in {list(A.columns)}")
    a = A[attribute]

    print(f"\nloaded in {time.time()-t0:.1f}s")
    print(f"  X : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  y : default rate {y.mean():.2%}")
    print(f"  A : {attribute} {dict(a.value_counts())}")

    leaked = [c for c in A.columns if c in X.columns]
    if leaked:
        raise SystemExit(f"FATAL: protected attribute(s) {leaked} present in X")
    print("  protected attributes absent from X — isolation holds")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---------------- PART A: SHAP ----------------
    print(f"\n{'-'*70}\nPART A — SHAP global importance")
    t = time.time()
    model = build_model(model_kind, X)
    model.fit(X, y)
    print(f"  fitted [{model_kind}] ({time.time()-t:.1f}s)")

    t = time.time()
    imp = global_importance(model, X, model_kind=model_kind,
                            max_rows=max_rows, background=background,
                            verbose=True)
    print(f"  ({time.time()-t:.1f}s)\n")
    print(imp[["rank", "feature", "shap_importance"]].head(15).to_string(index=False))

    out = imp.copy()
    out.insert(0, "model", model_kind)
    out.insert(0, "dataset", dataset)
    _append(RESULTS / "shap_importance.csv", out, ["dataset", "model"])
    print(f"\nwrote {RESULTS/'shap_importance.csv'}")

    # ---------------- PART B: cross-plot ----------------
    print(f"\n{'-'*70}\nPART B — leakage vs SHAP (RQ3)")
    print(f"  ranking ALL {X.shape[1]} features by leakage contribution...")
    t = time.time()
    leak = rank_leaky_features(X, a, top_k=X.shape[1])
    print(f"  ({time.time()-t:.1f}s)")

    lf = leak.copy()
    lf.insert(0, "attribute", attribute)
    lf.insert(0, "dataset", dataset)
    _append(RESULTS / "leaky_features_full.csv", lf, ["dataset", "attribute"])
    print(f"  wrote {RESULTS/'leaky_features_full.csv'}")

    cross = leakage_vs_shap(leak, imp)
    stats = rank_correlation(cross)

    print(f"\n  Spearman rho {stats['spearman_rho']:+.4f}  (p = {stats['spearman_p']:.3g})")
    print(f"  Pearson  r   {stats['pearson_r']:+.4f}  (p = {stats['pearson_p']:.3g})")
    print(f"\n  quadrant counts (median split on both axes):")
    for q, n in cross["quadrant"].value_counts().items():
        print(f"    {q:<22} {n:>4}")

    danger = cross[cross.quadrant == "leaky_and_relied_on"]
    print(f"\n  top of the problem quadrant — leaky AND relied upon:")
    print(danger.head(10)[["feature", "leakage_drop", "shap_importance",
                           "leakage_rank", "shap_rank"]].to_string(index=False))

    xf = cross.copy()
    xf.insert(0, "model", model_kind)
    xf.insert(0, "attribute", attribute)
    xf.insert(0, "dataset", dataset)
    _append(RESULTS / "leakage_vs_shap.csv", xf,
            ["dataset", "attribute", "model"])
    print(f"\nwrote {RESULTS/'leakage_vs_shap.csv'}")

    srow = pd.DataFrame([{"dataset": dataset, "attribute": attribute,
                          "model": model_kind, **stats}])
    _append(RESULTS / "leakage_vs_shap_stats.csv", srow,
            ["dataset", "attribute", "model"])
    print(f"wrote {RESULTS/'leakage_vs_shap_stats.csv'}")

    fig_path = FIGURES / f"leakage_vs_shap_{dataset}_{attribute}_{model_kind}.png"
    leakage_vs_shap_plot(cross, None, fig_path, dataset=dataset,
                         attribute=attribute, model_kind=model_kind,
                         n_annotate=n_annotate, log=log)
    print(f"  figure -> {fig_path}")

    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="home_credit", choices=list(LOADERS))
    ap.add_argument("--attribute", default="CODE_GENDER")
    ap.add_argument("--subsample", type=int, default=20_000)
    ap.add_argument("--full", action="store_true", help="use all rows")
    ap.add_argument("--model", default="gbm", choices=["logistic", "gbm", "mlp"])
    ap.add_argument("--max-rows", type=int, default=2_000,
                    help="rows explained by SHAP")
    ap.add_argument("--background", type=int, default=100,
                    help="background sample for linear/permutation explainers")
    ap.add_argument("--annotate", type=int, default=8,
                    help="features named in the problem quadrant")
    ap.add_argument("--linear", action="store_true",
                    help="linear axes; the default is symlog, because both "
                         "quantities span several orders of magnitude")
    args = ap.parse_args()

    run(args.dataset, args.attribute, None if args.full else args.subsample,
        args.model, args.max_rows, args.background, args.annotate,
        log=not args.linear)
