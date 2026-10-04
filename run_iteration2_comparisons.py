"""Equal-access comparison and benchmark datasets — iteration 2.

    python run_iteration2_comparisons.py training_only
    python run_iteration2_comparisons.py benchmarks

TRAINING-ONLY CONDITION
    The frozen 'latent' condition gave the proposed method the protected
    attribute at fitting while denying it to the four established methods, so
    it was not a like-for-like comparison. Under 'training_only' every method
    receives the attribute at fitting and none receives it at inference, which
    is exactly the access the proposed method has always had.

    The proposed method is not re-run. Its fit_A and infer_A are identical
    under 'latent' and 'training_only', so its frozen latent numbers ARE its
    training-only numbers.

    Two methods cannot be read as performance under this condition:
      reject_option              adjusts decisions at inference; without the
                                 attribute there it is inert
      disparate_impact_remover   scores raw features with a model fitted on
                                 repaired ones, a train/serve skew in the
                                 implementation rather than a property of
                                 Feldman et al.'s method

BENCHMARKS
    German Credit and Default of Credit Card Clients, explicit and latent, to
    establish whether the established methods behave as published when they do
    have the attribute.

Outputs under results/iteration2/ and results/iteration2/benchmarks/.
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
BENCH = OUT / "benchmarks"

ESTABLISHED = ["none", "reweighing", "disparate_impact_remover",
               "adversarial_debiasing", "reject_option"]

# methods whose training-only numbers must not be read as the method working
NOT_APPLICABLE_TRAINING_ONLY = {
    "reject_option": "requires A at inference",
    "disparate_impact_remover": "requires A at inference (train/serve skew)",
}


def training_only(subsample=8000, splits=5, repeats=5, model_kind="gbm"):
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for attribute in ("CODE_GENDER", "REGION_RATING_CLIENT"):
        t0 = time.time()
        print("=" * 74)
        print(f"TRAINING-ONLY ACCESS — home_credit / {attribute}")
        print("=" * 74)
        X, y, A = LOADERS["home_credit"](subsample=subsample)
        df = run_grid(X, y, A[attribute], attribute=attribute,
                      dataset="home_credit", methods=ESTABLISHED,
                      conditions=("training_only",), n_splits=splits,
                      n_repeats=repeats, model_kind=model_kind, verbose=False)
        allr.append(df)
        s = summarise(df)
        print(s[["method", "condition", "auc_mean", "dp_diff_mean",
                 "di_ratio_mean", "eo_diff_mean",
                 "probe_auc_after_mean"]].round(4).to_string(index=False))
        print(f"  ({time.time()-t0:.1f}s)\n")

    df = pd.concat(allr, ignore_index=True)
    df.to_csv(OUT / "training_only_folds.csv", index=False)
    summ = summarise(df)
    summ["applicability"] = summ.method.map(
        lambda m: NOT_APPLICABLE_TRAINING_ONLY.get(m, "applicable"))
    summ.to_csv(OUT / "training_only_summary.csv", index=False)
    print(f"wrote {OUT/'training_only_summary.csv'}")


def benchmarks(splits=5, repeats=2, model_kind="gbm"):
    BENCH.mkdir(parents=True, exist_ok=True)
    jobs = [("german_credit", "sex"), ("uci_default", "SEX")]
    allr = []
    for dataset, attribute in jobs:
        t0 = time.time()
        print("=" * 74)
        print(f"BENCHMARK — {dataset} / {attribute}")
        print("=" * 74)
        X, y, A = LOADERS[dataset]()
        if attribute not in A.columns:
            print(f"  SKIP: '{attribute}' not in {list(A.columns)}")
            continue
        print(f"  {X.shape[0]:,} rows x {X.shape[1]} features, "
              f"default rate {y.mean():.2%}")
        df = run_grid(X, y, A[attribute], attribute=attribute, dataset=dataset,
                      conditions=("explicit", "latent"), n_splits=splits,
                      n_repeats=repeats, model_kind=model_kind, verbose=False)
        allr.append(df)
        s = summarise(df)
        print(s[["method", "condition", "auc_mean", "dp_diff_mean",
                 "di_ratio_mean", "eo_diff_mean",
                 "probe_auc_after_mean"]].round(4).to_string(index=False))
        print(f"  ({time.time()-t0:.1f}s)\n")
        df.to_csv(BENCH / f"{dataset}_{attribute}_folds.csv", index=False)
        s.to_csv(BENCH / f"{dataset}_{attribute}_summary.csv", index=False)

    if allr:
        pd.concat(allr, ignore_index=True).to_csv(
            BENCH / "benchmarks_folds.csv", index=False)
        print(f"wrote {BENCH}")


def proxy_v2_comparable(subsample=8000, splits=5, repeats=5, model_kind="gbm"):
    """The chosen iteration-2 configuration under the comparison protocol.

    The validation/test runs measured it at 50k on a single held-out split.
    The comparison table is 8,000 rows over 25 cross-validation folds, so the
    two cannot be placed in adjacent rows without re-measuring. This is the
    same configuration, not a new one.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    allr = []
    for attribute in ("CODE_GENDER", "REGION_RATING_CLIENT"):
        t0 = time.time()
        print("=" * 74)
        print(f"PROXY_AWARE_V2 under comparison protocol — {attribute}")
        print("=" * 74)
        X, y, A = LOADERS["home_credit"](subsample=subsample)
        df = run_grid(X, y, A[attribute], attribute=attribute,
                      dataset="home_credit", methods=["proxy_aware_v2"],
                      conditions=("latent",), n_splits=splits,
                      n_repeats=repeats, model_kind=model_kind, verbose=False)
        allr.append(df)
        s = summarise(df)
        print(s[["method", "condition", "auc_mean", "dp_diff_mean",
                 "di_ratio_mean", "eo_diff_mean",
                 "probe_auc_after_mean"]].round(4).to_string(index=False))
        print(f"  ({time.time()-t0:.1f}s)\n")
    df = pd.concat(allr, ignore_index=True)
    df.to_csv(OUT / "proxy_v2_comparable_folds.csv", index=False)
    summarise(df).to_csv(OUT / "proxy_v2_comparable_summary.csv", index=False)
    print(f"wrote {OUT/'proxy_v2_comparable_summary.csv'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "training_only"
    if mode == "training_only":
        training_only()
    elif mode == "benchmarks":
        benchmarks()
    elif mode == "proxy_v2":
        proxy_v2_comparable()
    else:
        raise SystemExit("mode must be training_only | benchmarks | proxy_v2")
