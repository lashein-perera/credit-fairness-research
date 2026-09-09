"""Does Module 4 empty the danger quadrant? — Modules 4 x 5, for RQ3.

    python run_mitigation_explainability.py                            # gender
    python run_mitigation_explainability.py --attribute REGION_RATING_CLIENT

Re-runs the leakage-vs-SHAP cross analysis on the model AFTER proxy-aware
mitigation, and compares it against the unmitigated baseline.

PROTOCOL
    Everything is fitted on a training split and measured on a held-out test
    split, the baseline included, so the two sides are like for like.

FROZEN QUADRANT BOUNDARIES
    The quadrant cut-points are taken from the BASELINE and reused unchanged
    for every mitigated arm. Recomputing medians on mitigated data would move
    the boundary down with the data and leave roughly a fixed share of features
    in the top-right quadrant however well the mitigation worked, which would
    make the comparison meaningless.

    The leakage cut-point is additionally floored at the permutation noise
    level rather than taken at the median, because the leakage distribution is
    zero-inflated and its median sits at zero. See default_thresholds().

    A quadrant count is only as stable as its boundary, so the cut-point-free
    measures from residual_leakage() are reported alongside it and are the
    ones to quote.

ARMS, ORDERED BY HOW TARGETED THEY ACTUALLY ARE
    At the default configuration (top_k=15, max_iter=10) cumulative selection
    exhausts all 116 features before the leakage target is met, so 'full' and
    'random' both treat the entire feature space and differ only in the order
    they reach it. Targeting is inoperative there. The two restricted arms are
    the ones where it is live:

        single_pass   max_iter=1   ~15 features treated
        top_k=5       top_k=5      up to 50 features treated
        full          default      all 116 — targeting inoperative
        random        control      all 116, chosen at random

    If targeting carries the mechanism, the restricted arms should buy more
    leakage reduction per unit of accuracy than 'random'.

ACCURACY IS REPORTED ALONGSIDE
    An arm that treats fewer features leaves residual leakage closer to
    baseline almost by construction, so leakage figures alone would flatter the
    restricted arms. Test AUC and the aggregate probe AUC on the transformed
    test set are recorded so the trade-off is visible. The aggregate probe is
    harness._probe, the same measure as run_ablation.py, so these rows are
    directly comparable to results/ablation.csv.

Outputs
    results/mitigation_cross_analysis.csv    per-feature, per-arm
    results/mitigation_quadrant_summary.csv  quadrant counts per arm
    results/mitigation_transitions.csv       per-feature before -> after
    results/figures/leakage_vs_shap_beforeafter_*.png
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import _probe  # noqa: E402
from evaluation.metrics import auc_roc  # noqa: E402
from explain.shap_layer import (  # noqa: E402
    before_after_plot, default_thresholds, global_importance, leakage_vs_shap,
    leakage_vs_shap_quadrant, quadrant_transitions, rank_correlation,
    residual_leakage,
)
from leakage.probe import rank_leaky_features  # noqa: E402
from mitigation.proxy_aware import ProxyAwareMitigator  # noqa: E402
from models.baselines import build_model  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
RANDOM_STATE = 42
MIN_AUC_DROP = 0.002   # below this an AUC change is fold noise

# arm name -> ProxyAwareMitigator kwargs. Empty dict is the full method, as in
# run_ablation.py; 'random' is the decisive ablation control.
ARMS = {
    "single_pass": dict(max_iter=1),
    "top_k=5": dict(top_k=5),
    "full": dict(),
    "random": dict(random_features=True),
}

PINNED = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


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


def _analyse(model, X, a, model_kind, max_rows, thresholds=None):
    """SHAP importance and leakage for one fitted model on one feature frame.

    With `thresholds` None the frame is returned classified against its own
    medians, which the caller uses only to derive the frozen cut-points.
    """
    imp = global_importance(model, X, model_kind=model_kind, max_rows=max_rows)
    leak = rank_leaky_features(X, a, top_k=X.shape[1])
    return leakage_vs_shap(leak, imp, thresholds=thresholds)


def run(dataset, attribute, subsample, model_kind, max_rows, test_size):
    print("=" * 70)
    print(f"MITIGATION x EXPLAINABILITY — {dataset} / {attribute} / {model_kind}")
    print("=" * 70)

    t0 = time.time()
    loader = LOADERS[dataset]
    X, y, A = loader(subsample=subsample) if dataset == "home_credit" else loader()
    if attribute not in A.columns:
        raise SystemExit(f"attribute '{attribute}' not in {list(A.columns)}")
    a = A[attribute]

    leaked = [c for c in A.columns if c in X.columns]
    if leaked:
        raise SystemExit(f"FATAL: protected attribute(s) {leaked} present in X")

    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, a, test_size=test_size, stratify=y, random_state=RANDOM_STATE)
    for f in (Xtr, Xte, ytr, yte, atr, ate):
        f.reset_index(drop=True, inplace=True)

    print(f"\nloaded in {time.time()-t0:.1f}s")
    print(f"  X     : {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"  train : {len(Xtr):,}   test : {len(Xte):,}")
    print(f"  {attribute}: {dict(a.value_counts())}")
    print("  protected attributes absent from X — isolation holds")

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ---------------- baseline ----------------
    print(f"\n{'-'*70}\nBASELINE — no mitigation")
    t = time.time()
    base = build_model(model_kind, Xtr)
    base.fit(Xtr, ytr)
    auc_base = auc_roc(yte, base.predict_proba(Xte)[:, 1])
    probe_base = _probe(Xte, ate)
    before = _analyse(base, Xte, ate, model_kind, max_rows)
    thresholds = default_thresholds(before)
    before["quadrant"] = leakage_vs_shap_quadrant(before, thresholds)
    n_before = int((before.quadrant == "leaky_and_relied_on").sum())
    res_before = residual_leakage(before, thresholds[1])
    print(f"  ({time.time()-t:.1f}s)  leaky and relied upon: "
          f"{n_before}/{len(before)}")
    print(f"  frozen quadrant thresholds: leakage > {thresholds[0]:.3g} "
          f"(permutation noise floor), SHAP > {thresholds[1]:.3g}")
    print(f"  relied-upon features: {res_before['n_relied_on']}   "
          f"max leakage {res_before['max_leakage_relied_on']:.5f}   "
          f"total {res_before['total_leakage_relied_on']:.5f}")
    print(f"  test AUC {auc_base:.4f}   aggregate probe AUC {probe_base:.4f}")
    nuq_before = {c: int(Xte[c].nunique(dropna=False)) for c in Xte.columns}

    panels = [("baseline (no mitigation)", before)]
    rows, cross_all, trans_all = [], [], []

    st = rank_correlation(before)
    rows.append({"dataset": dataset, "attribute": attribute, "model": model_kind,
                 "arm": "baseline", "n_features": len(before),
                 "n_leaky_and_relied_on": n_before,
                 **{k: v for k, v in st.items() if k != "n_features"},
                 **res_before, "auc_test": auc_base,
                 "probe_auc_aggregate": probe_base, "auc_drop": 0.0,
                 "n_treated": 0, "left_danger": 0, "entered_danger": 0,
                 "n_leakage_emerged": 0})
    c = before.copy(); c.insert(0, "arm", "baseline")
    cross_all.append(c)

    # ---------------- mitigated arms ----------------
    for arm, kw in ARMS.items():
        print(f"\n{'-'*70}\n{arm.upper()}")
        t = time.time()
        mit = ProxyAwareMitigator(model_kind=model_kind, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mit.fit(Xtr, ytr, atr)
            Xte_t = mit.transform(Xte)
        auc_arm = auc_roc(yte, mit.predict_proba(Xte))
        probe_arm = _probe(Xte_t, ate)
        print(f"  fitted ({time.time()-t:.1f}s)  treated {len(mit.treated_)} features"
              f"  test AUC {auc_arm:.4f} ({auc_arm-auc_base:+.4f})"
              f"  aggregate probe AUC {probe_arm:.4f}")

        t = time.time()
        after = _analyse(mit.model_, Xte_t, ate, model_kind, max_rows,
                         thresholds=thresholds)
        n_after = int((after.quadrant == "leaky_and_relied_on").sum())
        res = residual_leakage(after, thresholds[1])
        print(f"  ({time.time()-t:.1f}s)  leaky and relied upon: "
              f"{n_after}/{len(after)}   ({n_after - n_before:+d} vs baseline)")
        print(f"  max leakage among relied-upon: "
              f"{res['max_leakage_relied_on']:.5f} "
              f"({100*(res['max_leakage_relied_on']/res_before['max_leakage_relied_on']-1):+.0f}%)   "
              f"total {res['total_leakage_relied_on']:.5f} "
              f"({100*(res['total_leakage_relied_on']/res_before['total_leakage_relied_on']-1):+.0f}%)")

        nuq_after = {c: int(Xte_t[c].nunique(dropna=False)) for c in Xte_t.columns}
        tr = quadrant_transitions(before, after, leakage_threshold=thresholds[0],
                                  n_unique_before=nuq_before,
                                  n_unique_after=nuq_after)
        left = int((tr.transition == "left_danger").sum())
        entered = int((tr.transition == "entered_danger").sum())
        emerged = int(tr.leakage_emerged.sum())
        print(f"  left danger quadrant: {left}    entered: {entered}")

        # Features that carried no recoverable signal before and do now. A
        # repair stratified on a predicted attribute can write stratum identity
        # into a near-constant column; the cardinality jump is the evidence.
        em = tr[tr.leakage_emerged].nlargest(3, "leakage_drop_after")
        print(f"  MANUFACTURED PROXIES: {emerged} features gained leakage from below "
              f"the noise floor")
        if not em.empty:
            print(em[["feature", "leakage_drop_before", "leakage_drop_after",
                      "shap_importance_after", "n_unique_before",
                      "n_unique_after"]].to_string(index=False))

        st = rank_correlation(after)
        rows.append({"dataset": dataset, "attribute": attribute, "model": model_kind,
                     "arm": arm, "n_features": len(after),
                     "n_leaky_and_relied_on": n_after,
                     **{k: v for k, v in st.items() if k != "n_features"},
                     **res, "auc_test": auc_arm,
                     "probe_auc_aggregate": probe_arm,
                     "auc_drop": auc_base - auc_arm,
                     "n_leakage_emerged": emerged,
                     "n_treated": len(mit.treated_),
                     "left_danger": left, "entered_danger": entered})

        c = after.copy(); c.insert(0, "arm", arm)
        cross_all.append(c)
        tr.insert(0, "arm", arm)
        trans_all.append(tr)
        panels.append((arm, after))

        pin = tr[tr.feature.isin(PINNED)]
        if not pin.empty:
            print(f"\n  EXT_SOURCE_* movement:")
            print(pin[["feature", "leakage_drop_before", "leakage_drop_after",
                       "shap_importance_before", "shap_importance_after",
                       "transition"]].to_string(index=False))

    # ---------------- outputs ----------------
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("QUADRANT SUMMARY (boundaries frozen at baseline)")
    print("=" * 70)
    print(summary[["arm", "n_leaky_and_relied_on", "left_danger",
                   "entered_danger", "n_treated", "spearman_rho",
                   "spearman_p"]].to_string(index=False))
    print("\nCUT-POINT-FREE MEASURES (leakage carried by relied-upon features)")
    print(summary[["arm", "n_treated", "max_leakage_relied_on",
                   "mean_leakage_relied_on",
                   "total_leakage_relied_on"]].to_string(index=False))
    print("\nTRADE-OFF — leakage reduction bought per point of accuracy")
    t = summary.copy()
    t["total_leak_reduction"] = (t.total_leakage_relied_on.iloc[0]
                                 - t.total_leakage_relied_on)
    # The ratio is only meaningful where accuracy was actually given up. Below
    # this floor the drop is indistinguishable from fold noise and can be
    # negative, which would flip the ratio's sign rather than shrink it.
    meaningful = t.auc_drop > MIN_AUC_DROP
    t["per_auc_point"] = (t.total_leak_reduction / t.auc_drop).where(meaningful)
    print(t[["arm", "auc_test", "auc_drop", "probe_auc_aggregate",
             "total_leak_reduction", "per_auc_point",
             "n_leakage_emerged"]].round(5).to_string(index=False))
    if (~meaningful).any():
        free = [a for a, m in zip(t.arm, meaningful) if not m and a != "baseline"]
        if free:
            print(f"  no measurable accuracy cost (auc_drop <= {MIN_AUC_DROP}), "
                  f"ratio undefined: {', '.join(free)}")

    _append(RESULTS / "mitigation_quadrant_summary.csv", summary,
            ["dataset", "attribute", "model"])
    xa = pd.concat(cross_all, ignore_index=True)
    xa.insert(0, "model", model_kind); xa.insert(0, "attribute", attribute)
    xa.insert(0, "dataset", dataset)
    _append(RESULTS / "mitigation_cross_analysis.csv", xa,
            ["dataset", "attribute", "model"])
    ta = pd.concat(trans_all, ignore_index=True)
    ta.insert(0, "model", model_kind); ta.insert(0, "attribute", attribute)
    ta.insert(0, "dataset", dataset)
    _append(RESULTS / "mitigation_transitions.csv", ta,
            ["dataset", "attribute", "model"])

    for f in ("mitigation_quadrant_summary", "mitigation_cross_analysis",
              "mitigation_transitions"):
        print(f"wrote {RESULTS/f}.csv")

    top = before.nlargest(4, "shap_importance").feature.tolist()
    highlight = list(dict.fromkeys(PINNED + top))
    out = FIGURES / f"leakage_vs_shap_beforeafter_{dataset}_{attribute}_{model_kind}.png"
    before_after_plot(panels, out, thresholds, dataset=dataset,
                      attribute=attribute, model_kind=model_kind,
                      highlight=highlight, ncols=3)
    print(f"  figure -> {out}")
    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="home_credit", choices=list(LOADERS))
    ap.add_argument("--attribute", default="CODE_GENDER")
    ap.add_argument("--subsample", type=int, default=20_000)
    ap.add_argument("--model", default="gbm", choices=["logistic", "gbm", "mlp"])
    ap.add_argument("--max-rows", type=int, default=2_000)
    ap.add_argument("--test-size", type=float, default=0.3)
    args = ap.parse_args()

    run(args.dataset, args.attribute, args.subsample, args.model,
        args.max_rows, args.test_size)
