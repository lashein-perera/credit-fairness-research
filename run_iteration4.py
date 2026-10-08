"""DSR iteration 4 — repair, verify, roll back — and the strong-probe measurement.

    python run_iteration4.py --mode heldout     # strong probe + held-out analysis
    python run_iteration4.py --mode folds       # 25-fold evaluation
    python run_iteration4.py --mode report      # criteria and final table

HELD-OUT ANALYSIS
    The split of the iteration-1 transition analysis: 50,000-row Home Credit
    sample, test_size=0.3, stratify=y, random_state=42, so the same 15,000
    held-out rows. The baseline cross-plot and its frozen noise-floor
    thresholds are recomputed and checked against the stored ones.

    One ProxyAwareRollback fit serves both arms. Its fitting up to the
    rollback step is iteration 1 exactly, so `repaired()` and
    `unrolled_model_` give the iteration-1 arm and `transform()` and `model_`
    the iteration-4 arm. The iteration-1 numbers are checked against
    results/mitigation_quadrant_summary.csv.

STRONG PROBE
    Module 1's audit probe: gradient boosting, 5-fold stratified CV
    (probe_auc), with shuffle_test as the control. For the three-group
    regional rating it scores one-vs-rest macro AUC, as the audit does; the
    binarised score (first group against the rest, the weak probe's target)
    is reported alongside.

25-FOLD EVALUATION
    The combined-table protocol: 8,000-row sample, 5x5 repeated stratified
    folds, training-only access.

OUTPUTS (results/iteration4/; earlier results untouched)
    strong_probe.csv              weak and strong probe, held-out split
    heldout_summary.csv           baseline, iteration 1, iteration 4
    heldout_transitions.csv       per-feature movement, both arms
    repair_cardinality_trace.csv  distinct values per repair round, binary features
    folds_<attribute>.csv         25-fold rows with fit diagnostics
    final_table.csv               25-fold means against iteration 1
    success_criteria.csv          pre-registered criteria (i)-(iv)
"""
from __future__ import annotations

import argparse
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import _probe, resolve_threshold, run_grid, summarise  # noqa: E402
from evaluation.metrics import (auc_roc, demographic_parity_difference,  # noqa: E402
                                disparate_impact_ratio,
                                equalized_odds_difference)
from leakage.probe import probe_auc, rank_leaky_features, shuffle_test  # noqa: E402
from mitigation.proxy_aware import ProxyAwareRollback  # noqa: E402
from models.baselines import build_model  # noqa: E402


# Module 5 is loaded from source rather than imported. On this machine the
# normal import of the `explain` package stalls indefinitely inside Python's
# code-loading path, which plain file reads bypass; the module itself is
# healthy and this route produces identical objects.
def _load_module_5():
    path = ROOT / "src" / "python" / "explain" / "shap_layer.py"
    mod = types.ModuleType("shap_layer")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


_m5 = _load_module_5()

RESULTS = ROOT / "results"
OUT = RESULTS / "iteration4"
RANDOM_STATE = 42
ATTRS = ["CODE_GENDER", "REGION_RATING_CLIENT"]
METHOD = "proxy_aware_rollback"

# iteration 1, the reference every criterion is measured against
IT1_EMERGED = {"CODE_GENDER": 24, "REGION_RATING_CLIENT": 23}
CRITERIA = {
    "CODE_GENDER": dict(emerged=12, probe=0.570, dp=0.020, auc=0.690),
    "REGION_RATING_CLIENT": dict(emerged=11, probe=0.608, dp=0.043, auc=0.669),
}


def _fairness(y_true, proba, y_fit, a):
    thr = resolve_threshold(proba, y_fit, None)
    pred = (np.asarray(proba) >= thr).astype(int)
    return {"auc": auc_roc(y_true, proba),
            "dp_diff": demographic_parity_difference(pred, a),
            "di_ratio": disparate_impact_ratio(pred, a),
            "eo_diff": equalized_odds_difference(y_true, pred, a)}


def _analyse(model, X, a, thresholds=None):
    imp = _m5.global_importance(model, X, model_kind="gbm", max_rows=2_000)
    leak = rank_leaky_features(X, a, top_k=X.shape[1])
    return _m5.leakage_vs_shap(leak, imp, thresholds=thresholds)


def _strong(X, a, label, attribute):
    """Audit probe on X, its shuffled control, and the binarised score."""
    a = pd.Series(a).reset_index(drop=True)
    X = X.reset_index(drop=True)
    actual = probe_auc(X, a, kind="gbm")
    control = shuffle_test(X, a, kind="gbm")
    a_str = a.astype(str)
    binary = (probe_auc(X, (a_str == sorted(a_str.unique())[0]).astype(int), kind="gbm")
              if a_str.nunique() > 2 else actual)
    return {"attribute": attribute, "features": label,
            "strong_probe_auc": actual["auc_mean"], "strong_probe_sd": actual["auc_std"],
            "strong_probe_binary_auc": binary["auc_mean"],
            "shuffled_control_auc": control["auc_mean"],
            "shuffled_control_sd": control["auc_std"]}


def heldout():
    OUT.mkdir(parents=True, exist_ok=True)
    stored = pd.read_csv(RESULTS / "mitigation_quadrant_summary.csv")
    combined = pd.read_csv(RESULTS / "iteration2" / "combined_comparison.csv")
    probes, summary, transitions, trace = [], [], [], []

    for attribute in ATTRS:
        t0 = time.time()
        print("=" * 74); print(f"HELD-OUT — {attribute}"); print("=" * 74, flush=True)
        X, y, A = LOADERS["home_credit"](subsample=50_000)
        Xtr, Xte, ytr, yte, atr, ate = train_test_split(
            X, y, A[attribute], test_size=0.3, stratify=y, random_state=RANDOM_STATE)
        for f in (Xtr, Xte, ytr, yte, atr, ate):
            f.reset_index(drop=True, inplace=True)
        print(f"  train {len(Xtr):,}  held-out {len(Xte):,}", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = build_model("gbm", Xtr)
            base.fit(Xtr, ytr)
            before = _analyse(base, Xte, ate)
        thresholds = _m5.default_thresholds(before)
        before["quadrant"] = _m5.leakage_vs_shap_quadrant(before, thresholds)
        nuq_before = {c: int(Xte[c].nunique(dropna=False)) for c in Xte.columns}
        st = stored[stored.attribute == attribute].set_index("arm")
        print(f"  baseline  ({time.time()-t0:.0f}s)  noise floor {thresholds[0]:.6f}  "
              f"leaky+relied {int((before.quadrant == 'leaky_and_relied_on').sum())} "
              f"(stored {int(st.loc['baseline', 'n_leaky_and_relied_on'])})", flush=True)

        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ProxyAwareRollback(model_kind="gbm")
            m.fit(Xtr, ytr, atr)
        d = m.diagnostics_
        print(f"  fitted ({time.time()-t:.0f}s)  stop: {d['stop_reason']}  "
              f"rounds {d['rounds']}  treated {d['n_treated']}  "
              f"rolled back {d['n_rolled_back']}  still emerged {d['n_still_emerged_after_rollback']}",
              flush=True)

        # Q4 evidence: distinct values of binary features after each round
        binary = [c for c in Xte.columns if nuq_before[c] <= 2]
        Xc = Xte.copy()
        for r, (tf, feats) in enumerate(m.transformers_, start=1):
            Xc = tf.transform(Xc, feats)
            for c in binary:
                trace.append({"attribute": attribute, "feature": c, "round": r,
                              "treated": c in feats,
                              "n_unique": int(Xc[c].nunique(dropna=False))})

        arms = {
            "iteration 1": (m.repaired(Xte), m.unrolled_model_),
            "iteration 4": (m.transform(Xte), m.model_),
        }
        weak_base = _probe(Xte, ate)
        fair_base = _fairness(yte, base.predict_proba(Xte)[:, 1], ytr, ate)
        summary.append({"attribute": attribute, "arm": "baseline", **fair_base,
                        "weak_probe": weak_base, "n_emerged": 0,
                        "n_leaky_and_relied_on": int((before.quadrant == "leaky_and_relied_on").sum())})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probes.append(_strong(Xte, ate, "original", attribute) | {"weak_probe": weak_base})

        for arm, (Xt, model) in arms.items():
            t = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fair = _fairness(yte, model.predict_proba(Xt)[:, 1], ytr, ate)
                weak = _probe(Xt, ate)
                after = _analyse(model, Xt, ate, thresholds=thresholds)
                nuq_after = {c: int(Xt[c].nunique(dropna=False)) for c in Xt.columns}
                tr = _m5.quadrant_transitions(before, after, leakage_threshold=thresholds[0],
                                              n_unique_before=nuq_before,
                                              n_unique_after=nuq_after)
                strong = _strong(Xt, ate, f"repaired, {arm}", attribute)
            n_em = int(tr.leakage_emerged.sum())
            summary.append({"attribute": attribute, "arm": arm, **fair, "weak_probe": weak,
                            "n_emerged": n_em,
                            "n_leaky_and_relied_on": int((after.quadrant == "leaky_and_relied_on").sum()),
                            "strong_probe": strong["strong_probe_auc"],
                            **({k: d[k] for k in ("stop_reason", "rounds", "n_treated",
                                                  "n_rolled_back", "n_emerged_validation",
                                                  "n_still_emerged_after_rollback")}
                               if arm == "iteration 4" else
                               {k: d[k] for k in ("stop_reason", "rounds", "n_treated")})})
            probes.append(strong | {"weak_probe": weak})
            tr.insert(0, "arm", arm); tr.insert(0, "attribute", attribute)
            transitions.append(tr)
            print(f"  {arm} ({time.time()-t:.0f}s)  AUC {fair['auc']:.4f}  weak {weak:.4f}  "
                  f"strong {strong['strong_probe_auc']:.4f}  emerged {n_em}", flush=True)
            if arm == "iteration 1":
                print(f"    stored iteration 1: AUC {st.loc['full', 'auc_test']:.4f}  "
                      f"weak {st.loc['full', 'probe_auc_aggregate']:.4f}  "
                      f"emerged {int(st.loc['full', 'n_leakage_emerged'])}", flush=True)

        ref = combined[(combined.attribute == attribute) & (combined.method.isin(["none", "proxy_aware"]))]
        for p in probes:
            if p["attribute"] == attribute:
                key = "none" if p["features"] == "original" else "proxy_aware"
                if p["features"] != "repaired, iteration 4":
                    p["weak_probe_25fold"] = float(ref[ref.method == key].probe_auc_after_mean.iloc[0])

        pd.DataFrame(probes).to_csv(OUT / "strong_probe.csv", index=False)
        pd.DataFrame(summary).to_csv(OUT / "heldout_summary.csv", index=False)
        pd.concat(transitions).to_csv(OUT / "heldout_transitions.csv", index=False)
        pd.DataFrame(trace).to_csv(OUT / "repair_cardinality_trace.csv", index=False)
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def folds(splits=5, repeats=5):
    OUT.mkdir(parents=True, exist_ok=True)
    for attribute in ATTRS:
        t0 = time.time()
        print("=" * 74); print(f"25 FOLDS — {attribute}"); print("=" * 74, flush=True)
        X, y, A = LOADERS["home_credit"](subsample=8000)
        df = run_grid(X, y, A[attribute], attribute=attribute, dataset="home_credit",
                      methods=[METHOD], conditions=("training_only",),
                      n_splits=splits, n_repeats=repeats, verbose=True)
        df.to_csv(OUT / f"folds_{attribute}.csv", index=False)
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def _relabel_stop(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ProxyAwareRollback._stop_reason's corrected ordering to rows written
    before it was fixed: a fit that used all 10 rounds ended on the round cap,
    even when the probe reached tau in that last round."""
    if "stop_reason" not in df or "rounds" not in df:
        return df
    capped = (df.stop_reason == "probe at or below tau") & (df.rounds >= 10)
    df.loc[capped, "stop_reason"] = "max rounds (probe reached tau in last round)"
    return df


def report():
    combined = pd.read_csv(RESULTS / "iteration2" / "combined_comparison.csv")
    held = _relabel_stop(pd.read_csv(OUT / "heldout_summary.csv"))
    held.to_csv(OUT / "heldout_summary.csv", index=False)
    rows, crit = [], []
    for attribute in ATTRS:
        df = _relabel_stop(pd.read_csv(OUT / f"folds_{attribute}.csv"))
        df.to_csv(OUT / f"folds_{attribute}.csv", index=False)
        s = summarise(df).iloc[0]
        ref = combined[(combined.attribute == attribute)
                       & (combined.method.isin(["none", "reweighing", "proxy_aware"]))
                       & (combined.condition == "training_only")]
        for _, r in ref.iterrows():
            rows.append({"attribute": attribute, "method": r.method,
                         "AUC": r.auc_mean, "DP diff": r.dp_diff_mean,
                         "DI ratio": r.di_ratio_mean, "EO diff": r.eo_diff_mean,
                         "probe AUC": r.probe_auc_after_mean})
        rows.append({"attribute": attribute, "method": METHOD,
                     "AUC": s.auc_mean, "DP diff": s.dp_diff_mean,
                     "DI ratio": s.di_ratio_mean, "EO diff": s.eo_diff_mean,
                     "probe AUC": s.probe_auc_after_mean})

        h = held[(held.attribute == attribute)].set_index("arm")
        c = CRITERIA[attribute]
        checks = [
            ("i_manufactured_proxies_halved", int(h.loc["iteration 4", "n_emerged"]) <= c["emerged"],
             f"{int(h.loc['iteration 4', 'n_emerged'])} vs iteration 1 "
             f"{int(h.loc['iteration 1', 'n_emerged'])} (limit {c['emerged']})"),
            ("ii_weak_probe_within_0.02", s.probe_auc_after_mean <= c["probe"],
             f"{s.probe_auc_after_mean:.4f} (limit {c['probe']:.3f})"),
            ("iii_dp_within_0.005", s.dp_diff_mean <= c["dp"],
             f"{s.dp_diff_mean:.4f} (limit {c['dp']:.3f})"),
            ("iv_auc_within_0.005", s.auc_mean >= c["auc"],
             f"{s.auc_mean:.4f} (limit {c['auc']:.3f})"),
        ]
        for name, ok, ev in checks:
            crit.append({"attribute": attribute, "criterion": name, "passed": bool(ok),
                         "evidence": ev})

        stops = df.stop_reason.value_counts().to_dict()
        print(f"\n{attribute}: iteration-1 stopping rule across 25 folds {stops}; "
              f"rounds {df.rounds.min()}-{df.rounds.max()}; "
              f"rolled back mean {df.n_rolled_back.mean():.1f}; "
              f"still emerged after rollback mean {df.n_still_emerged_after_rollback.mean():.1f}")

    final = pd.DataFrame(rows).round(4)
    final.to_csv(OUT / "final_table.csv", index=False)
    crit = pd.DataFrame(crit)
    crit.to_csv(OUT / "success_criteria.csv", index=False)
    print("\n" + final.to_string(index=False))
    print("\n" + crit.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heldout", "folds", "report"], required=True)
    args = ap.parse_args()
    {"heldout": heldout, "folds": folds, "report": report}[args.mode]()
