"""DSR iteration 2 of the proxy-aware mitigator — guarded repair.

    python run_iteration2.py --attribute CODE_GENDER
    python run_iteration2.py --attribute REGION_RATING_CLIENT

Iteration 1 reduced aggregate leakage but manufactured new proxies, exceeded the
accuracy budget, and left per-feature leakage among relied-upon features no
better than baseline. This iteration adds a degeneracy guard to the conditional
repair and selects among a small set of candidates on a validation split.

PROTOCOL
    The 50k subsample is split exactly as in iteration 1 (test_size=0.3,
    stratify=y, random_state=42) so the held-out test set is the same 15,000
    rows. The 35,000-row training portion is split again into fit/validation.
    Candidates are compared on VALIDATION ONLY; the winner is then run once on
    the untouched test split, against baseline thresholds and the noise-floor
    leakage cut-point from iteration 1.

OUTPUTS (results/iteration2/, iteration 1 untouched)
    validation_scores.csv     every candidate on the validation split
    test_results.csv          the chosen configuration on held-out test
    success_criteria.csv      pass/fail per pre-registered criterion
    transitions_<attr>.csv    per-feature movement for the chosen config
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import _probe, resolve_threshold  # noqa: E402
from evaluation.metrics import (auc_roc, demographic_parity_difference,  # noqa: E402
                                disparate_impact_ratio,
                                equalized_odds_difference)
# Module 5 is loaded from source rather than imported. On this machine the
# normal import of the `explain` package stalls indefinitely inside Python's
# code-loading path, which plain file reads bypass; the module itself is
# healthy and this route produces identical objects.
def _load_module_5():
    import types
    path = ROOT / "src" / "python" / "explain" / "shap_layer.py"
    mod = types.ModuleType("shap_layer")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


_m5 = _load_module_5()
default_thresholds = _m5.default_thresholds
global_importance = _m5.global_importance
leakage_vs_shap = _m5.leakage_vs_shap
leakage_vs_shap_quadrant = _m5.leakage_vs_shap_quadrant
quadrant_transitions = _m5.quadrant_transitions
residual_leakage = _m5.residual_leakage
from leakage.probe import rank_leaky_features  # noqa: E402
from mitigation.proxy_aware import ProxyAwareMitigator  # noqa: E402
from models.baselines import build_model  # noqa: E402

OUT = ROOT / "results" / "iteration2"
RANDOM_STATE = 42

# The agreed candidate list. All four carry max_accuracy_loss=0.02, which is
# success criterion (c) enforced during fitting rather than hoped for after it,
# and is the value config.py has always declared but never had wired up.
CANDIDATES = {
    "C1_guard99": dict(skip_dominant_share=0.99, max_accuracy_loss=0.02),
    "C2_guard95_binary": dict(skip_dominant_share=0.95, skip_min_unique=3,
                              max_accuracy_loss=0.02),
    "C3_guard99_partial50": dict(skip_dominant_share=0.99, repair_level=0.5,
                                 max_accuracy_loss=0.02),
    "C4_residual_guard99": dict(skip_dominant_share=0.99,
                                strategy="residualisation",
                                max_accuracy_loss=0.02),
}

MIN_PROBE_DROP = 0.15      # criterion (a)
MAX_AUC_DROP = 0.02        # criterion (c)


def _fairness(y_true, proba, y_fit, a):
    thr = resolve_threshold(proba, y_fit, None)
    pred = (np.asarray(proba) >= thr).astype(int)
    return {
        "auc": auc_roc(y_true, proba),
        "dp_diff": demographic_parity_difference(pred, a),
        "di_ratio": disparate_impact_ratio(pred, a),
        "eo_diff": equalized_odds_difference(y_true, pred, a),
    }


def _measure(model, X, a, model_kind, max_rows=2_000):
    imp = global_importance(model, X, model_kind=model_kind, max_rows=max_rows)
    leak = rank_leaky_features(X, a, top_k=X.shape[1])
    return leakage_vs_shap(leak, imp)


def evaluate(name, mit, Xfit, yfit, Xev, yev, aev, model_kind,
             base_cross, thresholds, base_res, base_fair, base_probe,
             nuq_before):
    """One candidate (or the baseline) measured on one evaluation split."""
    Xev_t = mit.transform(Xev) if mit is not None else Xev
    model = mit.model_ if mit is not None else _BASE[0]
    proba = (mit.predict_proba(Xev) if mit is not None
             else model.predict_proba(Xev)[:, 1])

    fair = _fairness(yev, proba, yfit, aev)
    probe = _probe(Xev_t, aev)

    cross = _measure(model, Xev_t, aev, model_kind)
    cross["quadrant"] = leakage_vs_shap_quadrant(cross, thresholds)
    res = residual_leakage(cross, thresholds[1])

    nuq_after = {c: int(Xev_t[c].nunique(dropna=False)) for c in Xev_t.columns}
    tr = quadrant_transitions(base_cross, cross, leakage_threshold=thresholds[0],
                              n_unique_before=nuq_before,
                              n_unique_after=nuq_after)
    emerged = int(tr.leakage_emerged.sum())

    row = {
        "candidate": name,
        "n_treated": len(getattr(mit, "treated_", [])) if mit is not None else 0,
        "n_skipped": len(getattr(mit, "skipped_", [])) if mit is not None else 0,
        "probe_auc": probe,
        "probe_drop": base_probe - probe,
        **fair,
        "auc_drop": base_fair["auc"] - fair["auc"],
        "dp_vs_base": fair["dp_diff"] - base_fair["dp_diff"],
        "eo_vs_base": fair["eo_diff"] - base_fair["eo_diff"],
        "n_leakage_emerged": emerged,
        "max_leakage_relied_on": res["max_leakage_relied_on"],
        "total_leakage_relied_on": res["total_leakage_relied_on"],
        "max_leak_vs_base": res["max_leakage_relied_on"] - base_res["max_leakage_relied_on"],
        "total_leak_vs_base": res["total_leakage_relied_on"] - base_res["total_leakage_relied_on"],
    }
    return row, cross, tr


_BASE = [None]


def run(attribute, subsample, model_kind, dataset="home_credit"):
    t0 = time.time()
    print("=" * 74)
    print(f"ITERATION 2 — {dataset} / {attribute} / {model_kind}")
    print("=" * 74)

    X, y, A = LOADERS[dataset](subsample=subsample)
    a = A[attribute]

    # identical split to iteration 1
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, a, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    # training portion split again for candidate selection
    Xfit, Xval, yfit, yval, afit, aval = train_test_split(
        Xtr, ytr, atr, test_size=0.3, stratify=ytr, random_state=RANDOM_STATE)
    for f in (Xtr, Xte, ytr, yte, atr, ate, Xfit, Xval, yfit, yval, afit, aval):
        f.reset_index(drop=True, inplace=True)

    print(f"  fit {len(Xfit):,}   validation {len(Xval):,}   "
          f"held-out test {len(Xte):,}   features {X.shape[1]}")
    print(f"  {attribute}: {dict(a.value_counts())}\n")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- PHASE 2a: validation ----------------
    print("-" * 74)
    print("PHASE 2a — candidate selection on VALIDATION split only")
    print("-" * 74)

    base_v = build_model(model_kind, Xfit); base_v.fit(Xfit, yfit)
    _BASE[0] = base_v
    pv = base_v.predict_proba(Xval)[:, 1]
    base_fair_v = _fairness(yval, pv, yfit, aval)
    base_probe_v = _probe(Xval, aval)
    base_cross_v = _measure(base_v, Xval, aval, model_kind)
    thr_v = default_thresholds(base_cross_v)
    base_cross_v["quadrant"] = leakage_vs_shap_quadrant(base_cross_v, thr_v)
    base_res_v = residual_leakage(base_cross_v, thr_v[1])
    nuq_v = {c: int(Xval[c].nunique(dropna=False)) for c in Xval.columns}

    print(f"  validation baseline: AUC {base_fair_v['auc']:.4f}  "
          f"probe {base_probe_v:.4f}  DP {base_fair_v['dp_diff']:.4f}  "
          f"EO {base_fair_v['eo_diff']:.4f}")
    print(f"  validation thresholds: leakage > {thr_v[0]:.3g}, SHAP > {thr_v[1]:.3g}\n")

    rows = []
    for name, kw in CANDIDATES.items():
        t = time.time()
        mit = ProxyAwareMitigator(model_kind=model_kind, **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mit.fit(Xfit, yfit, afit)
        r, _, _ = evaluate(name, mit, Xfit, yfit, Xval, yval, aval, model_kind,
                           base_cross_v, thr_v, base_res_v, base_fair_v,
                           base_probe_v, nuq_v)
        r["fit_s"] = round(time.time() - t, 1)
        rows.append(r)
        print(f"  {name:22} treated {r['n_treated']:3} skipped {r['n_skipped']:3}  "
              f"probe -{r['probe_drop']:.4f}  auc -{r['auc_drop']:+.4f}  "
              f"emerged {r['n_leakage_emerged']:2}  "
              f"DP {r['dp_vs_base']:+.4f}  EO {r['eo_vs_base']:+.4f}  "
              f"({r['fit_s']}s)")

    val = pd.DataFrame(rows)
    val.insert(0, "attribute", attribute)
    val.to_csv(OUT / f"validation_scores_{attribute}.csv", index=False)

    # selection: hard constraints first, then maximise leakage removed
    ok = val[(val.auc_drop <= MAX_AUC_DROP) & (val.n_leakage_emerged == 0)
             & (val.dp_vs_base < 0) & (val.eo_vs_base < 0)]
    relaxed = False
    if ok.empty:
        relaxed = True
        ok = val[(val.auc_drop <= MAX_AUC_DROP)]
        if ok.empty:
            ok = val
    chosen = ok.sort_values(["probe_drop", "total_leak_vs_base"],
                            ascending=[False, True]).iloc[0]
    print(f"\n  SELECTED: {chosen.candidate}"
          + ("   (no candidate met every hard constraint on validation; "
             "selected under relaxed rule)" if relaxed else ""))

    # ---------------- PHASE 2b: held-out test, once ----------------
    print("\n" + "-" * 74)
    print(f"PHASE 2b — {chosen.candidate} on held-out TEST split (single run)")
    print("-" * 74)

    base_t = build_model(model_kind, Xtr); base_t.fit(Xtr, ytr)
    _BASE[0] = base_t
    pt = base_t.predict_proba(Xte)[:, 1]
    base_fair_t = _fairness(yte, pt, ytr, ate)
    base_probe_t = _probe(Xte, ate)
    base_cross_t = _measure(base_t, Xte, ate, model_kind)
    thr_t = default_thresholds(base_cross_t)
    base_cross_t["quadrant"] = leakage_vs_shap_quadrant(base_cross_t, thr_t)
    base_res_t = residual_leakage(base_cross_t, thr_t[1])
    nuq_t = {c: int(Xte[c].nunique(dropna=False)) for c in Xte.columns}

    print(f"  baseline: AUC {base_fair_t['auc']:.4f}  probe {base_probe_t:.4f}  "
          f"DP {base_fair_t['dp_diff']:.4f}  EO {base_fair_t['eo_diff']:.4f}")
    print(f"  frozen thresholds: leakage > {thr_t[0]:.3g}, SHAP > {thr_t[1]:.3g}")

    mit = ProxyAwareMitigator(model_kind=model_kind,
                              **CANDIDATES[chosen.candidate])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mit.fit(Xtr, ytr, atr)
    test_row, cross_t, tr_t = evaluate(
        chosen.candidate, mit, Xtr, ytr, Xte, yte, ate, model_kind,
        base_cross_t, thr_t, base_res_t, base_fair_t, base_probe_t, nuq_t)

    base_row = {"candidate": "baseline", "n_treated": 0, "n_skipped": 0,
                "probe_auc": base_probe_t, "probe_drop": 0.0, **base_fair_t,
                "auc_drop": 0.0, "dp_vs_base": 0.0, "eo_vs_base": 0.0,
                "n_leakage_emerged": 0,
                "max_leakage_relied_on": base_res_t["max_leakage_relied_on"],
                "total_leakage_relied_on": base_res_t["total_leakage_relied_on"],
                "max_leak_vs_base": 0.0, "total_leak_vs_base": 0.0}

    test = pd.DataFrame([base_row, test_row])
    test.insert(0, "attribute", attribute)
    test.to_csv(OUT / f"test_results_{attribute}.csv", index=False)
    tr_t.insert(0, "attribute", attribute)
    tr_t.to_csv(OUT / f"transitions_{attribute}.csv", index=False)

    # ---------------- success criteria ----------------
    crit = [
        ("a_probe_drop_ge_0.15", test_row["probe_drop"] >= MIN_PROBE_DROP,
         f"{test_row['probe_drop']:.4f} >= {MIN_PROBE_DROP}"),
        ("b_dp_and_eo_below_baseline",
         bool(test_row["dp_vs_base"] < 0 and test_row["eo_vs_base"] < 0),
         f"DP {test_row['dp_vs_base']:+.4f}, EO {test_row['eo_vs_base']:+.4f}"),
        ("c_auc_drop_le_0.02", test_row["auc_drop"] <= MAX_AUC_DROP,
         f"{test_row['auc_drop']:.4f} <= {MAX_AUC_DROP}"),
        ("d_zero_manufactured_proxies", test_row["n_leakage_emerged"] == 0,
         f"{test_row['n_leakage_emerged']} emerged"),
        ("e_max_and_total_leak_below_baseline",
         bool(test_row["max_leak_vs_base"] < 0 and test_row["total_leak_vs_base"] < 0),
         f"max {test_row['max_leak_vs_base']:+.5f}, "
         f"total {test_row['total_leak_vs_base']:+.5f}"),
    ]
    cdf = pd.DataFrame([{"attribute": attribute, "candidate": chosen.candidate,
                         "criterion": c, "passed": bool(ok_), "evidence": ev}
                        for c, ok_, ev in crit])
    cdf.to_csv(OUT / f"success_criteria_{attribute}.csv", index=False)

    print(f"\n  RESULT: treated {test_row['n_treated']}, "
          f"skipped {test_row['n_skipped']}")
    print(f"  probe {base_probe_t:.4f} -> {test_row['probe_auc']:.4f}   "
          f"AUC {base_fair_t['auc']:.4f} -> {test_row['auc']:.4f}")
    print("\n" + "=" * 74)
    print("SUCCESS CRITERIA")
    print("=" * 74)
    for c, ok_, ev in crit:
        print(f"  [{'PASS' if ok_ else 'FAIL'}]  {c:38} {ev}")

    em = tr_t[tr_t.leakage_emerged].nlargest(5, "leakage_drop_after")
    print(f"\n  manufactured proxies: {test_row['n_leakage_emerged']}")
    if not em.empty:
        print(em[["feature", "leakage_drop_before", "leakage_drop_after",
                  "n_unique_before", "n_unique_after"]].to_string(index=False))
    fcm = tr_t[tr_t.feature == "FLAG_CONT_MOBILE"]
    if not fcm.empty:
        r = fcm.iloc[0]
        print(f"\n  FLAG_CONT_MOBILE: leakage {r.leakage_drop_before:.6f} -> "
              f"{r.leakage_drop_after:.6f}   unique {int(r.n_unique_before)} -> "
              f"{int(r.n_unique_after)}   (guarded: "
              f"{'FLAG_CONT_MOBILE' in mit.skipped_})")

    print(f"\ntotal {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribute", default="CODE_GENDER")
    ap.add_argument("--subsample", type=int, default=50_000)
    ap.add_argument("--model", default="gbm")
    args = ap.parse_args()
    run(args.attribute, args.subsample, args.model)
