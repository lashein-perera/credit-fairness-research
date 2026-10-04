"""Accuracy-budget trade-off curve for the guarded mitigator — iteration 2.

    python run_tradeoff.py

Holds the C1_guard99 configuration fixed and varies only max_accuracy_loss, to
show what leakage reduction costs in accuracy. Reference rows are the
unmitigated model and reweighing, both under training-only access.

Two protocols, kept separate on purpose:
    AUC / DP / DI / EO / probe_auc_after   25 cross-validation folds at 8,000
                                           rows, matching the combined tables
    emerged proxies                        one held-out split at the same
                                           8,000 rows; a per-feature leakage
                                           ranking per fold would cost more
                                           than the sweep itself

Outputs results/iteration2/tradeoff_*.csv and results/iteration2/figures/.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS  # noqa: E402
from evaluation.harness import _make, run_grid, summarise  # noqa: E402
from leakage.probe import rank_leaky_features  # noqa: E402
from models.baselines import build_model  # noqa: E402


def _load_module_5():
    import types
    path = ROOT / "src" / "python" / "explain" / "shap_layer.py"
    mod = types.ModuleType("shap_layer")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


_m5 = _load_module_5()

OUT = ROOT / "results" / "iteration2"
FIGS = OUT / "figures"
RANDOM_STATE = 42

# fallback set first, so the agreed minimum completes even if time runs out
BUDGETS = ["none", "0.01", "0.03", "0.02", "0.05"]
ATTRS = ["CODE_GENDER", "REGION_RATING_CLIENT"]


def emerged_for(budget, X, y, a, model_kind="gbm"):
    """Manufactured-proxy count for one budget, on a single held-out split."""
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, a, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    for f in (Xtr, Xte, ytr, yte, atr, ate):
        f.reset_index(drop=True, inplace=True)

    base = build_model(model_kind, Xtr); base.fit(Xtr, ytr)
    imp_b = _m5.global_importance(base, Xte, model_kind=model_kind, max_rows=1500)
    leak_b = rank_leaky_features(Xte, ate, top_k=Xte.shape[1])
    cross_b = _m5.leakage_vs_shap(leak_b, imp_b)
    thr = _m5.default_thresholds(cross_b)
    cross_b["quadrant"] = _m5.leakage_vs_shap_quadrant(cross_b, thr)

    mit = _make(f"proxy_aware_b{budget}", model_kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mit.fit(Xtr, ytr, atr)
        Xte_t = mit.transform(Xte)
    imp_a = _m5.global_importance(mit.model_, Xte_t, model_kind=model_kind, max_rows=1500)
    leak_a = rank_leaky_features(Xte_t, ate, top_k=Xte_t.shape[1])
    cross_a = _m5.leakage_vs_shap(leak_a, imp_a)
    cross_a["quadrant"] = _m5.leakage_vs_shap_quadrant(cross_a, thr)

    tr = _m5.quadrant_transitions(cross_b, cross_a, leakage_threshold=thr[0])
    return int(tr.leakage_emerged.sum()), len(mit.treated_), len(mit.skipped_)


def main(subsample=8000, splits=5, repeats=5, model_kind="gbm"):
    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    methods = [f"proxy_aware_b{b}" for b in BUDGETS]
    rows = []
    for attribute in ATTRS:
        X, y, A = LOADERS["home_credit"](subsample=subsample)
        a = A[attribute]
        print("=" * 74); print(f"TRADE-OFF SWEEP — {attribute}"); print("=" * 74)

        t0 = time.time()
        ref = run_grid(X, y, a, attribute=attribute, dataset="home_credit",
                       methods=["none", "reweighing"],
                       conditions=("training_only",), n_splits=splits,
                       n_repeats=repeats, model_kind=model_kind, verbose=False)
        print(f"  reference rows ({time.time()-t0:.0f}s)")

        for b, m in zip(BUDGETS, methods):
            t = time.time()
            df = run_grid(X, y, a, attribute=attribute, dataset="home_credit",
                          methods=[m], conditions=("training_only",),
                          n_splits=splits, n_repeats=repeats,
                          model_kind=model_kind, verbose=False)
            s = summarise(df).iloc[0]
            ne, ntr, nsk = emerged_for(b, X, y, a, model_kind)
            rows.append({"attribute": attribute, "budget": b,
                         "auc": round(s.auc_mean, 4), "dp_diff": round(s.dp_diff_mean, 4),
                         "di_ratio": round(s.di_ratio_mean, 4),
                         "eo_diff": round(s.eo_diff_mean, 4),
                         "probe_auc_after": round(s.probe_auc_after_mean, 4),
                         "n_emerged": ne, "n_treated": ntr, "n_skipped": nsk,
                         "runtime_s": round(time.time() - t, 1)})
            print(f"  budget {b:>5}  AUC {rows[-1]['auc']:.4f}  "
                  f"probe {rows[-1]['probe_auc_after']:.4f}  "
                  f"DP {rows[-1]['dp_diff']:.4f}  EO {rows[-1]['eo_diff']:.4f}  "
                  f"emerged {ne}  treated {ntr}  ({rows[-1]['runtime_s']}s)")
            pd.DataFrame(rows).to_csv(OUT / "tradeoff_curve.csv", index=False)

        for r in summarise(ref).itertuples():
            rows.append({"attribute": attribute, "budget": f"[ref] {r.method}",
                         "auc": round(r.auc_mean, 4), "dp_diff": round(r.dp_diff_mean, 4),
                         "di_ratio": round(r.di_ratio_mean, 4),
                         "eo_diff": round(r.eo_diff_mean, 4),
                         "probe_auc_after": round(r.probe_auc_after_mean, 4),
                         "n_emerged": 0, "n_treated": 0, "n_skipped": 0,
                         "runtime_s": 0.0})
        pd.DataFrame(rows).to_csv(OUT / "tradeoff_curve.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tradeoff_curve.csv", index=False)
    plot(df)
    print(f"\nwrote {OUT/'tradeoff_curve.csv'}")


def plot(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for attribute in ATTRS:
        d = df[df.attribute == attribute]
        if d.empty:
            continue
        sweep = d[~d.budget.str.startswith("[ref]")]
        refs = d[d.budget.str.startswith("[ref]")]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, ycol, ylab in [
                (axes[0], "probe_auc_after", "Residual leakage (probe AUC)"),
                (axes[1], "dp_diff", "Demographic parity difference")]:
            ax.plot(sweep.auc, sweep[ycol], "-o", color="#4C72B0",
                    label="guarded mitigator, varying budget")
            for r in sweep.itertuples():
                ax.annotate(r.budget, (r.auc, getattr(r, ycol)),
                            textcoords="offset points", xytext=(6, 5), fontsize=8,
                            color="#2F4B73")
            for r, mk in zip(refs.itertuples(), ["s", "^"]):
                ax.scatter(r.auc, getattr(r, ycol), marker=mk, s=70,
                           color="#C44E52",
                           label=r.budget.replace("[ref] ", "") + " (reference)")
            ax.set_xlabel("Credit-model AUC (higher is better)")
            ax.set_ylabel(ylab + " (lower is better)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Accuracy budget trade-off — home_credit / {attribute}\n"
                     "training-only access, 25 folds at 8,000 rows", fontsize=12)
        fig.tight_layout()
        out = FIGS / f"tradeoff_{attribute}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  figure -> {out}")


if __name__ == "__main__":
    main()
