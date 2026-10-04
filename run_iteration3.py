"""DSR iteration 3 and final cleanups.

    python run_iteration3.py reject_option   # A1: corrected reject-option
    python run_iteration3.py figures         # A2: redrawn trade-off figures
    python run_iteration3.py validate        # B: choose D1 or D2 on validation
    python run_iteration3.py evaluate        # B: run the chosen config once

ITERATION 3 HYPOTHESIS
    Proxy-aware repair is the only method that reduces leakage; reweighing is
    the only one that improved gender equalised odds under equal access. Repair
    alone regressed gender EO. Combining them -- repair, then reweigh on the
    repaired features -- should keep the leakage and DP gains while fixing EO.

CANDIDATES (no others)
    D1  unguarded repair (iteration-1 settings) + reweighing
    D2  guarded repair (iteration-2 C1_guard99) + reweighing

SUCCESS CRITERIA (fixed before running)
    (i)   EO no worse than unmitigated
    (ii)  DP lower than reweighing alone
    (iii) probe AUC lower than reweighing alone
    (iv)  AUC within 0.035 of unmitigated

SELECTION RULE (fixed before running)
    Validation data is disjoint from the 8,000 evaluation rows: drawn from a
    different subsample and filtered by row hash, so the choice never sees a
    row the final evaluation uses. 5-fold CV on that pool. Per attribute, the
    candidate passing more of (i)-(iv) wins; ties go to lower EO, then lower
    probe AUC.

All access is training-only: attribute at fitting, never at inference.
Outputs under results/iteration3/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS, load_home_credit  # noqa: E402
from evaluation.harness import run_grid, summarise  # noqa: E402

OUT = ROOT / "results" / "iteration3"
I2 = ROOT / "results" / "iteration2"
ATTRS = ["CODE_GENDER", "REGION_RATING_CLIENT"]
CANDIDATES = {"D1": "proxy_aware_rw_unguarded", "D2": "proxy_aware_rw_guarded"}
MAX_AUC_GAP = 0.035
EVAL_SUBSAMPLE = 8000


# --------------------------------------------------------------------------
def criteria(m: dict, none: dict, rw: dict) -> dict:
    return {
        "i_eo_no_worse_than_none": m["eo_diff_mean"] <= none["eo_diff_mean"],
        "ii_dp_below_reweighing": m["dp_diff_mean"] < rw["dp_diff_mean"],
        "iii_probe_below_reweighing": m["probe_auc_after_mean"] < rw["probe_auc_after_mean"],
        "iv_auc_within_0.035": (none["auc_mean"] - m["auc_mean"]) <= MAX_AUC_GAP,
    }


def _row(summary, method):
    return summary[summary.method == method].iloc[0].to_dict()


# --------------------------------------------------------------------------
def reject_option(splits=5, repeats=5):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for attribute in ATTRS:
        t0 = time.time()
        X, y, A = LOADERS["home_credit"](subsample=EVAL_SUBSAMPLE)
        df = run_grid(X, y, A[attribute], attribute=attribute,
                      dataset="home_credit", methods=["reject_option"],
                      conditions=("explicit",), n_splits=splits,
                      n_repeats=repeats, verbose=False)
        s = summarise(df)
        rows.append(s)
        print(f"{attribute}: " + s[["auc_mean", "dp_diff_mean", "di_ratio_mean",
                                    "eo_diff_mean"]].round(4).to_string(index=False)
              + f"  ({time.time()-t0:.0f}s)")
    pd.concat(rows, ignore_index=True).to_csv(
        OUT / "reject_option_fixed_summary.csv", index=False)
    print(f"wrote {OUT/'reject_option_fixed_summary.csv'}")


# --------------------------------------------------------------------------
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = pd.read_csv(I2 / "tradeoff_curve.csv")
    cc = pd.read_csv(I2 / "combined_comparison.csv")
    figs = I2 / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    for attribute in ATTRS:
        d = t[t.attribute == attribute]
        sweep = d[~d.budget.astype(str).str.startswith("[ref]")]
        refs = d[d.budget.astype(str).str.startswith("[ref]")]
        it1 = cc[(cc.attribute == attribute) & (cc.method == "proxy_aware")
                 & (cc.condition == "latent")].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax, ycol, it1col, ylab in [
                (axes[0], "probe_auc_after", "probe_auc_after_mean",
                 "Residual leakage (probe AUC)"),
                (axes[1], "dp_diff", "dp_diff_mean",
                 "Demographic parity difference")]:
            ax.scatter(sweep.auc, sweep[ycol], s=55, color="#4C72B0",
                       label="guarded mitigator, by accuracy budget", zorder=3)
            # points sit close together on y, so labels alternate above and
            # below in y order to stay legible
            for i, r in enumerate(sweep.sort_values(ycol).itertuples()):
                ax.annotate(f"budget {r.budget}", (r.auc, getattr(r, ycol)),
                            textcoords="offset points",
                            xytext=(7, -11 if i % 2 == 0 else 5),
                            fontsize=8, color="#2F4B73")
            for r, mk in zip(refs.itertuples(), ["s", "^"]):
                lbl = r.budget.replace("[ref] ", "")
                ax.scatter(r.auc, getattr(r, ycol), marker=mk, s=75,
                           color="#C44E52", label=f"{lbl} (reference)", zorder=3)
            ax.scatter(it1.auc_mean, it1[it1col], marker="D", s=75,
                       color="#55A868", label="unguarded mitigator (iteration 1)",
                       zorder=3)
            ax.annotate("iteration 1", (it1.auc_mean, it1[it1col]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8,
                        color="#2E6B3E")
            ax.set_xlabel("Credit-model AUC (higher is better)")
            ax.set_ylabel(ylab + " (lower is better)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"Accuracy budget trade-off — home_credit / {attribute}\n"
                     "training-only access, 25 folds at 8,000 rows", fontsize=12)
        fig.tight_layout()
        out = figs / f"tradeoff_{attribute}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"figure -> {out}")


# --------------------------------------------------------------------------
def validation_pool(n=EVAL_SUBSAMPLE, pool=14_000, seed=7):
    """Rows disjoint from the evaluation subsample, verified by row hash."""
    Xe, _, _ = load_home_credit(subsample=EVAL_SUBSAMPLE)
    seen = set(pd.util.hash_pandas_object(Xe, index=False).tolist())
    Xp, yp, Ap = load_home_credit(subsample=pool, random_state=seed)
    h = pd.util.hash_pandas_object(Xp, index=False).values
    keep = np.array([v not in seen for v in h])
    Xp = Xp[keep].reset_index(drop=True).iloc[:n]
    yp = yp[keep].reset_index(drop=True).iloc[:n]
    Ap = Ap[keep].reset_index(drop=True).iloc[:n]
    return Xp, yp, Ap, int((~keep).sum())


def validate(splits=5):
    OUT.mkdir(parents=True, exist_ok=True)
    Xv, yv, Av, dropped = validation_pool()
    print(f"validation pool: {len(Xv):,} rows, {dropped} overlapping rows "
          f"removed, overlap with evaluation set now 0")

    out, choice = [], {}
    for attribute in ATTRS:
        t0 = time.time()
        df = run_grid(Xv, yv, Av[attribute], attribute=attribute,
                      dataset="home_credit_validation",
                      methods=["none", "reweighing"] + list(CANDIDATES.values()),
                      conditions=("training_only",), n_splits=splits,
                      n_repeats=1, verbose=False)
        s = summarise(df)
        none, rw = _row(s, "none"), _row(s, "reweighing")
        scored = []
        for tag, method in CANDIDATES.items():
            m = _row(s, method)
            c = criteria(m, none, rw)
            scored.append({"attribute": attribute, "candidate": tag,
                           "method": method, "n_passed": sum(c.values()),
                           **{k: bool(v) for k, v in c.items()},
                           "auc": m["auc_mean"], "dp": m["dp_diff_mean"],
                           "eo": m["eo_diff_mean"],
                           "probe": m["probe_auc_after_mean"]})
        sc = pd.DataFrame(scored).sort_values(
            ["n_passed", "eo", "probe"], ascending=[False, True, True])
        choice[attribute] = sc.iloc[0].candidate
        ref = pd.DataFrame([
            {"attribute": attribute, "candidate": "[ref] none", "auc": none["auc_mean"],
             "dp": none["dp_diff_mean"], "eo": none["eo_diff_mean"],
             "probe": none["probe_auc_after_mean"]},
            {"attribute": attribute, "candidate": "[ref] reweighing", "auc": rw["auc_mean"],
             "dp": rw["dp_diff_mean"], "eo": rw["eo_diff_mean"],
             "probe": rw["probe_auc_after_mean"]}])
        out.append(pd.concat([ref, sc], ignore_index=True))
        print(f"\n{attribute}  ({time.time()-t0:.0f}s)")
        print(out[-1][["candidate", "n_passed", "auc", "dp", "eo", "probe"]]
              .round(4).to_string(index=False))
        print(f"  CHOSEN: {choice[attribute]}")

    pd.concat(out, ignore_index=True).to_csv(OUT / "validation_scores.csv", index=False)
    pd.Series(choice, name="candidate").to_csv(OUT / "chosen.csv")
    print(f"\nwrote {OUT/'validation_scores.csv'}")


# --------------------------------------------------------------------------
def evaluate(splits=5, repeats=5):
    choice = pd.read_csv(OUT / "chosen.csv", index_col=0)["candidate"].to_dict()
    tro = pd.read_csv(I2 / "training_only_summary.csv")
    v2 = pd.read_csv(I2 / "proxy_v2_comparable_summary.csv")
    main = {"CODE_GENDER": "main_comparison_summary_gender.csv",
            "REGION_RATING_CLIENT": "main_comparison_summary_region.csv"}

    tables, crit_rows = [], []
    for attribute in ATTRS:
        method = CANDIDATES[choice[attribute]]
        t0 = time.time()
        X, y, A = LOADERS["home_credit"](subsample=EVAL_SUBSAMPLE)
        df = run_grid(X, y, A[attribute], attribute=attribute,
                      dataset="home_credit", methods=[method],
                      conditions=("training_only",), n_splits=splits,
                      n_repeats=repeats, verbose=False)
        df.to_csv(OUT / f"combo_folds_{attribute}.csv", index=False)
        combo = summarise(df).iloc[0].to_dict()
        print(f"{attribute}: {choice[attribute]} evaluated ({time.time()-t0:.0f}s)")

        tr = tro[tro.attribute == attribute]
        none, rw = _row(tr, "none"), _row(tr, "reweighing")
        mm = pd.read_csv(ROOT / "results" / main[attribute])
        full = mm[(mm.method == "proxy_aware") & (mm.condition == "latent")].iloc[0].to_dict()
        guarded = v2[v2.attribute == attribute].iloc[0].to_dict()

        rows = [("none", none), ("reweighing", rw),
                ("proxy_aware full (iteration 1)", full),
                ("proxy_aware guarded (iteration 2)", guarded),
                (f"repair + reweighing, {choice[attribute]} (iteration 3)", combo)]
        tbl = pd.DataFrame([{"attribute": attribute, "method": n,
                             "AUC": r["auc_mean"], "DP diff": r["dp_diff_mean"],
                             "DI ratio": r["di_ratio_mean"], "EO diff": r["eo_diff_mean"],
                             "probe AUC": r["probe_auc_after_mean"]}
                            for n, r in rows]).round(4)
        tables.append(tbl)

        c = criteria(combo, none, rw)
        ev = {
            "i_eo_no_worse_than_none": f"EO {combo['eo_diff_mean']:.4f} vs none {none['eo_diff_mean']:.4f}",
            "ii_dp_below_reweighing": f"DP {combo['dp_diff_mean']:.4f} vs reweighing {rw['dp_diff_mean']:.4f}",
            "iii_probe_below_reweighing": f"probe {combo['probe_auc_after_mean']:.4f} vs reweighing {rw['probe_auc_after_mean']:.4f}",
            "iv_auc_within_0.035": f"AUC gap {none['auc_mean']-combo['auc_mean']:.4f} (limit {MAX_AUC_GAP})",
        }
        for k, v in c.items():
            crit_rows.append({"attribute": attribute, "candidate": choice[attribute],
                              "criterion": k, "passed": bool(v), "evidence": ev[k]})

    pd.concat(tables, ignore_index=True).to_csv(OUT / "final_table.csv", index=False)
    cr = pd.DataFrame(crit_rows)
    cr.to_csv(OUT / "success_criteria.csv", index=False)
    pd.set_option("display.width", 200)
    for t in tables:
        print("\n" + t.drop(columns=["attribute"]).to_string(index=False))
    print("\n" + cr.assign(passed=cr.passed.map({True: "PASS", False: "FAIL"}))
          [["attribute", "criterion", "passed", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    {"reject_option": reject_option, "figures": figures,
     "validate": validate, "evaluate": evaluate}[sys.argv[1]]()
