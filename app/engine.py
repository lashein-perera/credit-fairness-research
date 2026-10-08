"""Computation layer behind the dashboard.

Every number the dashboard shows is produced here, by calling the frozen
modules (artefact-v4-final) unchanged. Nothing in this file re-implements a
measurement: the probe, the leakage ranking, SHAP, the quadrant cut-points,
the mitigator and the fairness metrics are the same functions the experiments
used. This file only sequences them on a small subsample, and reads the
committed results/ tables for demo mode.

Kept free of Streamlit so it can be exercised from a plain Python session.
"""
from __future__ import annotations

import io
import json
import sys
import time
import types
import warnings
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

from data.loaders import LOADERS, PROTECTED  # noqa: E402
from evaluation.harness import PROXY_AWARE_V2, _probe, resolve_threshold  # noqa: E402
from evaluation.metrics import (auc_roc, demographic_parity_difference,  # noqa: E402
                                disparate_impact_ratio,
                                equalized_odds_difference)
from leakage.probe import (classify_leakage, probe_auc,  # noqa: E402
                           rank_leaky_features, shuffle_test)
from mitigation.proxy_aware import ProxyAwareMitigator  # noqa: E402
from models.baselines import build_model  # noqa: E402


# Module 5 is loaded from source rather than imported. On this machine the
# normal import of the `explain` package stalls indefinitely inside Python's
# code-loading path, which plain file reads bypass; the module itself is
# healthy and this route produces identical objects. Same workaround as the
# run_iteration*.py scripts.
def _load_module_5():
    path = ROOT / "src" / "python" / "explain" / "shap_layer.py"
    mod = types.ModuleType("shap_layer")
    mod.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


m5 = _load_module_5()

RESULTS = ROOT / "results"
RANDOM_STATE = 42
TEST_SIZE = 0.3
MODEL_KIND = "gbm"

DATASETS = {
    "Home Credit (sample)": "home_credit",
    "German Credit": "german_credit",
    "UCI Default of Credit Card Clients": "uci_default",
}
TARGETS = {
    "home_credit": "TARGET",
    "german_credit": "target",
    "uci_default": "default payment next month",
}

# full    = the iteration-1 mitigator at its default settings
# guarded = the iteration-2 configuration chosen on validation
MODES = {"full": {}, "guarded": dict(PROXY_AWARE_V2)}

# Plain-English explanations, shared by the tooltips and the report.
GLOSSARY = {
    "probe_auc": "How well a classifier can guess the protected attribute from "
                 "the other features. 0.50 = no better than a coin flip (no "
                 "leakage); above 0.65 = meaningful; above 0.80 = severe.",
    "control": "The same probe trained on a randomly shuffled copy of the "
               "attribute. It should sit near 0.50. If it does not, the "
               "measurement is faulty rather than the leakage real.",
    "balanced_acc": "Share of applicants whose group the probe guesses "
                    "correctly, averaged over groups so a majority group "
                    "cannot inflate it. 0.50 = chance.",
    "leakage": "Drop in probe AUC when this one feature is scrambled. The "
               "bigger the drop, the more the feature gives away about the "
               "protected attribute.",
    "shap": "Average size of this feature's push on the credit score "
            "(mean |SHAP|). The bigger it is, the more the credit model "
            "relies on the feature.",
    "danger": "Features above both cut-points: they reveal the protected "
              "attribute AND the credit model leans on them. These are where "
              "proxy discrimination can enter decisions.",
    "auc": "Credit-model accuracy: the chance a randomly chosen defaulter is "
           "scored riskier than a randomly chosen non-defaulter. 0.50 = "
           "random, 1.00 = perfect.",
    "dp": "Demographic parity difference: gap in approval rates between "
          "groups. 0 = every group approved at the same rate.",
    "di": "Disparate impact ratio: lowest group approval rate divided by the "
          "highest. 1 = equal; below 0.80 fails the four-fifths rule.",
    "eo": "Equalised odds difference: the largest gap between groups in "
          "error rates (defaulters missed, or non-defaulters wrongly "
          "flagged). 0 = equal error rates.",
    "probe_after": "Probe AUC measured on the features the credit model "
                   "actually receives. For the mitigated model these are the "
                   "repaired features.",
    "manufactured": "These are manufactured proxies: the repair itself wrote "
                    "group information into features that carried none. A jump "
                    "in the number of distinct values is the tell-tale sign.",
    "full": "Iteration-1 mitigator: repairs the 15 leakiest features per "
            "round, up to 10 rounds, and tolerates up to 0.08 AUC loss.",
    "weak_probe": "Logistic regression on a single 70/30 split: the check the "
                  "mitigator itself works against. It only sees straight-line "
                  "patterns.",
    "strong_probe": "Gradient boosting with 5-fold cross-validation, as in the "
                    "leakage audit. It can combine features in flexible ways.",
    "probe_gap": "The weak probe only spots simple straight-line patterns, which "
                 "is exactly what the repair is tuned to remove, while the strong "
                 "probe can combine features flexibly and so still finds much of "
                 "the information the repair merely disguised.",
    "risk_score": "The model's estimated chance that this applicant defaults. "
                  "Applicants at or above the decision threshold are declined.",
    "threshold": "The cut-off that declines the same share of applicants as "
                 "actually defaulted in the training data, the rule used "
                 "throughout the evaluation.",
    "push": "How much this feature pushed the applicant's risk up (+) or down "
            "(−), in the model's internal log-odds units (SHAP value).",
    "lender": "Fairness of the decisions in your uploaded column, measured "
              "against the protected attribute and the actual outcomes. These "
              "are your model's decisions, not this tool's.",
    "guarded": "Iteration-2 mitigator: same repair, but skips near-constant "
               "features (one value in more than 99% of rows) and stops if "
               "AUC falls by more than 0.02.",
}


# ==========================================================================
# 1. load data
# ==========================================================================
def _subsample(X, y, A, n_rows):
    """Stratified on the outcome, so the default rate is preserved."""
    if n_rows is None or n_rows >= len(X):
        return X.reset_index(drop=True), y.reset_index(drop=True), A.reset_index(drop=True)
    X, _, y, _, A, _ = train_test_split(X, y, A, train_size=n_rows, stratify=y,
                                        random_state=RANDOM_STATE)
    return X.reset_index(drop=True), y.reset_index(drop=True), A.reset_index(drop=True)


def load_builtin(key: str, n_rows: int | None):
    """(X, y, A) from a built-in dataset, via the frozen loaders."""
    if key == "home_credit":
        # the loader's own stratified subsample, so no 300k-row frame is kept
        return LOADERS[key](subsample=n_rows)
    X, y, A = LOADERS[key]()
    return _subsample(X, y, A, n_rows)


def builtin_attributes(key: str) -> list[str]:
    return list(PROTECTED[key])


def prepare_upload(df: pd.DataFrame, target: str, positive_value, attribute: str,
                   exclude: list[str], n_rows: int | None,
                   lender_col: str | None = None):
    """(X, y, A) from an uploaded table.

    `positive_value` is the target value that means default. The protected
    attribute, and any column the user marks as a near-duplicate of it or an
    identifier, is removed from X: the same isolation rule the loaders apply.

    `lender_col` holds the lender's own score or decision. It is carried in A,
    row-aligned through the subsample, and never enters X: it is an output of
    the lender's model, not a feature.
    """
    df = df[df[attribute].notna() & df[target].notna()].reset_index(drop=True)
    y = (df[target].astype(str) == str(positive_value)).astype(int)
    A = df[[attribute]].astype(str)
    if lender_col:
        A[lender_col] = df[lender_col].values
    drop = {target, attribute, *exclude} | ({lender_col} if lender_col else set())
    X = df.drop(columns=[c for c in drop if c in df.columns])
    return _subsample(X, y, A, n_rows)


def lender_audit(y, a, decisions, kind: str, decline_value=None) -> dict:
    """Group fairness of the lender's own decisions on the loaded rows.

    kind="decision": rows equal to `decline_value` are declines.
    kind="score":    higher means riskier; the same share is declined as
                     actually defaulted, the threshold rule the evaluation uses.
    """
    y = pd.Series(y).reset_index(drop=True)
    a = pd.Series(a).astype(str).reset_index(drop=True)
    d = pd.Series(decisions).reset_index(drop=True)
    if kind == "score":
        s = pd.to_numeric(d, errors="coerce")
        keep = s.notna()
        thr = resolve_threshold(s[keep].values, y[keep].values, None)
        declined = (s >= thr).astype(int)
        rule = (f"Scores at or above {thr:.4g} counted as declined: the same share "
                f"as actually defaulted ({y[keep].mean():.1%}).")
    else:
        keep = d.notna()
        declined = (d.astype(str) == str(decline_value)).astype(int)
        rule = f"Rows with the value '{decline_value}' counted as declined."
    y, a, declined = (v[keep].reset_index(drop=True) for v in (y, a, declined))
    groups = pd.DataFrame({"group": a, "approved": 1 - declined}).groupby("group").agg(
        applicants=("approved", "size"), approval_rate=("approved", "mean")).reset_index()
    return {"DP diff": demographic_parity_difference(declined, a),
            "DI ratio": disparate_impact_ratio(declined, a),
            "EO diff": equalized_odds_difference(y, declined, a),
            "approval_rate": float(1 - declined.mean()), "n": int(keep.sum()),
            "rule": rule, "groups": groups}


def check_inputs(X, y, a) -> list[str]:
    """Problems that would make the downstream measurements meaningless."""
    problems = []
    if y.nunique() != 2:
        problems.append("The outcome column needs both defaulters and non-defaulters.")
    elif y.value_counts().min() < 20:
        problems.append("Fewer than 20 defaulters (or non-defaulters) in the sample.")
    groups = pd.Series(a).nunique()
    if groups < 2:
        problems.append("The protected attribute has only one group in this sample.")
    elif groups > 10:
        problems.append(f"The protected attribute has {groups} distinct values. "
                        "Choose a categorical attribute (2–10 groups).")
    if X.shape[1] < 2:
        problems.append("At least two feature columns are needed.")
    return problems


def split(X, y, a):
    """The held-out split every step after the audit shares (70/30, stratified)."""
    parts = train_test_split(X, y, pd.Series(a), test_size=TEST_SIZE, stratify=y,
                             random_state=RANDOM_STATE)
    return [p.reset_index(drop=True) for p in parts]


# ==========================================================================
# 2. leakage audit
# ==========================================================================
def audit(X, a, probe: str = "gbm", progress=lambda msg, frac: None) -> dict:
    """Probe AUC, the shuffled-attribute control, and the per-feature ranking."""
    t0 = time.time()
    a = pd.Series(a).reset_index(drop=True)
    X = X.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        progress("Training the probe on the real attribute (5-fold)", 0.05)
        actual = probe_auc(X, a, kind=probe)
        progress("Training the probe on a shuffled attribute (control)", 0.40)
        control = shuffle_test(X, a, kind=probe)
        progress("Ranking features by leakage contribution", 0.70)
        ranking = rank_leaky_features(X, a, kind="gbm", top_k=X.shape[1])
    progress("Done", 1.0)
    return {"probe": probe, "actual": actual, "control": control,
            "verdict": classify_leakage(actual["auc_mean"]),
            "ranking": ranking, "runtime_s": time.time() - t0}


# ==========================================================================
# 3. explain
# ==========================================================================
def explain(X, y, a, progress=lambda msg, frac: None) -> dict:
    """Leakage vs SHAP on the held-out split, against the unmitigated model.

    Cut-points are Module 5's default_thresholds: median SHAP, and leakage at
    the permutation noise floor. The same thresholds are frozen and reused for
    the after-mitigation picture.
    """
    t0 = time.time()
    Xtr, Xte, ytr, yte, atr, ate = split(X, y, a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        progress("Training the credit model", 0.05)
        base = build_model(MODEL_KIND, Xtr)
        base.fit(Xtr, ytr)
        progress("Computing SHAP importance", 0.30)
        imp = m5.global_importance(base, Xte, model_kind=MODEL_KIND)
        progress("Measuring per-feature leakage on the held-out split", 0.55)
        leak = rank_leaky_features(Xte, ate, top_k=Xte.shape[1])
    cross = m5.leakage_vs_shap(leak, imp)
    thresholds = m5.default_thresholds(cross)
    cross["quadrant"] = m5.leakage_vs_shap_quadrant(cross, thresholds)
    progress("Done", 1.0)
    return {"cross": cross, "thresholds": thresholds,
            "stats": m5.rank_correlation(cross),
            "runtime_s": time.time() - t0}


def danger_features(cross: pd.DataFrame) -> pd.DataFrame:
    d = cross[cross["quadrant"] == "leaky_and_relied_on"]
    return d.sort_values("shap_importance", ascending=False)[
        ["feature", "leakage_drop", "shap_importance"]].reset_index(drop=True)


def cross_figure(cross: pd.DataFrame, thresholds, title: str = ""):
    """The Module 5 cross-plot, drawn by Module 5's own panel function."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    m5._draw_cross(ax, cross, thresholds, n_annotate=8)
    ax.set_xlabel("Leakage contribution (drop in probe AUC when the feature is scrambled)")
    ax.set_ylabel("SHAP importance (mean |SHAP|)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


def leakage_bar_figure(ranking: pd.DataFrame, top: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = ranking.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 0.26 * len(d) + 1.2))
    ax.barh(d["feature"], d["leakage_drop"], xerr=d["leakage_std"],
            color="#C44E52", alpha=0.85, error_kw={"lw": 0.8})
    ax.set_xlabel("Leakage contribution (drop in probe AUC when the feature is scrambled)")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ==========================================================================
# 4. mitigate
# ==========================================================================
def _outcomes(y_true, proba, y_fit, a) -> dict:
    """Decision threshold at the training base rate, as the harness does."""
    thr = resolve_threshold(proba, y_fit, None)
    pred = (np.asarray(proba) >= thr).astype(int)
    return {"AUC": auc_roc(y_true, proba),
            "DP diff": demographic_parity_difference(pred, a),
            "DI ratio": disparate_impact_ratio(pred, a),
            "EO diff": equalized_odds_difference(y_true, pred, a)}


def mitigate(X, y, a, mode: str, before: pd.DataFrame, thresholds,
             progress=lambda msg, frac: None) -> dict:
    """Fit the proxy-aware mitigator and compare against the unmitigated model.

    Training-only access: the mitigator sees the attribute while fitting and
    never at scoring. Manufactured proxies are flagged exactly as in
    run_mitigation_explainability.py: leakage rising from at-or-below the frozen
    noise floor to above it.
    """
    t0 = time.time()
    Xtr, Xte, ytr, yte, atr, ate = split(X, y, a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        progress("Training the unmitigated credit model", 0.03)
        base = build_model(MODEL_KIND, Xtr)
        base.fit(Xtr, ytr)
        row_before = _outcomes(yte, base.predict_proba(Xte)[:, 1], ytr, ate)
        row_before["probe AUC"] = _probe(Xte, ate)

        progress(f"Fitting the proxy-aware mitigator ({mode}); "
                 "this is the slow step", 0.10)
        mit = ProxyAwareMitigator(model_kind=MODEL_KIND, **MODES[mode])
        mit.fit(Xtr, ytr, atr)
        progress("Scoring the held-out split without the attribute", 0.60)
        Xte_t = mit.transform(Xte)
        row_after = _outcomes(yte, mit.predict_proba(Xte), ytr, ate)
        row_after["probe AUC"] = _probe(Xte_t, ate)

        progress("Re-measuring leakage and SHAP after mitigation", 0.70)
        imp = m5.global_importance(mit.model_, Xte_t, model_kind=MODEL_KIND)
        leak = rank_leaky_features(Xte_t, ate, top_k=Xte_t.shape[1])
        progress("Measuring leakage with the strong probe", 0.88)
        probes = _probe_table(row_before["probe AUC"], row_after["probe AUC"],
                              probe_auc(Xte, ate, kind="gbm")["auc_mean"],
                              probe_auc(Xte_t, ate, kind="gbm")["auc_mean"])
    after = m5.leakage_vs_shap(leak, imp, thresholds=thresholds)

    nuq_before = {c: int(Xte[c].nunique(dropna=False)) for c in Xte.columns}
    nuq_after = {c: int(Xte_t[c].nunique(dropna=False)) for c in Xte_t.columns}
    tr = m5.quadrant_transitions(before, after, leakage_threshold=thresholds[0],
                                 n_unique_before=nuq_before,
                                 n_unique_after=nuq_after)
    progress("Done", 1.0)

    table = pd.DataFrame([{"model": "Before (no mitigation)", **row_before},
                          {"model": f"After (proxy-aware, {mode})", **row_after}])
    return {"mode": mode, "table": table, "after": after,
            "manufactured": manufactured(tr),
            "n_treated": len(mit.treated_), "n_skipped": len(mit.skipped_),
            "budget_stop": any(h.get("rejected") for h in mit.history_),
            "danger_before": int((before["quadrant"] == "leaky_and_relied_on").sum()),
            "danger_after": int((after["quadrant"] == "leaky_and_relied_on").sum()),
            "probes": probes, "probes_source": None,
            "live": _bundle(base, mit, Xte, Xte_t, ytr, yte), "mitigator": mit,
            "source": "live", "runtime_s": time.time() - t0}


def _probe_table(weak_before, weak_after, strong_before, strong_after) -> pd.DataFrame:
    return pd.DataFrame([
        {"probe": "Weak: logistic regression", "before": weak_before, "after": weak_after},
        {"probe": "Strong: gradient boosting", "before": strong_before, "after": strong_after}])


def _bundle(base, mit, Xte, Xte_t, ytr, yte) -> dict:
    """What the borrower view needs: both models, both feature views, decisions."""
    pb = base.predict_proba(Xte)[:, 1]
    pa = mit.model_.predict_proba(Xte_t)[:, 1]
    return {"base": base, "mit_model": mit.model_, "Xte": Xte, "Xte_t": Xte_t,
            "yte": yte, "proba_before": pb, "proba_after": pa,
            "thr_before": resolve_threshold(pb, ytr, None),
            "thr_after": resolve_threshold(pa, ytr, None)}


# ==========================================================================
# borrower view
# ==========================================================================
def _plain(v):
    """A JSON- and display-friendly scalar."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating, float)):
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def borrower_index(bundle: dict) -> pd.DataFrame:
    """One line per held-out applicant, for the picker."""
    pb, pa = bundle["proba_before"], bundle["proba_after"]
    db = np.where(pb >= bundle["thr_before"], "declined", "approved")
    da = np.where(pa >= bundle["thr_after"], "declined", "approved")
    return pd.DataFrame({"applicant": np.arange(len(pb)), "before": db, "after": da,
                         "changed": db != da})


def borrower(bundle: dict, i: int, top: int = 6) -> dict:
    """One applicant: scores, decisions, threshold, drivers, values before and after."""
    xb = bundle["Xte"].iloc[[i]]
    xa = bundle["Xte_t"].iloc[[i]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vb, names, _ = m5.shap_matrix(bundle["base"], xb, model_kind=MODEL_KIND)
        va, names_a, _ = m5.shap_matrix(bundle["mit_model"], xa, model_kind=MODEL_KIND)
    pushes_b = dict(zip(names, vb[0]))
    pushes_a = dict(zip(names_a, va[0]))

    def drivers(pushes, frame):
        order = sorted(pushes, key=lambda f: abs(pushes[f]), reverse=True)[:top]
        return [{"feature": f, "value": _plain(frame.iloc[0][f]),
                 "push": round(float(pushes[f]), 4)} for f in order]

    before = drivers(pushes_b, xb)
    after = drivers(pushes_a, xa)
    key = list(dict.fromkeys([d["feature"] for d in before] + [d["feature"] for d in after]))
    values = [{"feature": f, "value before repair": _plain(xb.iloc[0][f]),
               "value after repair": _plain(xa.iloc[0][f])} for f in key]

    def side(p, thr):
        return {"risk_score": round(float(p), 4), "threshold": round(float(thr), 4),
                "decision": "declined" if p >= thr else "approved"}

    return {"applicant": int(i),
            "actual_outcome": "defaulted" if int(bundle["yte"].iloc[i]) == 1 else "repaid",
            "before": side(bundle["proba_before"][i], bundle["thr_before"]) | {"drivers": before},
            "after": side(bundle["proba_after"][i], bundle["thr_after"]) | {"drivers": after},
            "values": values}


DEMO_BORROWERS = ROOT / "data" / "demo"


def demo_borrowers(attribute: str) -> list[dict] | None:
    """Stored example applicants, built locally by app/build_demo_borrowers.py.

    Kept under data/ (not committed): they are rows of the Home Credit data,
    which its licence does not allow to be redistributed.
    """
    path = DEMO_BORROWERS / f"borrowers_{attribute}.json"
    return json.loads(path.read_text()) if path.exists() else None


# ==========================================================================
# repaired-data export
# ==========================================================================
EXPORT_NOTE = """REPAIRED LOAN DATA - read me first

repaired_data.csv holds the loaded applicants with the revealing features
repaired by the proxy-aware mitigator ({mode} mode), plus the outcome column
outcome_default (1 = defaulted). The protected attribute ({attribute}) is not
in the file and was used only while fitting the repair.

How a lender would use it
  Retrain your own credit model on these repaired features and the outcome,
  instead of on the original features. To score a new applicant, the same
  fitted repair must first be applied to their features; the repair is a
  fitted transformation and is not contained in this file.

What to keep in mind
  - {n_treated} of {n_features} features were repaired. Numeric features keep
    their units but their values are shifted; categories are replaced by
    numeric codes. Missing values were filled before repair.
  - The repair can create new proxies in features that had none
    (see "Manufactured proxies" in the dashboard).
  - A simple probe finds much less of {attribute} after repair, but a stronger
    probe still recovers a large part of it. Re-audit the retrained model.
  - Fitted on a {n_rows:,}-row sample in the dashboard, not the full data.
"""


def repaired_export(X: pd.DataFrame, y, mit, meta: dict, mode: str,
                    n_treated: int) -> bytes:
    """A zip of the repaired data and the explanatory note."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xr = mit.transform(X.reset_index(drop=True))
    Xr = Xr.assign(outcome_default=pd.Series(y).reset_index(drop=True).values)
    note = EXPORT_NOTE.format(mode=mode, attribute=meta["attribute"],
                              n_treated=n_treated, n_features=X.shape[1],
                              n_rows=len(X))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("repaired_data.csv", Xr.to_csv(index=False))
        z.writestr("README.txt", note)
    return buf.getvalue()


MANUFACTURED_LABELS = {"leakage_drop_before": "leakage before",
                       "leakage_drop_after": "leakage after",
                       "n_unique_before": "distinct values before",
                       "n_unique_after": "distinct values after"}


def manufactured(transitions: pd.DataFrame) -> pd.DataFrame:
    cols = ["feature", "leakage_drop_before", "leakage_drop_after",
            "n_unique_before", "n_unique_after"]
    d = transitions[transitions["leakage_emerged"].astype(bool)]
    return d.sort_values("leakage_drop_after", ascending=False)[
        [c for c in cols if c in d.columns]].reset_index(drop=True)


# ==========================================================================
# demo mode: the committed results, read back unchanged
# ==========================================================================
DEMO_ATTRIBUTES = ["CODE_GENDER", "REGION_RATING_CLIENT"]
DEMO_SOURCE = {
    "audit": "probe AUCs from results/leakage_table.csv (50,000-row Home "
             "Credit sample); feature ranking from "
             "results/leaky_features_full.csv (20,000-row explainability run)",
    "explain": "results/mitigation_cross_analysis.csv, baseline arm "
               "(held-out 15,000 rows of the 50,000-row sample)",
    "mitigate": "results/iteration3/final_table.csv (mean of 25 folds, 8,000 "
                "rows, training-only access); manufactured proxies from "
                "results/mitigation_transitions.csv and "
                "results/iteration2/transitions_<attribute>.csv (held-out 15,000 rows)",
}


def demo_audit(attribute: str) -> dict:
    t = pd.read_csv(RESULTS / "leakage_table.csv")
    t = t[(t.attribute == attribute) & (t.probe == "gbm")]
    actual = t[t.condition == "actual"].iloc[0].to_dict()
    control = t[t.condition == "shuffled"].iloc[0].to_dict()
    r = pd.read_csv(RESULTS / "leaky_features_full.csv")
    r = r[r.attribute == attribute].sort_values("rank").reset_index(drop=True)
    return {"probe": "gbm", "actual": actual, "control": control,
            "verdict": classify_leakage(actual["auc_mean"]),
            "ranking": r[["feature", "leakage_drop", "leakage_std", "rank"]],
            "runtime_s": 0.0}


def demo_explain(attribute: str) -> dict:
    c = pd.read_csv(RESULTS / "mitigation_cross_analysis.csv")
    cross = c[(c.attribute == attribute) & (c.arm == "baseline")].drop(
        columns=["dataset", "attribute", "model", "arm"]).reset_index(drop=True)
    thresholds = m5.default_thresholds(cross)
    cross["quadrant"] = m5.leakage_vs_shap_quadrant(cross, thresholds)
    return {"cross": cross, "thresholds": thresholds,
            "stats": m5.rank_correlation(cross), "runtime_s": 0.0}


def demo_mitigate(attribute: str, mode: str) -> dict:
    f = pd.read_csv(RESULTS / "iteration3" / "final_table.csv")
    f = f[f.attribute == attribute].set_index("method")
    label = {"full": "proxy_aware full (iteration 1)",
             "guarded": "proxy_aware guarded (iteration 2)"}[mode]
    cols = ["AUC", "DP diff", "DI ratio", "EO diff", "probe AUC"]
    table = pd.DataFrame([{"model": "Before (no mitigation)", **f.loc["none", cols].to_dict()},
                          {"model": f"After (proxy-aware, {mode})", **f.loc[label, cols].to_dict()}])

    if mode == "full":
        tr = pd.read_csv(RESULTS / "mitigation_transitions.csv")
        tr = tr[(tr.attribute == attribute) & (tr.arm == "full")]
        q = pd.read_csv(RESULTS / "mitigation_quadrant_summary.csv")
        q = q[(q.attribute == attribute)].set_index("arm")
        n_treated, n_skipped = int(q.loc["full", "n_treated"]), 0
        danger_before = int(q.loc["baseline", "n_leaky_and_relied_on"])
        danger_after = int(q.loc["full", "n_leaky_and_relied_on"])
    else:
        tr = pd.read_csv(RESULTS / "iteration2" / f"transitions_{attribute}.csv")
        t = pd.read_csv(RESULTS / "iteration2" / f"test_results_{attribute}.csv")
        t = t.set_index("candidate").loc["C1_guard99"]
        n_treated, n_skipped = int(t["n_treated"]), int(t["n_skipped"])
        danger_before = int((tr.quadrant_before == "leaky_and_relied_on").sum())
        danger_after = int((tr.quadrant_after == "leaky_and_relied_on").sum())
    probes, probes_source = None, None
    if mode == "full":
        sp = pd.read_csv(RESULTS / "iteration4" / "strong_probe.csv")
        sp = sp[sp.attribute == attribute].set_index("features")
        o, r = sp.loc["original"], sp.loc["repaired, iteration 1"]
        probes = _probe_table(o.weak_probe, r.weak_probe,
                              o.strong_probe_auc, r.strong_probe_auc)
        probes_source = ("results/iteration4/strong_probe.csv (held-out 15,000 rows "
                         "of the 50,000-row sample; both probes on the same rows)")
    return {"mode": mode, "table": table, "after": None,
            "probes": probes, "probes_source": probes_source,
            "manufactured": manufactured(tr),
            "n_treated": n_treated, "n_skipped": n_skipped, "budget_stop": False,
            "danger_before": danger_before, "danger_after": danger_after,
            "source": "demo", "runtime_s": 0.0}


# ==========================================================================
# 5. report
# ==========================================================================
def _md_table(df: pd.DataFrame, digits: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build_report(meta: dict, audit_res: dict | None, explain_res: dict | None,
                 mitigate_res: dict | None, lender_res: dict | None = None) -> str:
    """Plain-English audit report in Markdown."""
    L = ["# Proxy discrimination audit report", "",
         f"Generated {date.today().isoformat()} by the credit-fairness-research "
         "dashboard (artefact-v4-final modules).", "",
         "## Data", "",
         f"- **Dataset:** {meta['dataset']}",
         f"- **Rows analysed:** {meta['n_rows']:,}   **Features:** {meta['n_features']}",
         f"- **Protected attribute (audit only, never given to the credit model):** "
         f"{meta['attribute']}",
         f"- **Mode:** {meta['mode']}", ""]
    if meta.get("source"):
        L += [f"Results were read from the committed experiment tables: {meta['source']}.", ""]

    if lender_res:
        L += ["## Your model's decisions", "",
              f"> {GLOSSARY['lender']}", "",
              f"{lender_res['rule']} Overall approval rate "
              f"{lender_res['approval_rate']:.1%} across {lender_res['n']:,} applicants.", "",
              _md_table(pd.DataFrame([{k: lender_res[k] for k in ("DP diff", "DI ratio", "EO diff")}])), "",
              _md_table(lender_res["groups"].rename(columns={"approval_rate": "approval rate"})), ""]

    if audit_res:
        act, ctl = audit_res["actual"], audit_res["control"]
        probe = {"gbm": "gradient-boosting", "logistic": "logistic-regression"}[audit_res["probe"]]
        sound = abs(ctl["auc_mean"] - 0.5) <= 0.05
        L += ["## 1. Leakage audit", "",
              f"A {probe} probe predicts **{meta['attribute']}** from the "
              f"credit features with **AUC {act['auc_mean']:.3f}** "
              f"(± {act['auc_std']:.3f}); leakage is **{audit_res['verdict']}**. "
              f"With the attribute shuffled the same probe scores "
              f"**{ctl['auc_mean']:.3f}**, "
              + ("chance level, so the measurement is sound." if sound else
                 "not chance level, so treat the leakage figure with caution."), "",
              f"> {GLOSSARY['probe_auc']}", "",
              "Most revealing features:", "",
              _md_table(audit_res["ranking"].head(10)[["feature", "leakage_drop"]]
                        .rename(columns={"leakage_drop": "leakage contribution"}),
                        digits=5), ""]

    if explain_res:
        d = danger_features(explain_res["cross"])
        lx, ly = explain_res["thresholds"]
        L += ["## 2. Leaky and relied upon", "",
              f"{len(d)} of {len(explain_res['cross'])} features both reveal the "
              "attribute and drive the credit score (leakage above the permutation "
              f"noise floor, {lx:.2g}; SHAP importance above the median, {ly:.2g}). "
              "These are the features through which proxy discrimination can reach "
              "decisions.", ""]
        if len(d):
            L += [_md_table(d.rename(columns={"leakage_drop": "leakage contribution",
                                              "shap_importance": "SHAP importance"}),
                            digits=5), ""]

    if mitigate_res:
        L += [f"## 3. Mitigation ({mitigate_res['mode']})", "",
              f"> {GLOSSARY[mitigate_res['mode']]}", "",
              f"{mitigate_res['n_treated']} features repaired, "
              f"{mitigate_res['n_skipped']} skipped by the guard. The attribute "
              "was used while fitting only; scoring used none.", "",
              _md_table(mitigate_res["table"]), "",
              "- AUC: " + GLOSSARY["auc"],
              "- DP diff: " + GLOSSARY["dp"],
              "- DI ratio: " + GLOSSARY["di"],
              "- EO diff: " + GLOSSARY["eo"],
              "- probe AUC: " + GLOSSARY["probe_after"], ""]
        if mitigate_res.get("probes") is not None:
            L += ["### Leakage measured two ways", "",
                  _md_table(mitigate_res["probes"]), "",
                  GLOSSARY["probe_gap"], ""]
            if mitigate_res.get("probes_source"):
                L += [f"Source: {mitigate_res['probes_source']}.", ""]
        m = mitigate_res["manufactured"]
        L += ["### Manufactured proxies", "",
              f"{len(m)} features had no detectable leakage before mitigation and "
              "carry it afterwards. " + GLOSSARY["manufactured"], ""]
        if len(m):
            L += [_md_table(m.head(10).rename(columns=MANUFACTURED_LABELS), digits=5), ""]

    L += ["## Caveats", "",
          "- Live runs use a small subsample for speed; the dissertation's figures "
          "come from the full experiments (demo mode reproduces them).",
          "- The probe and the mitigator binarise the attribute (first group "
          "against the rest); fairness metrics compare every group.",
          "- The mitigator can manufacture new proxies (RESULTS_SUMMARY.md §7.1) "
          "and did not pass every pre-registered criterion (§4, §5).", ""]
    return "\n".join(L)
