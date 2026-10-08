"""Proxy-discrimination audit dashboard over the frozen artefact.

    streamlit run app/dashboard.py

A user interface only. Every measurement is made by app/engine.py, which calls
the artefact-v4-final modules unchanged. Demo mode reads the committed
results/ tables and computes nothing, so it cannot stall.

URL parameters, used for the viva and for the Chapter 4 screenshots:
    ?demo=1                       start in demo mode
    &attr=REGION_RATING_CLIENT    demo attribute
    &mode=guarded                 mitigation mode shown in demo mode
"""
from __future__ import annotations

import hashlib
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="Proxy Discrimination Audit", page_icon="⚖️",
                   layout="wide")

G = E.GLOSSARY
METRIC_HELP = {"AUC": G["auc"], "DP diff": G["dp"], "DI ratio": G["di"],
               "EO diff": G["eo"], "probe AUC": G["probe_after"]}
# direction in which each metric improves, for the delta arrows
LOWER_IS_BETTER = {"AUC": False, "DP diff": True, "DI ratio": False,
                   "EO diff": True, "probe AUC": True}


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def _init_state():
    qp = st.query_params
    if "demo" not in st.session_state:
        st.session_state.demo = qp.get("demo") == "1"
    if "demo_attr" not in st.session_state:
        st.session_state.demo_attr = (qp.get("attr") if qp.get("attr") in
                                      E.DEMO_ATTRIBUTES else E.DEMO_ATTRIBUTES[0])
    if "mode" not in st.session_state:
        st.session_state.mode = qp.get("mode") if qp.get("mode") in E.MODES else "guarded"
    for k in ("data", "meta", "audit", "explain", "mitigate", "lender"):
        st.session_state.setdefault(k, None)


def _clear_results():
    for k in ("audit", "explain", "mitigate"):
        st.session_state[k] = None


@st.cache_resource
def _store() -> dict:
    """Results cache shared across reruns and sessions, keyed by the data and settings.

    A plain dict rather than st.cache_data, so progress bars drawn while a
    step runs are not replayed on a cache hit.
    """
    return {}


def _cached(name: str, key: tuple, fn):
    store = _store()
    if (name, key) not in store:
        store[(name, key)] = fn()
    return store[(name, key)]


@st.cache_data(show_spinner=False)
def _builtin(key: str, n_rows: int):
    return E.load_builtin(key, n_rows)


@st.cache_data(show_spinner=False)
def _read_csv(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def _demo(step: str, attribute: str, mode: str = ""):
    if step == "audit":
        return E.demo_audit(attribute)
    if step == "explain":
        return E.demo_explain(attribute)
    return E.demo_mitigate(attribute, mode)


def _demo_meta() -> dict:
    return {"dataset": "Home Credit (50,000-row experimental sample)",
            "attribute": st.session_state.demo_attr, "n_rows": 50_000,
            "n_features": 116, "mode": "demo (precomputed results)",
            "source": None}


def _meta() -> dict | None:
    return _demo_meta() if st.session_state.demo else st.session_state.meta


def _need_data() -> bool:
    """True, after showing a pointer back to step 1, when nothing is loaded."""
    if st.session_state.demo or st.session_state.data is not None:
        return False
    st.info("No data loaded yet. Start at **1 · Load data**, or switch on "
            "**Demo mode** in the sidebar.")
    st.page_link(PAGES[0], label="Go to 1 · Load data", icon="➡️")
    return True


def _run_with_progress(label: str, name: str, key: tuple, fn):
    """Run one engine step with a progress bar, or return its cached result.

    `label` names the step ("Leakage audit"); the status line reads
    "<label>: running…" and then "<label> — done in Ns".
    """
    if (name, key) in _store():
        return _store()[(name, key)]
    with st.status(f"{label}: running…", expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        res = _cached(name, key, lambda: fn(lambda msg, frac: bar.progress(frac, text=msg)))
        status.update(label=f"{label} — done in {res['runtime_s']:.0f}s",
                      state="complete", expanded=False)
    return res


def _fit(n_rows: int, cap: int = 740) -> int:
    """Table height showing every row (35 px each, plus the header) up to a cap."""
    return min(35 * (n_rows + 1) + 3, cap)


def _show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _sidebar():
    with st.sidebar:
        st.markdown("### ⚖️ Proxy discrimination audit")
        st.toggle("Demo mode", key="demo",
                  help="Shows the precomputed results from the dissertation "
                       "experiments (results/ folder) instantly. Nothing is "
                       "recomputed, so it cannot fail or stall.")
        if st.session_state.demo:
            st.radio("Protected attribute", E.DEMO_ATTRIBUTES, key="demo_attr",
                     help="CODE_GENDER is the applicant's recorded gender; "
                          "REGION_RATING_CLIENT is the lender's rating of the "
                          "applicant's home region (1, 2 or 3).")
        meta = _meta()
        if meta:
            st.caption(f"**Data:** {meta['dataset']}  \n"
                       f"**Rows:** {meta['n_rows']:,} · **features:** {meta['n_features']}  \n"
                       f"**Protected attribute:** {meta['attribute']}")
        else:
            st.caption("No data loaded.")
        done = [s for s, k in (("audit", "audit"), ("explain", "explain"),
                               ("mitigate", "mitigate"))
                if st.session_state.demo or st.session_state[k] is not None]
        st.caption("**Completed:** " + (", ".join(done) if done else "none yet"))


# --------------------------------------------------------------------------
# 1. load data
# --------------------------------------------------------------------------
def page_load():
    st.title("1 · Load data")
    st.write("Choose the loan applications to audit, the outcome being "
             "predicted, and the protected attribute to check for. The "
             "protected attribute is **never given to the credit model**; it "
             "is used only to measure what the model's features reveal.")

    if st.session_state.demo:
        st.success("**Demo mode is on.** The dashboard is showing the "
                   "precomputed results for the 50,000-row Home Credit sample "
                   "used in the dissertation. Pick the protected attribute in "
                   "the sidebar, then move through the steps.")
        a = _demo("audit", st.session_state.demo_attr)["actual"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Applications", f"{int(a['n_samples']):,}")
        c2.metric("Credit features", int(a["n_features"]))
        c3.metric("Protected attribute", st.session_state.demo_attr)
        return

    source = st.radio("Data source", ["Built-in dataset", "Upload a CSV"],
                      horizontal=True)
    if source == "Built-in dataset":
        name = st.selectbox("Dataset", list(E.DATASETS),
                            help="Home Credit is the primary thin-file dataset; "
                                 "German Credit and UCI Default are benchmarks.")
        key = E.DATASETS[name]
        c1, c2 = st.columns(2)
        c1.selectbox("Outcome column (1 = defaulted)", [E.TARGETS[key]], disabled=True,
                     help="Fixed by the dataset's loader, which recodes it so "
                          "1 always means the applicant defaulted.")
        attr = c2.selectbox("Protected attribute", E.builtin_attributes(key),
                            help="Held out of the credit features and used for "
                                 "measurement only.")
        max_rows = 1_000 if key == "german_credit" else 50_000
        n_rows = st.number_input("Rows to sample", 500, max_rows,
                                 min(5_000, max_rows), step=500,
                                 help="A stratified random sample (default rate "
                                      "preserved). About 5,000 rows keeps each "
                                      "step under a minute.")
        if st.button("Load data", type="primary"):
            with st.spinner(f"Loading {name}…"):
                X, y, A = _builtin(key, int(n_rows))
            _set_data(X, y, A[attr], name, attr, ("builtin", key, attr, int(n_rows)))
    else:
        up = st.file_uploader("CSV file, one row per applicant", type="csv")
        if up is None:
            st.caption("The file needs an outcome column (did the applicant "
                       "default?) and a protected-attribute column with 2–10 "
                       "groups. Every other column is treated as a credit feature.")
            _show_loaded()
            return
        df = _read_csv(up.getvalue())
        cols = list(df.columns)
        c1, c2 = st.columns(2)
        target = c1.selectbox("Outcome column", cols,
                              help="The column recording whether the applicant "
                                   "defaulted.")
        values = sorted(df[target].dropna().astype(str).unique())[:50]
        positive = c2.selectbox("Value meaning 'defaulted'", values,
                                index=len(values) - 1 if values else 0,
                                help="Rows with this value count as defaults "
                                     "(1); all others as repaid (0).")
        attr = c1.selectbox("Protected attribute", [c for c in cols if c != target],
                            help="Held out of the credit features and used for "
                                 "measurement only.")
        lender = c2.selectbox("Your model's score or decision (optional)",
                              ["(none)"] + [c for c in cols if c not in (target, attr)],
                              help="A column holding your own credit model's output "
                                   "for each applicant. It is never used as a "
                                   "feature; the dashboard checks how fair those "
                                   "decisions are.")
        lender = None if lender == "(none)" else lender
        lender_kind, decline = None, None
        if lender:
            lender_kind = c1.radio("That column holds", ["score", "decision"],
                                   horizontal=True,
                                   format_func={"score": "A risk score (higher = riskier)",
                                                "decision": "A decision"}.get)
            if lender_kind == "decision":
                dvals = sorted(df[lender].dropna().astype(str).unique())[:50]
                decline = c2.selectbox("Value meaning 'declined'", dvals,
                                       help="Rows with this value count as declined; "
                                            "all others as approved.")
        exclude = c2.multiselect("Columns to leave out",
                                 [c for c in cols if c not in (target, attr, lender)],
                                 help="Identifiers, and any near-duplicate of the "
                                      "protected attribute (e.g. a title that "
                                      "states gender), so the audit measures "
                                      "proxies rather than restatements.")
        n_rows = st.number_input("Rows to sample", 200, max(len(df), 200),
                                 min(5_000, len(df)), step=500,
                                 help="A stratified random sample. About 5,000 "
                                      "rows keeps each step under a minute.")
        if st.button("Load data", type="primary"):
            X, y, A = E.prepare_upload(df, target, positive, attr, exclude, int(n_rows),
                                       lender_col=lender)
            digest = hashlib.sha1(up.getvalue()).hexdigest()
            _set_data(X, y, A[attr], up.name, attr,
                      ("upload", digest, target, positive, attr, tuple(exclude),
                       int(n_rows), lender, lender_kind, decline),
                      lender=(A[lender], lender_kind, decline) if lender else None)
    _show_loaded()


def _set_data(X, y, a, name, attr, key, lender=None):
    problems = E.check_inputs(X, y, a)
    if problems:
        for p in problems:
            st.error(p)
        return
    st.session_state.data = {"X": X, "y": y, "a": a, "key": key}
    st.session_state.lender = E.lender_audit(y, a, *lender) if lender else None
    st.session_state.meta = {"dataset": name, "attribute": attr,
                             "n_rows": len(X), "n_features": X.shape[1],
                             "mode": "live (computed on a subsample)", "source": None}
    _clear_results()


def _show_loaded():
    d = st.session_state.data
    if d is None:
        return
    X, y, a = d["X"], d["y"], d["a"]
    st.divider()
    st.subheader("Loaded")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(X):,}")
    c2.metric("Credit features", X.shape[1],
              help="Columns the credit model may use. The protected attribute "
                   "and its near-duplicates are already removed.")
    c3.metric("Default rate", f"{y.mean():.1%}")
    c4.metric("Groups in attribute", a.nunique())
    g = a.astype(str).value_counts().rename_axis(st.session_state.meta["attribute"])
    left, right = st.columns([1, 3])
    left.dataframe(g.rename("applicants").to_frame(), width="stretch")
    right.dataframe(X.head(8), width="stretch", height=300)
    _show_lender()
    st.page_link(PAGES[1], label="Next: 2 · Leakage audit", icon="➡️")


def _show_lender():
    lr = st.session_state.lender
    if not lr:
        return
    st.subheader("Your model's decisions")
    st.caption(G["lender"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval rate", f"{lr['approval_rate']:.1%}")
    c2.metric("DP diff", f"{lr['DP diff']:.4f}", help=G["dp"])
    c3.metric("DI ratio", f"{lr['DI ratio']:.4f}", help=G["di"])
    c4.metric("EO diff", f"{lr['EO diff']:.4f}", help=G["eo"])
    if lr["DI ratio"] < 0.8:
        st.warning("Your model's decisions fail the four-fifths rule "
                   f"(DI ratio {lr['DI ratio']:.2f}, below 0.80).")
    st.caption(f"{lr['rule']} Measured on the {lr['n']:,} loaded applicants.")
    st.dataframe(lr["groups"], hide_index=True, width="content", column_config={
        "approval_rate": st.column_config.NumberColumn("approval rate", format="%.3f")})


# --------------------------------------------------------------------------
# 2. leakage audit
# --------------------------------------------------------------------------
def page_audit():
    st.title("2 · Leakage audit")
    st.write("Can the protected attribute be **guessed from the credit "
             "features alone**? If so, a model that never sees it can still "
             "treat groups differently. A probe classifier is trained to make "
             "exactly that guess.")
    if _need_data():
        return

    meta = _meta()
    if st.session_state.demo:
        res = _demo("audit", meta["attribute"])
        st.caption(f"Precomputed: {E.DEMO_SOURCE['audit']}")
    else:
        probe = st.radio("Probe model", ["gbm", "logistic"], horizontal=True,
                         format_func={"gbm": "Gradient boosting (stronger)",
                                      "logistic": "Logistic regression (faster)"}.get,
                         help="A stronger probe finds more of the leakage; the "
                              "dissertation reports both.")
        d = st.session_state.data
        key = d["key"] + (probe,)
        if st.button("Run leakage audit", type="primary") or ("audit", key) in _store():
            st.session_state.audit = _run_with_progress(
                "Leakage audit", "audit", key,
                lambda p: E.audit(d["X"], d["a"], probe, progress=p))
        res = st.session_state.audit
        if res is None:
            return

    act, ctl = res["actual"], res["control"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probe AUC", f"{act['auc_mean']:.3f}", help=G["probe_auc"],
              delta=f"± {act['auc_std']:.3f} across folds", delta_color="off",
              delta_arrow="off")
    c2.metric("Shuffled control", f"{ctl['auc_mean']:.3f}", help=G["control"],
              delta="should be ≈ 0.50", delta_color="off", delta_arrow="off")
    c3.metric("Leakage level", res["verdict"].capitalize(),
              help="none < 0.55 ≤ mild < 0.65 ≤ meaningful < 0.80 ≤ severe")
    c4.metric("Balanced accuracy", f"{act['balanced_acc_mean']:.3f}",
              help=G["balanced_acc"])

    msg = (f"The credit features reveal **{meta['attribute']}** with probe AUC "
           f"**{act['auc_mean']:.3f}**: **{res['verdict']}** leakage.")
    {"severe": st.error, "meaningful": st.warning,
     "mild": st.info}.get(res["verdict"], st.success)(msg)
    if abs(ctl["auc_mean"] - 0.5) <= 0.05:
        st.success(f"Control passed: with the attribute shuffled the probe "
                   f"scores {ctl['auc_mean']:.3f}, i.e. chance. The leakage is real.")
    else:
        st.warning(f"Control did not return to chance ({ctl['auc_mean']:.3f}). "
                   "Treat the leakage figure with caution.")

    st.subheader("Most revealing features")
    top = st.slider("Features shown", 5, 30, 15)
    _show(E.leakage_bar_figure(res["ranking"], top=top))
    st.caption("Bar = drop in probe AUC when that feature alone is scrambled; "
               "whisker = variation across repeats. The longer the bar, the more "
               "the feature gives away about the protected attribute.")
    with st.expander("Full ranking"):
        st.dataframe(res["ranking"], width="stretch", hide_index=True,
                     column_config={"leakage_drop": st.column_config.NumberColumn(
                         "leakage contribution", help=G["leakage"], format="%.5f")})


# --------------------------------------------------------------------------
# 3. explain
# --------------------------------------------------------------------------
def page_explain():
    st.title("3 · Explain")
    st.write("Leakage only matters if the credit model **uses** the revealing "
             "features. This plot crosses what each feature gives away about "
             "the protected attribute (across) with how much the credit model "
             "relies on it (up, measured with SHAP).")
    if _need_data():
        return

    meta = _meta()
    if st.session_state.demo:
        res = _demo("explain", meta["attribute"])
        st.caption(f"Precomputed: {E.DEMO_SOURCE['explain']}")
    else:
        d = st.session_state.data
        if st.button("Run explanation", type="primary") or ("explain", d["key"]) in _store():
            st.session_state.explain = _run_with_progress(
                "Explanation", "explain", d["key"],
                lambda p: E.explain(d["X"], d["y"], d["a"], progress=p))
        res = st.session_state.explain
        if res is None:
            st.caption("Trains the credit model on 70% of the rows and explains "
                       "it on the other 30%.")
            return

    cross, (lx, ly) = res["cross"], res["thresholds"]
    danger = E.danger_features(cross)
    c1, c2, c3 = st.columns(3)
    c1.metric("Features analysed", len(cross))
    c2.metric("Leaky and relied upon", len(danger), help=G["danger"])
    c3.metric("Rank correlation (ρ)", f"{res['stats']['spearman_rho']:.3f}",
              help="Spearman correlation between leakage and SHAP importance. "
                   "Positive means the model leans hardest on the features "
                   "that give the attribute away.")

    left, right = st.columns([3, 2])
    with left:
        _show(E.cross_figure(cross, (lx, ly),
                             title=f"{meta['attribute']}: leakage vs model reliance"))
        st.caption(f"Dashed lines are the cut-points: leakage above the "
                   f"permutation noise floor ({lx:.2g}) and SHAP importance above "
                   f"the median ({ly:.2g}). Red points are in the top-right "
                   "quadrant. Both axes are symmetric-log so small and large "
                   "values are visible together.")
    with right:
        st.subheader("Leaky and relied upon")
        st.caption(G["danger"])
        st.dataframe(danger, width="stretch", hide_index=True, height=_fit(len(danger)),
                     column_config={
                         "leakage_drop": st.column_config.NumberColumn(
                             "leakage", help=G["leakage"], format="%.5f"),
                         "shap_importance": st.column_config.NumberColumn(
                             "SHAP importance", help=G["shap"], format="%.4f")})
    with st.expander("How to read the four quadrants"):
        st.markdown(
            "- **Top right — leaky and relied upon:** the problem. Removing these "
            "costs accuracy, so they must be repaired rather than dropped.\n"
            "- **Bottom right — leaky but unused:** can be dropped at no cost.\n"
            "- **Top left — relied upon, not leaky:** the model's safe workhorses.\n"
            "- **Bottom left — neither.**")


# --------------------------------------------------------------------------
# 4. mitigate
# --------------------------------------------------------------------------
def page_mitigate():
    st.title("4 · Mitigate")
    st.write("The proxy-aware mitigator repairs the revealing features so the "
             "protected attribute can no longer be recovered from them, then "
             "retrains the credit model. It uses the attribute **while "
             "fitting only**; scoring a new applicant needs no demographic data.")
    if _need_data():
        return

    # a copy outside the widget's own key, which Streamlit discards whenever
    # this page is not the one rendered
    mode = st.radio("Mode", list(E.MODES), index=list(E.MODES).index(st.session_state.mode),
                    key="_mode", horizontal=True,
                    on_change=lambda: st.session_state.update(mode=st.session_state._mode),
                    format_func={"full": "Full (iteration 1)",
                                 "guarded": "Guarded (iteration 2)"}.get,
                    help=f"Full: {G['full']}\n\nGuarded: {G['guarded']}")
    st.caption(G[mode])
    meta = _meta()
    if st.session_state.demo:
        res = _demo("mitigate", meta["attribute"], mode)
        st.caption(f"Precomputed: {E.DEMO_SOURCE['mitigate']}")
    else:
        d = st.session_state.data
        key = d["key"] + (mode,)
        if st.button("Run mitigation", type="primary") or ("mitigate", key) in _store():
            exp = _run_with_progress(
                "Explanation of the unmitigated model", "explain", d["key"],
                lambda p: E.explain(d["X"], d["y"], d["a"], progress=p))
            st.session_state.explain = exp
            st.session_state.mitigate = _run_with_progress(
                f"Proxy-aware mitigation ({mode})", "mitigate", key,
                lambda p: E.mitigate(d["X"], d["y"], d["a"], mode, exp["cross"],
                                     exp["thresholds"], progress=p))
        res = st.session_state.mitigate
        if res is None or res["mode"] != mode:
            st.caption("Fits on 70% of the rows and evaluates on the other 30%. "
                       "Full mode takes longest.")
            return

    before, after = res["table"].iloc[0], res["table"].iloc[1]
    st.subheader("Before and after")
    cols = st.columns(5)
    for col, m in zip(cols, ["AUC", "DP diff", "DI ratio", "EO diff", "probe AUC"]):
        col.metric(m, f"{after[m]:.4f}", delta=f"{after[m] - before[m]:+.4f}",
                   delta_color="inverse" if LOWER_IS_BETTER[m] else "normal",
                   help=f"{METRIC_HELP[m]} Before mitigation: {before[m]:.4f}.")
    st.caption("Large number = after mitigation; arrow = change from before. "
               "Green = fairer or more accurate; red = worse.")
    st.dataframe(res["table"], width="stretch", hide_index=True,
                 column_config={m: st.column_config.NumberColumn(m, help=h, format="%.4f")
                                for m, h in METRIC_HELP.items()})

    if res["n_treated"] == 0:
        st.info("No feature was repaired, so the model is unchanged. "
                + ("The first repair round cost more accuracy than this mode "
                   "allows, and was rolled back." if res["budget_stop"] else
                   "Leakage was already below the mitigator's target."))
    st.subheader("Leakage measured two ways")
    pr = res.get("probes")
    if pr is None:
        st.info("The strong-probe measurement is stored for full mode only "
                "(results/iteration4/strong_probe.csv). Choose full mode, or run "
                "guarded mode live.")
    else:
        for col, (_, r) in zip(st.columns(2), pr.iterrows()):
            col.metric(r.probe, f"{r.after:.3f}",
                       delta=f"{r.after - r.before:+.3f} from {r.before:.3f}",
                       delta_color="inverse",
                       help=G["weak_probe"] if r.probe.startswith("Weak") else G["strong_probe"])
        st.caption(G["probe_gap"])
        if res.get("probes_source"):
            st.caption(f"Precomputed: {res['probes_source']}. The weak-probe values "
                       "therefore differ slightly from the 25-fold table above.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Features repaired", res["n_treated"])
    c2.metric("Skipped by the guard", res["n_skipped"],
              help="Near-constant features the guarded mode will not touch.")
    c3.metric("Leaky and relied upon", res["danger_after"],
              delta=f"{res['danger_after'] - res['danger_before']:+d} vs before",
              delta_color="inverse", help=G["danger"])

    m = res["manufactured"]
    st.subheader("Manufactured proxies")
    if len(m):
        st.warning(f"**{len(m)} features** had no detectable leakage before "
                   "mitigation and carry it afterwards. " + G["manufactured"])
        st.dataframe(m, width="stretch", hide_index=True, height=_fit(len(m)), column_config={
            "leakage_drop_before": st.column_config.NumberColumn("leakage before", format="%.5f"),
            "leakage_drop_after": st.column_config.NumberColumn("leakage after", format="%.5f"),
            "n_unique_before": st.column_config.NumberColumn(
                "distinct values before", help="Number of distinct values in the column."),
            "n_unique_after": st.column_config.NumberColumn(
                "distinct values after", help="A sharp rise means the repair "
                                              "rewrote a near-constant column.")})
    else:
        st.success("No feature gained leakage from below the noise floor.")

    st.subheader("Download the repaired data")
    if st.session_state.demo:
        st.info("Available in live mode. Demo mode holds stored results only, "
                "not applicant data.")
        return
    d = st.session_state.data
    key = d["key"] + (mode,)
    if st.button("Prepare repaired data") or ("export", key) in _store():
        with st.spinner("Applying the repair to every loaded applicant…"):
            data = _cached("export", key, lambda: E.repaired_export(
                d["X"], d["y"], res["mitigator"], meta, mode, res["n_treated"]))
        st.download_button("Download repaired data (.zip)", data, type="primary",
                           file_name=f"repaired_{meta['attribute']}_{mode}.zip",
                           mime="application/zip")
    st.caption("A zip of the loaded applicants with the revealing features repaired "
               "(the protected attribute removed), and a note explaining that a "
               "lender would retrain its own credit model on it.")


# --------------------------------------------------------------------------
# 5. borrower view
# --------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "missing"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def page_borrower():
    st.title("5 · Borrower view")
    st.write("What the repair means for one applicant: their risk score and "
             "decision before and after repair, the decision threshold, the "
             "features that drove their score, and how their values changed.")
    if _need_data():
        return
    meta = _meta()
    if st.session_state.demo:
        rows = E.demo_borrowers(meta["attribute"])
        if not rows:
            st.info("No stored example applicants on this computer yet. They are "
                    "built locally, because they are real Home Credit rows that "
                    "may not be published: run `python app/build_demo_borrowers.py` "
                    "(about 12 minutes), then reload this page.")
            return
        st.caption("Precomputed examples from the held-out 15,000 applicants of the "
                   "50,000-row sample, with the iteration-1 full repair. Applicants "
                   "are identified only by their position in that held-out set.")
        pick = st.radio("Example applicant", range(len(rows)),
                        format_func=lambda i: f"{rows[i]['label']} "
                                              f"(applicant {rows[i]['applicant']})")
        b = rows[pick]
    else:
        res = st.session_state.mitigate
        if res is None or "live" not in res:
            st.info("Run **4 · Mitigate** first: this view compares its two models.")
            st.page_link(PAGES[3], label="Go to 4 · Mitigate", icon="➡️")
            return
        idx = E.borrower_index(res["live"]).set_index("applicant", drop=False)
        n_changed = int(idx.changed.sum())
        only = st.toggle("Only applicants whose decision the repair changed",
                         value=n_changed > 0)
        pool = idx[idx.changed] if only and n_changed else idx
        st.caption(f"{n_changed} of {len(idx):,} held-out applicants get a different "
                   f"decision after the {res['mode']} repair.")
        i = st.selectbox("Applicant (position in the held-out 30%)",
                         pool.applicant.tolist(),
                         format_func=lambda j: f"Applicant {j}: "
                                               f"{idx.at[j, 'before']} → {idx.at[j, 'after']}")
        b = _cached("borrower", st.session_state.data["key"] + (res["mode"], int(i)),
                    lambda: E.borrower(res["live"], int(i)))

    st.caption(f"Actual outcome: **{b['actual_outcome']}**.")
    for col, side, title in zip(st.columns(2), (b["before"], b["after"]),
                                ("Before repair", "After repair")):
        with col:
            st.subheader(title)
            c1, c2 = st.columns(2)
            c1.metric("Risk score", f"{side['risk_score']:.3f}", help=G["risk_score"])
            c2.metric("Decision", side["decision"].capitalize(),
                      delta=f"threshold {side['threshold']:.3f}", delta_color="off",
                      delta_arrow="off", help=G["threshold"])
            drivers = pd.DataFrame(side["drivers"]).assign(
                value=lambda d: d["value"].map(_fmt))
            st.dataframe(drivers, hide_index=True, width="stretch", column_config={
                "feature": st.column_config.TextColumn("top features driving the score"),
                "push": st.column_config.NumberColumn("push on risk", help=G["push"],
                                                      format="%+.3f")})
    st.subheader("Key feature values before and after repair")
    vals = pd.DataFrame(b["values"])
    for c in ("value before repair", "value after repair"):
        vals[c] = vals[c].map(_fmt)
    st.dataframe(vals, hide_index=True, width="stretch")
    st.caption("Repair moves each value to the same position in a distribution "
               "shared by all applicants: numeric features keep their units but "
               "their values shift, and categories become numeric codes. Missing "
               "values are filled before repair.")


# --------------------------------------------------------------------------
# 6. report
# --------------------------------------------------------------------------
def page_report():
    st.title("6 · Report")
    st.write("A plain-language audit report of everything run so far, ready to "
             "file or share.")
    meta = _meta()
    if meta is None:
        _need_data()
        return
    meta = dict(meta)
    if st.session_state.demo:
        attr, mode = meta["attribute"], st.session_state.mode
        res = (_demo("audit", attr), _demo("explain", attr), _demo("mitigate", attr, mode))
        meta["source"] = "; ".join(E.DEMO_SOURCE.values())
    else:
        res = (st.session_state.audit, st.session_state.explain, st.session_state.mitigate)
        if not any(res):
            st.info("Run at least one step (2–4) to include its results.")
    report = E.build_report(meta, *res, lender_res=None if st.session_state.demo
                            else st.session_state.lender)
    st.download_button("Download report (Markdown)", report, type="primary",
                       file_name=f"proxy_audit_{meta['attribute']}.md",
                       mime="text/markdown")
    with st.container(border=True):
        st.markdown(report)


PAGES = [
    st.Page(page_load, title="1 · Load data", url_path="load", default=True),
    st.Page(page_audit, title="2 · Leakage audit", url_path="audit"),
    st.Page(page_explain, title="3 · Explain", url_path="explain"),
    st.Page(page_mitigate, title="4 · Mitigate", url_path="mitigate"),
    st.Page(page_borrower, title="5 · Borrower view", url_path="borrower"),
    st.Page(page_report, title="6 · Report", url_path="report"),
]

_init_state()
nav = st.navigation(PAGES)
nav.run()
# after the page, so the status it shows reflects what the page just did;
# its widgets still take effect first, through session state
_sidebar()
