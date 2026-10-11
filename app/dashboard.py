"""ProxyLens: a proxy-discrimination audit for credit data.

    streamlit run app/dashboard.py

A user interface only. Every measurement is made by app/engine.py, which calls
the artefact-v4-final modules unchanged. The sample audit reads the committed
results/ tables and computes nothing, so it cannot stall.

Flow: start -> setup -> run (leakage audit, then explanation) -> results
workspace with the tabs Overview, Leakage, Explanation, Repair, Applicant
lookup and Report.

URL parameters, used for the viva and for the Chapter 4 screenshots:
    ?demo=1                       open the sample audit
    &attr=REGION_RATING_CLIENT    sample-audit attribute
    &mode=full | guarded          sample audit with that repair applied
    &tab=Repair                   open a results tab
The page addresses of the earlier dashboard (/audit, /mitigate, /borrower, ...)
redirect to the matching tab.
"""
from __future__ import annotations

import hashlib
import html
import io
import re
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import engine as E

APP = "ProxyLens"
TAGLINE = ("Checks whether your credit features quietly reveal a protected "
           "attribute such as gender, shows which ones your model relies on, "
           "and repairs them.")

st.set_page_config(page_title=APP, page_icon=":material/policy:", layout="wide")

G = E.GLOSSARY
METRIC_HELP = {"AUC": G["auc"], "DP diff": G["dp"], "DI ratio": G["di"],
               "EO diff": G["eo"], "probe AUC": G["probe_after"]}
# direction in which each metric improves, for the delta arrows
LOWER_IS_BETTER = {"AUC": False, "DP diff": True, "DI ratio": False,
                   "EO diff": True, "probe AUC": True}

TABS = ["Overview", "Leakage", "Explanation", "Repair", "Applicant lookup", "Report"]
# Module 1's leakage bands (classify_leakage), as status labels and colours
LEVELS = {"none": ("LOW", "green", ":material/check_circle:"),
          "mild": ("MODERATE", "orange", ":material/warning:"),
          "meaningful": ("HIGH", "red", ":material/warning:"),
          "severe": ("SEVERE", "red", ":material/error:")}
RATING_NOTE = ("Ratings use the probe-AUC bands of the methodology: below 0.55 "
               "**LOW**, 0.55–0.65 **MODERATE**, 0.65–0.80 **HIGH**, 0.80 and "
               "above **SEVERE**. 0.50 means the probe is only guessing.")
ATTR_NAMES = {"CODE_GENDER": "Gender", "REGION_RATING_CLIENT": "Region rating",
              "sex": "Sex", "age_group": "Age group", "SEX": "Sex",
              "MARRIAGE": "Marital status"}
CHECK_ICON = {"ok": ":green[:material/check_circle:]",
              "warn": ":orange[:material/warning:]",
              "fail": ":red[:material/cancel:]"}
FRIENDLY = ("This usually means a column holds values the model cannot use, such "
            "as free text or identifiers, or that the sample is too small. Try "
            "leaving such columns out, or sampling more rows.")

CSS = """<style>
.pl-brand {display:flex; align-items:center; gap:.55rem; font-weight:700;
           font-size:1.3rem; color:var(--pl-ink)}
.pl-brand .pl-logo {display:inline-flex; width:2rem; height:2rem; border-radius:.55rem;
           background:#1E40AF; color:#fff; align-items:center; justify-content:center;
           font-size:1.05rem}
.pl-brand small {font-weight:500; font-size:.85rem; color:#64748B; margin-left:.25rem}
:root {--pl-ink:#0F172A}
.pl-hero {padding:2.2rem 0 1.2rem}
.pl-hero h1 {font-size:2.6rem; margin:0 0 .4rem; letter-spacing:-.02em;
             display:flex; align-items:center; gap:.8rem}
.pl-hero .pl-logo {display:inline-flex; width:3rem; height:3rem; border-radius:.8rem;
             background:#1E40AF; color:#fff; align-items:center; justify-content:center;
             font-size:1.6rem}
.pl-hero p {font-size:1.2rem; color:#475569; max-width:46rem; margin:0}
.pl-stepper {display:flex; align-items:center; gap:.6rem; margin:.35rem 0 .2rem}
.pl-step {display:flex; align-items:center; gap:.45rem; font-weight:600;
          color:#94A3B8; white-space:nowrap; font-size:.95rem}
.pl-dot {width:1.65rem; height:1.65rem; border-radius:50%; display:inline-flex;
         align-items:center; justify-content:center; border:2px solid #CBD5E1;
         font-size:.8rem}
.pl-step.done {color:#1E40AF}
.pl-step.done .pl-dot {background:#1E40AF; border-color:#1E40AF; color:#fff}
.pl-step.current {color:#0F172A}
.pl-step.current .pl-dot {border-color:#1E40AF; color:#1E40AF;
                          box-shadow:0 0 0 4px rgba(30,64,175,.14)}
.pl-step em {font-style:normal; font-weight:500; font-size:.78rem; color:#64748B}
.pl-line {flex:1; height:2px; background:#E2E8F0; min-width:1rem}
.pl-kv span {display:block; font-size:.72rem; color:#64748B; text-transform:uppercase;
             letter-spacing:.05em; margin-bottom:.1rem}
.pl-kv b {font-size:.98rem; color:#0F172A; font-weight:600}
.pl-how b {display:block; margin:.2rem 0 .15rem}
.pl-how span {color:#475569; font-size:.92rem}
[data-testid="stHeaderActionElements"] {display:none}
.pl-how .pl-num {display:inline-flex; width:1.6rem; height:1.6rem; border-radius:50%;
                 background:#E0E7FF; color:#1E40AF; font-weight:700; font-size:.85rem;
                 align-items:center; justify-content:center}
</style>"""


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def _init_state():
    """Session defaults, read once from the URL the session started on."""
    ss = st.session_state
    if ss.get("ready"):
        return
    qp = st.query_params
    mode = qp.get("mode") if qp.get("mode") in E.MODES else None
    ss.ready = True
    ss.demo = qp.get("demo") == "1"
    ss.demo_attr = qp.get("attr") if qp.get("attr") in E.DEMO_ATTRIBUTES \
        else E.DEMO_ATTRIBUTES[0]
    ss.demo_repair = mode if ss.demo else None
    ss.repair_mode = mode or "full"
    ss.report_done = False
    for k in ("data", "meta", "lender", "audit", "explain", "mitigate", "pending",
              "upload", "toast"):
        ss[k] = None


def _reset():
    """Forget the current audit; setup choices and the uploaded file are kept."""
    ss = st.session_state
    ss.demo, ss.demo_repair, ss.report_done = False, None, False
    for k in ("data", "meta", "lender", "audit", "explain", "mitigate", "pending"):
        ss[k] = None


def _demo_params() -> dict:
    ss = st.session_state
    qp = {"demo": "1"}
    if ss.demo_attr != E.DEMO_ATTRIBUTES[0]:
        qp["attr"] = ss.demo_attr
    if ss.demo_repair:
        qp["mode"] = ss.demo_repair
    return qp


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
def _prepare(raw: bytes, target, positive, attribute, exclude, n_rows, lender):
    return E.prepare_upload(_read_csv(raw), target, positive, attribute,
                            list(exclude), n_rows, lender_col=lender)


@st.cache_data(show_spinner=False)
def _demo(step: str, attribute: str, mode: str = ""):
    if step == "audit":
        return E.demo_audit(attribute)
    if step == "explain":
        return E.demo_explain(attribute)
    return E.demo_mitigate(attribute, mode)


def _meta() -> dict | None:
    ss = st.session_state
    if not ss.demo:
        return ss.meta
    a = _demo("audit", ss.demo_attr)["actual"]
    return {"dataset": "Home Credit (50,000-row experimental sample)",
            "attribute": ss.demo_attr, "n_rows": int(a["n_samples"]),
            "n_features": int(a["n_features"]),
            "mode": "sample audit (precomputed results)", "source": None}


def _results():
    """(audit, explain, repair) of the current audit, stored or live."""
    ss = st.session_state
    if ss.demo:
        attr = ss.demo_attr
        mit = _demo("mitigate", attr, ss.demo_repair) if ss.demo_repair else None
        return _demo("audit", attr), _demo("explain", attr), mit
    return ss.audit, ss.explain, ss.mitigate


def _attr_name(attr: str) -> str:
    return ATTR_NAMES.get(attr, attr)


def _n(k: int, noun: str) -> str:
    """'1 feature', '3 features'."""
    return f"{k} {noun}{'' if k == 1 else 's'}"


# --------------------------------------------------------------------------
# shared pieces
# --------------------------------------------------------------------------
def _stage(name: str, key: tuple, fn):
    """Run one engine step under a progress bar, or return its cached result."""
    if (name, key) in _store():
        st.caption("Already computed for these settings; reusing the result.")
        return _store()[(name, key)]
    bar = st.progress(0.0, text="Starting")
    res = _cached(name, key, lambda: fn(lambda msg, frac: bar.progress(frac, text=msg)))
    bar.progress(1.0, text=f"Done in {res['runtime_s']:.0f} s")
    return res


def _fail(what: str, err: Exception):
    st.error(f"**{what} stopped.** {FRIENDLY}", icon=":material/error:")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(err)), language=None)


def _fit(n_rows: int, cap: int = 740) -> int:
    """Table height showing every row (35 px each, plus the header) up to a cap."""
    return min(35 * (n_rows + 1) + 3, cap)


def _show(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _goto(tab: str):
    st.session_state.tab = tab
    st.session_state.scroll_top = True


def _next(label: str, tab: str):
    st.space("small")
    st.button(label, type="primary", icon=":material/arrow_forward:",
              icon_position="right", on_click=_goto, args=(tab,), key=f"next_{tab}")


def _source(text: str):
    with st.expander("Data source", icon=":material/database:"):
        st.markdown(text)


def _brand(right=None):
    left, rgt = st.columns([3, 2], vertical_alignment="center")
    left.html(f"<div class='pl-brand'><span class='pl-logo'>◎</span>{APP}"
              "<small>proxy-discrimination audit</small></div>")
    with rgt, st.container(horizontal=True, horizontal_alignment="right"):
        with st.popover("About", icon=":material/info:", type="tertiary"):
            _about()
        if right:
            right()


def _about():
    st.markdown(
        f"**{APP}** audits credit data for *proxy discrimination*: features that "
        "reveal a protected attribute the credit model is never given.\n\n"
        "- **Leakage audit**: a probe classifier tries to guess the attribute "
        "from the features; a shuffled-attribute control checks the measurement.\n"
        "- **Explanation**: SHAP shows which revealing features the credit model "
        "relies on.\n"
        "- **Repair**: the proxy-aware mitigator rewrites those features, using "
        "the attribute while fitting only.\n\n"
        "Built on the frozen research modules (tag `artefact-v4-final`). Runs on "
        "this computer: uploaded files are not sent anywhere.\n\n"
        "Research prototype for the dissertation *Detecting and Mitigating Proxy "
        "Discrimination in Alternative-Data Credit Scoring for Thin-File Borrowers*.")


def _new_audit_button():
    if st.button("New audit", icon=":material/add:"):
        _reset()
        st.query_params.clear()
        st.switch_page(START)


def _scroll_top():
    st.html("<script>(window.parent || window).document.querySelectorAll("
            "'[data-testid=\"stMain\"], section.main').forEach(e => e.scrollTo(0, 0));"
            "</script>", unsafe_allow_javascript=True)


# --------------------------------------------------------------------------
# 1. start
# --------------------------------------------------------------------------
def page_start():
    ss = st.session_state
    if ss.demo:
        st.switch_page(RESULTS, query_params=_demo_params())
    st.html(CSS)
    with st.container(horizontal=True, horizontal_alignment="right"):
        with st.popover("About", icon=":material/info:", type="tertiary"):
            _about()
    st.html(f"<div class='pl-hero'><h1><span class='pl-logo'>◎</span>{APP}</h1>"
            f"<p>{TAGLINE}</p></div>")

    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True, height="stretch"):
        st.markdown("### :material/upload_file: Audit my data")
        st.write("Upload a CSV of loan applications, or use a built-in dataset "
                 "(Home Credit, German Credit or UCI Default of Credit Card "
                 "Clients). The audit runs on this computer.")
        if st.button("Audit my data", type="primary", icon=":material/arrow_forward:",
                     icon_position="right", width="stretch"):
            st.switch_page(SETUP)
    with c2, st.container(border=True, height="stretch"):
        st.markdown("### :material/travel_explore: Explore a sample audit")
        st.write("A finished audit of 50,000 Home Credit applicants, using the "
                 "stored results of the dissertation experiments. Opens "
                 "instantly; nothing is computed.")
        if st.button("Explore a sample audit", icon=":material/arrow_forward:",
                     icon_position="right", width="stretch"):
            ss.demo = True
            st.switch_page(RESULTS, query_params=_demo_params())

    st.space("medium")
    st.markdown("##### How it works")
    steps = [("Data", "Choose the applications, the outcome and the protected attribute."),
             ("Audit", "Measure how well the features reveal the attribute, and "
                       "which of them the credit model relies on."),
             ("Repair", "Optionally rewrite the revealing features and see what "
                        "it costs in accuracy."),
             ("Report", "Download a plain-language audit report.")]
    for i, (col, (title, text)) in enumerate(zip(st.columns(4), steps), start=1):
        col.html(f"<div class='pl-how'><span class='pl-num'>{i}</span>"
                 f"<b>{title}</b><span>{text}</span></div>")


# --------------------------------------------------------------------------
# 2. setup
# --------------------------------------------------------------------------
def page_setup():
    ss = st.session_state
    st.html(CSS)

    def back():
        if st.button("Back", icon=":material/arrow_back:", type="tertiary"):
            st.switch_page(START)

    _brand(back)
    st.title("Set up your audit")
    st.write("Choose the loan applications, the outcome being predicted and the "
             "protected attribute to check for. The protected attribute is "
             "**never given to the credit model**; it is used only to measure what "
             "the other features reveal.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        data_box = st.container(border=True)
        cols_box = st.container(border=True)
        adv_box = st.expander("Advanced settings", icon=":material/tune:")
    with data_box:
        st.markdown("#### 1 · Data")
        src = st.segmented_control(
            "Data source", ["builtin", "upload"], default="builtin", required=True,
            key="s_src", persist_state="session",
            format_func={"builtin": "Built-in dataset", "upload": "Upload a CSV"}.get)
    with cols_box:
        st.markdown("#### 2 · Columns")

    if src == "builtin":
        checks, payload = _setup_builtin(data_box, cols_box, adv_box)
    else:
        checks, payload = _setup_upload(data_box, cols_box, adv_box)

    with right, st.container(border=True):
        st.markdown("#### Ready to run?")
        for status, text in checks:
            st.markdown(f"{CHECK_ICON[status]} {text}")
        blocked = payload is None or any(s == "fail" for s, _ in checks)
        if blocked:
            st.caption("Fix the items marked with a cross to continue.")
        if st.button("Run audit", type="primary", icon=":material/play_arrow:",
                     width="stretch", disabled=blocked):
            ss.pending = payload
            st.switch_page(RUN)
        st.caption("Runs the leakage audit, then the explanation, one after the "
                   "other. You decide on repair afterwards.")


def _advanced(adv_box, max_rows: int, key: str) -> tuple[int, str]:
    with adv_box:
        n_rows = st.number_input(
            "Rows to sample", min(200, max_rows), max_rows, min(5_000, max_rows),
            step=500, key=f"s_rows_{key}", persist_state="session",
            help="A random sample that keeps the default rate. About 5,000 rows "
                 "keeps each step to about a minute or less.")
        probe = st.radio(
            "Leakage probe", ["gbm", "logistic"], horizontal=True, key="s_probe",
            persist_state="session",
            format_func={"gbm": "Gradient boosting (methodology default)",
                         "logistic": "Logistic regression (faster, weaker)"}.get,
            help="The classifier that tries to guess the protected attribute. "
                 "Gradient boosting is the probe used in the methodology; a "
                 "weaker probe finds less of the leakage.")
    return int(n_rows), probe


def _attr_checks(a: pd.Series, attr: str) -> list:
    groups = a.dropna().astype(str).value_counts()
    if len(groups) < 2:
        return [("fail", f"Protected attribute **{attr}** has only one group, so "
                         "there is nothing to compare.")]
    if len(groups) > 10:
        return [("fail", f"Protected attribute **{attr}** has {len(groups)} different "
                         "values. Choose a categorical attribute with 2–10 groups "
                         "(for example gender or an age band); a number such as "
                         "exact age needs grouping first.")]
    shown = ", ".join(map(str, groups.index[:4])) + (", …" if len(groups) > 4 else "")
    return [("ok", f"Protected attribute **{attr}**: {len(groups)} groups ({shown}).")]


def _setup_builtin(data_box, cols_box, adv_box):
    with data_box:
        name = st.selectbox("Dataset", list(E.DATASETS), key="s_dataset",
                            persist_state="session",
                            help="Home Credit is the primary thin-file dataset; German "
                                 "Credit and UCI Default are benchmarks.")
    key = E.DATASETS[name]
    n_rows, probe = _advanced(adv_box, 1_000 if key == "german_credit" else 50_000, key)
    with data_box:
        try:
            with st.spinner(f"Loading {name}…"):
                X, y, A = _builtin(key, n_rows)
        except Exception as e:
            _fail(f"Loading {name}", e)
            return [("fail", f"{name} could not be loaded on this computer.")], None
        st.caption(f"{len(X):,} applicants sampled · default rate {y.mean():.1%}")

    with cols_box:
        c1, c2 = st.columns(2)
        c1.selectbox("Outcome column", [f"{E.TARGETS[key]} (1 = defaulted)"], disabled=True,
                     help="Fixed by the dataset's loader, which recodes it so 1 "
                          "always means the applicant defaulted.")
        attr = c2.selectbox("Protected attribute", E.builtin_attributes(key),
                            key=f"s_attr_{key}", persist_state="session",
                            format_func=lambda c: f"{_attr_name(c)} ({c})",
                            help="Held out of the credit features and used for "
                                 "measurement only.")
        c1.selectbox("Your model's score or decision", ["Not available for built-in data"],
                     disabled=True, help="Upload a CSV to check your own model's "
                                         "decisions alongside the audit.")
        exclude = c2.multiselect("Columns to leave out", list(X.columns),
                                 key=f"s_excl_{key}", persist_state="session",
                                 help="Identifiers, or anything you do not want the "
                                      "credit model to use. The protected attribute "
                                      "and its near-duplicates are already removed.")

    X2 = X.drop(columns=list(exclude))
    a = A[attr]
    checks = [("ok", f"**{name}**: {len(X):,} applicants, default rate {y.mean():.1%}."),
              ("ok", f"Outcome **{E.TARGETS[key]}**, recoded so 1 = defaulted.")]
    checks += _attr_checks(a, attr)
    if X2.shape[1] < 2:
        checks.append(("fail", "At least two feature columns are needed."))
    else:
        checks.append(("ok", f"**{X2.shape[1]}** columns used as credit features "
                             f"({len(exclude)} left out)."))
    checks += [("fail", p) for p in E.check_inputs(X2, y, a)]
    payload = {"X": X2, "y": y, "a": a, "lender": None, "probe": probe,
               "key": ("builtin", key, attr, n_rows, tuple(sorted(exclude))),
               "meta": {"dataset": name, "attribute": attr, "n_rows": len(X2),
                        "n_features": X2.shape[1],
                        "mode": "live (computed on a subsample)", "source": None}}
    return checks, payload


def _guess(cols: list[str], pattern: str) -> int:
    for i, c in enumerate(cols):
        if re.search(pattern, c, re.I):
            return i
    return 0


def _setup_upload(data_box, cols_box, adv_box):
    ss = st.session_state
    with data_box:
        up = st.file_uploader("CSV file, one row per applicant", type="csv", key="s_file")
        if up is not None:
            ss.upload = {"name": up.name, "bytes": up.getvalue()}
        if ss.upload is None:
            st.caption("The file needs an outcome column (did the applicant "
                       "default?) and a protected-attribute column with 2–10 "
                       "groups. Every other column is treated as a credit feature "
                       "unless you leave it out.")
            _advanced(adv_box, 50_000, "none")
            return [("fail", "Upload a CSV file to continue.")], None
        raw, fname = ss.upload["bytes"], ss.upload["name"]
        try:
            df = _read_csv(raw)
        except Exception as e:
            _advanced(adv_box, 50_000, "none")
            return [("fail", f"**{fname}** could not be read as a CSV file "
                             f"({type(e).__name__}). Save it as comma-separated "
                             "values and upload it again.")], None
        st.caption(f"Using **{fname}**: {len(df):,} rows, {df.shape[1]} columns.")
    if len(df) == 0 or df.shape[1] < 4:
        _advanced(adv_box, 50_000, "none")
        return [("fail", "The file needs an outcome column, a protected attribute "
                         "and at least two feature columns.")], None

    dg = hashlib.sha1(raw).hexdigest()[:12]
    n_rows, probe = _advanced(adv_box, max(len(df), 200), dg)
    cols = list(df.columns)
    with cols_box:
        c1, c2 = st.columns(2)
        target = c1.selectbox("Outcome column", cols, key=f"s_target_{dg}",
                              index=_guess(cols, r"outcome|target|default"),
                              persist_state="session",
                              help="The column recording whether the applicant defaulted.")
        values = sorted(df[target].dropna().astype(str).unique())
        positive = c2.selectbox("Value meaning 'defaulted'", values[:50],
                                index=min(len(values), 50) - 1 if values else 0,
                                key=f"s_pos_{dg}_{target}", persist_state="session",
                                help="Rows with this value count as defaults (1); "
                                     "all others as repaid (0).")
        attr = c1.selectbox("Protected attribute", [c for c in cols if c != target],
                            key=f"s_attr_{dg}", persist_state="session",
                            help="Held out of the credit features and used for "
                                 "measurement only.")
        lender = c2.selectbox("Your model's score or decision (optional)",
                              ["(none)"] + [c for c in cols if c not in (target, attr)],
                              key=f"s_lender_{dg}", persist_state="session",
                              help="A column holding your own credit model's output for "
                                   "each applicant. It is never used as a feature; the "
                                   "audit checks how fair those decisions are.")
        lender = None if lender == "(none)" else lender
        kind, decline = None, None
        if lender:
            kind = c1.radio("That column holds", ["score", "decision"], horizontal=True,
                            key=f"s_kind_{dg}", persist_state="session",
                            format_func={"score": "A risk score (higher = riskier)",
                                         "decision": "A decision"}.get)
            if kind == "decision":
                dvals = sorted(df[lender].dropna().astype(str).unique())[:50]
                decline = c2.selectbox("Value meaning 'declined'", dvals,
                                       key=f"s_decline_{dg}_{lender}",
                                       persist_state="session",
                                       help="Rows with this value count as declined; "
                                            "all others as approved.")
        exclude = st.multiselect("Columns to leave out",
                                 [c for c in cols if c not in (target, attr, lender)],
                                 key=f"s_excl_{dg}", persist_state="session",
                                 help="Identifiers, and any near-duplicate of the "
                                      "protected attribute (e.g. a title that states "
                                      "gender), so the audit measures proxies rather "
                                      "than restatements.")

    checks = _upload_checks(df, target, positive, attr, lender, kind, decline, exclude)
    if any(s == "fail" for s, _ in checks):
        return checks, None
    X, y, A = _prepare(raw, target, positive, attr, tuple(exclude), n_rows, lender)
    checks += [("fail", p) for p in E.check_inputs(X, y, A[attr])]
    payload = {"X": X, "y": y, "a": A[attr], "probe": probe,
               "lender": (A[lender], kind, decline) if lender else None,
               "key": ("upload", dg, target, positive, attr, tuple(sorted(exclude)),
                       n_rows, lender, kind, decline),
               "meta": {"dataset": fname, "attribute": attr, "n_rows": len(X),
                        "n_features": X.shape[1],
                        "mode": "live (computed on a subsample)", "source": None}}
    return checks, payload


def _upload_checks(df, target, positive, attr, lender, kind, decline, exclude) -> list:
    checks = []
    out = df[target].dropna()
    k = out.nunique()
    numeric = pd.api.types.is_numeric_dtype(out)
    if k < 2:
        checks.append(("fail", f"Outcome **{target}** has only one value, so there is "
                               "nothing to predict."))
    elif k > 20:
        what = "a measurement" if numeric else "free text or a category"
        checks.append(("fail", f"**{target}** has {k:,} different values and looks like "
                               f"{what}, not an outcome. Choose the column that records "
                               "whether the applicant defaulted."))
    else:
        n_pos = int((out.astype(str) == str(positive)).sum())
        if k == 2:
            checks.append(("ok", f"Outcome **{target}**: '{positive}' counts as "
                                 f"defaulted ({n_pos:,} rows, {n_pos / len(out):.1%})."))
        else:
            checks.append(("warn", f"Outcome **{target}** has {k} values, not two. Rows "
                                   f"equal to '{positive}' count as defaulted and the "
                                   f"other {k - 1} values as repaid."))
        if min(n_pos, len(out) - n_pos) < 20:
            checks.append(("fail", "Fewer than 20 defaulters (or non-defaulters): too "
                                   "few to train a credit model."))
    missing = int(df[target].isna().sum() + df[attr].isna().sum())
    if missing:
        checks.append(("warn", f"{missing:,} rows have no outcome or no protected "
                               "attribute and will be skipped."))
    checks += _attr_checks(df[attr], attr)

    if lender:
        vals = df[lender].dropna()
        if kind == "score":
            share = pd.to_numeric(vals, errors="coerce").notna().mean() if len(vals) else 0
            if share < 0.95:
                checks.append(("fail", f"**{lender}** is not numeric, so it cannot be a "
                                       "score. If it holds approve/decline labels, "
                                       "choose 'A decision'."))
            else:
                checks.append(("ok", f"Your model's score **{lender}**: the same share "
                                     "is declined as actually defaulted."))
        elif vals.nunique() < 2:
            checks.append(("fail", f"Decision column **{lender}** has only one value."))
        else:
            checks.append(("ok", f"Your model's decision **{lender}**: '{decline}' "
                                 "counts as declined."))

    feats = [c for c in df.columns if c not in {target, attr, lender, *exclude}]
    if len(feats) < 2:
        checks.append(("fail", "At least two feature columns are needed."))
    else:
        checks.append(("ok", f"**{len(feats)}** columns used as credit features "
                             f"({len(exclude)} left out)."))
    ids = [c for c in feats if re.search(r"(^|_)id($|_)|^id|_id$", c, re.I)
           and df[c].nunique() > 0.95 * len(df)]
    if ids:
        checks.append(("warn", f"**{', '.join(ids)}** looks like an identifier (almost "
                               "every value unique). Consider leaving it out."))
    return checks


# --------------------------------------------------------------------------
# 3. run
# --------------------------------------------------------------------------
def page_run():
    ss = st.session_state
    st.html(CSS)
    _brand()
    p = ss.pending
    if p is None:
        st.info("There is no audit waiting to run.", icon=":material/info:")
        if st.button("Set up an audit", type="primary"):
            st.switch_page(SETUP)
        return

    meta, attr = p["meta"], p["meta"]["attribute"]
    st.title("Running your audit")
    st.caption(f"{meta['dataset']} · {meta['n_rows']:,} applicants · "
               f"{meta['n_features']} features · protected attribute {attr}")
    with st.status("Running the audit…", expanded=True) as status:
        st.markdown(f"{CHECK_ICON['ok']} **Step 1 of 3 · Data checked.** "
                    f"{meta['n_rows']:,} applicants, {meta['n_features']} features, "
                    f"default rate {p['y'].mean():.1%}.")
        try:
            lender = E.lender_audit(p["y"], p["a"], *p["lender"]) if p["lender"] else None
            st.markdown(f"**Step 2 of 3 · Leakage audit.** Can {attr} be guessed from "
                        "the features alone?")
            audit = _stage("audit", p["key"] + (p["probe"],),
                           lambda pr: E.audit(p["X"], p["a"], p["probe"], progress=pr))
            st.markdown("**Step 3 of 3 · Explanation.** Which revealing features does "
                        "the credit model rely on?")
            explain = _stage("explain", p["key"],
                             lambda pr: E.explain(p["X"], p["y"], p["a"], progress=pr))
        except Exception as e:
            status.update(label="The audit stopped", state="error", expanded=True)
            _fail("The audit", e)
            if st.button("Back to setup", icon=":material/arrow_back:"):
                st.switch_page(SETUP)
            return
        total = audit["runtime_s"] + explain["runtime_s"]
        status.update(label=f"Audit complete in {total:.0f} s", state="complete")

    ss.data = {k: p[k] for k in ("X", "y", "a", "key")}
    ss.meta, ss.lender, ss.audit, ss.explain = meta, lender, audit, explain
    ss.mitigate, ss.pending, ss.demo, ss.report_done = None, None, False, False
    ss.toast = f"Audit complete in {total:.0f} s"
    st.switch_page(RESULTS)


# --------------------------------------------------------------------------
# 4. results workspace
# --------------------------------------------------------------------------
def page_results():
    ss = st.session_state
    st.html(CSS)
    _brand(_new_audit_button)
    if not ss.demo and ss.audit is None:
        st.space("medium")
        st.info("No audit has been run in this session yet.", icon=":material/info:")
        c = st.container(horizontal=True)
        if c.button("Set up an audit", type="primary"):
            st.switch_page(SETUP)
        if c.button("Explore a sample audit"):
            ss.demo = True
            st.switch_page(RESULTS, query_params=_demo_params())
        return
    if ss.toast:
        st.toast(ss.toast, icon=":material/check_circle:")
        ss.toast = None

    _stepper()
    _summary_bar()
    tabs = st.tabs(TABS, key="tab", bind="query-params")
    for t, fn in zip(tabs, (tab_overview, tab_leakage, tab_explain, tab_repair,
                            tab_applicant, tab_report)):
        if t.open:
            with t:
                fn()
    if ss.pop("scroll_top", False):
        _scroll_top()


def _stepper():
    ss = st.session_state
    audit, explain, mit = _results()
    audited = audit is not None and explain is not None
    states = ["done", "done" if audited else "current",
              "done" if mit else ("current" if audited else "todo"),
              "done" if ss.report_done else ("current" if mit else "todo")]
    note = {"done": "done", "current": "next", "todo": ""}
    steps = [f"<div class='pl-step {s}'><span class='pl-dot'>{'✓' if s == 'done' else i}"
             f"</span>{name} <em>{note[s]}</em></div>"
             for i, (name, s) in enumerate(zip(["Data", "Audit", "Repair", "Report"],
                                               states), start=1)]
    st.html("<div class='pl-stepper'>" + "<div class='pl-line'></div>".join(steps)
            + "</div>")


def _kv(col, label: str, value):
    col.html(f"<div class='pl-kv'><span>{html.escape(label)}</span>"
             f"<b>{html.escape(str(value))}</b></div>")


def _attr_changed():
    st.query_params["attr"] = st.session_state.demo_attr


def _summary_bar():
    ss = st.session_state
    meta = _meta()
    _, _, mit = _results()
    attr = meta["attribute"]
    with st.container(border=True):
        cols = st.columns([2.4, 0.9, 0.9, 2.2, 2.2], vertical_alignment="center")
        _kv(cols[0], "Dataset", meta["dataset"])
        _kv(cols[1], "Rows", f"{meta['n_rows']:,}")
        _kv(cols[2], "Features", meta["n_features"])
        if ss.demo:
            cols[3].html("<div class='pl-kv'><span>Protected attribute</span></div>")
            cols[3].segmented_control(
                "Protected attribute", E.DEMO_ATTRIBUTES, key="demo_attr",
                required=True, persist_state="session", on_change=_attr_changed,
                format_func=_attr_name, label_visibility="collapsed")
        else:
            _kv(cols[3], "Protected attribute", f"{_attr_name(attr)} ({attr})"
                if _attr_name(attr) != attr else attr)
        with cols[4]:
            _kv(st, "Status", "Sample audit · stored results" if ss.demo
                else "Your data · computed live")
            if mit is None:
                st.badge("Audited · repair not applied", color="blue",
                         icon=":material/fact_check:")
            else:
                st.badge(f"Repair applied · {mit['mode']} mode", color="violet",
                         icon=":material/build:")


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
def _card(title: str, level: tuple, headline: str, detail: str):
    label, color, icon = level
    with st.container(border=True, height="stretch"):
        st.caption(title.upper())
        st.badge(label, color=color, icon=icon)
        st.markdown(f"**{headline}**")
        st.caption(detail)


def _probe_pair(mit: dict):
    """(weak before, weak after, strong before, strong after or None)."""
    pr = mit.get("probes")
    if pr is not None:
        w, s = pr.iloc[0], pr.iloc[1]
        return w.before, w.after, s.before, s.after
    t = mit["table"]["probe AUC"]
    return t.iloc[0], t.iloc[1], None, None


def _overview_facts():
    meta = _meta()
    audit, explain, mit = _results()
    name = _attr_name(meta["attribute"])
    danger = E.danger_features(explain["cross"])
    return meta, audit, explain, mit, name, danger


def tab_overview():
    ss = st.session_state
    meta, audit, explain, mit, name, danger = _overview_facts()
    act, ctl = audit["actual"], audit["control"]
    lr = None if ss.demo else ss.lender

    cards = []
    recovered = audit["verdict"] != "none"
    detail = (f"Probe AUC {act['auc_mean']:.3f} vs {ctl['auc_mean']:.3f} for the "
              "shuffled control (0.500 = guessing).")
    if abs(ctl["auc_mean"] - 0.5) > 0.05:
        detail += " The control did not return to chance: treat this with caution."
    cards.append(("Leakage", LEVELS[audit["verdict"]],
                  f"{name} {'can' if recovered else 'cannot'} be recovered from the "
                  "credit features", detail))

    rho = explain["stats"]["spearman_rho"]
    if len(danger):
        level = LEVELS[audit["verdict"] if recovered else "mild"]
        head = (f"{_n(len(danger), 'feature')} both reveal{'s' if len(danger) == 1 else ''} "
                f"{name} and drive{'s' if len(danger) == 1 else ''} the credit score")
    else:
        level, head = LEVELS["none"], f"The model does not rely on the features that reveal {name}"
    cards.append(("Model reliance", level, head,
                  f"Out of {len(explain['cross'])} features. Rank correlation between "
                  f"leakage and model reliance: ρ = {rho + 0.0:.2f}.".replace("-0.00", "0.00")))

    if mit is None:
        cards.append(("Leakage after repair", ("NOT APPLIED", "gray", ":material/pending:"),
                      "No repair applied yet",
                      "Open the Repair tab to choose full or guarded mode."))
    elif mit["n_treated"] == 0:
        cards.append(("Leakage after repair", LEVELS[audit["verdict"]],
                      "Nothing was repaired",
                      "The first repair round cost more accuracy than this mode "
                      "allows and was undone." if mit["budget_stop"] else
                      "Leakage was already below the repair's target."))
    else:
        wb, wa, sb, sa = _probe_pair(mit)
        after = sa if sa is not None else wa
        verdict = E.classify_leakage(after)
        head = (f"A strong probe still recovers {name}" if sa is not None and verdict != "none"
                else f"{name} {'can' if verdict != 'none' else 'can no longer'} be "
                     "recovered after repair")
        d_auc = mit["table"]["AUC"].iloc[1] - mit["table"]["AUC"].iloc[0]
        detail = f"Weak probe {wb:.3f} → {wa:.3f}"
        detail += (f"; strong probe {sb:.3f} → {sa:.3f}." if sa is not None else
                   ". The strong probe is stored for full mode only.")
        n_m = len(mit["manufactured"])
        detail += (f" {n_m} manufactured {'proxy' if n_m == 1 else 'proxies'}; "
                   f"credit-model AUC {d_auc:+.3f}.")
        cards.append(("Leakage after repair", LEVELS[verdict], head, detail))

    if lr:
        fails = lr["DI ratio"] < 0.8
        cards.append(("Your model's decisions",
                      LEVELS["meaningful"] if fails else LEVELS["none"],
                      f"Your decisions {'fail' if fails else 'pass'} the four-fifths rule",
                      f"DI ratio {lr['DI ratio']:.3f} (pass ≥ 0.80) · DP diff "
                      f"{lr['DP diff']:.3f} · EO diff {lr['EO diff']:.3f}."))

    st.subheader("Verdict")
    for col, c in zip(st.columns(len(cards)), cards):
        with col:
            _card(*c)

    st.subheader("What this means")
    st.markdown("\n".join(f"- {s}" for s in _meaning(audit, danger, mit, name, lr)))
    rec, label, tab = _recommendation(audit, danger, mit, name, lr)
    st.info(f"**Recommended next action.** {rec}", icon=":material/assistant_direction:")
    if lr:
        _show_lender(lr)
    with st.expander("How the ratings work", icon=":material/help:"):
        st.markdown(RATING_NOTE + " The model-reliance card takes the leakage rating "
                    "when at least one feature is both leaky and relied upon, and is "
                    "LOW when none is. Your model's decisions are rated against the "
                    "four-fifths rule: a disparate-impact ratio below 0.80 is HIGH.")
    _next(f"Next: {label}", tab)


def _meaning(audit, danger, mit, name, lr) -> list[str]:
    auc = audit["actual"]["auc_mean"]
    out = []
    if audit["verdict"] != "none":
        out.append(f"The credit model never sees {name}, but the other features give "
                   f"it away: a probe guesses it with AUC {auc:.3f}, where 0.500 is "
                   "guessing. A model trained on these features can therefore treat "
                   "groups differently without being told who is in which group.")
    else:
        out.append(f"The credit features reveal little about {name} (probe AUC {auc:.3f}), so "
                   "proxy discrimination through them is unlikely.")
    if len(danger):
        top = ", ".join(f"`{f}`" for f in danger["feature"].head(3))
        out.append(f"{len(danger)} of the revealing features "
                   f"{'is one' if len(danger) == 1 else 'are ones'} the credit model "
                   f"relies on, led by {top}. These are the routes by which {name} can "
                   "reach lending decisions.")
    else:
        out.append(f"The credit model leans on features that reveal little about {name}.")
    if mit is not None:
        if mit["n_treated"] == 0:
            out.append(f"The {mit['mode']} repair changed nothing, so the model is "
                       "unchanged.")
        else:
            wb, wa, sb, sa = _probe_pair(mit)
            s = f"The {mit['mode']} repair rewrote {mit['n_treated']} features. "
            if sa is not None and E.classify_leakage(sa) != "none":
                s += (f"Simple patterns were removed (weak probe {wb:.3f} → {wa:.3f}), "
                      f"but a stronger probe still finds {name} (AUC {sa:.3f}): much of "
                      "the signal was disguised rather than removed. ")
            else:
                s += f"The probe's AUC fell from {wb:.3f} to {wa:.3f}. "
            n_m = len(mit["manufactured"])
            if n_m:
                s += (f"The repair also wrote {name} information into "
                      f"{_n(n_m, 'feature')} that carried none before (manufactured "
                      f"{'proxy' if n_m == 1 else 'proxies'}). ")
            d = mit["table"]["AUC"].iloc[1] - mit["table"]["AUC"].iloc[0]
            out.append(s + f"Credit-model AUC changed by {d:+.3f}.")
    if lr:
        g = lr["groups"].sort_values("approval_rate")
        lo, hi = g.iloc[0], g.iloc[-1]
        out.append(f"Your own decisions approve {name} group '{lo.group}' at "
                   f"{lo.approval_rate:.1%} and '{hi.group}' at {hi.approval_rate:.1%} "
                   f"(DI ratio {lr['DI ratio']:.3f}), which "
                   f"{'fails' if lr['DI ratio'] < 0.8 else 'passes'} the four-fifths rule.")
    return out


def _recommendation(audit, danger, mit, name, lr) -> tuple[str, str, str]:
    lender = ("Your own decisions fail the four-fifths rule; investigate them before "
              "relying on them. " if lr and lr["DI ratio"] < 0.8 else "")
    if mit is None:
        if audit["verdict"] == "none" and not len(danger):
            return (lender + f"The credit features reveal little about {name}, so no repair "
                    "is needed. Download the report for your records.",
                    "get the report", "Report")
        return (lender + f"Review which features reveal {name} and whether the model "
                "relies on them, then decide on repair.", "review leakage", "Leakage")
    wb, wa, sb, sa = _probe_pair(mit)
    after = sa if sa is not None else wa
    if mit["n_treated"] and (E.classify_leakage(after) != "none" or len(mit["manufactured"])):
        rec = (f"The repair is partial. Do not treat repaired data as free of {name}: "
               "review the manufactured proxies, re-audit any model retrained on it, "
               "and file the report.")
    elif mit["n_treated"]:
        rec = "The repair removed the measurable leakage. File the report with the repaired data."
    else:
        rec = "Nothing was repaired. Try the other mode in the Repair tab, or file the report."
    return lender + rec, "get the report", "Report"


def _show_lender(lr: dict):
    st.subheader("Your model's decisions")
    st.caption(G["lender"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval rate", f"{lr['approval_rate']:.1%}", border=True)
    c2.metric("DP diff", f"{lr['DP diff']:.4f}", help=G["dp"], border=True)
    c3.metric("DI ratio", f"{lr['DI ratio']:.4f}", help=G["di"], border=True)
    c4.metric("EO diff", f"{lr['EO diff']:.4f}", help=G["eo"], border=True)
    st.dataframe(lr["groups"], hide_index=True, width="content", column_config={
        "approval_rate": st.column_config.NumberColumn("approval rate", format="%.3f")})
    st.caption(f"{lr['rule']} Measured on the {lr['n']:,} loaded applicants.")


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def tab_leakage():
    ss = st.session_state
    meta = _meta()
    res, _, _ = _results()
    attr = meta["attribute"]
    st.markdown(f"Can **{_attr_name(attr)}** be guessed from the credit features "
                "alone? If so, a model that never sees it can still treat groups "
                "differently. A probe classifier is trained to make exactly that guess.")
    act, ctl = res["actual"], res["control"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probe AUC", f"{act['auc_mean']:.3f}", help=G["probe_auc"], border=True,
              delta=f"± {act['auc_std']:.3f} across folds", delta_color="off",
              delta_arrow="off")
    c2.metric("Shuffled control", f"{ctl['auc_mean']:.3f}", help=G["control"],
              border=True, delta="should be ≈ 0.50", delta_color="off", delta_arrow="off")
    label = LEVELS[res["verdict"]][0]
    c3.metric("Leakage level", label, border=True,
              help="LOW < 0.55 ≤ MODERATE < 0.65 ≤ HIGH < 0.80 ≤ SEVERE")
    c4.metric("Balanced accuracy", f"{act['balanced_acc_mean']:.3f}", border=True,
              help=G["balanced_acc"])

    msg = (f"The credit features reveal **{attr}** with probe AUC "
           f"**{act['auc_mean']:.3f}**: **{label}** leakage.")
    {"severe": st.error, "meaningful": st.warning,
     "mild": st.info}.get(res["verdict"], st.success)(msg)
    if abs(ctl["auc_mean"] - 0.5) <= 0.05:
        st.success(f"Control passed: with the attribute shuffled the probe scores "
                   f"{ctl['auc_mean']:.3f}, i.e. chance. The leakage is real.")
    else:
        st.warning(f"Control did not return to chance ({ctl['auc_mean']:.3f}). Treat "
                   "the leakage figure with caution.")

    st.subheader("Most revealing features")
    top = st.slider("Features shown", 5, 30, 15)
    _show(E.leakage_bar_figure(res["ranking"], top=top))
    st.caption("Bar = drop in probe AUC when that feature alone is scrambled; whisker "
               "= variation across repeats. The longer the bar, the more the feature "
               "gives away about the protected attribute.")
    with st.expander("Full ranking", icon=":material/list:"):
        st.dataframe(res["ranking"], width="stretch", hide_index=True,
                     column_config={"leakage_drop": st.column_config.NumberColumn(
                         "leakage contribution", help=G["leakage"], format="%.5f")})
    probe = {"gbm": "gradient-boosting", "logistic": "logistic-regression"}[res["probe"]]
    _source(f"Precomputed: {E.DEMO_SOURCE['audit']}." if ss.demo else
            f"Computed live on this session's {meta['n_rows']:,}-row sample: a {probe} "
            "probe with 5-fold cross-validation, the same probe on a shuffled copy of "
            "the attribute, and a gradient-boosting permutation ranking.")
    _next("Next: see what the model relies on", "Explanation")


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------
def tab_explain():
    ss = st.session_state
    meta = _meta()
    _, res, _ = _results()
    st.markdown("Leakage only matters if the credit model **uses** the revealing "
                "features. This plot crosses what each feature gives away about the "
                "protected attribute (across) with how much the credit model relies on "
                "it (up, measured with SHAP).")
    cross, (lx, ly) = res["cross"], res["thresholds"]
    danger = E.danger_features(cross)
    c1, c2, c3 = st.columns(3)
    c1.metric("Features analysed", len(cross), border=True)
    c2.metric("Leaky and relied upon", len(danger), help=G["danger"], border=True)
    c3.metric("Rank correlation (ρ)", f"{res['stats']['spearman_rho']:.3f}", border=True,
              help="Spearman correlation between leakage and SHAP importance. "
                   "Positive means the model leans hardest on the features that "
                   "give the attribute away.")

    left, right = st.columns([3, 2])
    with left:
        _show(E.cross_figure(cross, (lx, ly),
                             title=f"{meta['attribute']}: leakage vs model reliance"))
        st.caption(f"Dashed lines are the cut-points: leakage above the permutation "
                   f"noise floor ({lx:.2g}) and SHAP importance above the median "
                   f"({ly:.2g}). Red points are in the top-right quadrant. Both axes "
                   "are symmetric-log so small and large values are visible together.")
    with right:
        st.markdown("##### Leaky and relied upon")
        st.caption(G["danger"])
        st.dataframe(danger, width="stretch", hide_index=True, height=_fit(len(danger)),
                     column_config={
                         "leakage_drop": st.column_config.NumberColumn(
                             "leakage", help=G["leakage"], format="%.5f"),
                         "shap_importance": st.column_config.NumberColumn(
                             "SHAP importance", help=G["shap"], format="%.4f")})
    with st.expander("How to read the four quadrants", icon=":material/help:"):
        st.markdown(
            "- **Top right — leaky and relied upon:** the problem. Removing these "
            "costs accuracy, so they must be repaired rather than dropped.\n"
            "- **Bottom right — leaky but unused:** can be dropped at no cost.\n"
            "- **Top left — relied upon, not leaky:** the model's safe workhorses.\n"
            "- **Bottom left — neither.**")
    _source(f"Precomputed: {E.DEMO_SOURCE['explain']}." if ss.demo else
            "Computed live: a gradient-boosting credit model trained on 70% of this "
            "session's sample; SHAP importance and per-feature leakage measured on the "
            "other 30%.")
    _next("Next: decide on repair", "Repair")


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------
def tab_repair():
    ss = st.session_state
    meta = _meta()
    _, explain, applied = _results()
    name = _attr_name(meta["attribute"])

    c1, c2 = st.columns(2)
    with c1, st.container(border=True, height="stretch"):
        st.markdown("##### :material/build: What repair does")
        st.markdown(f"Rewrites the features that reveal {name} so that every group's "
                    "values follow one shared distribution, then retrains the credit "
                    f"model on them. {name} is used **while fitting only**: scoring a "
                    "new applicant needs no demographic data.")
    with c2, st.container(border=True, height="stretch"):
        st.markdown("##### :material/balance: What it costs")
        st.markdown("- Some accuracy: full mode accepts up to 0.08 AUC, guarded mode "
                    "up to 0.02.\n"
                    "- It can **manufacture new proxies** in features that carried no "
                    "group information before.\n"
                    "- It removes simple patterns better than complex ones: a strong "
                    "probe may still recover the attribute.\n"
                    "- Repaired values shift; categories become numeric codes.")

    st.subheader("Choose a mode")
    mode = st.radio("Repair mode", list(E.MODES), key="repair_mode",
                    persist_state="session", label_visibility="collapsed",
                    format_func={"full": "Full (default)",
                                 "guarded": "Guarded (protects accuracy)"}.get,
                    captions=[G["full"], G["guarded"]])
    same = applied is not None and applied["mode"] == mode
    label = ("Repair applied" if same else
             "Apply repair" if applied is None else f"Apply {mode} repair instead")
    if st.button(label, type="primary", icon=":material/build:", disabled=same):
        if ss.demo:
            ss.demo_repair = mode
            st.query_params["mode"] = mode
            ss.toast = f"{mode.capitalize()} repair applied (stored results)"
            st.rerun()
        res = _apply_repair(mode, explain)
        if res is not None:
            ss.mitigate, ss.report_done = res, False
            ss.toast = f"{mode.capitalize()} repair applied in {res['runtime_s']:.0f} s"
            st.rerun()
        return
    if applied is None:
        st.caption("Fits on 70% of the rows and evaluates on the other 30%. Full mode "
                   "takes longest." if not ss.demo else
                   "The sample audit shows the stored results of the experiments for "
                   "the mode you apply.")
        return
    if not same:
        st.caption(f"The results below are for **{applied['mode']}** mode.")
    st.divider()
    _repair_results(applied, meta)
    _next("Next: look up an applicant", "Applicant lookup")


def _apply_repair(mode: str, explain: dict):
    d = st.session_state.data
    with st.status(f"Applying the {mode} repair…", expanded=True) as status:
        try:
            res = _stage("mitigate", d["key"] + (mode,),
                         lambda p: E.mitigate(d["X"], d["y"], d["a"], mode,
                                              explain["cross"], explain["thresholds"],
                                              progress=p))
        except Exception as e:
            status.update(label="The repair stopped", state="error", expanded=True)
            _fail("The repair", e)
            return None
        status.update(label=f"Repair done in {res['runtime_s']:.0f} s", state="complete",
                      expanded=False)
    return res


def _repair_results(res: dict, meta: dict):
    ss = st.session_state
    before, after = res["table"].iloc[0], res["table"].iloc[1]
    st.subheader(f"Before and after · {res['mode']} mode")
    for col, m in zip(st.columns(5), ["AUC", "DP diff", "DI ratio", "EO diff", "probe AUC"]):
        col.metric(m, f"{after[m]:.4f}", delta=f"{after[m] - before[m]:+.4f}", border=True,
                   delta_color="inverse" if LOWER_IS_BETTER[m] else "normal",
                   help=f"{METRIC_HELP[m]} Before repair: {before[m]:.4f}.")
    st.caption("Large number = after repair; arrow = change from before. Green = fairer "
               "or more accurate; red = worse.")
    st.dataframe(res["table"], width="stretch", hide_index=True,
                 column_config={m: st.column_config.NumberColumn(m, help=h, format="%.4f")
                                for m, h in METRIC_HELP.items()})

    if res["n_treated"] == 0:
        st.info("No feature was repaired, so the model is unchanged. "
                + ("The first repair round cost more accuracy than this mode allows, "
                   "and was rolled back." if res["budget_stop"] else
                   "Leakage was already below the mitigator's target."))

    st.subheader("Leakage measured two ways")
    pr = res.get("probes")
    if pr is None:
        st.info("The strong-probe measurement is stored for full mode only. Apply "
                "full mode to see it.", icon=":material/info:")
    else:
        for col, (_, r) in zip(st.columns(2), pr.iterrows()):
            col.metric(r.probe, f"{r.after:.3f}", border=True,
                       delta=f"{r.after - r.before:+.3f} from {r.before:.3f}",
                       delta_color="inverse",
                       help=G["weak_probe"] if r.probe.startswith("Weak") else G["strong_probe"])
        st.caption(G["probe_gap"])
        if res.get("probes_source"):
            st.caption("The weak-probe values come from a different run than the "
                       "25-fold table above, so they differ slightly.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Features repaired", res["n_treated"], border=True)
    c2.metric("Skipped by the guard", res["n_skipped"], border=True,
              help="Near-constant features the guarded mode will not touch.")
    c3.metric("Leaky and relied upon", res["danger_after"], border=True,
              delta=f"{res['danger_after'] - res['danger_before']:+d} vs before",
              delta_color="inverse", help=G["danger"])

    m = res["manufactured"]
    st.subheader("Manufactured proxies")
    if len(m):
        st.warning(f"**{_n(len(m), 'feature')}** had no detectable leakage before "
                   "repair and carr" + ("ies" if len(m) == 1 else "y")
                   + " it afterwards. " + G["manufactured"])
        st.dataframe(m, width="stretch", hide_index=True, height=_fit(len(m)), column_config={
            "leakage_drop_before": st.column_config.NumberColumn("leakage before", format="%.5f"),
            "leakage_drop_after": st.column_config.NumberColumn("leakage after", format="%.5f"),
            "n_unique_before": st.column_config.NumberColumn(
                "distinct values before", help="Number of distinct values in the column."),
            "n_unique_after": st.column_config.NumberColumn(
                "distinct values after", help="A sharp rise means the repair rewrote a "
                                              "near-constant column.")})
    else:
        st.success("No feature gained leakage from below the noise floor.")

    st.subheader("Download the repaired data")
    if ss.demo:
        st.info("Available when you audit your own data. The sample audit holds "
                "stored results only, not applicant data.", icon=":material/info:")
    else:
        d = ss.data
        key = d["key"] + (res["mode"],)
        ready = ("export", key) in _store()
        if not ready:
            slot = st.empty()
            if slot.button("Prepare repaired data", icon=":material/table:"):
                slot.empty()
                ready = True
        if ready:
            try:
                with st.spinner("Applying the repair to every loaded applicant…"):
                    data = _cached("export", key, lambda: E.repaired_export(
                        d["X"], d["y"], res["mitigator"], meta, res["mode"],
                        res["n_treated"]))
            except Exception as e:
                _fail("Preparing the repaired data", e)
            else:
                st.download_button("Download repaired data (.zip)", data,
                                   icon=":material/download:", on_click="ignore",
                                   file_name=f"repaired_{meta['attribute']}_{res['mode']}.zip",
                                   mime="application/zip")
        st.caption("A zip of the loaded applicants with the revealing features repaired "
                   "(the protected attribute removed), and a note explaining that a "
                   "lender would retrain its own credit model on it.")

    if ss.demo:
        src = E.DEMO_SOURCE["mitigate"]
        if res.get("probes_source"):
            src += f"; leakage measured two ways from {res['probes_source']}"
        _source(f"Precomputed: {src}.")
    else:
        _source(f"Computed live on this session's {meta['n_rows']:,}-row sample: the "
                "repair and the credit model are fitted on 70% of the rows and "
                "evaluated on the other 30%, which never see the protected attribute.")


# --------------------------------------------------------------------------
# Applicant lookup
# --------------------------------------------------------------------------
def _fmt(v):
    if v is None:
        return "missing"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    return str(v)


def tab_applicant():
    ss = st.session_state
    meta = _meta()
    st.markdown("What the repair means for one applicant: their risk score and "
                "decision before and after repair, the decision threshold, the "
                "features that drove their score, and how their values changed.")
    if ss.demo:
        rows = E.demo_borrowers(meta["attribute"])
        if not rows:
            st.info("No stored example applicants on this computer yet. They are built "
                    "locally, because they are real Home Credit rows that may not be "
                    "published: run `python app/build_demo_borrowers.py` (about 12 "
                    "minutes), then reload this page.", icon=":material/info:")
            _next("Next: get the report", "Report")
            return
        pick = st.radio("Example applicant", range(len(rows)), horizontal=True,
                        format_func=lambda i: f"{rows[i]['label']} "
                                              f"(applicant {rows[i]['applicant']})")
        b = rows[pick]
        src = ("Precomputed examples from the held-out 15,000 applicants of the "
               "50,000-row sample, with the iteration-1 full repair. Applicants are "
               "identified only by their position in that held-out set. Built locally "
               "by `app/build_demo_borrowers.py` and not published, because Home "
               "Credit's licence does not allow its rows to be redistributed.")
    else:
        res = ss.mitigate
        if res is None or "live" not in res:
            st.info("Apply a repair first: this view compares the credit model before "
                    "and after it.", icon=":material/info:")
            st.button("Go to Repair", type="primary", icon=":material/arrow_forward:",
                      icon_position="right", on_click=_goto, args=("Repair",))
            return
        idx = E.borrower_index(res["live"]).set_index("applicant", drop=False)
        n_changed = int(idx.changed.sum())
        only = st.toggle("Only applicants whose decision the repair changed",
                         value=n_changed > 0)
        pool = idx[idx.changed] if only and n_changed else idx
        st.caption(f"{n_changed} of {len(idx):,} held-out applicants get a different "
                   f"decision after the {res['mode']} repair.")
        i = st.selectbox("Applicant (position in the held-out 30%)", pool.applicant.tolist(),
                         format_func=lambda j: f"Applicant {j}: "
                                               f"{idx.at[j, 'before']} → {idx.at[j, 'after']}")
        try:
            b = _cached("borrower", ss.data["key"] + (res["mode"], int(i)),
                        lambda: E.borrower(res["live"], int(i)))
        except Exception as e:
            _fail("Explaining this applicant", e)
            return
        src = (f"Computed live: the held-out 30% of this session's sample, scored by the "
               f"credit model before and after the {res['mode']} repair.")

    st.caption(f"Actual outcome: **{b['actual_outcome']}**.")
    for col, side, title in zip(st.columns(2), (b["before"], b["after"]),
                                ("Before repair", "After repair")):
        with col, st.container(border=True):
            st.markdown(f"##### {title}")
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
    st.markdown("##### Key feature values before and after repair")
    vals = pd.DataFrame(b["values"])
    for c in ("value before repair", "value after repair"):
        vals[c] = vals[c].map(_fmt)
    st.dataframe(vals, hide_index=True, width="stretch")
    st.caption("Repair moves each value to the same position in a distribution shared "
               "by all applicants: numeric features keep their units but their values "
               "shift, and categories become numeric codes. Missing values are filled "
               "before repair.")
    _source(src)
    _next("Next: get the report", "Report")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _report_downloaded():
    st.session_state.report_done = True


def tab_report():
    ss = st.session_state
    meta = dict(_meta())
    audit, explain, mit = _results()
    if ss.demo:
        meta["source"] = "; ".join(E.DEMO_SOURCE.values())
    report = E.build_report(meta, audit, explain, mit,
                            lender_res=None if ss.demo else ss.lender)
    # the engine's report names the research repository; the product is ProxyLens
    report = report.replace("by the credit-fairness-research dashboard", f"by {APP}")
    st.markdown("A plain-language audit report of everything run so far, ready to "
                "file or share.")
    st.download_button("Download report (Markdown)", report, type="primary",
                       icon=":material/download:", on_click=_report_downloaded,
                       file_name=f"proxylens_audit_{meta['attribute']}.md",
                       mime="text/markdown")
    if mit is None:
        st.caption("Covers the leakage audit and the explanation. Apply a repair to "
                   "include it.")
    with st.container(border=True, height=720):
        st.markdown(report)


# --------------------------------------------------------------------------
# navigation
# --------------------------------------------------------------------------
def _legacy(tab: str | None):
    """A page address of the earlier dashboard, sent to its new place."""
    def redirect():
        ss = st.session_state
        if tab is None and not ss.demo:
            st.switch_page(SETUP)
        qp = _demo_params() if ss.demo else {}
        if tab:
            qp["tab"] = tab
        st.switch_page(RESULTS, query_params=qp)
    return redirect


START = st.Page(page_start, title=APP, url_path="start", default=True)
SETUP = st.Page(page_setup, title=f"Set up · {APP}", url_path="setup")
RUN = st.Page(page_run, title=f"Running · {APP}", url_path="run")
RESULTS = st.Page(page_results, title=f"Results · {APP}", url_path="results")
LEGACY = [st.Page(_legacy(tab), title=APP, url_path=path)
          for path, tab in (("load", None), ("audit", "Leakage"),
                            ("explain", "Explanation"), ("mitigate", "Repair"),
                            ("borrower", "Applicant lookup"), ("report", "Report"))]

_init_state()
st.navigation([START, SETUP, RUN, RESULTS, *LEGACY], position="hidden").run()
