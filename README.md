# Detecting and Mitigating Proxy Discrimination in Alternative-Data Credit Scoring for Thin-File Borrowers

Final-year research project — BSc (Hons) Computing, NIBM
**Author:** Lashein Perera (COBSCCOMP242P-014)
**Supervisor:** Dr. Thisara Weerasinghe

---

## Overview

Lenders increasingly score thin-file and unbanked applicants using **alternative data** — location, device, mobile/airtime usage, utility payments and behavioural signals. These features expand access to credit, but they also act as **proxies** for protected attributes such as gender, region and ethnicity. A model that is never given a protected attribute can therefore still discriminate through it.

Established bias-mitigation methods were developed and benchmarked on datasets where the protected attribute is **explicitly observed**. In alternative-data scoring it is **latent** — hidden inside ordinary features. This project asks whether those methods still work in that setting, and proposes a proxy-aware alternative that does not require the protected attribute at inference time.

## Research Objectives

1. **Identify** and quantify the extent to which alternative-data features encode protected attributes in thin-file credit datasets.
2. **Compare** the effectiveness of established bias-mitigation techniques under explicit versus latent protected-attribute conditions.
3. **Design** a proxy-aware, explainable fairness framework that detects leaky features and mitigates proxy-driven bias.
4. **Develop and evaluate** the framework as a working artefact, validated through benchmark experiments and structured expert review.

## Research Questions

1. To what extent do alternative-data features act as measurable proxies for protected attributes in thin-file credit datasets?
2. How does the effectiveness of established bias-mitigation techniques differ when the protected attribute is latent rather than explicit?
3. What framework design can detect and mitigate proxy-driven discrimination while preserving predictive performance and explainability?
4. How effective is the developed framework when evaluated against benchmark baselines and assessed by domain experts?

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  ProxyLens — Streamlit dashboard        app/dashboard.py │
│  start · setup · run · results tabs · report download    │
└───────────────┬──────────────────────────────────────────┘
                │  presentation only
┌───────────────▼──────────────────────────────────────────┐
│  Computation layer                      app/engine.py    │
│  sequences the modules; reads results/ for the sample    │
└───────────────┬──────────────────────────────────────────┘
                │  calls them unchanged
┌───────────────▼──────────────────────────────────────────┐
│  Python modules (frozen at artefact-v4-final)            │
│  1. Proxy Leakage Detector   src/python/leakage          │
│  2. Credit Models            src/python/models           │
│  3. Mitigation Bank          src/python/mitigation       │
│  4. Proxy-Aware Mitigator    src/python/mitigation       │
│  5. Explainability (SHAP)    src/python/explain          │
│  6. Evaluation Harness       src/python/evaluation       │
│     Data loaders             src/python/data             │
└───────────────┬──────────────────────────────────────────┘
                │  experiment scripts (run_*.py)
┌───────────────▼──────────────────────────────────────────┐
│  results/   frozen result tables and figures             │
│             (read back by the dashboard's sample audit)  │
└──────────────────────────────────────────────────────────┘
```

`src/java/audit-service/` is an **unused early skeleton** from the project's first week: a Spring Boot REST service whose endpoints were never implemented. The artefact does not use it; the dashboard took its place.

## Repository Layout

| Path | Contents |
|---|---|
| `app/` | ProxyLens dashboard (`dashboard.py`), its computation layer (`engine.py`) and the sample-applicant builder |
| `src/python/data/` | Loaders and the preprocessing pipeline |
| `src/python/leakage/` | Probe classifiers measuring protected-attribute leakage |
| `src/python/models/` | Baseline credit-scoring models |
| `src/python/mitigation/` | Established mitigation methods and the proxy-aware mitigator |
| `src/python/explain/` | SHAP explainability layer |
| `src/python/evaluation/` | Metrics and the experiment harness |
| `run_*.py` | Experiment scripts that produce the tables in `results/` |
| `results/` | Result tables and publication figures |
| `tests/` | Unit tests, including protected-attribute isolation |
| `docs/screenshots/` | Screenshots of each dashboard screen, used in Chapter 4 |
| `notebooks/` | Exploratory analysis only |
| `data/` | Datasets — **not committed**, see `data/README.md` |
| `src/java/audit-service/` | Unused early skeleton (Spring Boot stubs, never implemented); not part of the artefact |

---

## Setup

```bash
conda create -n creditfair python=3.11
conda activate creditfair
pip install -r requirements.txt
```

Then download the datasets as described in [`data/README.md`](data/README.md).

Run the tests:

```bash
pytest tests/
```

## Dashboard

**ProxyLens** is a Streamlit interface for running an audit without writing code. It is a user interface only: every number comes from the frozen modules under `src/python/` (tag `artefact-v4-final`), called through `app/engine.py`.

```bash
conda activate creditfair
streamlit run app/dashboard.py
```

The browser opens at <http://localhost:8501> on the start screen, which offers two choices:

- **Audit my data**: a single setup screen. Pick a built-in dataset (Home Credit, German Credit, UCI Default of Credit Card Clients) or upload a CSV; choose the outcome column, the protected attribute, optionally **your own model's score or decision**, and any columns to leave out. A "Ready to run?" checklist validates the choices (for example an outcome column that is not binary, or an attribute with more than ten groups) and keeps **Run audit** disabled until they pass. Rows to sample (5,000) and the leakage probe (gradient boosting, the methodology's probe) are under *Advanced settings*.
- **Explore a sample audit**: the stored dissertation results for the 50,000-row Home Credit sample, read from `results/`. Nothing is loaded or trained, so it opens instantly and works without the datasets downloaded.

**Run audit** runs the leakage audit and then the explanation, with a progress bar for each, and opens the results workspace. A stepper (Data → Audit → Repair → Report) and a summary bar (dataset, rows, features, protected attribute, status) stay at the top; the results are in tabs:

| Tab | What it shows |
|---|---|
| Overview | Verdict cards with a text rating (LOW / MODERATE / HIGH / SEVERE, from the probe-AUC bands), what the results mean, and a recommended next action. If a lender score or decision column was uploaded, the fairness of those decisions (approval rate, DP diff, DI ratio, EO diff) |
| Leakage | Probe AUC with the shuffled-attribute control, and the most revealing features |
| Explanation | Leakage vs SHAP cross-plot, and the features that are both leaky and relied upon |
| Repair | What repair does and costs, a choice of full (default) or guarded mode, then **Apply repair**: before/after AUC, DP diff, DI ratio, EO diff; leakage measured with the **weak** (logistic) and **strong** (gradient-boosting) probe; manufactured proxies; and, for your own data, a **download of the repaired data** with a note on retraining |
| Applicant lookup | One applicant's risk score and decision before and after repair, the decision threshold, the features driving their score, and their key values before and after repair |
| Report | Preview and download of a plain-language audit report (Markdown) |

Each tab ends with a button to the next step. Provenance (which `results/` file or live computation a number comes from) is in each tab's *Data source* expander.

Direct links, used in the dissertation:

| Link | Opens |
|---|---|
| <http://localhost:8501/?demo=1> | The sample audit (gender) |
| `?demo=1&attr=REGION_RATING_CLIENT` | The sample audit for region rating |
| `?demo=1&mode=full` (or `guarded`) | The sample audit with that repair applied |
| `?demo=1&tab=Repair` | A particular tab (Overview, Leakage, Explanation, Repair, Applicant+lookup, Report) |

The page addresses of the earlier dashboard (`/audit`, `/explain`, `/mitigate`, `/borrower`, `/report`) redirect to the matching tab.

The applicant lookup's sample examples are real Home Credit applicants, which the data licence does not allow to be published, so they are not in the repository. Build them once on a machine that has the data (about 12 minutes):

```bash
python app/build_demo_borrowers.py
```

They are written to `data/demo/`, which git ignores. Without them, every other tab still works in the sample audit.

On an Apple M1 laptop with 8 GB of memory, at the default 5,000 Home Credit rows, every step finishes in under a minute (measured for both attributes): loading 4–5 s, leakage audit 20–28 s, explanation 6–7 s, full repair 33–40 s, guarded repair 28–52 s, repaired-data export 12–16 s; the report and an applicant lookup take under a second. Results are cached, so revisiting a tab or re-running with the same settings is instant. Screenshots of each screen are in [`docs/screenshots/`](docs/screenshots/).

## Datasets

| Dataset | Role |
|---|---|
| Home Credit Default Risk | Primary — thin-file applicants with alternative-style features |
| UCI German Credit | Benchmark contrast — explicit protected attributes |
| UCI Default of Credit Card Clients | Secondary benchmark |

## Key Design Rule

The protected attribute is **withheld from every model** and retained **for evaluation only**. `tests/test_protected_attribute_isolation.py` enforces this and will fail if it is ever violated.

## Status

Complete. Final release `artefact-v4-ui2-r1` (dashboard); results frozen at `artefact-v4-final`.

## Licence

MIT — see [LICENSE](LICENSE).
