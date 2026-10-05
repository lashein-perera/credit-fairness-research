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
┌───────────────────────────────────────────────────────┐
│  Java Spring Boot — Fairness Audit Service            │
│  REST API · scoring · audit report · explanations     │
└───────────────┬───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│  Python ML core                                        │
│  1. Proxy Leakage Detector   src/python/leakage        │
│  2. Credit Models            src/python/models         │
│  3. Mitigation Bank          src/python/mitigation     │
│  4. Proxy-Aware Mitigator    src/python/mitigation     │
│  5. Explainability (SHAP)    src/python/explain        │
│  6. Evaluation Harness       src/python/evaluation     │
└────────────────────────────────────────────────────────┘
```

## Repository Layout

| Path | Contents |
|---|---|
| `src/python/data/` | Loaders and the preprocessing pipeline |
| `src/python/leakage/` | Probe classifiers measuring protected-attribute leakage |
| `src/python/models/` | Baseline credit-scoring models |
| `src/python/mitigation/` | Established mitigation methods and the proxy-aware mitigator |
| `src/python/explain/` | SHAP explainability layer |
| `src/python/evaluation/` | Metrics and the experiment harness |
| `src/java/audit-service/` | Spring Boot REST service and dashboard |
| `notebooks/` | Exploratory analysis only |
| `results/` | Result tables and publication figures |
| `docs/` | Dissertation drafts |
| `tests/` | Unit tests, including protected-attribute isolation |
| `data/` | Datasets — **not committed**, see `data/README.md` |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download the datasets as described in [`data/README.md`](data/README.md).

Run the tests:

```bash
pytest tests/
```

## Dashboard

A Streamlit interface for running an audit without writing code. It is a user interface only: every number comes from the frozen modules under `src/python/` (tag `artefact-v3-final`), called through `app/engine.py`.

```bash
streamlit run app/dashboard.py
```

The browser opens at <http://localhost:8501>. Work through the five screens in the sidebar:

| Screen | What it does |
|---|---|
| 1 · Load data | Pick Home Credit, German Credit or UCI Default, or upload a CSV; choose the outcome column and the protected attribute; sample ~5,000 rows |
| 2 · Leakage audit | Probe AUC with the shuffled-attribute control, and a bar chart of the most revealing features |
| 3 · Explain | Leakage vs SHAP cross-plot, and the features that are both leaky and relied upon |
| 4 · Mitigate | Proxy-aware mitigator in full or guarded mode: before/after AUC, DP diff, DI ratio, EO diff and probe AUC, plus any manufactured proxies |
| 5 · Report | Download a plain-language audit report (Markdown) |

**Demo mode** — the sidebar toggle, or <http://localhost:8501/?demo=1> — shows the precomputed dissertation results from `results/` instantly. Nothing is loaded or trained, so it cannot stall, and it works without the datasets downloaded. Add `&attr=REGION_RATING_CLIENT` or `&mode=full` to open on a particular attribute or mitigation mode.

Live runs take about 5–20 s per step at 5,000 rows on an Apple-silicon laptop; results are cached, so revisiting a screen or re-running with the same settings is instant. Screenshots of each screen are in [`docs/screenshots/`](docs/screenshots/).

## Datasets

| Dataset | Role |
|---|---|
| Home Credit Default Risk | Primary — thin-file applicants with alternative-style features |
| UCI German Credit | Benchmark contrast — explicit protected attributes |
| UCI Default of Credit Card Clients | Secondary benchmark |

## Key Design Rule

The protected attribute is **withheld from every model** and retained **for evaluation only**. `tests/test_protected_attribute_isolation.py` enforces this and will fail if it is ever violated.

## Status

Project in progress. Target completion: 1 October 2026.

## Licence

MIT — see [LICENSE](LICENSE).
