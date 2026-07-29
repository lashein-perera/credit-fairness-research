# Datasets

Datasets are **not committed** to this repository. Download them into this folder before running any experiments.

| Dataset | Source | Expected local path |
|---|---|---|
| Home Credit Default Risk | Kaggle — `competitions/home-credit-default-risk` | `data/home_credit/` |
| UCI German Credit (Statlog) | UCI Machine Learning Repository | `data/german_credit/` |
| UCI Default of Credit Card Clients | UCI Machine Learning Repository | `data/uci_default/` |

## Download record

Fill this in as you download — the dissertation methodology chapter needs it.

| Dataset | Downloaded on | Version / notes |
|---|---|---|
| Home Credit Default Risk | | |
| UCI German Credit | | |
| UCI Default of Credit Card Clients | | |

## Protected attributes

Recorded here for evaluation use only. These columns are **removed before model training**.

| Dataset | Protected attribute columns |
|---|---|
| Home Credit | `CODE_GENDER`, `REGION_RATING_CLIENT`, `REGION_RATING_CLIENT_W_CITY` |
| German Credit | personal status / sex attribute, age |
| UCI Default | `SEX`, `EDUCATION`, `MARRIAGE` |
