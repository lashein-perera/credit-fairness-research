## Download record

| Dataset | Downloaded on | Version / notes |
|---|---|---|
| Home Credit Default Risk | 2026-08-02 | Kaggle competition `home-credit-default-risk`. Primary file `application_train.csv`, 166 MB, 307,511 rows × 122 columns. Verified: default rate 7.7%; `CODE_GENDER` F 66% / M 34%; `REGION_RATING_CLIENT` levels 1/2/3 at 11%/74%/16%. |
| UCI German Credit | 2026-08-02 | UCI ML Repository dataset 144, Statlog (German Credit Data). File `german.data`, 1,000 rows × 21 columns. Sex not a standalone column — derived from personal-status codes A91–A94 (A92 = female, n=310; A91/A93/A94 = male, n=690). Age range 19–75. Attribute definitions in `german.doc`. |
| UCI Default of Credit Card Clients | 2026-08-02 | UCI ML Repository dataset 350. File `default of credit card clients.xls`, 30,000 rows × 25 columns. Header on row 2 — load with `header=1`. Protected columns `SEX`, `EDUCATION`, `MARRIAGE` present. |

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
