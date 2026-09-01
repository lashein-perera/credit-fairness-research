# Results

Generated outputs. Result tables and figures are committed so they can be cited in the dissertation.

---

## Module 1 — Proxy Leakage Detection

**Run date:** 31 August 2026
**Data:** Home Credit Default Risk, stratified 50,000-row subsample, default rate 8.07%
**Method:** probe classifiers predicting the protected attribute from the remaining features, cross-validated over 5 stratified folds. Operationalises the predictability test of Feldman et al. (2015).

### Headline results

| Attribute | Probe | AUC | Balanced acc. | Verdict |
|---|---|---|---|---|
| CODE_GENDER | logistic | 0.7840 ± 0.0036 | 0.6822 | meaningful |
| CODE_GENDER | gradient boosting | **0.8885 ± 0.0031** | 0.7868 | **severe** |
| REGION_RATING_CLIENT | logistic | 0.7815 ± 0.0039 | 0.4820 | meaningful |
| REGION_RATING_CLIENT | gradient boosting | **0.8264 ± 0.0031** | 0.5564 | **severe** |

### Shuffle controls — both passed

| Attribute | Control AUC | Result |
|---|---|---|
| CODE_GENDER | 0.5008 ± 0.0049 | PASS — collapses to chance |
| REGION_RATING_CLIENT | 0.5007 ± 0.0038 | PASS — collapses to chance |

With the protected attribute randomly permuted, probe performance must fall to chance. It does. The measurements are therefore properties of the data, not artefacts of the implementation.

### Top leaking features

**Gender**

| Rank | Feature | Leakage drop | Interpretation |
|---|---|---|---|
| 1 | OCCUPATION_TYPE | 0.1345 | Occupational segregation |
| 2 | OWN_CAR_AGE | 0.0397 | Vehicle ownership patterns |
| 3 | EXT_SOURCE_1 | 0.0292 | **External credit score carries gender** |
| 4 | AMT_INCOME_TOTAL | 0.0271 | Income disparity |
| 5 | DAYS_BIRTH | 0.0153 | Age–gender interaction |
| 6 | NAME_FAMILY_STATUS | 0.0128 | Marital status |
| 7 | FLAG_DOCUMENT_8 | 0.0117 | Which documents were submitted |

**Region**

| Rank | Feature | Leakage drop | Interpretation |
|---|---|---|---|
| 1 | EXT_SOURCE_2 | 0.0685 | **External credit score carries region** |
| 2 | HOUR_APPR_PROCESS_START | 0.0638 | **Time of day of application** |
| 3 | AMT_INCOME_TOTAL | 0.0170 | Regional income variation |
| 4 | TOTALAREA_MODE | 0.0108 | Property size |
| 5 | WALLSMATERIAL_MODE | 0.0081 | Construction materials |
| 6 | FLOORSMAX_AVG | 0.0059 | Building height |

---

## Two versions of the region result — both reportable

### v1 — `leakage_table_v1_with_population.csv`
`REGION_POPULATION_RELATIVE` retained in the feature set. Probe AUC **1.0000**, balanced accuracy 0.9997.

This is not a usable fairness measurement, because regional population density is close to a restatement of the region rating itself: permutation importance attributed 0.4638 to that single feature while every other feature scored below 0.00001.

It is nonetheless a finding worth reporting. **A single ordinary geographic variable encodes the protected attribute completely.** No fairness method that leaves such a variable in the feature set can prevent regional discrimination, whatever it does to the decision threshold.

### v2 — `leakage_table.csv` (current)
`REGION_POPULATION_RELATIVE` excluded alongside `REGION_RATING_CLIENT_W_CITY`. Probe AUC **0.8264**.

This is the honest measure of leakage through features that are not near-duplicates of the attribute.

**Report both.** v1 demonstrates the extreme case; v2 gives the defensible measurement.

---

## Observations for Chapter 5

**External credit scores leak protected attributes.** `EXT_SOURCE_1` ranks third for gender and `EXT_SOURCE_2` ranks first for region. These are purpose-built creditworthiness scores supplied by third-party providers — precisely the signals a lender would treat as clean and neutral. Both carry protected information.

**Leakage is substantially non-linear.** For gender, logistic regression reaches 0.784 while gradient boosting reaches 0.889. A linear audit would materially understate the exposure.

**Behavioural metadata leaks geography.** `HOUR_APPR_PROCESS_START` — the time of day an application is submitted — is the second strongest predictor of region rating.

**AUC and balanced accuracy diverge for region.** AUC 0.826 against balanced accuracy 0.556 reflects the three-class imbalance, with 74% of applicants in rating 2. Both are reported; relying on AUC alone would overstate the practical recoverability of the attribute.

---

## File index

| File | Produced | Contents |
|---|---|---|
| `leakage_table.csv` | Week 3 | Probe AUC per attribute and probe type (v2, current) |
| `leakage_table_v1_with_population.csv` | Week 3 | As above, retaining REGION_POPULATION_RELATIVE |
| `leaky_features_ranked.csv` | Week 3 | Features ranked by leakage contribution |
| `figures/leakage_home_credit_CODE_GENDER.png` | Week 3 | Top leaking features, gender |
| `figures/leakage_home_credit_REGION_RATING_CLIENT.png` | Week 3 | Top leaking features, region |

### Still to come
`baselines.csv`, `main_comparison.csv`, `proxy_aware.csv`, `ablation.csv`, `final_benchmark.csv`

---

## Reproducing

```bash
conda activate creditfair
python run_leakage_experiment.py            # 50,000-row subsample
python run_leakage_experiment.py --full     # all 307,511 rows
```

Protected attributes are excluded from the feature matrix by `src/python/data/loaders.py`. The runner aborts if any protected column reaches `X`.