# Results Summary — Artefact v3

Detecting and Mitigating Proxy Discrimination in Alternative-Data Credit Scoring
for Thin-File Borrowers.

Covers the frozen iteration-1 artefact and the iteration-2 and iteration-3
design cycles. Every
number here is reproducible from the committed CSVs under `results/`.

## Contents

1. [Conditions](#1-conditions)
2. [Combined comparison](#2-combined-comparison)
3. [Accuracy-budget trade-off](#3-accuracy-budget-trade-off)
4. [Iteration 2 candidate selection and criteria](#4-iteration-2-candidate-selection-and-criteria)
5. [Iteration 3: repair plus reweighing](#5-iteration-3-repair-plus-reweighing)
6. [Benchmark datasets](#6-benchmark-datasets)
7. [Known defects](#7-known-defects)
8. [Protocol notes](#8-protocol-notes)

---

## 1. Conditions

| Condition | Attribute at fitting | Attribute at inference |
|---|---|---|
| `explicit` | yes | yes |
| `training_only` | yes | no |
| `latent` | no | no |

`training_only` was added in iteration 2. The frozen `latent` condition gave the
proposed method the attribute at fitting while denying it to the four
established methods, so it was **not a like-for-like comparison**. Under
`training_only` every method has exactly the access the proposed method always
had: `harness.run_fold` sets `fit_A = Atr` and `infer_A = None` for all.

The proposed method is not re-run across these conditions — its `fit_A` and
`infer_A` are identical under `latent` and `training_only`, so one set of
numbers serves both.

**The central empirical finding.** Under `latent`, all four established methods
are numerically identical to no mitigation on every metric, to four decimal
places. They fall back (`fell_back = 1.0` on every fold) because each requires
the attribute it has been denied. That is the thesis claim, and it is visible
directly in the tables below.

---

## 2. Combined comparison

Home Credit, 8,000 rows, 25 cross-validation folds, gradient-boosting model.
`probe AUC` is residual leakage after mitigation — how well a probe recovers the
protected attribute from the features the model scores.

### CODE_GENDER

| method | condition | AUC | DP diff | DI ratio | EO diff | probe AUC | note |
|---|---|---|---|---|---|---|---|
| none | explicit | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| none | training_only | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| none | latent | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| reweighing | explicit | 0.7109 | 0.0226 | 0.9757 | 0.066 | 0.7397 |  |
| reweighing | training_only | 0.7109 | 0.0226 | 0.9757 | 0.066 | 0.7397 |  |
| reweighing | latent | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| adversarial_debiasing | explicit | 0.6769 | 0.029 | 0.9693 | 0.0906 | 0.7397 | CORRECTED: adversary gradient scaled |
| adversarial_debiasing | training_only | 0.6769 | 0.029 | 0.9693 | 0.0906 | 0.7397 | CORRECTED: adversary gradient scaled |
| adversarial_debiasing | latent | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 | CORRECTED: adversary gradient scaled |
| disparate_impact_remover | explicit | 0.714 | 0.0289 | 0.9691 | 0.0708 | 0.7397 |  |
| disparate_impact_remover | training_only | 0.714 | 0.0289 | 0.9691 | 0.0708 | 0.7397 | n/a - needs A at inference (skew) |
| disparate_impact_remover | latent | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| reject_option | explicit | 0.7085 | 0.0334 | 0.9639 | 0.0742 | 0.7397 | CORRECTED: band centred on operating threshold |
| reject_option | training_only | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 | n/a - needs A at inference |
| reject_option | latent | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |  |
| proxy_aware | training_only | 0.6953 | 0.0147 | 0.9841 | 0.0825 | 0.5498 | A at fit only - identical to latent by construction |
| proxy_aware | latent | 0.6953 | 0.0147 | 0.9841 | 0.0825 | 0.5498 |  |
| proxy_aware_v2 | training_only | 0.7057 | 0.0238 | 0.9744 | 0.0705 | 0.6723 | A at fit only - identical to latent by construction |
| proxy_aware_v2 | latent | 0.7057 | 0.0238 | 0.9744 | 0.0705 | 0.6723 |  |

### REGION_RATING_CLIENT

| method | condition | AUC | DP diff | DI ratio | EO diff | probe AUC | note |
|---|---|---|---|---|---|---|---|
| none | explicit | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| none | training_only | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| none | latent | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| reweighing | explicit | 0.7036 | 0.0835 | 0.9136 | 0.2598 | 0.8337 |  |
| reweighing | training_only | 0.7036 | 0.0835 | 0.9136 | 0.2598 | 0.8337 |  |
| reweighing | latent | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| adversarial_debiasing | explicit | 0.6828 | 0.1481 | 0.8515 | 0.3528 | 0.8337 | CORRECTED: adversary gradient scaled |
| adversarial_debiasing | training_only | 0.6828 | 0.1481 | 0.8515 | 0.3528 | 0.8337 | CORRECTED: adversary gradient scaled |
| adversarial_debiasing | latent | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 | CORRECTED: adversary gradient scaled |
| disparate_impact_remover | explicit | 0.7119 | 0.0928 | 0.9048 | 0.2448 | 0.8337 |  |
| disparate_impact_remover | training_only | 0.7119 | 0.0928 | 0.9048 | 0.2448 | 0.8337 | n/a - needs A at inference (skew) |
| disparate_impact_remover | latent | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| reject_option | explicit | 0.706 | 0.0465 | 0.9513 | 0.1398 | 0.8337 | CORRECTED: band centred on operating threshold |
| reject_option | training_only | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 | n/a - needs A at inference |
| reject_option | latent | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |  |
| proxy_aware | training_only | 0.674 | 0.0382 | 0.9594 | 0.1528 | 0.5882 | A at fit only - identical to latent by construction |
| proxy_aware | latent | 0.674 | 0.0382 | 0.9594 | 0.1528 | 0.5882 |  |
| proxy_aware_v2 | training_only | 0.7045 | 0.0829 | 0.914 | 0.2362 | 0.7678 | A at fit only - identical to latent by construction |
| proxy_aware_v2 | latent | 0.7045 | 0.0829 | 0.914 | 0.2362 | 0.7678 |  |

**Reading these tables.**

- **Leakage is the clear separation.** `probe AUC` is **0.7397 / 0.8337 for every
  established method under every condition** — identical to no mitigation.
  Reweighing reweights rows, adversarial debiasing changes the model, neither
  alters the features, so the attribute stays exactly as recoverable. Only the
  proxy-aware mitigator moves it.
- **Reweighing is a genuine competitor on outcome fairness.** Given equal
  training-only access it beats the proposed method on equalised odds and
  accuracy for gender. It cannot reduce leakage at all.
- **Two methods cannot be read as performance under `training_only`** — see
  [Known defects](#7-known-defects).
- **Two rows are corrected re-runs.** Adversarial debiasing (all conditions) and
  reject-option (explicit) were defective as originally shipped; the tables show
  corrected numbers and [Known defects](#7-known-defects) keeps the originals.
- **Corrected reject-option is strong on region when it has the attribute at
  inference** — EO 0.3074 -> 0.1398, better than the proposed method's 0.1528 —
  and slightly harmful on gender. It remains inapplicable without the attribute
  at scoring time, which is the deployment condition this research addresses.

---

## 3. Accuracy-budget trade-off

The chosen guard (`skip_dominant_share=0.99`) held fixed; only
`max_accuracy_loss` varies. Training-only access, same 25 folds at 8,000 rows.
`emerged` is the manufactured-proxy count, measured on a single held-out split
at the same sample size (a per-fold leakage ranking would cost more than the
sweep itself).

### CODE_GENDER

| budget | AUC | DP diff | DI ratio | EO diff | probe AUC | emerged | treated | skipped |
|---|---|---|---|---|---|---|---|---|
| none | 0.7056 | 0.0234 | 0.9748 | 0.0726 | 0.6557 | 15 | 97 | 19 |
| 0.01 | 0.7088 | 0.0258 | 0.9723 | 0.0704 | 0.7073 | 11 | 30 | 19 |
| 0.03 | 0.7057 | 0.0225 | 0.9758 | 0.0607 | 0.6674 | 15 | 97 | 19 |
| 0.02 | 0.7057 | 0.0238 | 0.9744 | 0.0705 | 0.6723 | 16 | 90 | 19 |
| 0.05 | 0.7058 | 0.023 | 0.9753 | 0.0697 | 0.6574 | 15 | 97 | 19 |
| [ref] none | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 | 0 | 0 | 0 |
| [ref] reweighing | 0.7109 | 0.0226 | 0.9757 | 0.066 | 0.7397 | 0 | 0 | 0 |

### REGION_RATING_CLIENT

| budget | AUC | DP diff | DI ratio | EO diff | probe AUC | emerged | treated | skipped |
|---|---|---|---|---|---|---|---|---|
| none | 0.6994 | 0.0788 | 0.9182 | 0.2191 | 0.7585 | 15 | 97 | 19 |
| 0.01 | 0.7037 | 0.0815 | 0.9155 | 0.2374 | 0.7868 | 15 | 97 | 19 |
| 0.03 | 0.6986 | 0.0813 | 0.9158 | 0.2273 | 0.7619 | 15 | 97 | 19 |
| 0.02 | 0.7045 | 0.0829 | 0.914 | 0.2362 | 0.7678 | 15 | 97 | 19 |
| 0.05 | 0.6992 | 0.0789 | 0.918 | 0.2197 | 0.7577 | 15 | 97 | 19 |
| [ref] none | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 | 0 | 0 | 0 |
| [ref] reweighing | 0.7036 | 0.0835 | 0.9136 | 0.2598 | 0.8337 | 0 | 0 | 0 |

**What the curve shows.**

- **Accuracy is essentially flat across budgets** (gender 0.7056–0.7088; region
  0.6986–0.7045) while leakage varies substantially. At this sample size the
  budget is not buying back meaningful accuracy — it only limits how far the
  repair iterates.
- **Returns diminish sharply past 0.03.** Gender probe AUC moves 0.6674 → 0.6557
  between budget 0.03 and no budget at all.
- **The guard, not the budget, is what costs leakage reduction.** With no budget
  the guarded mitigator reaches probe 0.6557 on gender; the unguarded iteration-1
  mitigator reached 0.5498. Skipping 19 degenerate features leaves roughly 0.10
  of probe AUC on the table. That is the price of closing one failure mode.
- **Manufactured proxies barely respond to the budget** (15–16 on gender across
  every setting but the most restrictive). Emergence is not a function of how far
  the repair iterates.

Figures: `results/iteration2/figures/tradeoff_*.png` — labelled scatter points,
with the unguarded iteration-1 mitigator marked separately for comparison.

---

## 4. Iteration 2 candidate selection and criteria

Four candidates compared on a validation split carved out of the training
portion; the winner run once on the held-out test split, against frozen baseline
thresholds and the noise-floor leakage cut-point.

### CODE_GENDER

| candidate | treated | skipped | probe drop | AUC drop | emerged | DP vs base | EO vs base |
|---|---|---|---|---|---|---|---|
| C1_guard99 | 97 | 19 | 0.07111 | 0.00386 | 16 | -0.00715 | -0.03391 |
| C2_guard95_binary | 76 | 40 | 0.00409 | 0.0052 | 21 | -0.00757 | -0.01801 |
| C3_guard99_partial50 | 97 | 19 | 0.01475 | 0.00538 | 10 | -0.0021 | -0.04186 |
| C4_residual_guard99 | 97 | 19 | -0.00758 | 0.00478 | 19 | -0.00631 | -0.04104 |

### REGION_RATING_CLIENT

| candidate | treated | skipped | probe drop | AUC drop | emerged | DP vs base | EO vs base |
|---|---|---|---|---|---|---|---|
| C1_guard99 | 97 | 19 | 0.08038 | 0.00248 | 25 | -0.00344 | -0.08658 |
| C2_guard95_binary | 76 | 40 | 0.04112 | 0.00462 | 17 | -0.0124 | -0.03051 |
| C3_guard99_partial50 | 97 | 19 | 0.03902 | 0.00214 | 19 | -0.00737 | 0.03608 |
| C4_residual_guard99 | 97 | 19 | 0.01164 | 0.00801 | 21 | -0.0042 | 0.01546 |

No candidate met every hard constraint — zero emerged proxies was never
achieved — so selection fell to the relaxed rule: accuracy drop within budget,
then maximum leakage removed. **C1_guard99 was chosen independently for both
attributes.**

C2's wider guard skips 40 features and collapses leakage reduction on gender
(probe drop 0.0041); C4 (residualisation) *increased* gender leakage.
Over-guarding removes the method's reach.

### Pre-registered success criteria

### CODE_GENDER — chosen config `C1_guard99`

| criterion | passed | evidence |
|---|---|---|
| a_probe_drop_ge_0.15 | FAIL | 0.0665 >= 0.15 |
| b_dp_and_eo_below_baseline | FAIL | DP -0.0024, EO +0.0141 |
| c_auc_drop_le_0.02 | PASS | 0.0158 <= 0.02 |
| d_zero_manufactured_proxies | FAIL | 17 emerged |
| e_max_and_total_leak_below_baseline | PASS | max -0.01893, total -0.00563 |

### REGION_RATING_CLIENT — chosen config `C1_guard99`

| criterion | passed | evidence |
|---|---|---|
| a_probe_drop_ge_0.15 | FAIL | 0.0590 >= 0.15 |
| b_dp_and_eo_below_baseline | PASS | DP -0.0094, EO -0.0039 |
| c_auc_drop_le_0.02 | PASS | 0.0160 <= 0.02 |
| d_zero_manufactured_proxies | FAIL | 15 emerged |
| e_max_and_total_leak_below_baseline | FAIL | max +0.00018, total +0.00650 |

**Two of five pass on each attribute.** Iteration 2 does not deliver a working
mitigator against the bar set in advance.

The accuracy budget delivered criterion (c) for the first time, but probe
reduction fell from 0.2165 / 0.2680 in iteration 1 to 0.0665 / 0.0590. Iteration
1 was not badly tuned — it was *buying* its leakage reduction with the accuracy
that criterion (c) forbids. **Criteria (a) and (c) may be jointly unsatisfiable
by this mechanism**, which is a finding rather than a failure to conceal.

---

## 5. Iteration 3: repair plus reweighing

**Hypothesis.** Proxy-aware repair is the only method that reduces leakage;
reweighing is the only one that improved gender equalised odds under equal
access. Repair alone regressed gender EO (0.0671 -> 0.0825). Repairing the
features and then training on them with Kamiran-Calders weights should keep the
leakage and DP gains while repairing that regression.

**Candidates** (no others): D1 = unguarded repair (iteration-1 settings) +
reweighing; D2 = guarded repair (iteration-2 C1_guard99) + reweighing. The
weighting delegates to the bank's own `Reweighing` class, so it is the
established method applied to repaired features. Training-only access.

**Selection, fixed before running.** Validation rows are disjoint from the
8,000 evaluation rows: a different subsample, filtered by row hash (347
overlapping rows removed). 5-fold CV on that pool. Per attribute, the candidate
passing more of (i)-(iv) wins; ties go to lower EO, then lower probe AUC.

### CODE_GENDER — validation

| candidate | criteria passed | AUC | DP diff | EO diff | probe AUC |
|---|---|---|---|---|---|
| [ref] none |  | 0.681 | 0.0212 | 0.0612 | 0.7098 |
| [ref] reweighing |  | 0.6863 | 0.0099 | 0.0582 | 0.7098 |
| D1 | 4/4 | 0.6746 | 0.0094 | 0.0333 | 0.5157 |
| D2 | 3/4 | 0.6727 | 0.0096 | 0.069 | 0.6414 |

### REGION_RATING_CLIENT — validation

| candidate | criteria passed | AUC | DP diff | EO diff | probe AUC |
|---|---|---|---|---|---|
| [ref] none |  | 0.681 | 0.0913 | 0.2082 | 0.8097 |
| [ref] reweighing |  | 0.6787 | 0.0732 | 0.2 | 0.8097 |
| D2 | 4/4 | 0.669 | 0.0559 | 0.1301 | 0.7429 |
| D1 | 4/4 | 0.6554 | 0.049 | 0.1832 | 0.5845 |

D1 chosen for gender (4/4 vs 3/4). Region tied at 4/4; D2 chosen on lower EO.

**Final evaluation** — chosen configuration run once, 25 folds at 8,000 rows,
against the existing reference rows measured under the same protocol.

### CODE_GENDER

| method | AUC | DP diff | DI ratio | EO diff | probe AUC |
|---|---|---|---|---|---|
| none | 0.7086 | 0.0307 | 0.967 | 0.0671 | 0.7397 |
| reweighing | 0.7109 | 0.0226 | 0.9757 | 0.066 | 0.7397 |
| proxy_aware full (iteration 1) | 0.6953 | 0.0147 | 0.9841 | 0.0825 | 0.5498 |
| proxy_aware guarded (iteration 2) | 0.7057 | 0.0238 | 0.9744 | 0.0705 | 0.6723 |
| repair + reweighing, D1 (iteration 3) | 0.6929 | 0.0153 | 0.9835 | 0.0763 | 0.5498 |

### REGION_RATING_CLIENT

| method | AUC | DP diff | DI ratio | EO diff | probe AUC |
|---|---|---|---|---|---|
| none | 0.7086 | 0.0941 | 0.9033 | 0.3074 | 0.8337 |
| reweighing | 0.7036 | 0.0835 | 0.9136 | 0.2598 | 0.8337 |
| proxy_aware full (iteration 1) | 0.674 | 0.0382 | 0.9594 | 0.1528 | 0.5882 |
| proxy_aware guarded (iteration 2) | 0.7045 | 0.0829 | 0.914 | 0.2362 | 0.7678 |
| repair + reweighing, D2 (iteration 3) | 0.6966 | 0.0691 | 0.928 | 0.2092 | 0.7678 |

### Success criteria (fixed in advance)

### CODE_GENDER — chosen `D1`

| criterion | passed | evidence |
|---|---|---|
| i_eo_no_worse_than_none | FAIL | EO 0.0763 vs none 0.0671 |
| ii_dp_below_reweighing | PASS | DP 0.0153 vs reweighing 0.0226 |
| iii_probe_below_reweighing | PASS | probe 0.5498 vs reweighing 0.7397 |
| iv_auc_within_0.035 | PASS | AUC gap 0.0157 (limit 0.035) |

### REGION_RATING_CLIENT — chosen `D2`

| criterion | passed | evidence |
|---|---|---|
| i_eo_no_worse_than_none | PASS | EO 0.2092 vs none 0.3074 |
| ii_dp_below_reweighing | PASS | DP 0.0691 vs reweighing 0.0835 |
| iii_probe_below_reweighing | PASS | probe 0.7678 vs reweighing 0.8337 |
| iv_auc_within_0.035 | PASS | AUC gap 0.0120 (limit 0.035) |

### Verdict: the hypothesis is not supported

- **Gender fails exactly the criterion the hypothesis targeted.** Reweighing
  moved EO from 0.0825 to 0.0763 — a real improvement — but it stays worse than
  unmitigated (0.0671). The regression is reduced, not repaired.
- **The validation result did not replicate.** On validation, D1 reached gender
  EO 0.0333 against an unmitigated 0.0612 and passed all four criteria. On the
  evaluation set it fails (i). One 8,000-row pool over five folds was not enough
  to estimate a gender EO difference of this size reliably.
- **Region's 4/4 is not evidence for the combination.** The iteration-1
  mitigator **alone** also passes all four region criteria (EO 0.1528, DP 0.0382,
  probe 0.5882, AUC gap 0.0346 — just inside the 0.035 limit) and beats D2 on
  every fairness metric. The
  criteria compare against reweighing and no mitigation, not against repair
  alone, so they cannot attribute a pass to the combination. D2's only advantage
  over iteration 1 on region is accuracy (0.6966 vs 0.6740).
- **D1's probe AUC equals iteration 1's exactly** (0.5498 gender) — expected,
  since reweighing changes weights, not features. A useful sanity check that the
  composition behaves as designed.

- **Across both attributes, the combination's pass/fail profile is identical to
  repair alone.** Iteration 1 by itself scores 3/4 on gender, failing the same EO
  criterion, and 4/4 on region. Iteration 3 changed the magnitudes, not a single
  verdict.

**What iteration 3 does establish:** reweighing composes with repair without
undoing its leakage reduction, and it partially offsets repair's EO cost on
gender. It does not deliver the fix it was designed for.

---

## 6. Benchmark datasets

Explicit and latent conditions, 5x2 folds.

### German Credit (sex)

| method | condition | AUC | DP diff | DI ratio | EO diff | probe AUC |
|---|---|---|---|---|---|---|
| adversarial_debiasing | explicit | 0.7537 | 0.1156 | 0.8537 | 0.214 | 0.666 |
| adversarial_debiasing | latent | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| disparate_impact_remover | explicit | 0.7743 | 0.0493 | 0.9318 | 0.094 | 0.666 |
| disparate_impact_remover | latent | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| none | explicit | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| none | latent | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| proxy_aware | latent | 0.743 | 0.0536 | 0.9279 | 0.1086 | 0.5892 |
| reject_option | explicit | 0.7701 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| reject_option | latent | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |
| reweighing | explicit | 0.7736 | 0.048 | 0.9346 | 0.0939 | 0.666 |
| reweighing | latent | 0.7705 | 0.0527 | 0.9284 | 0.0989 | 0.666 |

### Default of Credit Card Clients (SEX)

| method | condition | AUC | DP diff | DI ratio | EO diff | probe AUC |
|---|---|---|---|---|---|---|
| adversarial_debiasing | explicit | 0.5625 | 0.0438 | 0.9456 | 0.059 | 0.5689 |
| adversarial_debiasing | latent | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| disparate_impact_remover | explicit | 0.7808 | 0.0418 | 0.9475 | 0.0414 | 0.5689 |
| disparate_impact_remover | latent | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| none | explicit | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| none | latent | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| proxy_aware | latent | 0.7748 | 0.0341 | 0.9569 | 0.0298 | 0.5298 |
| reject_option | explicit | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| reject_option | latent | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |
| reweighing | explicit | 0.7827 | 0.0392 | 0.9507 | 0.0383 | 0.5689 |
| reweighing | latent | 0.7826 | 0.0419 | 0.9474 | 0.0418 | 0.5689 |

Baseline AUC is **0.7705** on German Credit and **0.7826** on Default of Credit
Card Clients, both within the band usually reported for these datasets with
gradient boosting. Specific published figures should be checked against the
source papers rather than quoted from here.

The latent-condition collapse reproduces on both: every established method
returns exactly the unmitigated numbers.

**Coverage gap.** Module 1 leakage measurement and the main explicit-vs-latent
comparison were run on **Home Credit only**. These benchmark runs are the sole
multi-dataset evidence; the leakage tables, ablation and cross-plots are
single-dataset. Chapter 3 should not claim three-dataset validation beyond what
this section covers.

---

## 7. Known defects

### 7.1 Manufactured proxies — unresolved

The conditional repair **creates** proxies in features that had none. It
percentile-maps each feature within strata of a *predicted* attribute; a feature
whose distribution gives it nothing to repair receives stratum identity instead.

`FLAG_CONT_MOBILE` is the canonical case — 99.79% constant, two distinct values,
zero baseline leakage:

| arm | treated | leakage after | SHAP after | distinct values |
|---|---|---|---|---|
| iteration 1, full (116 features) | yes | 0.135 / 0.094 | 0.042 / 0.116 | 2 -> 30 / 41 |
| iteration 1, random (116) | yes | 0.013 / 0.010 | 0.066 / 0.093 | 2 -> 174 / 410 |
| iteration 2, guarded | **no** | **0.000** | **0.000** | **2 -> 2** |

(gender / region)

**The guard fixed its target and did not fix the phenomenon.** 17 and 15 features
still emerge in iteration 2, led by `FLAG_OWN_CAR` (2 -> 477 distinct) and
`FLAG_EMP_PHONE` (2 -> 817) — binary features at dominant shares 0.66 and 0.82,
below any threshold that does not also exclude genuine proxies. High-cardinality
features emerge too (`DAYS_REGISTRATION`, 12,664 distinct), which no degeneracy
guard can exclude. Emergence has a second route through the percentile remap
itself.

Recorded per feature per arm in `mitigation_transitions.csv` and
`iteration2/transitions_*.csv` via `leakage_emerged`, `n_unique_before`,
`n_unique_after`.

### 7.2 Disparate impact remover — train/serve skew

`DisparateImpactRemover.predict_proba` ignores `A` and scores **raw** features,
while the model was fitted on **repaired** features. That is a defect in this
implementation, not a property of Feldman et al.'s method, and it depresses the
method's numbers under every condition. Its `training_only` row is marked
not-applicable for this reason.

### 7.3 Adversarial debiasing — corrected in iteration 2

The shipped predictor update subtracted the adversary's gradient at a fixed
weight with no regard to the predictor gradient's magnitude, so the adversary
term dominated and the predictor never fit the outcome. Diagnosis on Home
Credit, gender:

| configuration | test AUC |
|---|---|
| gbm (what other methods use) | 0.7205 |
| sklearn logistic | 0.7145 |
| hand-rolled logistic, 120 steps, adversary off | 0.7121 |
| hand-rolled logistic, 2000 steps, adversary off | 0.7140 |
| **as originally shipped** | **0.6134** |

Model family cost ~0.006 AUC and step count ~0.002 — neither explains the
collapse. The adversary term destroyed ~0.10 AUC on its own.

**Original (defective) numbers, retained for the record:**

| attribute | condition | AUC | DP diff | DI ratio | EO diff |
|---|---|---|---|---|---|
| CODE_GENDER | explicit / training-only | 0.6134 | 0.0523 | 0.9453 | 0.1255 |
| REGION_RATING_CLIENT | explicit / training-only | 0.6665 | 0.1527 | 0.8469 | 0.3717 |

The symptom worth noting: the method scored **worse with** the protected
attribute (0.6134) than **without** it (0.7086, where it falls back to a plain
model). A fairness method harmed by receiving the attribute is a red flag.

**Fix applied:** the adversary term is rescaled to the norm of the predictor
gradient, so neither side dominates by accident — the role Zhang et al.'s
annealing schedule plays in the original. One setting, no hyperparameter search.
`AdversarialDebiasing(scale_adversary=False)` reproduces the original numbers.

Corrected numbers are in the combined tables above. The method remains weak:
still worse than no mitigation on gender EO and on both region fairness metrics.

### 7.4 Module 5 import hang — environment, workaround in place

`from explain.shap_layer import ...` stalls indefinitely on the development
machine. The module is healthy — a brand-new module imports in 0.00 s and this
module's source `exec`s and runs in 0.48 s — but Python's import machinery blocks
reading the `explain` package, with the sandbox on or off, and from a local
non-synced copy as well as the project path. The cause was not identified; it is
local, not a property of the code.

`run_iteration2.py` and `run_tradeoff.py` load Module 5 by reading and `exec`-ing
its source, which bypasses the stalled path and produces identical objects.
`run_explainability.py` and `run_mitigation_explainability.py` still use the
normal import and will not run on that machine until it is resolved.

The same machine intermittently failed `git status` with `mmap failed:
Operation canceled` during iteration 3, succeeding on retry. No repository file
was found to be evicted or dataless. Both symptoms point to local filesystem
instability rather than to anything in the repository; the artefact was pushed
to `origin` so that a copy exists off this machine.

### 7.5 Dead configuration constants

`config.py` declares `DEFAULT_TOP_K = 5`, `DEFAULT_TAU = 0.55` and
`DEFAULT_MAX_ACCURACY_LOSS = 0.02`. These are **never imported anywhere** and
disagree with the class defaults the experiments actually used (`top_k=15`,
`max_accuracy_loss=0.08`). Iteration 2 adopted `max_accuracy_loss=0.02`
explicitly, matching the declared-but-unused value.

### 7.6 Targeting is inoperative at the default configuration

At `top_k=15, max_iter=10`, cumulative selection exhausts all 116 features before
the leakage target is met, so the `full` and `random` ablation arms treat the
same set and differ only in order. `results/ablation.csv` records
`n_treated = 116.0` for both. The ablation as configured cannot discriminate
targeted from random selection; only `single_pass` (15 features) and `top_k=5`
(50 features) are genuinely targeted, and both underperform random selection.

### 7.7 Reject-option classification — corrected in iteration 3

In the explicit condition the method returned results identical to no
mitigation on every metric, to four decimals, on both attributes — although it
receives the attribute at inference and should change decisions.

**Cause: the band was centred on the wrong threshold.** Kamiran, Karim & Zhang
(2012) define the critical region as a band around the classifier's decision
boundary. The implementation hard-coded that centre at 0.5. On Home Credit,
with a default rate of about 8%, only about 1% of applicants score above 0.5,
while the evaluation decides at the base-rate quantile — about 0.18. The band
([0.45, 0.55] on gender) therefore sat entirely among applicants already
rejected and moved each from one rejected score to another: **zero decisions
changed** on a held-out split, on either attribute. Group advantage was also
read off approval rates at 0.5, where both groups are about 99% approved.

`harness.resolve_threshold`'s own docstring records exactly why 0.5 is
degenerate on this data; the reject-option class had not adopted that rule.

**Fix:** centre the band on the base-rate quantile of the training scores — the
same rule the harness decides with. Band grid, search and objective unchanged.
With it, 38 (gender) and 36 (region) decisions change on the same split.
`RejectOptionClassification(align_threshold=False)` reproduces the originals.

| attribute | version | AUC | DP diff | DI ratio | EO diff |
|---|---|---|---|---|---|
| CODE_GENDER | original | 0.7086 | 0.0307 | 0.967 | 0.0671 |
| CODE_GENDER | corrected | 0.7085 | 0.0334 | 0.9639 | 0.0742 |
| REGION_RATING_CLIENT | original | 0.7086 | 0.0941 | 0.9033 | 0.3074 |
| REGION_RATING_CLIENT | corrected | 0.706 | 0.0465 | 0.9513 | 0.1398 |

The corrected method is **strong on region** (EO and DP both roughly halved)
and **mildly harmful on gender** (DP and EO both worse): the band is chosen to
minimise disparity on training data, and where baseline disparity is small it
can overshoot on test. Its training-only and latent rows are unchanged — without
the attribute at inference it cannot act.

**Not re-run:** the German Credit and Default of Credit Card Clients benchmark
rows for reject-option were produced before this fix and remain the defective
numbers.

---

## 8. Protocol notes

**Result sets use different protocols and their baselines legitimately differ.**

| Artefact | Sample | Split | Quadrant boundary |
|---|---|---|---|
| RQ3 cross-plots (`leakage_vs_shap*.csv`) | 20k | fit and measure on same data | median split |
| Mitigation analysis (`mitigation_*.csv`) | 50k | held-out test | noise-floored |
| Iteration 2 test (`iteration2/test_results_*.csv`) | 50k | held-out test | noise-floored |
| Combined tables, trade-off | 8k | 25 CV folds | n/a |

The gender danger-quadrant count reads **41/116** in `leakage_vs_shap.csv` and
**17/116** in `mitigation_quadrant_summary.csv`. Both are correct for their
protocol. State the protocol change when quoting both.

**Probe AUC has two distinct measurements.** `leakage_table.csv` uses a 5-fold
cross-validated gradient-boosting probe over all classes (gender 0.889, region
0.826, shuffle control 0.501). `probe_auc_after` uses `harness._probe`: a
logistic single-split probe, binarised for region. Their absolute levels are not
comparable — the first is the leakage magnitude to cite, the second is for
relative before/after comparison under a constant instrument.

**A median split inverts before/after comparisons on the leakage axis.** That
distribution is zero-inflated (~35 of 116 features exactly zero, ~35 negative),
so its median sits at zero and mitigation appears to make things worse while
leakage genuinely falls. `explain/shap_layer.py::default_thresholds` floors the
cut at twice the median permutation standard deviation.
