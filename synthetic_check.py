"""Validate the probe on synthetic data with KNOWN leakage.

Three scenarios:
  1. Features genuinely encode gender  -> probe AUC must be high
  2. Features are pure noise           -> probe AUC must be ~0.5
  3. Shuffle control on scenario 1     -> must collapse to ~0.5
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'src/python')
from leakage.probe import probe_auc, shuffle_test, rank_leaky_features, classify_leakage

rng = np.random.default_rng(0)
N = 4000

# 66/34 split, mirroring Home Credit
gender = pd.Series(rng.choice(['F','M'], size=N, p=[0.66,0.34]))
is_f = (gender == 'F').values

def build(leaky: bool):
    d = {}
    if leaky:
        # ORGANIZATION_TYPE: occupational segregation
        d['ORGANIZATION_TYPE'] = np.where(
            rng.random(N) < np.where(is_f, 0.75, 0.25),
            'School', 'Construction')
        # income: correlated with gender
        d['AMT_INCOME_TOTAL'] = np.where(is_f, 1.4, 1.9)*1e5 + rng.normal(0,4e4,N)
        # car ownership
        d['FLAG_OWN_CAR'] = np.where(
            rng.random(N) < np.where(is_f, 0.25, 0.55), 'Y','N')
    else:
        d['ORGANIZATION_TYPE'] = rng.choice(['School','Construction'], N)
        d['AMT_INCOME_TOTAL'] = rng.normal(1.6e5, 5e4, N)
        d['FLAG_OWN_CAR'] = rng.choice(['Y','N'], N)
    # genuine noise columns present in both
    for i in range(6):
        d[f'NOISE_{i}'] = rng.normal(0,1,N)
    d['REGION_POPULATION'] = rng.random(N)
    df = pd.DataFrame(d)
    df.loc[rng.random(N) < 0.05, 'AMT_INCOME_TOTAL'] = np.nan   # realistic NaN
    return df

print("="*62)
print("SCENARIO 1 — features DO encode gender (expect high AUC)")
print("="*62)
X = build(leaky=True)
for k in ['logistic','gbm']:
    r = probe_auc(X, gender, kind=k)
    print(f"  {k:9s} AUC {r['auc_mean']:.4f} ±{r['auc_std']:.4f}   "
          f"bal-acc {r['balanced_acc_mean']:.4f}   -> {classify_leakage(r['auc_mean'])}")

print("\n" + "="*62)
print("SCENARIO 2 — features are noise (expect AUC ~0.50)")
print("="*62)
Xn = build(leaky=False)
r = probe_auc(Xn, gender, kind='gbm')
print(f"  gbm       AUC {r['auc_mean']:.4f} ±{r['auc_std']:.4f}   "
      f"-> {classify_leakage(r['auc_mean'])}")

print("\n" + "="*62)
print("SCENARIO 3 — SHUFFLE CONTROL on scenario 1 (expect ~0.50)")
print("="*62)
r = shuffle_test(X, gender, kind='gbm')
print(f"  gbm       AUC {r['auc_mean']:.4f} ±{r['auc_std']:.4f}   "
      f"-> {classify_leakage(r['auc_mean'])}")

print("\n" + "="*62)
print("FEATURE RANKING on scenario 1 (planted features should top the list)")
print("="*62)
print(rank_leaky_features(X, gender, top_k=6).to_string(index=False))
