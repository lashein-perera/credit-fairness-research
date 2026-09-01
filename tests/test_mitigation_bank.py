"""Verify each mitigation method reduces disparity on data with planted bias."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'src/python')
from mitigation.bank import BANK, build
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0); N = 4000
a = pd.Series(rng.choice(['F','M'], N, p=[0.66,0.34])); f = (a=='F').values

# features that leak gender AND relate to default
occ    = np.where(rng.random(N) < np.where(f,0.75,0.25), 'School','Construction')
income = np.where(f, 1.4, 1.9)*1e5 + rng.normal(0,4e4,N)
car    = np.where(rng.random(N) < np.where(f,0.25,0.55), 'Y','N')
risk   = rng.normal(0,1,N)

X = pd.DataFrame({'OCC':occ,'INCOME':income,'CAR':car,'RISK':risk,
                  'N1':rng.normal(0,1,N),'N2':rng.normal(0,1,N)})
# default depends on risk + a gender-correlated push (the planted unfairness)
logit = -2.2 + 0.9*risk + 0.7*f
y = pd.Series(rng.binomial(1, 1/(1+np.exp(-logit))))

Xtr,Xte,ytr,yte,atr,ate = train_test_split(X,y,a,test_size=0.3,stratify=y,random_state=1)
Xtr,Xte = Xtr.reset_index(drop=True),Xte.reset_index(drop=True)
ytr,yte = ytr.reset_index(drop=True),yte.reset_index(drop=True)
atr,ate = atr.reset_index(drop=True),ate.reset_index(drop=True)

def dpd(proba, a, thr=0.5):
    pred = (proba>=thr).astype(int)
    ap = pd.Series((pred==0).astype(int)).reset_index(drop=True)
    a = pd.Series(a).reset_index(drop=True)
    r = {g: ap[a==g].mean() for g in a.unique()}
    return max(r.values())-min(r.values())

print(f"{'method':30s} {'cond':8s} {'AUC':>7s} {'DPD':>7s} {'fallback':>9s}")
print("-"*68)
for name in BANK:
    for cond, A in (("explicit", atr), ("latent", None)):
        m = build(name, model_kind="gbm")
        m.fit(Xtr, ytr, A)
        Ate = ate if (cond=="explicit") else None
        p = m.predict_proba(Xte, Ate)
        print(f"{name:30s} {cond:8s} {roc_auc_score(yte,p):7.4f} {dpd(p,ate):7.4f} {str(m.fell_back_):>9s}")
