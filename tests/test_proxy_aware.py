"""Test the proxy-aware mitigator against the bank, under both conditions."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'src/python')
from mitigation.bank import BANK, build
from mitigation.proxy_aware import ProxyAwareMitigator
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0); N = 4000
a = pd.Series(rng.choice(['F','M'], N, p=[0.66,0.34])); f = (a=='F').values
occ    = np.where(rng.random(N) < np.where(f,0.75,0.25), 'School','Construction')
income = np.where(f, 1.4, 1.9)*1e5 + rng.normal(0,4e4,N)
carage = np.where(f, 3.0, 8.0) + rng.normal(0,2,N)
risk   = rng.normal(0,1,N)
X = pd.DataFrame({'OCC':occ,'INCOME':income,'CARAGE':carage,'RISK':risk,
                  'N1':rng.normal(0,1,N),'N2':rng.normal(0,1,N)})
logit = -2.2 + 0.9*risk + 0.7*f
y = pd.Series(rng.binomial(1, 1/(1+np.exp(-logit))))

Xtr,Xte,ytr,yte,atr,ate = train_test_split(X,y,a,test_size=0.3,stratify=y,random_state=1)
for d in (Xtr,Xte): d.reset_index(drop=True,inplace=True)
ytr,yte,atr,ate = [s.reset_index(drop=True) for s in (ytr,yte,atr,ate)]

def dpd(p,a,thr=0.5):
    ap = pd.Series(((p>=thr).astype(int)==0).astype(int)).reset_index(drop=True)
    a = pd.Series(a).reset_index(drop=True)
    r=[ap[(a==g).values].mean() for g in a.unique()]; return max(r)-min(r)

def probe(Xd, ad):
    from models.baselines import build_preprocessor
    from sklearn.linear_model import LogisticRegression
    ab=(pd.Series(ad).astype(str)=='F').astype(int).values
    Z=build_preprocessor(Xd,dense=True).fit_transform(Xd)
    Ztr,Zte,atr_,ate_=train_test_split(Z,ab,test_size=.3,stratify=ab,random_state=42)
    c=LogisticRegression(max_iter=1000).fit(Ztr,atr_)
    return roc_auc_score(ate_,c.predict_proba(Zte)[:,1])

print(f"{'method':26s} {'cond':8s} {'AUC':>7s} {'DPD':>7s} {'leak_after':>11s}")
print("-"*64)
for name in BANK:
    for cond,A in (("explicit",atr),("latent",None)):
        m=build(name,model_kind="gbm"); m.fit(Xtr,ytr,A)
        p=m.predict_proba(Xte, ate if cond=="explicit" else None)
        print(f"{name:26s} {cond:8s} {roc_auc_score(yte,p):7.4f} {dpd(p,ate):7.4f} {probe(Xte,ate):11.4f}")

print("-"*64)
for cond,A in (("explicit",atr),("latent",atr)):   # ours uses A at FIT only
    m=ProxyAwareMitigator(model_kind="gbm",top_k=2,tau=0.55)
    m.fit(Xtr,ytr,A)
    p=m.predict_proba(Xte)                          # NO attribute at inference
    lk=probe(m.transform(Xte),ate)
    print(f"{'proxy_aware (ours)':26s} {cond:8s} {roc_auc_score(yte,p):7.4f} {dpd(p,ate):7.4f} {lk:11.4f}")
    print(f"   treated: {m.treated_}   leakage {m.history_[0]['leakage']:.4f} -> {m.final_leakage_:.4f}")
    break

print("\n=== ABLATION: targeted vs random feature selection ===")
for rf in (False, True):
    m=ProxyAwareMitigator(model_kind="gbm",top_k=2,tau=0.55,random_features=rf)
    m.fit(Xtr,ytr,atr); p=m.predict_proba(Xte)
    lab = "random" if rf else "targeted"
    print(f"  {lab:9s} AUC {roc_auc_score(yte,p):.4f}  DPD {dpd(p,ate):.4f}  "
          f"leak {m.final_leakage_:.4f}  treated {m.treated_}")
