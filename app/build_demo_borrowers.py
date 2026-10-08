"""Build the stored example applicants for the borrower view in demo mode.

    python app/build_demo_borrowers.py

Uses the held-out split of the iteration-1 analysis (50,000-row Home Credit
sample, test_size=0.3, stratify=y, random_state=42) and the iteration-1
full-mode mitigator. For each protected attribute it keeps four applicants:
declined before and approved after repair, the reverse, and one each whose
decision did not change. Applicants are identified only by their position in
the held-out split.

Written to data/demo/, which is not committed: the examples are rows of the
Home Credit data, and its licence does not allow redistribution. About six
minutes per attribute.
"""
import json
import time
import warnings

import numpy as np
from sklearn.model_selection import train_test_split

import engine as E

LABELS = {("declined", "approved"): "Declined before repair, approved after",
          ("approved", "declined"): "Approved before repair, declined after",
          ("approved", "approved"): "Approved before and after repair",
          ("declined", "declined"): "Declined before and after repair"}


def build(attribute: str) -> list[dict]:
    X, y, A = E.LOADERS["home_credit"](subsample=50_000)
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, A[attribute], test_size=0.3, stratify=y, random_state=E.RANDOM_STATE)
    Xtr, Xte, ytr, yte, atr = (v.reset_index(drop=True) for v in (Xtr, Xte, ytr, yte, atr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = E.build_model(E.MODEL_KIND, Xtr)
        base.fit(Xtr, ytr)
        mit = E.ProxyAwareMitigator(model_kind=E.MODEL_KIND)
        mit.fit(Xtr, ytr, atr)
        Xte_t = mit.transform(Xte)
    bundle = E._bundle(base, mit, Xte, Xte_t, ytr, yte)
    idx = E.borrower_index(bundle)
    change = np.abs(bundle["proba_after"] - bundle["proba_before"])
    idx["change"] = change
    out = []
    for (b, a), label in LABELS.items():
        pool = idx[(idx.before == b) & (idx.after == a)]
        if pool.empty:
            continue
        # changed decisions: the clearest case; unchanged: a typical one, at the
        # median score change, rather than one the repair barely moved
        pool = pool.sort_values("change", ascending=False)
        pick = (pool.iloc[0] if b != a else pool.iloc[len(pool) // 2]).applicant
        out.append({"label": label, **E.borrower(bundle, int(pick))})
    return out


if __name__ == "__main__":
    E.DEMO_BORROWERS.mkdir(parents=True, exist_ok=True)
    for attribute in E.DEMO_ATTRIBUTES:
        t = time.time()
        rows = build(attribute)
        path = E.DEMO_BORROWERS / f"borrowers_{attribute}.json"
        path.write_text(json.dumps(rows, indent=1))
        print(f"{attribute}: {len(rows)} applicants -> {path} ({time.time()-t:.0f}s)", flush=True)
