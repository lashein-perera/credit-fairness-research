"""Verify the environment and datasets are correctly set up.

Run from the repository root:
    python verify_setup.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

OK, WARN, FAIL = "  [ok]  ", " [warn] ", " [FAIL] "
issues = []


def check_packages():
    print("\n--- packages ---")
    required = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    optional = [
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("shap", "shap"),
        ("fairlearn", "fairlearn"),
        ("aif360", "aif360"),
    ]
    for mod, name in required:
        try:
            __import__(mod)
            print(OK + name)
        except ImportError:
            print(FAIL + name + "  -> required")
            issues.append(f"install {name}")
    for mod, name in optional:
        try:
            __import__(mod)
            print(OK + name)
        except ImportError:
            print(WARN + name + "  -> needed later, not blocking today")


def check_home_credit():
    print("\n--- Home Credit Default Risk (primary) ---")
    f = DATA / "home_credit" / "application_train.csv"
    if not f.exists():
        print(FAIL + "application_train.csv not found")
        print("        expected at: data/home_credit/application_train.csv")
        issues.append("download Home Credit")
        return
    import pandas as pd
    df = pd.read_csv(f, nrows=5000)
    print(OK + f"loaded, {len(df.columns)} columns")

    expected_rows = 307511
    size_mb = f.stat().st_size / 1e6
    print(OK + f"file size {size_mb:.0f} MB (expect ~166 MB)")

    for col in ["TARGET", "CODE_GENDER", "REGION_RATING_CLIENT",
                "REGION_RATING_CLIENT_W_CITY"]:
        if col in df.columns:
            print(OK + f"column {col}")
        else:
            print(FAIL + f"column {col} missing")
            issues.append(f"Home Credit missing {col}")

    print("\n        protected attribute distribution (first 5000 rows):")
    print("        CODE_GENDER:", dict(df["CODE_GENDER"].value_counts()))
    print("        REGION_RATING_CLIENT:", dict(df["REGION_RATING_CLIENT"].value_counts().sort_index()))
    rate = df["TARGET"].mean()
    print(f"        default rate: {rate:.1%} (expect ~8%)")


def check_german():
    print("\n--- German Credit (benchmark) ---")
    f = DATA / "german_credit" / "german.data"
    if not f.exists():
        print(FAIL + "german.data not found")
        print("        expected at: data/german_credit/german.data")
        issues.append("download German Credit")
        return
    import pandas as pd
    df = pd.read_csv(f, sep=r"\s+", header=None)
    print(OK + f"loaded, {df.shape[0]} rows x {df.shape[1]} columns")
    if df.shape == (1000, 21):
        print(OK + "shape correct (1000 x 21)")
    else:
        print(WARN + f"expected 1000 x 21, got {df.shape}")
    # Attribute 9 (index 8) is personal status and sex; attribute 13 (index 12) is age
    print("        col 8 (personal status/sex):", dict(df[8].value_counts()))
    print(f"        col 12 (age): min {df[12].min()}, max {df[12].max()}")


def check_uci_default():
    print("\n--- UCI Default of Credit Card Clients (secondary) ---")
    candidates = list((DATA / "uci_default").glob("*.xls*")) if (DATA / "uci_default").exists() else []
    if not candidates:
        print(FAIL + "no .xls/.xlsx found")
        print("        expected at: data/uci_default/default of credit card clients.xls")
        issues.append("download UCI Default")
        return
    import pandas as pd
    f = candidates[0]
    try:
        df = pd.read_excel(f, header=1)
    except ImportError:
        print(FAIL + "reading .xls needs xlrd -> pip install xlrd openpyxl")
        issues.append("pip install xlrd openpyxl")
        return
    print(OK + f"loaded {f.name}, {df.shape[0]} rows x {df.shape[1]} columns")
    for col in ["SEX", "EDUCATION", "MARRIAGE"]:
        print((OK if col in df.columns else FAIL) + f"column {col}")


def check_gitignore():
    print("\n--- git hygiene ---")
    gi = ROOT / ".gitignore"
    if gi.exists() and "data/*" in gi.read_text():
        print(OK + "data/ is gitignored")
    else:
        print(FAIL + "data/ NOT ignored — do not commit until fixed")
        issues.append("fix .gitignore")


if __name__ == "__main__":
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    print(f"python {sys.version.split()[0]}")
    print(f"repo   {ROOT}")

    check_packages()
    check_home_credit()
    check_german()
    check_uci_default()
    check_gitignore()

    print("\n" + "=" * 60)
    if issues:
        print(f"{len(issues)} item(s) outstanding:")
        for i in issues:
            print("  - " + i)
    else:
        print("All checks passed. Environment and data ready.")
    print("=" * 60)
