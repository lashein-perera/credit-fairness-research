"""Central configuration: paths, protected attributes, experiment constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

# Protected attributes are held out of training and used for evaluation ONLY.
PROTECTED_ATTRIBUTES = {
    "home_credit": ["CODE_GENDER", "REGION_RATING_CLIENT"],
    "german_credit": ["sex", "age_group"],
    "uci_default": ["SEX", "MARRIAGE"],
}

# Leakage thresholds (probe AUC)
LEAKAGE_NONE = 0.55
LEAKAGE_MEANINGFUL = 0.65
LEAKAGE_SEVERE = 0.80

# Proxy-aware mitigator defaults
DEFAULT_TOP_K = 5
DEFAULT_TAU = 0.55
DEFAULT_MAX_ACCURACY_LOSS = 0.02
