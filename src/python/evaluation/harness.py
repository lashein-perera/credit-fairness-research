"""Module 6 — Evaluation Harness.

Takes one configuration, returns one comparable results row:

    dataset | method | condition | AUC | KS | Brier |
    DP_diff | EO_diff | DI_ratio | probe_AUC_after | runtime

`probe_AUC_after` carries the central argument: it measures whether leakage
survived the mitigation.

Conditions
----------
    'explicit' : protected attribute available during training/mitigation
    'latent'   : protected attribute withheld — the realistic setting
"""
import pandas as pd


def run_configuration(dataset: str, method: str, condition: str,
                      n_splits: int = 5, n_repeats: int = 10) -> dict:
    """Run one cell of the experiment grid with repeated stratified CV."""
    raise NotImplementedError("Week 4")


def run_grid(datasets: list[str], methods: list[str],
             conditions: list[str]) -> pd.DataFrame:
    """Run the full experiment grid and return all rows."""
    raise NotImplementedError("Week 4")


def wilcoxon_compare(results: pd.DataFrame, method_a: str, method_b: str):
    """Paired Wilcoxon signed-rank test. Report effect size alongside the p-value."""
    raise NotImplementedError("Week 7")
