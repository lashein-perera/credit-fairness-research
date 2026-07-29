"""Guards the central design rule of this research.

The protected attribute must NEVER appear in the training feature matrix. If it
leaks in, every fairness and leakage result in the dissertation is invalid.

These tests are expected to FAIL until the loaders are implemented in Week 2.
"""
import pytest

from src.python import config


@pytest.mark.parametrize("dataset", list(config.PROTECTED_ATTRIBUTES))
def test_protected_columns_absent_from_features(dataset):
    """No protected attribute column may survive into X."""
    from src.python.data.loaders import LOADERS

    X, y, A = LOADERS[dataset]()
    for col in config.PROTECTED_ATTRIBUTES[dataset]:
        assert col not in X.columns, (
            f"Protected attribute '{col}' leaked into the feature matrix "
            f"for dataset '{dataset}'. All downstream results are invalid."
        )


@pytest.mark.parametrize("dataset", list(config.PROTECTED_ATTRIBUTES))
def test_protected_frame_is_populated(dataset):
    """A must actually contain the protected attributes, for evaluation use."""
    from src.python.data.loaders import LOADERS

    X, y, A = LOADERS[dataset]()
    assert not A.empty, f"No protected attributes returned for '{dataset}'."


@pytest.mark.parametrize("dataset", list(config.PROTECTED_ATTRIBUTES))
def test_row_alignment(dataset):
    """X, y and A must stay row-aligned or evaluation silently mismatches."""
    from src.python.data.loaders import LOADERS

    X, y, A = LOADERS[dataset]()
    assert len(X) == len(y) == len(A)
