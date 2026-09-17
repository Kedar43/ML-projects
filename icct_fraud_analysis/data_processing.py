"""Data loading and preprocessing helpers for the fraud analysis project."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the fraud transaction dataset from a CSV file."""
    return pd.read_csv(path)


def missing_values(data: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return data.isna().sum()


def fraud_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Return rows labelled as fraudulent."""
    return data.loc[data["isFraud"] == 1].copy()


def outlier_bounds(values: pd.Series) -> tuple[float, float]:
    """Calculate IQR-based lower and upper bounds for a numeric series."""
    q1, q3 = np.percentile(values.dropna(), [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def cap_outliers(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cap extreme values in selected columns using IQR bounds.

    The input dataframe is copied so the original data remains unchanged.
    """
    result = data.copy()
    for column in columns:
        lower, upper = outlier_bounds(result[column])
        result[column] = np.clip(result[column], lower, upper)
    return result
