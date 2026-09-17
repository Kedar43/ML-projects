"""Exploratory analysis helpers for the fraud detection sample."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data_processing import NUMERIC_FEATURES, load_data, missing_values, fraud_transactions


def fraud_summary(data):
    """Summarize transaction count and fraud rate by transaction type."""
    summary = (
        data.groupby("type")
        .agg(
            transactions=("isFraud", "size"),
            fraudulent=("isFraud", "sum"),
        )
        .reset_index()
    )
    summary["fraud_rate"] = summary["fraudulent"] / summary["transactions"]
    return summary.sort_values("fraud_rate", ascending=False)


def numeric_summary(data):
    """Return descriptive statistics for the numeric transaction fields."""
    return data[NUMERIC_FEATURES].describe().T


def plot_transaction_amounts(data, output_path):
    """Save a histogram of transaction amounts to a PNG file."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data["amount"], bins=50)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_analysis(data_path):
    """Run core quality checks and exploratory summaries."""
    data = load_data(data_path)
    return {
        "shape": data.shape,
        "missing_values": missing_values(data),
        "fraud_count": len(fraud_transactions(data)),
        "fraud_summary": fraud_summary(data),
        "numeric_summary": numeric_summary(data),
    }
