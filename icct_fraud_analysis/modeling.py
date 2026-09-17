"""Model training and evaluation for the fraud analysis sample."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "isFraud"
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select numeric predictors and the fraud target."""
    features = data[NUMERIC_FEATURES].copy()
    target = data[TARGET].copy()
    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the dataset into stratified training and test sets."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[StandardScaler, LogisticRegression]:
    """Scale numeric features and train a logistic regression classifier."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_scaled, y_train)
    return scaler, model


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate the classifier and return standard classification metrics."""
    predictions = model.predict(scaler.transform(x_test))
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }


def run_model(path: str | Path) -> dict[str, object]:
    """Load data, train the model, and return evaluation results."""
    data = pd.read_csv(path)
    features, target = prepare_features(data)
    x_train, x_test, y_train, y_test = split_data(features, target)
    scaler, model = train_logistic_regression(x_train, y_train)
    return evaluate_model(model, scaler, x_test, y_test)
