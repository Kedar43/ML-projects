"""Fraud classification models and evaluation utilities."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "isFraud"
FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]


def prepare_model_data(data):
    """Select numeric predictors and the fraud target."""
    features = data[FEATURES].copy()
    target = data[TARGET].astype(int).copy()
    return features, target


def split_data(features, target, test_size=0.2, random_state=42):
    """Create a stratified train/test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_models():
    """Create two baseline classifiers for comparison."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, x_test, y_test):
    """Return classification metrics for a fitted model."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
    }


def compare_models(data):
    """Train both baseline models and compare their fraud-detection metrics."""
    features, target = prepare_model_data(data)
    x_train, x_test, y_train, y_test = split_data(features, target)
    results = {}

    for name, model in build_models().items():
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_test, y_test)

    return results


def run_model_comparison(path):
    """Load a CSV dataset and run the baseline model comparison."""
    data = pd.read_csv(path)
    return compare_models(data)
