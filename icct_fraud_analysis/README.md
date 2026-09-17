# Fraud Detection Analysis

A Python project for exploring transaction-level fraud data, preparing features, comparing baseline classification models, and evaluating fraud-detection performance.

## Scope

The project covers data loading and quality checks, fraud-rate analysis by transaction type, numerical summaries, IQR-based outlier handling, model preparation, stratified train/test splitting, and comparison of Logistic Regression and Random Forest classifiers using fraud-detection metrics.

## Structure

- `data_processing.py` contains data loading, quality checks, fraud filtering, and outlier-processing functions.
- `analysis.py` contains exploratory analysis, fraud summaries, numerical summaries, and visualizations.
- `modeling.py` contains feature preparation, train/test splitting, baseline model training, and model evaluation.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## Data

The dataset is not included in this repository. Place the CSV file locally and pass its path to the analysis and modeling workflows.
