# Fraud Detection Analysis

A refactored version of the original fraud-detection notebook, organized into reusable Python modules for analysis and experimentation.

## Scope

The analysis works with transaction-level data containing transaction type, amount, account balances, and fraud labels. The original notebook explored the dataset, inspected missing values and class distribution, analyzed numerical features, and applied IQR-based outlier handling.

## Structure

- `data_processing.py` contains reusable data-loading, validation, fraud-filtering, and outlier-processing functions.
- `analysis.py` will contain exploratory analysis and visualizations.
- `modeling.py` will contain model preparation and evaluation.
- `main.py` will provide the runnable workflow.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## Data

The dataset is not included in this repository. Place the CSV file locally and pass its path to the analysis workflow.
