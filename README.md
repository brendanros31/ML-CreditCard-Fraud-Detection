# ML-CreditCard-Fraud-Detection

This project implements a complete machine‑learning pipeline for binary classification on transaction data, including data loading, preprocessing, model training, and evaluation. It compares three models: Decision Tree, Random Forest, and XGBoost.

---

## Features

- YAML‑based configuration for dataset paths.
- Data loading and cleaning utilities.
- Feature scaling for selected numeric columns.
- Train–test splitting with a reusable helper.
- Multiple model backends: Decision Tree, Random Forest, XGBoost.
- Evaluation utilities with optional plots (confusion matrix, ROC AUC).
- Modular code structure under `src/` for easy extension.

---

## Project structure
.
├── config/
│   └── config.yaml
│
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── evaluate.py
│   └── utils.py
│
├── EDA.ipynb
└── main.py


- `config/config.yaml`  
  Contains configuration options such as data paths, model types, and model paramaters.

- `src/data_loader.py`  
  Functions to load the dataset and perform basic cleaning.

- `src/features.py`  
  Functions for feature engineering, and train–test splitting.

- `src/models.py`  
  Methods to build models by name and train them on provided data.

- `src/evaluate.py`  
  Functions to evaluate models on a test set and generate metrics and plots.

- `src/utils.py`  
  Helper utilities, including a heatmap plotting function.

---

## Installation

1. Clone the repository:
    git clone .git
    cd 

2. (Recommended) Create and activate a virtual environment:
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate


3. Install dependencies (adjust to your environment):
    pip install -r requirements.txt

---

## Usage

Run the main pipeline script from the project root:
    python main.py


The script will:

1. Load the YAML config from `config/config.yaml`.
2. Load the dataset from `config['data']['raw_path']`.
3. Clean the data (e.g. remove null values).
4. Plot a correlation heatmap (not shown interactively if `show=False`).
5. Scale the `Time` and `Amount` features.
6. Split the data into train and test sets.
7. Build and train three models:
   - Decision Tree
   - Random Forest
   - XGBoost
8. Evaluate each model on the test set and print metrics to the console.

---

## Plots

Inside the `__main__()` function there is a flag:
    plots = False

Set this to `True` to enable evaluation plots:

- Confusion matrix
- ROC AUC curve