# Predicting Concrete Compressive Strength

This repository contains a reproducible machine-learning mini-project for predicting concrete compressive strength from mixture proportions and curing age. It compares a standardized Ridge-regression baseline with tuned random-forest and gradient-boosting regressors.

## Problem, inputs, and output

Estimating compressive strength before destructive testing can support preliminary mixture screening and quality-control planning. Each observation has eight numerical inputs:

- cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate, and fine aggregate, all in kg/m3;
- curing age in days.

The output is concrete compressive strength in MPa. The model is intended as a data-driven screening tool, not a substitute for laboratory testing or design-code verification.

## Dataset

The project uses the **UCI Concrete Compressive Strength** dataset created by I-Cheng Yeh:

- 1,030 observations;
- eight numerical input features and one continuous target;
- no missing values;
- DOI: <https://doi.org/10.24432/C5PK67>;
- UCI page: <https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength>;
- license: Creative Commons Attribution 4.0.

The original archive and readme are retained under `data/raw/`. The normalized CSV used by the scripts is under `data/processed/`. The download script can recreate these files from the official UCI URL.

## Repository structure

```text
concrete-strength-ml/
├── data/raw/                 # UCI archive, Excel data, and dataset readme
├── data/processed/           # normalized CSV used for training
├── examples/                 # example JSON input
├── figures/                  # report-ready evaluation figure
├── models/                   # locally generated model files (not required in Git)
├── report/                   # two-page LaTeX report and compiled PDF
├── results/                  # metrics, predictions, and validation searches
├── src/                      # download, train, and inference code
├── tests/                    # lightweight reproducibility tests
└── requirements.txt
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate
```

Install the pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproduce the project

From the repository root, download and normalize the data:

```bash
python -m src.download_data
```

Train both models, tune them on the validation set, evaluate once on the held-out test set, and regenerate all result files:

```bash
python -m src.train --data data/processed/concrete_strength.csv
```

Run the tests:

```bash
python -m pytest -q
```

Run a prediction using the supplied example:

```bash
python -m src.predict --input examples/example_input.json
```

The command prints a JSON object containing `predicted_compressive_strength_mpa`. To use a new mixture, copy the example JSON and replace all eight values while keeping the same keys and units.

## Experimental design

- Random seed: `42`.
- Split: 720 training, 155 validation, and 155 test observations (approximately 70/15/15).
- Baseline: standardized Ridge regression; `alpha` selected from `[0.01, 0.1, 1, 10, 100]` using validation RMSE.
- Tree model 1: random forest with 400 trees; maximum depth, minimum leaf size, and feature subsampling selected using validation RMSE.
- Tree model 2: gradient boosting; number of estimators, learning rate, tree depth, and minimum leaf size selected using validation RMSE.
- Gradient boosting had the lowest validation RMSE and was therefore saved as `models/best_model.joblib` for inference.
- After model selection, each model is refitted on the combined training and validation data and evaluated once on the untouched test set.
- Metrics: mean absolute error (MAE), root mean squared error (RMSE), and coefficient of determination (`R2`).

## Main results

| Model | Test MAE (MPa) | Test RMSE (MPa) | Test R2 |
|---|---:|---:|---:|
| Ridge regression | 7.825 | 9.698 | 0.641 |
| Random forest | 3.488 | 4.937 | 0.907 |
| Gradient boosting | **2.778** | **4.161** | **0.934** |

Both tree ensembles improve on the linear baseline. Gradient boosting also outperforms the random forest on the held-out test data and was selected before test evaluation because it had the lowest validation RMSE. Permutation importance identifies age and cement content as the two most influential variables for the selected model. Full numerical outputs are available in `results/metrics.json`, `results/test_predictions.csv`, and `results/feature_importance.csv`.

## Limitations

The dataset is relatively small and does not include several practical factors, such as curing temperature, specimen geometry, testing standard, or material-source variability. Random train/test splitting also measures interpolation within the historical data distribution rather than performance on a genuinely new laboratory or concrete family. Predictions should therefore be validated experimentally before practical use.

## Reproducibility notes

The reference run used Python 3.11.7, NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.2.2, Matplotlib 3.8.0, joblib 1.2.0, and xlrd 2.0.1 on Windows. The software versions required for reproduction are pinned in `requirements.txt`.

## External resources

- Dataset: Yeh, I.-C. (1998), *Concrete Compressive Strength*, UCI Machine Learning Repository, DOI 10.24432/C5PK67.
- Machine-learning implementation uses pandas, NumPy, Matplotlib, joblib, and scikit-learn.

