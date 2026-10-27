from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_utils import FEATURES, TARGET, load_dataset


SEED = 42


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def split_data(frame: pd.DataFrame):
    train_val, test = train_test_split(frame, test_size=0.15, random_state=SEED)
    train, validation = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=SEED
    )
    return train, validation, test


def tune_ridge(x_train, y_train, x_val, y_val):
    candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
    records = []
    for alpha in candidates:
        model = Pipeline(
            [("scale", StandardScaler()), ("regressor", Ridge(alpha=alpha))]
        )
        model.fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val))
        records.append({"alpha": alpha, **score})
    best = min(records, key=lambda row: row["rmse"])
    return best, records


def tune_forest(x_train, y_train, x_val, y_val):
    records = []
    for max_depth, min_leaf, max_features in product(
        [None, 12, 20], [1, 2], [1.0, "sqrt"]
    ):
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            max_features=max_features,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val))
        records.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                "max_features": max_features,
                **score,
            }
        )
    best = min(records, key=lambda row: row["rmse"])
    return best, records


def tune_gradient_boosting(x_train, y_train, x_val, y_val):
    records = []
    for n_estimators, learning_rate, max_depth, min_leaf in product(
        [200, 400], [0.03, 0.05, 0.1], [2, 3], [1, 3]
    ):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            loss="squared_error",
            random_state=SEED,
        )
        model.fit(x_train, y_train)
        score = metrics(y_val, model.predict(x_val))
        records.append(
            {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_leaf,
                **score,
            }
        )
    best = min(records, key=lambda row: row["rmse"])
    return best, records


def make_figures(
    output: Path,
    y_test: pd.Series,
    ridge_pred: np.ndarray,
    forest_pred: np.ndarray,
    boosting_pred: np.ndarray,
    importance: pd.DataFrame,
    selected_name: str,
) -> None:
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.45), dpi=200)

    low = float(
        min(y_test.min(), ridge_pred.min(), forest_pred.min(), boosting_pred.min())
    )
    high = float(
        max(y_test.max(), ridge_pred.max(), forest_pred.max(), boosting_pred.max())
    )
    axes[0].scatter(y_test, ridge_pred, s=15, alpha=0.55, label="Ridge")
    axes[0].scatter(y_test, forest_pred, s=15, alpha=0.65, label="Random forest")
    axes[0].scatter(
        y_test,
        boosting_pred,
        s=15,
        alpha=0.65,
        marker="x",
        linewidths=0.8,
        label="Gradient boosting",
    )
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set_xlabel("Measured strength (MPa)")
    axes[0].set_ylabel("Predicted strength (MPa)")
    axes[0].set_title("Test-set predictions")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    ordered = importance.sort_values("importance_mean")
    axes[1].barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color="#4C78A8",
        alpha=0.9,
    )
    axes[1].set_xlabel("Decrease in test R2 after permutation")
    axes[1].set_title(f"{selected_name} feature importance")
    axes[1].grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output / "model_evaluation.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate concrete-strength models")
    parser.add_argument(
        "--data",
        default="data/processed/concrete_strength.csv",
        help="path to processed CSV",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path
    frame = load_dataset(data_path)
    train, validation, test = split_data(frame)

    x_train, y_train = train[FEATURES], train[TARGET]
    x_val, y_val = validation[FEATURES], validation[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]

    ridge_best, ridge_search = tune_ridge(x_train, y_train, x_val, y_val)
    forest_best, forest_search = tune_forest(x_train, y_train, x_val, y_val)
    boosting_best, boosting_search = tune_gradient_boosting(
        x_train, y_train, x_val, y_val
    )

    train_val = pd.concat([train, validation], axis=0)
    x_train_val, y_train_val = train_val[FEATURES], train_val[TARGET]

    ridge = Pipeline(
        [
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=ridge_best["alpha"])),
        ]
    )
    ridge.fit(x_train_val, y_train_val)
    forest = RandomForestRegressor(
        n_estimators=400,
        max_depth=forest_best["max_depth"],
        min_samples_leaf=forest_best["min_samples_leaf"],
        max_features=forest_best["max_features"],
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(x_train_val, y_train_val)
    boosting = GradientBoostingRegressor(
        n_estimators=int(boosting_best["n_estimators"]),
        learning_rate=boosting_best["learning_rate"],
        max_depth=int(boosting_best["max_depth"]),
        min_samples_leaf=int(boosting_best["min_samples_leaf"]),
        loss="squared_error",
        random_state=SEED,
    )
    boosting.fit(x_train_val, y_train_val)

    ridge_pred = ridge.predict(x_test)
    forest_pred = forest.predict(x_test)
    boosting_pred = boosting.predict(x_test)
    test_metrics = {
        "ridge_regression": metrics(y_test, ridge_pred),
        "random_forest": metrics(y_test, forest_pred),
        "gradient_boosting": metrics(y_test, boosting_pred),
    }

    validation_candidates = {
        "random_forest": forest_best["rmse"],
        "gradient_boosting": boosting_best["rmse"],
    }
    selected_key = min(validation_candidates, key=validation_candidates.get)
    selected_model = forest if selected_key == "random_forest" else boosting
    selected_name = (
        "Random forest" if selected_key == "random_forest" else "Gradient boosting"
    )

    result_dir = root / "results"
    model_dir = root / "models"
    figure_dir = root / "figures"
    for directory in (result_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(ridge, model_dir / "ridge.joblib")
    joblib.dump(forest, model_dir / "random_forest.joblib")
    joblib.dump(boosting, model_dir / "gradient_boosting.joblib")
    joblib.dump(selected_model, model_dir / "best_model.joblib")

    permutation = permutation_importance(
        selected_model,
        x_test,
        y_test,
        scoring="r2",
        n_repeats=30,
        random_state=SEED,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(result_dir / "feature_importance.csv", index=False)

    predictions = pd.DataFrame(
        {
            "row_index": y_test.index,
            "measured_mpa": y_test.to_numpy(),
            "ridge_prediction_mpa": ridge_pred,
            "random_forest_prediction_mpa": forest_pred,
            "gradient_boosting_prediction_mpa": boosting_pred,
        }
    ).sort_values("row_index")
    predictions.to_csv(result_dir / "test_predictions.csv", index=False)

    summary = {
        "random_seed": SEED,
        "n_total": len(frame),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "ridge_best_validation": ridge_best,
        "forest_best_validation": forest_best,
        "boosting_best_validation": boosting_best,
        "selected_model_by_validation": selected_key,
        "test_metrics": test_metrics,
    }
    (result_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"model": "Ridge regression", **test_metrics["ridge_regression"]},
            {"model": "Random forest", **test_metrics["random_forest"]},
            {"model": "Gradient boosting", **test_metrics["gradient_boosting"]},
        ]
    ).to_csv(result_dir / "metrics.csv", index=False)
    pd.DataFrame(ridge_search).to_csv(result_dir / "ridge_validation_search.csv", index=False)
    pd.DataFrame(forest_search).to_csv(result_dir / "forest_validation_search.csv", index=False)
    pd.DataFrame(boosting_search).to_csv(
        result_dir / "boosting_validation_search.csv", index=False
    )
    make_figures(
        figure_dir,
        y_test,
        ridge_pred,
        forest_pred,
        boosting_pred,
        importance,
        selected_name,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

