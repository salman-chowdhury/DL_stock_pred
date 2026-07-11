from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    lags: int = 20
    train_window: int = 750
    refit_interval: int = 20
    test_year: int = 2025
    tree_seeds: tuple[int, ...] = (17, 42, 91)


def _feature(values: np.ndarray, position: int, lags: int) -> np.ndarray:
    return values[position - lags : position][::-1]


def _training_matrix(
    values: np.ndarray, stop: int, config: BaselineConfig
) -> tuple[np.ndarray, np.ndarray]:
    first = max(config.lags, stop - config.train_window)
    positions = range(first, stop)
    X = np.stack([_feature(values, position, config.lags) for position in positions])
    y = values[first:stop]
    return X, y


def evaluate_walk_forward(frame: pd.DataFrame, config: BaselineConfig) -> pd.DataFrame:
    """Evaluate baselines with every model fit restricted to observations before the target."""
    values = frame["close"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["date"]).reset_index(drop=True)
    test_positions = [
        position
        for position in range(config.lags, len(values))
        if dates.iloc[position].year == config.test_year
    ]
    if not test_positions:
        raise ValueError(f"No observations found for test year {config.test_year}")

    linear: LinearRegression | None = None
    forests: list[RandomForestRegressor] = []
    rows: list[dict[str, object]] = []
    for offset, position in enumerate(test_positions):
        if linear is None or offset % config.refit_interval == 0:
            X_train, y_train = _training_matrix(values, position, config)
            linear = LinearRegression().fit(X_train, y_train)
            forests = [
                RandomForestRegressor(
                    n_estimators=80,
                    max_depth=6,
                    min_samples_leaf=3,
                    random_state=seed,
                    n_jobs=-1,
                ).fit(X_train, y_train)
                for seed in config.tree_seeds
            ]

        feature = _feature(values, position, config.lags).reshape(1, -1)
        tree_predictions = [float(model.predict(feature)[0]) for model in forests]
        rows.append(
            {
                "date": dates.iloc[position],
                "actual": values[position],
                "previous_actual": values[position - 1],
                "naive_persistence": values[position - 1],
                "moving_average_20": float(np.mean(values[position - config.lags : position])),
                "linear_regression": float(linear.predict(feature)[0]),
                "random_forest": float(np.mean(tree_predictions)),
                **{
                    f"random_forest_seed_{seed}": prediction
                    for seed, prediction in zip(config.tree_seeds, tree_predictions, strict=True)
                },
            }
        )
    return pd.DataFrame.from_records(rows)


def forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    previous_actual: np.ndarray,
    mase_scale: float,
) -> dict[str, float]:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mase": float(mae / mase_scale) if mase_scale > 0 else float("nan"),
        "directional_accuracy": float(
            np.mean(np.sign(predicted - previous_actual) == np.sign(actual - previous_actual))
        ),
    }


def summarise_baselines(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    config: BaselineConfig,
) -> pd.DataFrame:
    pretest = frame.loc[pd.to_datetime(frame["date"]).dt.year < config.test_year, "close"]
    mase_scale = float(np.mean(np.abs(np.diff(pretest.to_numpy(dtype=float)))))
    actual = predictions["actual"].to_numpy(dtype=float)
    previous = predictions["previous_actual"].to_numpy(dtype=float)
    models = ["naive_persistence", "moving_average_20", "linear_regression", "random_forest"]
    metrics = {
        model: forecast_metrics(
            actual,
            predictions[model].to_numpy(dtype=float),
            previous,
            mase_scale,
        )
        for model in models
    }
    naive_mae = metrics["naive_persistence"]["mae"]
    seed_maes = [
        forecast_metrics(
            actual,
            predictions[f"random_forest_seed_{seed}"].to_numpy(dtype=float),
            previous,
            mase_scale,
        )["mae"]
        for seed in config.tree_seeds
    ]
    rows = []
    for model in models:
        rows.append(
            {
                "model": model,
                **metrics[model],
                "mae_improvement_vs_naive": float((naive_mae - metrics[model]["mae"]) / naive_mae),
                "mae_std_across_seeds": (
                    float(np.std(seed_maes, ddof=1)) if model == "random_forest" else 0.0
                ),
                "observations": len(predictions),
            }
        )
    return pd.DataFrame.from_records(rows)
