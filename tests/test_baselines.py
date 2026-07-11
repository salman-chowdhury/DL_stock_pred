import numpy as np
import pandas as pd

from dl_stock_pred.baselines import BaselineConfig, evaluate_walk_forward, summarise_baselines


def synthetic_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=900, freq="D")
    values = 100 + np.arange(900) * 0.2 + np.sin(np.arange(900) / 10)
    return pd.DataFrame({"date": dates, "close": values})


def test_walk_forward_baselines_and_metrics() -> None:
    frame = synthetic_frame()
    config = BaselineConfig(
        lags=5,
        train_window=100,
        refit_interval=30,
        test_year=2025,
        tree_seeds=(1, 2),
    )
    predictions = evaluate_walk_forward(frame, config)
    summary = summarise_baselines(frame, predictions, config)

    assert not predictions.empty
    assert (predictions["naive_persistence"] == predictions["previous_actual"]).all()
    assert set(summary["model"]) == {
        "naive_persistence",
        "moving_average_20",
        "linear_regression",
        "random_forest",
    }
    assert summary["mae"].notna().all()
    assert summary.loc[summary["model"] == "random_forest", "mae_std_across_seeds"].iloc[0] >= 0
