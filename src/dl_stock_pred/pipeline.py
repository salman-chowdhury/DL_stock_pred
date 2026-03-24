from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, validate_experiment_config
from .data import load_index_dataframe, prepare_supervised_data
from .models import RecurrentRegressor
from .train import (
    EvaluationOutput,
    Metrics,
    compute_metrics,
    evaluate_model,
    make_loader,
    resolve_device,
    set_seed,
    train_with_early_stopping,
)


def _serialize_config(config: ExperimentConfig) -> dict[str, Any]:
    raw = asdict(config)
    data_files = {
        key: str(path)
        for key, path in config.data_files.items()
    }
    raw["data_files"] = data_files
    raw["output_dir"] = str(config.output_dir)
    return raw


def _iter_hyperparams(config: ExperimentConfig) -> list[dict[str, Any]]:
    combos = [
        {
            "hidden_size": hidden,
            "num_layers": layers,
            "learning_rate": lr,
            "dropout": dropout,
        }
        for hidden, layers, lr, dropout in product(
            config.search.hidden_sizes,
            config.search.num_layers,
            config.search.learning_rates,
            config.search.dropout,
        )
    ]

    if config.max_trials_per_model is not None:
        return combos[: config.max_trials_per_model]
    return combos


def _save_predictions(
    output_path: Path,
    dates: pd.Series,
    previous_close: np.ndarray,
    eval_output: EvaluationOutput,
) -> None:
    actual_change = eval_output.y_true - previous_close
    predicted_change = eval_output.y_pred - previous_close
    df = pd.DataFrame(
        {
            "date": dates.astype("datetime64[ns]"),
            "previous_close": previous_close,
            "actual": eval_output.y_true,
            "predicted": eval_output.y_pred,
            "actual_change": actual_change,
            "predicted_change": predicted_change,
            "actual_direction": np.sign(actual_change).astype(int),
            "predicted_direction": np.sign(predicted_change).astype(int),
            "direction_hit": (np.sign(actual_change) == np.sign(predicted_change)),
            "residual": eval_output.y_pred - eval_output.y_true,
        }
    )
    df.to_csv(output_path, index=False)


def _save_plot(
    output_path: Path,
    title: str,
    dates: pd.Series,
    eval_output: EvaluationOutput,
) -> None:
    cache_dir = Path(".cache/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(dates.values, eval_output.y_true, label="actual", linewidth=2)
    plt.plot(dates.values, eval_output.y_pred, label="predicted", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _runtime_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
            "torch": _package_version("torch"),
            "dl-stock-pred": _package_version("dl-stock-pred"),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_file_metadata(data_files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        symbol: {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for symbol, path in data_files.items()
    }


def _last_close_unscaled(
    X: np.ndarray,
    target_scaler,
    close_feature_index: int,
) -> np.ndarray:
    close_scaled = X[:, -1, close_feature_index]
    return target_scaler.inverse_transform(close_scaled.reshape(-1, 1)).reshape(-1)


def _baseline_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    target_scaler,
    close_feature_index: int,
) -> tuple[np.ndarray, Metrics]:
    previous_close = _last_close_unscaled(
        X=X,
        target_scaler=target_scaler,
        close_feature_index=close_feature_index,
    )
    y_true = target_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
    metrics = compute_metrics(y_true=y_true, y_pred=previous_close)
    return previous_close, metrics


def _directional_accuracy(
    previous_close: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    true_direction = np.sign(y_true - previous_close)
    pred_direction = np.sign(y_pred - previous_close)
    return float(np.mean(true_direction == pred_direction) * 100.0)


def _dataset_profile_row(
    symbol: str,
    df: pd.DataFrame,
    prepared,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "rows": len(df),
        "start_date": df["date"].min().date().isoformat(),
        "end_date": df["date"].max().date().isoformat(),
        "feature_columns": ",".join(prepared.feature_columns),
        "has_volume": "volume" in df.columns and df["volume"].notna().any(),
        "has_change_pct": "change_pct" in df.columns and df["change_pct"].notna().any(),
        "n_train_sequences": int(prepared.train_X.shape[0]),
        "n_val_sequences": int(prepared.val_X.shape[0]),
        "n_test_sequences": int(prepared.test_X.shape[0]),
        "train_target_start": prepared.train_dates.min().date().isoformat(),
        "train_target_end": prepared.train_dates.max().date().isoformat(),
        "val_target_start": prepared.val_dates.min().date().isoformat(),
        "val_target_end": prepared.val_dates.max().date().isoformat(),
        "test_target_start": prepared.test_dates.min().date().isoformat(),
        "test_target_end": prepared.test_dates.max().date().isoformat(),
    }


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    validate_experiment_config(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "runtime": _runtime_info(),
        "data_files": _data_file_metadata(config.data_files),
        "config": _serialize_config(config),
    }
    (output_root / "run_config.json").write_text(json.dumps(metadata, indent=2))

    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    champion_rows: list[dict[str, Any]] = []

    for symbol, csv_path in config.data_files.items():
        index_dir = output_root / symbol
        index_dir.mkdir(parents=True, exist_ok=True)

        df = load_index_dataframe(Path(csv_path), symbol=symbol)
        prepared = prepare_supervised_data(
            df=df,
            feature_candidates=list(config.feature_candidates),
            target_column=config.target_column,
            split_cfg=config.split,
            train_cfg=config.train,
        )
        dataset_rows.append(_dataset_profile_row(symbol=symbol, df=df, prepared=prepared))

        train_loader = make_loader(
            prepared.train_X,
            prepared.train_y,
            batch_size=config.train.batch_size,
            shuffle=True,
        )
        val_loader = make_loader(
            prepared.val_X,
            prepared.val_y,
            batch_size=config.train.batch_size,
            shuffle=False,
        )
        test_loader = make_loader(
            prepared.test_X,
            prepared.test_y,
            batch_size=config.train.batch_size,
            shuffle=False,
        )

        hyperparams = _iter_hyperparams(config)
        close_feature_index = prepared.feature_columns.index("close")
        val_previous_close, val_baseline_metrics = _baseline_evaluation(
            X=prepared.val_X,
            y=prepared.val_y,
            target_scaler=prepared.target_scaler,
            close_feature_index=close_feature_index,
        )
        test_previous_close, test_baseline_metrics = _baseline_evaluation(
            X=prepared.test_X,
            y=prepared.test_y,
            target_scaler=prepared.target_scaler,
            close_feature_index=close_feature_index,
        )

        per_model_best: list[dict[str, Any]] = []

        for model_type in config.model_types:
            best_trial: dict[str, Any] | None = None

            for trial_id, hp in enumerate(hyperparams, start=1):
                model = RecurrentRegressor(
                    model_type=model_type,
                    input_size=len(prepared.feature_columns),
                    hidden_size=int(hp["hidden_size"]),
                    num_layers=int(hp["num_layers"]),
                    dropout=float(hp["dropout"]),
                )

                train_out = train_with_early_stopping(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                    learning_rate=float(hp["learning_rate"]),
                    max_epochs=config.train.max_epochs,
                    patience=config.train.patience,
                    weight_decay=config.train.weight_decay,
                )

                val_eval = evaluate_model(
                    model=model,
                    loader=val_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                )
                test_eval = evaluate_model(
                    model=model,
                    loader=test_loader,
                    device=device,
                    target_scaler=prepared.target_scaler,
                )

                candidate = {
                    "symbol": symbol,
                    "model_type": model_type,
                    "trial_id": trial_id,
                    "params": hp,
                    "history": train_out.history,
                    "val_eval": val_eval,
                    "test_eval": test_eval,
                    "val_rmse": val_eval.metrics.rmse,
                    "test_rmse": test_eval.metrics.rmse,
                    "test_mae": test_eval.metrics.mae,
                    "test_mape": test_eval.metrics.mape,
                    "epochs_trained": int(train_out.history["epoch"].max()),
                }

                if best_trial is None or candidate["val_rmse"] < best_trial["val_rmse"]:
                    best_trial = candidate

            assert best_trial is not None

            model_tag = f"{symbol}_{model_type}"
            best_trial["history"].to_csv(index_dir / f"history_{model_tag}.csv", index=False)
            _save_predictions(
                output_path=index_dir / f"predictions_{model_tag}.csv",
                dates=prepared.test_dates,
                previous_close=test_previous_close,
                eval_output=best_trial["test_eval"],
            )

            if config.save_plots:
                _save_plot(
                    output_path=index_dir / f"plot_{model_tag}.png",
                    title=f"{symbol.upper()} - {model_type.upper()} (test year {config.split.test_year})",
                    dates=prepared.test_dates,
                    eval_output=best_trial["test_eval"],
                )

            record = {
                "symbol": symbol,
                "model_type": model_type,
                "window_size": config.train.window_size,
                "batch_size": config.train.batch_size,
                "hidden_size": int(best_trial["params"]["hidden_size"]),
                "num_layers": int(best_trial["params"]["num_layers"]),
                "learning_rate": float(best_trial["params"]["learning_rate"]),
                "dropout": float(best_trial["params"]["dropout"]),
                "epochs_trained": best_trial["epochs_trained"],
                "val_rmse": best_trial["val_rmse"],
                "test_rmse": best_trial["test_rmse"],
                "test_mae": best_trial["test_mae"],
                "test_mape_pct": best_trial["test_mape"],
                "val_naive_baseline_rmse": val_baseline_metrics.rmse,
                "test_naive_baseline_rmse": test_baseline_metrics.rmse,
                "val_rmse_minus_naive": best_trial["val_rmse"] - val_baseline_metrics.rmse,
                "test_rmse_minus_naive": best_trial["test_rmse"] - test_baseline_metrics.rmse,
                "val_rmse_vs_naive_pct": (
                    (best_trial["val_rmse"] / val_baseline_metrics.rmse) - 1.0
                ) * 100.0,
                "test_rmse_vs_naive_pct": (
                    (best_trial["test_rmse"] / test_baseline_metrics.rmse) - 1.0
                ) * 100.0,
                "val_directional_accuracy_pct": _directional_accuracy(
                    previous_close=val_previous_close,
                    y_true=best_trial["val_eval"].y_true,
                    y_pred=best_trial["val_eval"].y_pred,
                ),
                "test_directional_accuracy_pct": _directional_accuracy(
                    previous_close=test_previous_close,
                    y_true=best_trial["test_eval"].y_true,
                    y_pred=best_trial["test_eval"].y_pred,
                ),
                "beats_val_naive_baseline": bool(best_trial["val_rmse"] < val_baseline_metrics.rmse),
                "beats_test_naive_baseline": bool(best_trial["test_rmse"] < test_baseline_metrics.rmse),
                "feature_columns": ",".join(prepared.feature_columns),
                "n_train": prepared.train_X.shape[0],
                "n_val": prepared.val_X.shape[0],
                "n_test": prepared.test_X.shape[0],
            }

            summary_rows.append(record)
            per_model_best.append(record)

        champion = min(per_model_best, key=lambda x: x["val_rmse"])
        champion_rows.append(champion)
        (index_dir / "champion.json").write_text(json.dumps(champion, indent=2))

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["symbol", "val_rmse", "test_rmse"]
    )
    summary_df.to_csv(output_root / "summary.csv", index=False)
    pd.DataFrame(dataset_rows).sort_values("symbol").to_csv(output_root / "dataset_profile.csv", index=False)
    pd.DataFrame(champion_rows).sort_values("symbol").to_csv(output_root / "champions.csv", index=False)

    # A compact human-readable report
    report_lines = [
        "# Experiment Summary",
        "",
        "Lower RMSE is better. Negative `test_rmse_minus_naive` means the neural model beat the",
        "persistence baseline that predicts the next close as the most recent observed close.",
        "",
    ]
    for symbol, symbol_df in summary_df.groupby("symbol"):
        champ = symbol_df.nsmallest(1, "val_rmse").iloc[0]
        baseline_delta = float(champ["test_rmse_minus_naive"])
        baseline_status = "yes" if bool(champ["beats_test_naive_baseline"]) else "no"
        report_lines.extend(
            [
                f"## {symbol}",
                f"- Champion model: `{champ['model_type']}`",
                f"- Val RMSE: `{champ['val_rmse']:.4f}`",
                f"- Test RMSE: `{champ['test_rmse']:.4f}`",
                f"- Val naive baseline RMSE: `{champ['val_naive_baseline_rmse']:.4f}`",
                f"- Test naive baseline RMSE: `{champ['test_naive_baseline_rmse']:.4f}`",
                f"- Test MAE: `{champ['test_mae']:.4f}`",
                f"- Test MAPE: `{champ['test_mape_pct']:.2f}%`",
                f"- Val directional accuracy: `{champ['val_directional_accuracy_pct']:.2f}%`",
                f"- Test directional accuracy: `{champ['test_directional_accuracy_pct']:.2f}%`",
                f"- Test RMSE minus naive baseline: `{baseline_delta:.4f}`",
                f"- Beats naive baseline: `{baseline_status}`",
                "",
            ]
        )

    (output_root / "SUMMARY.md").write_text("\n".join(report_lines))

    return summary_df
