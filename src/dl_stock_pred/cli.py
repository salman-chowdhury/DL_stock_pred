from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .config import ExperimentConfig
from .pipeline import run_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate RNN/LSTM/GRU models for index close forecasting."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write all results. Defaults to outputs/run_<timestamp>",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early stopping patience.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override sequence window size.",
    )
    parser.add_argument(
        "--max-trials-per-model",
        type=int,
        default=None,
        help="Limit number of hyperparameter combinations per model for quick runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for torch (e.g., cpu, cuda, auto).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated subset of symbols to run (e.g., sp500,dowjones).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated subset of models to run (e.g., gru,lstm).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for reproducible runs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving prediction plots.",
    )
    return parser.parse_args(argv)


def _parse_csv_values(raw: str | None) -> list[str] | None:
    if raw is None:
        return None

    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def _validate_subset(
    requested: list[str] | None,
    available: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    available_tuple = tuple(available)
    if requested is None:
        return available_tuple

    invalid = [item for item in requested if item not in available_tuple]
    if invalid:
        raise ValueError(
            f"Unknown {label}: {', '.join(invalid)}. Available {label}: {', '.join(available_tuple)}"
        )

    deduped: list[str] = []
    for item in requested:
        if item not in deduped:
            deduped.append(item)

    return tuple(deduped)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path(f"outputs/run_{timestamp}")

    config = ExperimentConfig(output_dir=output_dir)

    if args.max_trials_per_model is not None:
        config.max_trials_per_model = args.max_trials_per_model

    if args.max_epochs is not None:
        config.train.max_epochs = args.max_epochs

    if args.patience is not None:
        config.train.patience = args.patience

    if args.batch_size is not None:
        config.train.batch_size = args.batch_size

    if args.window_size is not None:
        config.train.window_size = args.window_size

    if args.device is not None:
        config.train.device = args.device

    selected_symbols = _validate_subset(
        requested=_parse_csv_values(args.symbols),
        available=tuple(config.data_files.keys()),
        label="symbols",
    )
    config.data_files = {symbol: config.data_files[symbol] for symbol in selected_symbols}

    config.model_types = _validate_subset(
        requested=_parse_csv_values(args.models),
        available=tuple(config.model_types),
        label="models",
    )

    if args.seed is not None:
        config.train.seed = args.seed

    if args.no_plots:
        config.save_plots = False

    return config


def main() -> None:
    args = parse_args()
    try:
        config = build_config(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summary = run_experiment(config)
    print("\nTop models by validation RMSE:")
    print(summary[["symbol", "model_type", "val_rmse", "test_rmse"]].head(12).to_string(index=False))
    print(f"\nSaved outputs to: {config.output_dir}")


if __name__ == "__main__":
    main()
