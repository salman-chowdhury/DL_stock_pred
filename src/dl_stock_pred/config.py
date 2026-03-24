from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class SplitConfig:
    """Chronological split configuration by target year."""

    train_end_year: int = 2023
    val_year: int = 2024
    test_year: int = 2025


@dataclass
class TrainingConfig:
    """Model training controls."""

    window_size: int = 20
    batch_size: int = 64
    max_epochs: int = 60
    patience: int = 10
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"


@dataclass
class SearchSpace:
    """Hyperparameter grid for recurrent models."""

    hidden_sizes: Sequence[int] = (32, 64, 128)
    num_layers: Sequence[int] = (1, 2)
    learning_rates: Sequence[float] = (1e-3, 5e-4)
    dropout: Sequence[float] = (0.0,)


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    data_files: dict[str, Path] = field(
        default_factory=lambda: {
            "sp500": Path("data/raw/sp500_5y.csv"),
            "nasdaq100": Path("data/raw/nasdaq100_5y.csv"),
            "dowjones": Path("data/raw/dowjones_5y.csv"),
        }
    )
    feature_candidates: Sequence[str] = (
        "open",
        "high",
        "low",
        "close",
        "range_hl",
        "return_1d",
        "change_pct_lag1",
        "volume",
    )
    target_column: str = "close"
    model_types: Sequence[str] = ("gru", "lstm", "rnn")
    output_dir: Path = Path("outputs/latest")
    max_trials_per_model: int | None = None
    save_plots: bool = True

    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    search: SearchSpace = field(default_factory=SearchSpace)


def validate_experiment_config(config: ExperimentConfig) -> None:
    if config.train.window_size <= 0:
        raise ValueError("window_size must be positive.")
    if config.train.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.train.max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    if config.train.patience <= 0:
        raise ValueError("patience must be positive.")
    if config.max_trials_per_model is not None and config.max_trials_per_model <= 0:
        raise ValueError("max_trials_per_model must be positive when provided.")
    if not config.model_types:
        raise ValueError("At least one model type must be configured.")
    if not config.data_files:
        raise ValueError("At least one dataset must be configured.")

    split = config.split
    if not (split.train_end_year < split.val_year < split.test_year):
        raise ValueError(
            "Split years must be strictly ordered as train_end_year < val_year < test_year."
        )

    missing_files = [str(path) for path in config.data_files.values() if not Path(path).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing data files: {', '.join(missing_files)}")
