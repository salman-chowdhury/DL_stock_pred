import pytest

from dl_stock_pred.cli import build_config, parse_args
from dl_stock_pred.config import validate_experiment_config


def test_build_config_can_filter_symbols_and_models() -> None:
    args = parse_args(["--symbols", "sp500,dowjones", "--models", "gru,lstm", "--seed", "7"])

    config = build_config(args)

    assert tuple(config.data_files.keys()) == ("sp500", "dowjones")
    assert config.model_types == ("gru", "lstm")
    assert config.train.seed == 7


def test_build_config_rejects_unknown_symbol() -> None:
    args = parse_args(["--symbols", "sp500,unknown"])

    with pytest.raises(ValueError, match="Unknown symbols"):
        build_config(args)


def test_build_config_rejects_empty_csv_value() -> None:
    args = parse_args(["--models", " , "])

    with pytest.raises(ValueError, match="Expected at least one comma-separated value"):
        build_config(args)


def test_validate_experiment_config_rejects_bad_split_order() -> None:
    args = parse_args([])
    config = build_config(args)
    config.split.train_end_year = 2024
    config.split.val_year = 2024

    with pytest.raises(ValueError, match="Split years must be strictly ordered"):
        validate_experiment_config(config)
