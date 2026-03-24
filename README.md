# DL Stock Prediction

Time-series forecasting project built around daily US index data already included in the repository. The active codebase packages a small but complete experiment pipeline for recurrent neural networks (`RNN`, `LSTM`, `GRU`) with chronological splits, leakage-safe preprocessing, benchmark comparison, tests, and saved artifacts.

This is not a live trading system and it does not claim profitable predictive performance. It is a cleaned, reproducible ML project that trains and evaluates next-step close-price models on the bundled historical CSV files.

## What The Repo Currently Does

- Loads and normalizes three local datasets:
  - `data/raw/sp500_5y.csv`
  - `data/raw/nasdaq100_5y.csv`
  - `data/raw/dowjones_5y.csv`
- Handles mixed raw schemas such as `Close/Last` vs `Price`, optional volume, and optional percent-change fields.
- Engineers a small feature set from OHLCV-style data:
  - `open`, `high`, `low`, `close`
  - `range_hl`
  - `return_1d`
  - `change_pct_lag1`
  - `volume` when available
- Builds sliding-window supervised sequences for next-step close prediction.
- Splits data chronologically by target year:
  - train: `<= 2023`
  - validation: `2024`
  - test: `2025`
- Trains `GRU`, `LSTM`, and `RNN` regressors with early stopping over a small hyperparameter grid.
- Writes reproducible artifacts for each run:
  - `run_config.json`
  - `summary.csv`
  - `champions.csv`
  - `dataset_profile.csv`
  - `SUMMARY.md`
  - per-symbol histories, predictions, plots, and `champion.json`

## What It Does Not Do

- No live market data ingestion
- No backtesting or trading strategy simulation
- No feature store, model registry, or deployment service
- No claim that the neural models outperform a naive persistence baseline

That last point matters. The project now reports a naive baseline explicitly because daily price-level forecasting is difficult, and a portfolio project is stronger when it shows benchmark discipline rather than only model training.

## Evaluation Workflow

- Model selection is based on validation RMSE only.
- Final reporting keeps test results separate and compares them against a naive persistence baseline.
- Feature scaling and target scaling are both fit on training-period rows only, then applied to validation and test periods.
- Saved prediction files now include `previous_close`, price changes, predicted direction, actual direction, and per-row direction hits.

Details are documented in [`docs/evaluation.md`](/Users/salman/dev/DL_stock_pred/docs/evaluation.md).

## Data Included In The Repo

The bundled datasets cover approximately `2020-05-22` through `2025-05-20`.

| Symbol | File | Rows | Notes |
| --- | --- | ---: | --- |
| `sp500` | `data/raw/sp500_5y.csv` | 1255 | OHLC data |
| `nasdaq100` | `data/raw/nasdaq100_5y.csv` | 1255 | OHLC data |
| `dowjones` | `data/raw/dowjones_5y.csv` | 1256 | OHLC + volume + change % |

## Repository Layout

```text
src/dl_stock_pred/
  cli.py          # CLI and run configuration
  config.py       # dataclass config objects
  data.py         # CSV normalization, cleaning, feature engineering, sequence prep
  models.py       # RNN/LSTM/GRU regressors
  train.py        # training loop, metrics, early stopping
  pipeline.py     # experiment orchestration and artifact writing

data/raw/         # bundled source CSVs
tests/            # lightweight regression tests
docs/             # generated sample outputs + resume bullets
archive/legacy/   # preserved old coursework material, not used by the active pipeline
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -e .
```

## Run

Full run across all symbols and all model types:

```bash
.venv/bin/python -m dl_stock_pred
```

Faster smoke run:

```bash
.venv/bin/python -m dl_stock_pred \
  --max-epochs 3 \
  --patience 2 \
  --max-trials-per-model 1 \
  --output-dir outputs/portfolio_sample
```

Targeted run for iteration:

```bash
.venv/bin/python -m dl_stock_pred \
  --symbols sp500,dowjones \
  --models gru,lstm \
  --max-epochs 5 \
  --max-trials-per-model 1 \
  --seed 42
```

Disable plots:

```bash
.venv/bin/python -m dl_stock_pred --no-plots
```

## Test

```bash
.venv/bin/python -m pytest
```

Current tests cover:

- schema normalization against the bundled raw CSVs
- non-empty chronological train/validation/test sequence splits
- CLI subset selection validation

## Example Verified Output

A bounded run was generated from the current repository and copied into `docs/sample_outputs/`.

- Summary tables: [`docs/sample_outputs/champions.csv`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/champions.csv)
- Dataset profile: [`docs/sample_outputs/dataset_profile.csv`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/dataset_profile.csv)
- Run notes: [`docs/sample-run.md`](/Users/salman/dev/DL_stock_pred/docs/sample-run.md)
- Evaluation notes: [`docs/evaluation.md`](/Users/salman/dev/DL_stock_pred/docs/evaluation.md)

Champion plots from that verified run:

- [`docs/sample_outputs/sp500_champion_gru.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/sp500_champion_gru.png)
- [`docs/sample_outputs/nasdaq100_champion_gru.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/nasdaq100_champion_gru.png)
- [`docs/sample_outputs/dowjones_champion_rnn.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/dowjones_champion_rnn.png)

## Practical Limitations

- The feature set is intentionally simple and derived only from the included OHLCV-like columns.
- Model search is a small grid, not a large-scale tuning workflow.
- The project predicts next-step close price, not return direction, volatility, or trading PnL.
- A quick verified run did not beat the naive persistence baseline on any of the three indices, which is reported explicitly in the generated artifacts.
- Directional accuracy is included for inspection, but this repository still evaluates primary model selection with validation RMSE.

## Notes

- The active project lives under `src/dl_stock_pred/`.
- `archive/legacy/` is kept for provenance only and is not part of the active package.
- Resume-oriented bullets based strictly on the current repository are in [`docs/resume-bullets.md`](/Users/salman/dev/DL_stock_pred/docs/resume-bullets.md).

## License

MIT
