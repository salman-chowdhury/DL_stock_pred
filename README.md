# Time-Series Forecasting: Recurrent Models and Baselines

Time-series forecasting study for major US indices using recurrent models (**RNN, LSTM, GRU**) and explicit classical baselines with chronological, reproducible evaluation.

**Review the [rolling-origin baseline report](evaluation/baselines-v1/report.md) first.** Persistence had the lowest MAE among the four tested classical baselines on each of the three datasets. The recurrent pipeline and baseline study are separate experiments; this result does not establish a recurrent model win or a profitable trading strategy.

## Project Highlights

- End-to-end forecasting pipeline with clear train/validation/test split by year.
- Robust CSV ingestion across mixed schemas (`Close/Last` vs `Price`, optional volume/change columns).
- Feature engineering + leakage-safe scaling (fit on train period only).
- Hyperparameter search with early stopping.
- Per-model metrics and saved predictions/plots.
- Rolling-origin persistence, moving-average, linear-regression, and random-forest baselines.
- MAE, RMSE, MASE, directional accuracy, naive-relative improvement, and seed variability.
- Tests for data loading and split integrity.

## Data

Raw CSV files are stored in:

- `data/raw/sp500_5y.csv`
- `data/raw/nasdaq100_5y.csv`
- `data/raw/dowjones_5y.csv`

Expected coverage: approximately **2020-05-22 to 2025-05-20** (daily rows).

## Forecasting Setup

- **Target:** next-step close price.
- **Window size:** configurable (default `20` days).
- **Split (by target year):**
  - Train: `<= 2023`
  - Validation: `2024`
  - Test: `2025`
- **Models:** `gru`, `lstm`, `rnn`

## Repository Structure

```text
src/dl_stock_pred/
  config.py        # dataclass configs
  data.py          # data loading, cleaning, feature engineering, sequence prep
  models.py        # recurrent regressor models
  train.py         # training/evaluation utils + early stopping
  pipeline.py      # experiment runner and artifact writing
  cli.py           # command-line interface

scripts/run_experiment.py

tests/test_data_pipeline.py

data/raw/
```

## Quick Start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2) Run full experiment

```bash
dl-stock-pred
```

### 3) Run a fast smoke experiment

```bash
dl-stock-pred --max-epochs 3 --patience 2 --max-trials-per-model 1 --no-plots
```

## Outputs

Each run writes artifacts to `outputs/run_<timestamp>/` (or your custom `--output-dir`):

- `run_config.json`
- `summary.csv`
- `SUMMARY.md`
- Per index folder with:
  - `history_<symbol>_<model>.csv`
  - `predictions_<symbol>_<model>.csv`
  - `plot_<symbol>_<model>.png` (unless `--no-plots`)
  - `champion.json`

## Baseline evaluation

```bash
python scripts/run_baselines.py
```

The [committed 2025 rolling-origin report](evaluation/baselines-v1/report.md) evaluates all four baselines on all three index datasets. Linear and tree models use only prior targets and refit every 20 observations. See the [case study](docs/case-study.md) for the architecture and interpretation limits.

## Tests

```bash
ruff check .
pytest -q
```

Current tests validate:

- schema normalization for all three raw datasets,
- non-empty train/val/test sequence splits,
- correct chronological split assignment.

## Notes on Refactor

This repository was fully cleaned and restructured from a mixed collection of scripts/notebooks/archives into a packaged pipeline with a CLI, explicit splits and tests.

Legacy coursework artifacts are preserved in `archive/legacy/` and are not used by the active pipeline.

## Forecast error is not trading profit

Index prices are highly autocorrelated, so persistence can produce low next-level error without a useful return signal. This project does not claim profitability: it does not model execution timing, transaction costs, slippage, turnover, position sizing, drawdown, or risk-adjusted performance.

## License

MIT
