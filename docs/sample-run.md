# Sample Run

This repository now includes a verified bounded run generated from the bundled CSV files already present in `data/raw/`.

## Command Used

```bash
.venv/bin/python -m dl_stock_pred \
  --max-epochs 3 \
  --patience 2 \
  --max-trials-per-model 1 \
  --output-dir outputs/portfolio_sample
```

## Data Scope

- Symbols: `sp500`, `nasdaq100`, `dowjones`
- Date coverage: `2020-05-22` to `2025-05-20`
- Split years:
  - train targets through `2023`
  - validation targets in `2024`
  - test targets in `2025`

## Champion Models From The Verified Run

| Symbol | Champion | Val RMSE | Test RMSE | Val Naive RMSE | Test Naive RMSE | Test Directional Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dowjones` | `rnn` | 2381.52 | 4025.05 | 290.48 | 585.18 | 47.37% |
| `nasdaq100` | `gru` | 1488.22 | 2386.49 | 220.14 | 396.98 | 43.16% |
| `sp500` | `gru` | 443.00 | 668.07 | 43.52 | 91.49 | 42.11% |

The purpose of this sample run is verification and documentation, not to present state-of-the-art performance. The key portfolio-quality point is that the project now records baseline comparison explicitly, so weak model performance is visible instead of being hidden.

## Included Sample Artifacts

- [`docs/sample_outputs/champions.csv`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/champions.csv)
- [`docs/sample_outputs/dataset_profile.csv`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/dataset_profile.csv)
- [`docs/sample_outputs/sp500_champion_gru.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/sp500_champion_gru.png)
- [`docs/sample_outputs/nasdaq100_champion_gru.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/nasdaq100_champion_gru.png)
- [`docs/sample_outputs/dowjones_champion_rnn.png`](/Users/salman/dev/DL_stock_pred/docs/sample_outputs/dowjones_champion_rnn.png)
