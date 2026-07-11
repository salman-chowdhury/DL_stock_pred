# Forecasting Baselines v1

Rolling-origin evaluation on the 2025 portion of each committed index dataset. Linear and random-forest models use only earlier targets, a 750-observation training window, 20 close-price lags, and refit every 20 observations. Random-forest predictions average seeds 17, 42, and 91.

| Index | Model | MAE | RMSE | MASE | Direction accuracy | MAE improvement vs naive | Seed MAE std |
|---|---|---:|---:|---:|---:|---:|---:|
| sp500 | naive_persistence | 59.91 | 91.49 | 1.804 | 0.000 | 0.000 | 0.000 |
| sp500 | moving_average_20 | 164.61 | 208.54 | 4.958 | 0.453 | -1.748 | 0.000 |
| sp500 | linear_regression | 65.53 | 94.55 | 1.973 | 0.463 | -0.094 | 0.000 |
| sp500 | random_forest | 70.88 | 100.70 | 2.135 | 0.537 | -0.183 | 0.636 |
| nasdaq100 | naive_persistence | 266.72 | 396.98 | 1.743 | 0.000 | 0.000 | 0.000 |
| nasdaq100 | moving_average_20 | 744.74 | 933.24 | 4.867 | 0.453 | -1.792 | 0.000 |
| nasdaq100 | linear_regression | 290.26 | 406.82 | 1.897 | 0.442 | -0.088 | 0.000 |
| nasdaq100 | random_forest | 310.36 | 429.76 | 2.028 | 0.474 | -0.164 | 0.810 |
| dowjones | naive_persistence | 389.89 | 585.18 | 1.674 | 0.000 | 0.000 | 0.000 |
| dowjones | moving_average_20 | 1072.30 | 1316.56 | 4.605 | 0.453 | -1.750 | 0.000 |
| dowjones | linear_regression | 416.14 | 604.20 | 1.787 | 0.453 | -0.067 | 0.000 |
| dowjones | random_forest | 435.30 | 639.26 | 1.869 | 0.526 | -0.116 | 12.373 |

## Interpretation

- `MASE < 1` indicates lower absolute error than the in-sample one-step naive scale.
- Relative improvement is calculated against the 2025 persistence MAE for the same index.
- Seed variability applies only to the random forest; other baselines are deterministic.
- Low forecast error does not include transaction costs, slippage, position sizing, risk, or tradability.
- Directional accuracy near 0.5 and low price error can coexist because persistence is already a strong level forecast.

## Failure analysis

Predicting tomorrow's index level with low RMSE does not establish a profitable trading strategy. Index levels are highly autocorrelated, so a persistence model can achieve low error while offering no reliable return signal. A trading claim would require return forecasts, a predeclared execution rule, realistic costs and slippage, risk-adjusted out-of-sample results, and testing across market regimes. None is inferred here.
