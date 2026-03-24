# Evaluation Notes

This project now makes the ML evaluation workflow explicit instead of leaving it implicit in the training code.

## Split Logic

- Raw data is ordered chronologically.
- Supervised samples use a trailing window of past observations to predict the next close.
- Sample assignment is based on the target timestamp, not on the start of the window:
  - train targets: year `<= 2023`
  - validation targets: year `2024`
  - test targets: year `2025`

That means a validation or test sample may use recent prior history from the immediately preceding period, which is correct for a rolling forecasting setup.

## Leakage Guardrails

- Feature scaling uses `RobustScaler` fit only on training-period rows.
- Target scaling uses `RobustScaler` fit only on training-period targets.
- Validation and test sets never influence scaler fitting or model selection.
- Config validation rejects invalid split ordering such as overlapping train/validation/test years.

## Model Selection

- The pipeline trains `gru`, `lstm`, and `rnn` candidates.
- The best configuration for each model family is chosen by validation RMSE.
- The champion for each index is the per-symbol model with the lowest validation RMSE.
- Test metrics are reported only after that selection step.

## Baseline

Every run compares learned models against a naive persistence baseline:

- prediction for next close = most recent observed close in the input window

This is intentionally simple, but it is the correct first benchmark for daily price-level forecasting. A portfolio project is more credible when it records whether neural models actually outperform that baseline.

## Saved Metrics

Run summaries now include:

- validation and test RMSE
- validation and test naive-baseline RMSE
- RMSE gap versus the naive baseline
- validation and test directional accuracy
- flags for whether the model beat the naive baseline on validation or test

Per-prediction CSVs include:

- `previous_close`
- `actual`
- `predicted`
- `actual_change`
- `predicted_change`
- `actual_direction`
- `predicted_direction`
- `direction_hit`
- `residual`
