#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dl_stock_pred.baselines import BaselineConfig, evaluate_walk_forward, summarise_baselines
from dl_stock_pred.data import load_index_dataframe

DATA_FILES = {
    "sp500": Path("data/raw/sp500_5y.csv"),
    "nasdaq100": Path("data/raw/nasdaq100_5y.csv"),
    "dowjones": Path("data/raw/dowjones_5y.csv"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe rolling baseline evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/baselines-v1"))
    parser.add_argument("--test-year", type=int, default=2025)
    args = parser.parse_args()
    config = BaselineConfig(test_year=args.test_year)
    summary_frames = []
    prediction_frames = []

    for symbol, path in DATA_FILES.items():
        frame = load_index_dataframe(path, symbol)
        predictions = evaluate_walk_forward(frame, config)
        summary = summarise_baselines(frame, predictions, config)
        predictions.insert(0, "symbol", symbol)
        summary.insert(0, "symbol", symbol)
        prediction_frames.append(predictions)
        summary_frames.append(summary)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    table = "\n".join(
        f"| {row.symbol} | {row.model} | {row.mae:.2f} | {row.rmse:.2f} | "
        f"{row.mase:.3f} | {row.directional_accuracy:.3f} | "
        f"{row.mae_improvement_vs_naive:.3f} | {row.mae_std_across_seeds:.3f} |"
        for row in summary.itertuples()
    )
    report = "\n".join(
        [
            "# Forecasting Baselines v1",
            "",
            "Rolling-origin evaluation on the 2025 portion of each committed index dataset. "
            "Linear and random-forest models use only earlier targets, a 750-observation "
            "training window, 20 close-price lags, and refit every 20 observations. "
            "Random-forest predictions average seeds 17, 42, and 91.",
            "",
            "| Index | Model | MAE | RMSE | MASE | Direction accuracy | MAE improvement "
            "vs naive | Seed MAE std |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            table,
            "",
            "## Interpretation",
            "",
            "- `MASE < 1` indicates lower absolute error than the in-sample one-step naive scale.",
            "- Relative improvement is calculated against the 2025 persistence MAE for the "
            "same index.",
            "- Seed variability applies only to the random forest; other baselines are "
            "deterministic.",
            "- Low forecast error does not include transaction costs, slippage, position "
            "sizing, risk, or tradability.",
            "- Directional accuracy near 0.5 and low price error can coexist because "
            "persistence is already a strong level forecast.",
            "",
            "## Failure analysis",
            "",
            "Predicting tomorrow's index level with low RMSE does not establish a profitable "
            "trading strategy. Index levels are highly autocorrelated, so a persistence model "
            "can achieve low error while offering no reliable return signal. A trading claim "
            "would require return forecasts, a predeclared execution rule, realistic costs and "
            "slippage, risk-adjusted out-of-sample results, and testing across market regimes. "
            "None is inferred here.",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {len(summary)} baseline rows to {args.output_dir}")


if __name__ == "__main__":
    main()
