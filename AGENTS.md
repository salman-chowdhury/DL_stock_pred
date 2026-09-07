# AI Contributor Handoff

## Purpose

This repository is a reproducible time-series forecasting project, not evidence of a profitable trading strategy.

## Start here

1. Read `README.md` and inspect the existing experiment runner and tests.
2. Install with `pip install -r requirements.txt` and `pip install -e .` where required.
3. Run `pytest` and the documented smoke experiment before pushing.

## Current highest-priority work

1. Add naïve persistence, moving-average, linear and tree-based baselines.
2. Add rolling or walk-forward evaluation.
3. Report MAE/RMSE, MASE where appropriate, directional accuracy and baseline-relative improvement.
4. Add repeated-run variability or uncertainty for stochastic neural models.
5. Separate price-level, return, direction and volatility targets explicitly.
6. Add failure analysis covering regime shifts, leakage risks and why low error does not establish profitability.
7. Add experiment tracking only if it improves reproducibility without overcomplicating the project.

## Non-negotiable rules

- Preserve chronological splits and leakage-safe preprocessing.
- Never fit scalers or feature transformations on future data.
- Do not describe model output as investment advice or proven trading alpha.
- Include simple baselines before presenting complex models as improvements.
- Record seeds, data range, features, target, split and hyperparameters with results.
- Do not fabricate transaction-cost or profitability claims.

## Repository naming

`DL_stock_pred` is understandable but informal and narrowly framed. Before a polished public release, consider renaming it to `time-series-forecasting-lab` or `financial-forecasting-benchmarks`. After renaming, update badges, clone commands, package metadata and portfolio links.

## Definition of done

- Tests and smoke experiment pass.
- New models are compared against simple baselines.
- Results include data period and evaluation protocol.
- Limitations and failure modes are documented.
- Claims remain statistical rather than financial.
