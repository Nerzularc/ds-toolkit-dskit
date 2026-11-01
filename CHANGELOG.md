# Changelog

## 0.1.0 — 2026-11-01

First public release.

- `profile`: EDA report with data-quality alerts, quality score, correlations, target analysis, leakage check
- `validate`: YAML data contracts (not null, unique, regex, allowed values, ranges, dates, freshness, row expressions), `--suggest`, CI exit codes
- `clean`: auditable cleaning with change log; Indonesian/European number formats; label unification
- `baseline` / `predict`: auto task detection, 4 models × 5-fold CV, hold-out evaluation, threshold table, permutation importance, model card, saved pipeline
- `forecast`: rolling-origin backtest of naive, seasonal naive, moving average and trend + seasonality models; empirical intervals
- `abtest` / `samplesize`: SRM check, frequentist + Bayesian results, detectable effect, guardrails, Holm-adjusted segments
- `rfm`: RFM scores and 9 actionable segments
- `new`: client project scaffold with remote-work documents
