# C1 — GIFT-Eval Setup & Smoke Test (SUCCESS)

Stage C1 goal: stand up the official **GIFT-Eval** evaluation pipeline locally and prove it runs
end-to-end on one tiny CPU baseline — *before* we connect ERA to it (C2+).

## TL;DR
- **Status: SUCCESS.** The official Naive baseline ran on `m4_weekly` and produced metrics that
  match the official notebook **to the decimal**.
- Everything is isolated in `/Users/zhangweikun/era/gift-eval` (separate repo + venv + data); the
  ERA implementation dir was not polluted and the ERA Python env was not modified.

## What was run
| | |
|---|---|
| Command | `python -u run_naive_smoke.py` (in the gift-eval venv) |
| Dataset / config | `m4_weekly` → `m4_weekly/W/short` (359 univariate series; `pred_len=13`, `windows=1`) |
| Baseline/model | **Naive** (`statsforecast.models.Naive`), CPU-only |
| Metric produced | full GIFT-Eval metric set (MSE/MAE/MASE/MAPE/sMAPE/MSIS/RMSE/NRMSE/ND/CRPS) |
| Headline numbers | `MASE=2.7773`, `RMSE=673.44`, `CRPS (mean_weighted_sum_quantile_loss)=0.0609` |
| Runtime | ~1.3 s eval |
| Output | `all_results.csv` (this folder) |

## Environment
- macOS 26.5.1, Apple Silicon (arm64), MacBook Air.
- **Separate** venv at `/Users/zhangweikun/era/gift-eval/.venv`, Python 3.12.13.
- Key pins: `gluonts 0.15.1`, `numpy 1.26.4`, `pandas 2.3.3`, `datasets 2.17.1`, `statsforecast 2.0.3`.
  (A separate env was mandatory — ERA uses numpy 2.5 / pandas 3.0, which conflict with these pins.)

## Files in this folder
- `README_C1.md` — this summary.
- `setup_notes.md` — detailed step-by-step + repo inspection answers + problems/fixes.
- `commands_used.sh` — exact commands, copy-paste reproducible.
- `environment_info.txt` — platform, python, full `pip freeze`.
- `smoke_test_output.txt` — captured stdout of the smoke run.
- `all_results.csv` — the official-format metric output (1 dataset row).
- `run_naive_smoke.py` — the smoke-test script (extracted from official `naive.ipynb`).

## Remaining blockers
- None for C1. One operational note: native baseline libs (`statsforecast`/`numba`) **segfault inside
  the agent command sandbox** — run python steps in a normal terminal. Not an issue for real use.

---

## The GIFT-Eval evaluation interface (plain English)

**1. What does an input example look like?**
A single time series ("item"): a numeric `target` array sampled at a fixed `freq` (e.g. weekly
`W-SUN`), with a `start` timestamp and an `item_id`. Some datasets are multivariate (the loader can
split them into univariate series). The `Dataset` class auto-splits each series into
train / validation / test windows: the model sees history up to a cutoff and must forecast the next
`prediction_length` steps (here 13). For `m4_weekly` there's 1 forecast window per series, 359 series.

**2. What does the model/baseline output?**
For each test window, a forecast over the next `prediction_length` steps — both a **point forecast**
(mean / median) and a set of **quantiles** (0.1…0.9) for probabilistic scoring. In gluonts terms it
returns a `Forecast`/`QuantileForecast` per series.

**3. What metric is computed?**
gluonts `evaluate_model` aggregates 11 metrics over all series/windows: MSE, MAE, **MASE**, MAPE,
sMAPE, MSIS, RMSE, NRMSE, ND, and **mean_weighted_sum_quantile_loss** (a CRPS-style probabilistic
score). The public leaderboard ranks primarily by **MASE** and **CRPS**.

**4. Lower or higher better?**
**Lower is better** for every GIFT-Eval metric (they're forecast errors). This is the opposite of the
ERA toy tasks so far (regression used `-RMSE`, breast cancer used ROC-AUC, both "higher better").
When wrapping as an ERA score we'll **negate** (e.g. score = `-MASE` or `-CRPS`) so ERA's
"higher is better" search still applies.

**5. What would ERA need to generate later (C2+)?**
ERA's LLM would write a Python forecasting function — conceptually
`train_and_predict(train_series, prediction_length, freq) -> forecast` — that produces point +
quantile forecasts for each series. The sandbox runs it; we score with the **same** gluonts
`evaluate_model` so results are leaderboard-comparable. ERA then tree-searches over candidate
forecasting programs (e.g. better imputation, seasonality handling, model choice) to drive the
metric down, exactly as it improved code on the toy tasks.

**6. What files/functions would we wrap into a scorable task in C2?**
- `gift_eval.data.Dataset` — gives `training_dataset`, `validation_dataset`, `test_data`,
  `prediction_length`, `freq`, `season_length`, `windows`.
- `gluonts.model.evaluate_model` + the metric list — the scoring core (reuse verbatim for fairness).
- The predictor pattern in `run_naive_smoke.py` (`StatsForecastPredictor`/`NaivePredictor`) — the
  template for turning "a model" into a gluonts `Predictor` that `evaluate_model` can score. ERA-
  generated code would slot in where `NaivePredictor` is, and we'd return a single scalar
  (e.g. `-MASE` on the **validation** split) as the ERA reward, keeping `test_data` for final report.

## Ready for C2?
**Yes.** The pipeline installs, data loads, and the official metric reproduces exactly. C2 = wrap a
small GIFT-Eval subset (start with `m4_weekly`, maybe add `bizitobs_l2c/H`) as an ERA scorable task:
expose a `build_task()` that loads the `Dataset`, and a scorer that runs `evaluate_model` and returns
`-MASE` (or `-CRPS`) so ERA can optimize it — mirroring how `compare_era_vs_bon*.py` wrapped the toy
tasks.
