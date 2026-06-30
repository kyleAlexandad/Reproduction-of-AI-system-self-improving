# C2 — GIFT-Eval ERA-Scorable Task Wrapper (SUCCESS)

Stage C2 goal: wrap the GIFT-Eval `m4_weekly/W/short` setting as an **ERA-scorable task** —
take a simple candidate forecasting function, score it with the *same* official gluonts pipeline
used in C1, and return a single scalar reward. **No Gemini, no ERA search, no best-of-N, no full
benchmark** — this is just the scoring interface + a baseline smoke test.

## TL;DR
- **Status: SUCCESS.** All three baseline candidates evaluate end-to-end and return rewards.
- The **naive candidate reproduces the C1 official Naive point metrics essentially exactly**
  (MASE/RMSE match to ~6 significant figures), confirming the evaluation path is faithful.
- Invalid candidates (crash / wrong shape / NaN / no `forecast`) are caught cleanly and marked
  `valid=false, reward=-inf` with a descriptive error.

## How to run
```bash
cd /Users/zhangweikun/era/implementation
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_m4_weekly_task.py
```
(Run with the **GIFT-Eval venv**, not the ERA env. Run outside the agent sandbox — native libs
segfault under it; a normal terminal is fine.)

To score one arbitrary candidate (how C3/ERA will call it):
```bash
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_m4_weekly_task.py \
    --candidates /path/to/candidate.py
```

## Candidate interface (chosen)
A candidate `.py` file defines:
```python
def forecast(context, prediction_length, freq, metadata=None):
    # context: 1D numpy array of past target values (NO future labels)
    # prediction_length: int horizon (13 here)
    # freq: str, e.g. "W-SUN"
    # metadata: dict {item_id, season_length, context_length, prediction_length, freq}
    # returns: 1D numpy array of length prediction_length (point forecasts)
```
- **No leakage by construction:** gluonts splits each series into input/label; the wrapper only
  passes the *input* (`entry["target"]`) to `forecast`, never the label window.
- Point-only forecasts are adapted to GIFT-Eval's probabilistic metrics by `FunctionPredictor`,
  which emits a degenerate `QuantileForecast` (all quantiles + mean = the point forecast).
- This keeps the candidate dead-simple for an LLM to generate, while still producing the full
  official metric set.

## Results (m4_weekly/W/short, prediction_length=13, season_length=1)

| candidate | valid | MASE | CRPS | RMSE | reward (=-MASE) | runtime |
|---|---|---|---|---|---|---|
| candidate_naive | ✅ | **2.777295** | 0.063399 | 673.4428 | **-2.777295** | 0.16 s |
| candidate_seasonal_naive | ✅ | 9.577987 | 0.132493 | 1397.8209 | -9.577987 | 0.16 s |
| candidate_moving_average | ✅ | 3.420637 | 0.078397 | 802.6817 | -3.420637 | 0.16 s |

Ranking by reward: **naive > moving_average > seasonal_naive**. (For M4-weekly the series are
near-random-walk, so plain naive is hard to beat; a 52-week seasonal tile and an 8-step MA both do
worse — exactly the kind of signal ERA will later try to improve on.)

### Invalid-candidate handling (verified separately)
| broken candidate | result |
|---|---|
| raises an exception | `valid=false` · `INVALID: candidate raised RuntimeError: boom` |
| returns wrong length | `valid=false` · `INVALID: expected forecast of length 13, got 18` |
| returns NaN/inf | `valid=false` · `INVALID: forecast contains NaN or inf` |
| no `forecast` function | `valid=false` · `LOAD_ERROR: ... does not define a 'forecast' function` |

All return `reward = -inf` so ERA search can keep going (same sentinel pattern as the toy tasks).

## Sanity check vs C1 (Naive)
| metric | C2 candidate_naive | C1 official Naive | match? |
|---|---|---|---|
| MASE | 2.7772950477 | 2.777295047362 | ✅ identical to ~10 s.f. |
| RMSE | 673.44276802 | 673.442756230 | ✅ ~6 s.f. |
| MSE[mean] | 453525.16180 | 453525.145918 | ✅ ~7 s.f. |
| CRPS | 0.063399 | 0.060870 | ⚠️ differs — explained below |

**Why MASE/RMSE differ only in the ~6th–7th digit:** C1's `StatsForecastPredictor` casts targets to
`float32`; our candidate computes in `float64`. The forecast values are identical (both repeat the
last observation); the tiny gap is pure floating-point precision. This confirms the evaluation path
is the same.

**Why CRPS differs (0.0634 vs 0.0609):** C1's `statsforecast` Naive produces *genuine prediction
intervals* (spread quantiles), so its `mean_weighted_sum_quantile_loss` rewards a calibrated
distribution. Our point-only candidate uses **degenerate** quantiles (all equal to the point), so the
quantile loss collapses to the normalized MAE — and indeed our naive CRPS `0.06339865` equals C1's
`ND[0.5]=0.06339865` exactly. This is expected and harmless: **we default the ERA reward to `-MASE`**
(a point metric), which is unaffected. If we later want ERA to optimize probabilistic CRPS properly,
the candidate interface can be extended to return quantiles (documented as future work).

## Files
- `../../gift_eval_m4_weekly_task.py` — the wrapper (FunctionPredictor + scorer + CLI). Importable:
  `load_task()`, `evaluate_candidate()`, `load_candidate_fn()`.
- `candidate_naive.py`, `candidate_seasonal_naive.py`, `candidate_moving_average.py` — baselines.
- `candidate_results.json`, `candidate_results.csv` — metrics + reward per candidate.
- `logs/<candidate>.log` — per-candidate JSON record.
- `run_output.txt` — captured stdout of the run.
- `commands_used.sh`, `environment_info.txt`.

## Design constraints honored
- Only `m4_weekly/W/short`; CPU-only; no full-benchmark download; no foundation-model deps; no
  Gemini / ERA search. C1 files untouched. The third-party `gift-eval` clone/venv/data stay outside
  the ERA repo and `/gift-eval/` remains in `.gitignore`.

## Ready for C3?
**Yes.** We now have a clean function: candidate file → `evaluate_candidate()` → `{valid, MASE, CRPS,
RMSE, reward=-MASE}`. C3 can have Gemini generate `forecast(...)` candidate files and feed them into
FUTS tree search (and a best-of-N control), using `reward` exactly like the toy-task executors did.
Recommended C3 touchpoints: reuse `load_task()` once per run, call `evaluate_candidate()` per
generated candidate, treat `-inf` as an invalid leaf, and (optionally) score on the **validation**
split during search while reserving `test_data` for the final report.
