# C6A — Dataset-Parametric GIFT-Eval Scorer + ERA Scripts (DONE)

**Goal:** make the GIFT-Eval scorer and the ERA / best-of-N scripts **dataset-parametric**, so the
same pipeline can run on `m4_hourly/H/short` (the C5-recommended target where naive is beatable),
not only `m4_weekly/W/short`. **Scope:** code preparation + non-Gemini scorer validation only — no
Gemini, no ERA search, no best-of-N, no repeated/foundation/full-benchmark runs.

## Files created / modified
**Created**
- `implementation/gift_eval_task.py` — generalized scorer (`--dataset --freq --term --candidates
  --out-dir`). Reuses the verified C2 machinery (`load_task` + `FunctionPredictor` + official gluonts
  `evaluate_model`) from `gift_eval_m4_weekly_task.py`, made selectable by dataset. Same candidate
  interface and reward (`-MASE`); invalid → `valid=false`, `reward=-inf`.
- 5 baseline candidates in this folder: `candidate_naive.py`, `candidate_seasonal_naive.py`
  (period inferred from freq → 24 for hourly), `candidate_moving_average.py`,
  `candidate_damped_trend.py`, `candidate_ensemble_naive_trend.py`.

**Modified (dataset-parametric, backward compatible)**
- `implementation/gift_eval_era_search.py` (C3 controller): added `--dataset/--freq/--term`; default
  scorer is now `gift_eval_task.py`; `score_program` forwards dataset args to the scorer subprocess;
  the prompt is **dataset-aware** (per-dataset measured facts); the initial seed is **dataset-aware**
  (naive for most, **seasonal-naive for m4_hourly**); results/labels use the chosen config.
- `implementation/gift_eval_compare_era_vs_bon.py` (C4 controller): same `--dataset/--freq/--term`
  + dataset-aware prompt/seed, forwarding dataset args to the scorer.
- `gift_eval_m4_weekly_task.py` (C1–C5) is **left intact** for backward compatibility.

## How the generalized scorer is called
```bash
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_task.py \
    --dataset m4_hourly --freq H --term short --candidates /path/to/candidate.py
# old setting still works:
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_task.py \
    --dataset m4_weekly --freq W --term short --candidates /path/to/candidate.py
```
(Run with the GIFT-Eval venv, in a normal terminal — native libs segfault under an agent sandbox.)

## Validation — does m4_hourly match C5?  ✅ exact match
`hourly_results/candidate_results.csv` (config `m4_hourly/H/short`, 414 series, pred_len 48):

| baseline | C6A MASE | C5 MASE | match |
|---|---:|---:|:--:|
| naive | 11.607688 | 11.6077 | ✅ |
| seasonal_naive (period 24) | **1.193210** | **1.1932** | ✅ |
| moving_average | 11.278355 | 11.2784 | ✅ |
| damped_trend | 12.037122 | 12.0371 | ✅ |
| ensemble_naive_trend | 11.690169 | 11.6902 | ✅ |

## Validation — does old m4_weekly still work?  ✅ exact match
`weekly_results/candidate_results.csv` (config `m4_weekly/W/short`):

| baseline | C6A MASE | C2/C5 MASE | match |
|---|---:|---:|:--:|
| naive | 2.777295 | 2.777295 | ✅ |
| seasonal_naive | 9.577987 | 9.577987 | ✅ |
| moving_average | 3.420637 | 3.420637 | ✅ |
| damped_trend | 2.803882 | 2.803882 | ✅ |
| ensemble_naive_trend | 2.781831 | 2.781831 | ✅ |

All baselines `valid=True`; the generalized scorer reproduces both datasets to the decimal.

## Dataset-aware prompt + seed (for C6B)
- For **m4_hourly**, `conservative_v2` states the measured facts: **naive is weak (~11.61) but
  seasonal-naive with daily period 24 is strong (~1.19)** → it tells the model that daily (24) /
  weekly (168) seasonality may help, includes a seasonal worked template, and asks it to do better.
  (Weekly keeps the naive-anchored conservative prompt; hospital has its own facts; unknown datasets
  get a generic robust prompt.)
- **Seed selection (corrected):** the seed is chosen by `--initial_seed {naive,seasonal_naive}` and
  the **default is `naive` (the weak last-value baseline) for EVERY dataset, including m4_hourly.**
  The main m4_hourly experiment deliberately starts from naive (MASE ~11.61) so we can test whether
  ERA improves from a weak seed and discovers seasonal behaviour. `--initial_seed seasonal_naive`
  is available only as an optional strong-baseline / oracle run.

## Exact commands for C6B (ERA + ERA-vs-best-of-N on m4_hourly) — RUN MANUALLY
```bash
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash

# C6B-1 (MAIN): ERA tree search on m4_hourly from the WEAK naive seed
python gift_eval_era_search.py \
    --dataset m4_hourly --freq H --term short \
    --initial_seed naive \
    --iterations 10 --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c6b_era_m4_hourly_naive_iter10

# C6B-2 (MAIN): ERA vs best-of-N on m4_hourly from the WEAK naive seed (equal budget)
python gift_eval_compare_era_vs_bon.py \
    --dataset m4_hourly --freq H --term short \
    --initial_seed naive \
    --N 10 --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c6b_era_vs_bon_m4_hourly_naive_N10

# OPTIONAL (strong-baseline / oracle): seed from seasonal-naive instead (NOT the main experiment)
#   python gift_eval_era_search.py --dataset m4_hourly --freq H --term short \
#       --initial_seed seasonal_naive --iterations 10 --model gemini-2.5-flash \
#       --out_dir saved_runs/gift_eval_c6b_era_m4_hourly_seasonal_iter10
#   python gift_eval_compare_era_vs_bon.py --dataset m4_hourly --freq H --term short \
#       --initial_seed seasonal_naive --N 10 --model gemini-2.5-flash \
#       --out_dir saved_runs/gift_eval_c6b_era_vs_bon_m4_hourly_seasonal_N10
```
The controllers run in the ERA env (`python`); they call the GIFT-Eval venv scorer internally via
subprocess (you do not invoke the venv yourself for C6B).

## Status — ready for manual C6B
**Yes.** The scorer is dataset-parametric and validated to match C5 on m4_hourly and stay
backward-compatible on m4_weekly; the ERA/best-of-N controllers accept `--dataset/--freq/--term`
with a dataset-aware prompt and seed. Next: run the C6B commands above (ERA should be able to
actually beat naive on m4_hourly, where the strong baseline is seasonal-naive ~1.19).

## Files in this folder
- `README_C6A.md`, `commands_used.sh`, `environment_info.txt`
- `candidate_{naive,seasonal_naive,moving_average,damped_trend,ensemble_naive_trend}.py`
- `hourly_results/` and `weekly_results/` — each with `candidate_results.{json,csv}` + `logs/`
