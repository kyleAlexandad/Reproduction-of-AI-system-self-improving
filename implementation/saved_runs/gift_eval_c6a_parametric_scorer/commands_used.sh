#!/usr/bin/env bash
# C6A — dataset-parametric GIFT-Eval scorer + baseline validation (NO Gemini/ERA/best-of-N).
# Run MANUALLY in a normal terminal. Scoring uses the GIFT-Eval venv; the ERA controllers
# (C6B) run in the ERA env but call the GIFT-Eval venv scorer internally via subprocess.

GE=/Users/zhangweikun/era/gift-eval/.venv/bin/python
cd /Users/zhangweikun/era/implementation
C6A=saved_runs/gift_eval_c6a_parametric_scorer

# --- Validate the generalized scorer on m4_hourly/H/short (primary C6 target) ---
"$GE" -u gift_eval_task.py --dataset m4_hourly --freq H --term short \
  --out-dir $C6A/hourly_results \
  --candidates \
    $C6A/candidate_naive.py \
    $C6A/candidate_seasonal_naive.py \
    $C6A/candidate_moving_average.py \
    $C6A/candidate_damped_trend.py \
    $C6A/candidate_ensemble_naive_trend.py

# --- Confirm backward compatibility on m4_weekly/W/short ---
"$GE" -u gift_eval_task.py --dataset m4_weekly --freq W --term short \
  --out-dir $C6A/weekly_results \
  --candidates \
    $C6A/candidate_naive.py \
    $C6A/candidate_seasonal_naive.py \
    $C6A/candidate_moving_average.py \
    $C6A/candidate_damped_trend.py \
    $C6A/candidate_ensemble_naive_trend.py

# Score a single candidate (how C6B calls it internally):
#   "$GE" -u gift_eval_task.py --dataset m4_hourly --freq H --term short \
#         --candidates /path/to/candidate.py --out-dir /tmp/scratch
