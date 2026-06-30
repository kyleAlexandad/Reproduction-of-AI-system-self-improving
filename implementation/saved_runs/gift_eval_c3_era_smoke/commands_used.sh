#!/usr/bin/env bash
# C3 — ERA/FUTS tree search over GIFT-Eval forecasting candidates (m4_weekly/W/short).
#
# Run these MANUALLY in a normal terminal (NOT inside any agent sandbox — the GIFT-Eval
# scorer subprocess uses native libs that segfault under the sandbox).
#
# The ERA search script runs in the ERA env (`python`); it scores each candidate by
# launching the C2 scorer in the GIFT-Eval venv via subprocess. You only set the ERA env vars.

# ---------------------------------------------------------------------------
# 1) SMOKE TEST — 5 iterations (10 if you want a touch more; 5 recommended first)
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_era_search.py \
    --iterations 5 \
    --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c3_era_smoke

# ---------------------------------------------------------------------------
# 2) C3.1 — 10 iterations with the improved CONSERVATIVE prompt (now the default)
#    (run this next; conservative_v2 is the default --prompt_version)
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_era_search.py \
    --iterations 10 \
    --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c3_era_iter10_conservative

# Optional A/B: re-run the OLD prompt for comparison:
#   python gift_eval_era_search.py --iterations 10 --model gemini-2.5-flash \
#       --prompt_version baseline --out_dir saved_runs/gift_eval_c3_era_iter10_baseline
# Optional: seed from the conservative candidate instead of naive:
#   python gift_eval_era_search.py --iterations 10 --model gemini-2.5-flash \
#       --initial_candidate initial_candidate_conservative.py \
#       --out_dir saved_runs/gift_eval_c3_era_iter10_consseed

# Notes:
#  - Gemini calls this run = --iterations (one per expansion). The initial naive candidate
#    is scored locally (no Gemini call).
#  - Other options: --gift_eval_python, --scorer_script, --temperature, --seed, --c_puct.
#  - To re-score a single candidate by hand (no Gemini), use the C2 scorer directly:
#      /Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_m4_weekly_task.py \
#          --candidates saved_runs/gift_eval_c3_era_smoke/best_candidate.py \
#          --out-dir /tmp/c3_recheck
