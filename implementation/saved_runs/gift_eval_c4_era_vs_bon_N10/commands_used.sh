#!/usr/bin/env bash
# C4 — fair GIFT-Eval ERA vs best-of-N (m4_weekly/W/short). Run MANUALLY in a normal terminal
# (the GIFT-Eval scorer subprocess uses native libs that segfault under the agent sandbox).
# The script runs in the ERA env (`python`); it scores candidates via the GIFT-Eval venv subprocess.

# ---------------------------------------------------------------------------
# 1) SMOKE — N=10 per method (20 Gemini calls total)
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_compare_era_vs_bon.py \
    --N 10 \
    --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c4_era_vs_bon_N10

# ---------------------------------------------------------------------------
# 2) OPTIONAL — N=20 per method (40 Gemini calls total)
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_compare_era_vs_bon.py \
    --N 20 \
    --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c4_era_vs_bon_N20

# Notes:
#  - Both methods use the SAME model, prompt (conservative_v2), naive seed, scorer and reward.
#    Equal budget: N Gemini calls each. Only parent selection differs (tree vs always-seed).
#  - Other options: --prompt_version {baseline,conservative_v2}, --seed, --c_puct,
#    --gift_eval_python, --scorer_script.
