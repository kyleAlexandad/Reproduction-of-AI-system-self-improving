#!/usr/bin/env bash
# GIFT-Eval C2 — scorable task wrapper + baseline-candidate smoke test.
# Run the python step OUTSIDE any sandbox (native libs segfault under the agent sandbox).
set -euo pipefail

ERA_IMPL=/Users/zhangweikun/era/implementation
GIFT_PY=/Users/zhangweikun/era/gift-eval/.venv/bin/python   # GIFT-Eval venv (NOT the ERA env)

# Evaluate the three baseline candidates (default set) on m4_weekly/W/short:
cd "$ERA_IMPL"
"$GIFT_PY" -u gift_eval_m4_weekly_task.py

# Or evaluate an arbitrary candidate file (this is how C3/ERA will call it):
# "$GIFT_PY" -u gift_eval_m4_weekly_task.py \
#     --candidates /path/to/candidate.py \
#     --out-dir "$ERA_IMPL/saved_runs/gift_eval_c2_task_wrapper"

# Outputs (in saved_runs/gift_eval_c2_task_wrapper/):
#   candidate_results.json   candidate_results.csv   logs/<candidate>.log   run_output.txt
