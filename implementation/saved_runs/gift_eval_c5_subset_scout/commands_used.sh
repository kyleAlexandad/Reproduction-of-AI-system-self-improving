#!/usr/bin/env bash
# C5 — GIFT-Eval subset scouting (cheap hand-written baselines only; NO Gemini/ERA/best-of-N).
# Run MANUALLY in a normal terminal (GIFT-Eval native libs segfault under the agent sandbox).
# The scout imports gluonts/gift_eval via the C2 wrapper, so it MUST run with the GIFT-Eval venv.

# ---------------------------------------------------------------------------
# 0. (one-time) Download the small candidate datasets (leaf configs, all tiny).
#    Only these small subsets were downloaded — NOT the full benchmark.
#      m4_hourly    ~1.5 MB  (414 series, hourly)
#      hospital     ~0.28 MB (767 series, monthly)
#      covid_deaths ~0.23 MB (266 series, daily)
#    (m4_weekly was already downloaded in C1.)
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/gift-eval
source .venv/bin/activate
for ds in m4_hourly hospital covid_deaths; do
    hf download Salesforce/GiftEval --repo-type=dataset --include "$ds/*" --local-dir ./data
done

# ---------------------------------------------------------------------------
# 1. Run the scout (default set: m4_weekly, m4_hourly, hospital, covid_deaths).
# ---------------------------------------------------------------------------
cd /Users/zhangweikun/era/implementation
/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_subset_scout.py

# Optional: scout a custom list of leaf datasets (must already be downloaded):
#   /Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_subset_scout.py \
#       --datasets m4_hourly hospital --term short
