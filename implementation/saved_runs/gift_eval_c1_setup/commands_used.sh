#!/usr/bin/env bash
# GIFT-Eval C1 setup + smoke test — exact commands used.
# NOTE: native baseline libs (statsforecast/numba/coreforecast) SEGFAULT under the
# Cursor sandbox; run the python steps in a normal terminal (outside any sandbox).
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Locations
#   Repo  : /Users/zhangweikun/era/gift-eval         (cloned, isolated)
#   Venv  : /Users/zhangweikun/era/gift-eval/.venv   (Python 3.12, separate from ERA)
#   Data  : /Users/zhangweikun/era/gift-eval/data    (only m4_weekly downloaded)
#   Out   : /Users/zhangweikun/era/implementation/saved_runs/gift_eval_c1_setup
# ---------------------------------------------------------------------------

# 1. Clone the official repo (shallow)
cd /Users/zhangweikun/era
git clone --depth 1 https://github.com/SalesforceAIResearch/gift-eval.git gift-eval

# 2. Create an ISOLATED venv (do NOT reuse the ERA env: GIFT-Eval needs numpy 1.26 + gluonts 0.15)
cd /Users/zhangweikun/era/gift-eval
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install GIFT-Eval core (editable) + the Naive baseline dep (statsforecast).
#    We deliberately skip the heavy `.[baseline]` extras (torch/lightning) — not needed for Naive.
pip install -e .
pip install statsforecast

# 4. Download ONLY the smallest dataset (m4_weekly, ~1.5 MB) instead of the full benchmark.
mkdir -p data
hf download Salesforce/GiftEval --repo-type=dataset --include "m4_weekly/*" --local-dir ./data

# 5. Point GIFT-Eval at the local data dir.
echo "GIFT_EVAL=/Users/zhangweikun/era/gift-eval/data" > .env

# 6. Run the smoke test (official Naive baseline, 1 dataset). Writes all_results.csv to the C1 out dir.
python -u run_naive_smoke.py
