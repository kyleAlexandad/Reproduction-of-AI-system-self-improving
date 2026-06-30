# GIFT-Eval C1 — Setup Notes

> Stage **C1 only**: install the official GIFT-Eval pipeline, run the smallest official
> baseline on one tiny dataset, and confirm the evaluation produces correct metrics.
> No ERA, no Gemini, no tree search, no full benchmark. Date: 2026-06-30.

---

## 1. Official repo inspection (Step 1)

Repo: **`SalesforceAIResearch/gift-eval`** (cloned shallow to `/Users/zhangweikun/era/gift-eval`).

| Question | Answer |
|---|---|
| Recommended Python | `requires-python = ">=3.10"` (repo authors use 3.10; we used **3.12.13**, works fine). |
| Install (explore only) | `pip install -e .` |
| Install (baselines) | `pip install -e .[baseline]` → adds `torch`, `lightning`, `statsforecast`, `tensorboard`. **We skipped this**; for the Naive baseline only `statsforecast` is needed. |
| Expected data location | A local dir pointed to by env var `GIFT_EVAL` (set in `.env`). Loaded via `datasets.load_from_disk(GIFT_EVAL/<name>)`. |
| How to get data | `huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir PATH` (full benchmark = 28 dataset families). We downloaded **only `m4_weekly`** with `--include "m4_weekly/*"`. |
| Output format | A CSV `results/<model>/all_results.csv`: 1 row per `dataset/freq/term` config, 15 columns (4 meta + 11 metrics). Full leaderboard submission = 98 rows. |
| Baseline scripts | Jupyter notebooks under `notebooks/`: `naive.ipynb` (statistical, lightest), `feedforward.ipynb` (DL), `moirai.ipynb` / `chronos.ipynb` / many FM notebooks. Plus `cli/analysis.py` (Hydra) for time-series feature analysis (not forecasting eval). |
| GPU / HF / large downloads? | **No GPU needed** for Naive (pure CPU statistical model). HF download required for data, but a single dataset is tiny (~1.5 MB) and unauthenticated download works. Foundation-model notebooks (Moirai/Chronos) would need big model downloads + ideally GPU — **avoided**. |

Core deps (from `pyproject.toml`): `pandas>=2`, `gluonts~=0.15.1`, `numpy~=1.26`, `einops==0.7.*`,
`python-dotenv==1.0.0`, `hydra-core==1.3`, `datasets~=2.17.1`, `orjson`, `matplotlib~=3.9`,
`tsfeatures`, `ray`, `scipy~=1.11`.

## 2. Environment (Step 2)

**A separate venv was required** — GIFT-Eval pins `numpy~=1.26` and `gluonts~=0.15.1`, but the
main ERA env has `numpy 2.5` / `pandas 3.0`. Forcing GIFT-Eval into the ERA env would break ERA.

- Venv: `/Users/zhangweikun/era/gift-eval/.venv` (Homebrew `python3.12` → 3.12.13).
- Installed: `pip install -e .` then `pip install statsforecast`.
- Result: `gluonts 0.15.1`, `numpy 1.26.4`, `pandas 2.3.3`, `datasets 2.17.1`, `statsforecast 2.0.3`,
  `scipy 1.11.4`, `ray 2.56.0`. Full list in `environment_info.txt`.
- The main ERA env (`/opt/homebrew` Python 3.12 site-packages) was **not touched**, and the main ERA
  implementation directory was **not polluted** (all GIFT-Eval code lives under `/Users/zhangweikun/era/gift-eval`).

## 3. Smoke test (Step 3)

Smallest official baseline = **Naive** (`statsforecast.models.Naive`), CPU-only.
Smallest dataset = **`m4_weekly`** (359 univariate series, ~1.5 MB) — the exact dataset the official
`naive.ipynb` demonstrates.

Script: `run_naive_smoke.py` (copy saved in this folder). It is a faithful extraction of the official
`naive.ipynb` wrapper (`ModelConfig` / `StatsForecastPredictor` / `NaivePredictor`) restricted to one
dataset, writing `all_results.csv` here. Run with:

```bash
cd /Users/zhangweikun/era/gift-eval && source .venv/bin/activate && python -u run_naive_smoke.py
```

Dataset config produced: `m4_weekly/W/short` — `freq=W-SUN`, `prediction_length=13`, `windows=1`,
`season_length=1`, `target_dim=1`. Eval finished in ~1.3s.

### Result vs official notebook (validation)
The numbers reproduce the official `naive.ipynb` output **exactly**:

| metric | our run | official notebook |
|---|---|---|
| MSE[mean] | 453525.1459 | 453525.145918 |
| MASE[0.5] | 2.777295 | 2.777295 |
| RMSE[mean] | 673.4428 | 673.442756 |
| mean_weighted_sum_quantile_loss | 0.060870 | 0.060870 |

→ The evaluation pipeline is installed and correct.

## 4. Problems encountered + fixes

1. **Segfault (exit 139) under the agent sandbox.** Running any script that imports
   `statsforecast`/`numba`/`coreforecast` crashed with SIGSEGV *before the first print* when run
   inside the Cursor command sandbox (seccomp blocks a syscall these native/JIT libs make).
   - **Fix:** run the python steps outside the sandbox (normal terminal). Imports + eval then work.
     This is an agent-sandbox artifact only; a normal user terminal is unaffected.
2. **`huggingface-cli` vs `hf`.** The README shows `huggingface-cli download`; current
   `huggingface-hub 1.21` prefers `hf download`. Used `hf download ... --include "m4_weekly/*"`.
   Unauthenticated download works (just a rate-limit warning; set `HF_TOKEN` for higher limits).
3. **pandas `FutureWarning`** on `res[metric][0]` (Series positional indexing). Cosmetic; fixed in the
   script to use `.iloc[0]`.

## 5. Not done (intentionally, out of C1 scope)
- No full 98-config benchmark run (only 1 dataset downloaded/run).
- No `.[baseline]` torch/lightning install, no foundation models (Moirai/Chronos), no GPU.
- No ERA / Gemini / best-of-N / tree search wiring (that is C2+).
