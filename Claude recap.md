
# Claude Recap

> Last updated: 2026-06-30. Repo: `/Users/zhangweikun/era` (git root).
> Working subfolder for all reproduction code: `/Users/zhangweikun/era/implementation`.
> Latest milestone: **GIFT-Eval Stage C: C1–C5 done** — C1 setup, C2 scorer, C3 ERA search, C4 ERA
> vs best-of-N (all on m4_weekly where naive is near-optimal: ERA tied naive and beat best-of-N at
> N=10/20). **C5 subset scouting DONE** → found subsets where naive is NOT dominant: **`m4_hourly`**
> (seasonal naive 1.19 vs naive 11.61, ~10× headroom — primary C6 target) and **`hospital`** (MA
> 0.81 vs naive 0.97 — secondary). Root **README.md** has the full Stage-3 section. **Next: C6 =
> run ERA on `m4_hourly` where it can actually beat naive.** Earlier: Stage 1 (regression) + Stage 2
> (breast cancer) complete (ERA beat best-of-N 3/3 each).
>
> **C6A DONE** (dataset-parametric scorer + ERA scripts): new `gift_eval_task.py` (`--dataset/--freq/
> --term`); `gift_eval_era_search.py` + `gift_eval_compare_era_vs_bon.py` take `--dataset/--freq/
> --term` + `--initial_seed {naive,seasonal_naive}` (DEFAULT naive for EVERY dataset).
> **C6B DONE — the MAIN POSITIVE GIFT-Eval result:** on m4_hourly/H/short from the WEAK naive seed
> (MASE 11.61), ERA discovered DAILY SEASONALITY (period 24) and reached **MASE 1.1384** (10-iter
> ERA-only, even beating the seasonal-naive reference 1.1932); and ERA **beat best-of-N**
> (1.1932 vs 1.3651) under equal budget. So ERA's tree-search advantage now holds on a real
> benchmark AND clears the naive baseline (unlike m4_weekly where naive was unbeatable).
> **C6B N=20 numbers confirmed present** in both this recap (§13.9) and root README (§C5–C6):
> seed naive 11.6077, ERA best-gen 1.1932, BoN best-gen 1.3651, 20/0 both, winner ERA.
>
> **D0 DONE (2026-06-30) — next-stage RECON only (no Gemini/ERA/BoN/downloads):** scRNA-seq batch
> integration. The upstream ERA task already lives locally at
> `implementation/notebooks/single_cell_batch_integration.ipynb` (this repo root IS the
> `google-research/era` clone; `origin=google-research/era.git`). It needs a SEPARATE isolated env
> (`scanpy`/`anndata`/`scib` + R/`kBET`, numpy<2) — incompatible with BOTH the ERA env (numpy 2.5)
> and the gift-eval venv. Full findings + D1 plan in **§14**.
>
> **D1 DONE (2026-06-30) — scRNA-env built + synthetic Python-only smoke test PASSES (no download,
> no R, no Gemini).** New env `/Users/zhangweikun/era/scRNA-env` (Python 3.11.15; numpy 1.26.4,
> anndata 0.11.3, scanpy 1.10.4, scib 1.1.7). New script `implementation/scrna_synthetic_smoke.py`
> validates the full plumbing on synthetic data. Details in **§14.8**.
>
> **D2A DONE (2026-06-30) — synthetic scRNA ERA-scorable wrapper + controller (code-ready; Gemini NOT
> run yet).** GIFT-Eval-style two-env split: `implementation/scrna_synthetic_task.py` (scorer, runs in
> scRNA-env; reward = reduced score, higher better) + `implementation/scrna_era_search.py` (ERA-env
> FUTS/Gemini controller, scores via subprocess into scRNA-env). Non-Gemini bridge validated; user
> runs the 5-iter Gemini smoke manually. `/scRNA-env/` added to `.gitignore`. Details in **§14.9**.
> **D2A Gemini runs (user):** 5-iter PCA 1.0699→**1.2793**; 10-iter 1.0699→1.2525 (valid/invalid 10/1)
> — the synthetic LLM→scorer→FUTS loop works.
>
> **D3A DONE (2026-06-30) — REAL scRNA data path + tiny real-data smoke (no download/R/scIB/Gemini).**
> New `implementation/scrna_realdata_smoke.py` loads the real `*-train-dataset.h5ad` (counts in
> `layers/counts`), subsamples to ~100 cells, runs the D1/D2A candidates + reduced score. Real data is
> NOT present → the script exits with exact Kaggle download + placement instructions; the real code
> path was validated on a fake structured h5ad. Details in **§14.10**.
>
> DOCS POLICY (user instruction, 2026-06-30): going forward do NOT create new per-stage markdown
> files. Only maintain (1) this `Claude recap.md` and (2) ONE overall/final report md (the root
> `README.md`). Existing per-stage READMEs (C1..C6A) may stay/ be edited but no NEW ones.
>
> NOTE: C5 + C6A changes are currently **uncommitted** (those tasks did not request a push).
>
> **READ THIS FIRST — two-environment rule (new in Stage C):** ERA and GIFT-Eval need
> *incompatible* Python envs. ERA code uses the system brew Python; GIFT-Eval uses its OWN venv at
> `/Users/zhangweikun/era/gift-eval/.venv`. Never mix them. Details in **§13**.
>
> Baseline commits on `repro/main`: `fbc0746`, `b14696a`. GIFT-Eval Stage C work is currently
> **uncommitted** (see `git status`).

---

## 1. Project Overview

This project reproduces and lightly extends the official **`google-research/era`** code for the
Nature paper *"An AI system to help scientists write expert-level empirical software"* (ERA).

- **ERA loop:** an LLM (Gemini) writes/edits a Python `train_and_predict` function → a local
  sandbox executes it → it is auto-scored → **Flat UCB Tree Search (FUTS)** selects a parent and
  iterates to improve the code.
- **Demo task:** `playground_s3e1.py` — a Kaggle / California-Housing-style **regression** problem.
- **Metric:** **negative RMSE** (`-RMSE`); **higher (closer to 0) is better**.
- **Main goal of this work:** (a) get the official demo running locally, (b) answer the research
  question *"is ERA tree search actually better than best-of-N independent sampling?"*, and
  (c) make that answer credible via repeated runs + a small model ablation, then publish a clean
  reproduction repo with reports.

---

## 2. Current Project State

### What works now
- **Official demo runs end-to-end:** `python playground_s3e1.py` (needs `GEMINI_API_KEY`).
- **ERA vs best-of-N single comparison:** `compare_era_vs_bon.py` — verified fair, runs, saves JSON + plot + report.
- **Repeated evaluation:** `repeat_era_vs_bon.py` — run for real (flash-lite, N=10, 3 repeats); **ERA won 3/3**.
- **Stage 2 — second benchmark complete:** `playground_breast_cancer.py` (sklearn breast cancer, ROC-AUC).
  ERA lifted a 0.8468 baseline to ~0.99. ERA vs best-of-N on `gemini-2.5-flash`: single ERA **0.9943** >
  BoN 0.9841; repeated N=10×3 **ERA 3/3**, avg **0.9888** vs 0.9808, **0/30 invalid**. Scripts:
  `compare_era_vs_bon_breast_cancer.py`, `repeated_breast_cancer_era_vs_bon.py`.
- **Reports compiled:** Part 1 + Part 2 LaTeX reports compiled to PDF via Tectonic.
- **Consolidated `README.md`** at repo root + pushed to the user's GitHub repo.
- **Python toolchain fixed:** native arm64 Python 3.12.13 with all packages.
- **GIFT-Eval Stage C (the current focus) — see §13 for full detail:**
  - **C1 ✅** official GIFT-Eval installed in its own venv; Naive baseline reproduces the official
    `naive.ipynb` numbers exactly on `m4_weekly/W/short` (MASE 2.7773).
  - **C2 ✅** ERA-scorable wrapper `gift_eval_m4_weekly_task.py` turns a simple `forecast(...)`
    candidate file into `reward = -MASE`; 3 baseline candidates pass + invalid-handling verified.
  - **C3 (code-ready, NOT run)** `gift_eval_era_search.py` wires the C2 scorer into `futs.search`
    + Gemini. Compiles; cross-env subprocess plumbing tested (no Gemini). User runs it manually.

### Partially working / not yet run
- **GIFT-Eval C3 not executed yet** — the active next action is for the user to run
  `gift_eval_era_search.py` manually (it spends Gemini calls). See §13 / §12.
- **`model_ablation.py`** (flash-lite vs flash) is written + smoke-tested but **NOT run for real** →
  `saved_runs/model_ablation/` does not exist. Now a *secondary* (toy-task) TODO; the user pivoted
  to GIFT-Eval.
- **`compare_era_vs_bon.py --repeats`** flag exists (an earlier mean±std version) but the *canonical*
  repeated path is now `repeat_era_vs_bon.py`. Both coexist.

### Broken / uncertain
- **Gemini availability is flaky:** intermittent `429` (rate limit) and `503 UNAVAILABLE`
  ("high demand") errors. Mitigated by retries + `SafeGenerator`, but a sustained outage will still
  make runs slow or all-`-inf`.
- **`futs_test.py` cannot run as-is:** it needs `absl-py`, which is **not installed**.
- **GIFT-Eval native libs SEGFAULT inside the Cursor agent sandbox** (`statsforecast`/`numba`/
  `coreforecast` crash with exit 139 *before any output*). They run fine in a normal terminal. So
  any GIFT-Eval execution (C1/C2/C3 scorer) must be run outside the agent sandbox / by the user.

### Important assumptions / constraints
- **`sandbox.py` is NOT secure** — it executes LLM-generated code directly on the machine. Toy use only.
- **API calls cost money / quota** → keep `N` small. Cost per comparison run = `2 * N * num_repeats`.
- **Do NOT default to Gemini Pro.** Default model is `gemini-2.5-flash-lite`.
- Single small stochastic runs on a toy task → results are "consistent," not a tight statistical claim.

---

## 3. Completed Changes

### Environment / toolchain (machine-level, this session)
- **Fixed broken Python.** `python3` was aliased in `~/.zshrc` to `/usr/local/bin/python3.10`, an
  **x86_64** binary that can't run on this arm64 Mac (`bad CPU type in executable`). Installed native
  **Homebrew at `/opt/homebrew`** + `python@3.12` (3.12.13). Edited `~/.zshrc`: removed the bad alias,
  added `eval "$(/opt/homebrew/bin/brew shellenv)"` and
  `export PATH="/opt/homebrew/opt/python@3.12/libexec/bin:$PATH"`.
- **Installed packages** into brew Python 3.12 (`pip install --break-system-packages ...`):
  `pandas 3.0.4, numpy 2.5.0, scikit-learn 1.9.0, google-genai 2.10.0, matplotlib 3.11.0, tabulate 0.10.0`.
- **Installed Tectonic** (`brew install tectonic`) at `/opt/homebrew/bin/tectonic` for LaTeX → PDF.

### Files MODIFIED (originals — kept working)
- **`implementation/llm.py`**
  - Model default changed `gemini-2.5-flash-image` → **`gemini-2.5-flash`** (the image model has
    **zero free-tier quota** → permanent 429).
  - Added `GEMINI_MODEL` env override + `import os`; `model_name=None` resolves to env or default.
  - Broadened retry: was only `"429"`; now retries transient **5xx/UNAVAILABLE/overloaded** too, so a
    momentary 503 doesn't crash a multi-call run. Still raises on permanent errors (e.g. 400).
- **`implementation/sandbox.py`**
  - Was an intentional stub (`run()` raised `NotImplementedError`, no constructor) → caused
    `TypeError: Sandbox() takes no arguments`. **Rewritten** as a minimal local executor:
    `Sandbox(timeout_seconds=60)`; `run(program, function_to_run, test_input, timeout_seconds)` writes
    a temp script, runs it via `subprocess` with `sys.executable`, passes input + returns result via
    base64/pickle. **Clearly documented as INSECURE.**
- **`implementation/playground_s3e1.py`** — one line only: `run_experiment(iterations=10)` → `=30`
  (the user's change for the 30-iteration run). Otherwise untouched.

### Files CREATED (in `implementation/`)
- **`compare_era_vs_bon.py`** — ERA vs best-of-N single comparison harness. Core functions reused by
  the other scripts: `run_era`, `run_best_of_n`, `running_best`, `SafeGenerator`,
  `aggregate_running_bests`, `make_plot`, `_json_safe`, `PROBLEM_DESCRIPTION`, `INITIAL_CODE`.
  Flags: `--n`, `--repeats`.
- **`repeat_era_vs_bon.py`** — repeated comparison. Key importable fn `run_experiment_repeats(model, n,
  num_repeats, api_key)`; helpers `count_failures`, `decide_winner`; plots `plot_repeated_curves`,
  `plot_final_best_bars`; `write_summary_md` (Chinese). Flags: `--model` (default `gemini-2.5-flash-lite`),
  `--N` (10), `--num_repeats` (3), `--out-dir`, `--plot-only`.
- **`model_ablation.py`** — flash-lite vs flash ablation; imports `run_experiment_repeats`. Flags:
  `--models` (default the two), `--N` (10), `--num_repeats` (1), `--out-dir`. Pro excluded by default.
- **`plot_progress.py`** — robust progress-figure plotter for Part 1 (handles list/dict JSON shapes).
- **`playground_breast_cancer.py`** — Stage-2 benchmark (sklearn breast cancer, ROC-AUC, higher better).
  Weak `DecisionTreeClassifier(max_depth=1)` baseline; prompt explicitly states the test CSV has no
  `target` column (fixed an earlier `KeyError: "['target'] not found in axis"`).
- **`compare_era_vs_bon_breast_cancer.py`** — Stage-2 single ERA vs best-of-N; exposes
  `build_task` / `run_one_comparison` (reused by the repeated script).
- **`repeated_breast_cancer_era_vs_bon.py`** — Stage-2 repeated comparison
  (reuses the single script + `repeat_era_vs_bon.save_summary_csv`).

### Outputs CREATED (in `implementation/saved_runs/` and `implementation/results/`)
- `results/futs_progress.json` — 31 points from the 30-iteration demo run.
- `saved_runs/playground_s3e1_iter30/` — `progress.{png,pdf}`, `report.{tex,pdf}` (Part 1, English).
- `saved_runs/era_vs_bon/` — `era_vs_bon.{png,pdf}`, `report.{tex,pdf}`, `results.json` (Part 2, single run).
- `saved_runs/repeated_era_vs_bon/` — `results.json`, `summary.csv`, `summary.md` (Chinese),
  `repeated_curves.{png,pdf}`, `final_best_scores.{png,pdf}` (Part 3 repeated run).
- `saved_runs/phase3_summary.md` — Chinese advisor summary (updated with 3/3 result).

### Repo-root changes (this session)
- **`README.md`** — replaced with a consolidated reproduction summary (front page) embedding both
  reports' results + figures; updated with the repeated-run 3/3 result.
- **`README_UPSTREAM.md`** — the original google-research ERA README, preserved via `git mv`.
- **`.gitignore`** — created (excludes `__pycache__/`, `*.py[cod]`, `.venv/`, `.DS_Store`, etc.).

### Git
- Added remote **`repro`** = `https://github.com/kyleAlexandad/Reproduction-of-AI-system-self-improving.git`
  (separate from `origin` = upstream `google-research/era`).
- Two commits pushed to `repro/main`: `fbc0746` (initial reproduction), `b14696a` (repeated results).
- Push auth works via macOS keychain credential helper.

---

## 4. Key Files and Their Roles

### Entry points / runnable scripts (`implementation/`)
| File | Role |
|------|------|
| `playground_s3e1.py` | Official ERA demo. `run_experiment(iterations=30)`. Defines `prepare_data()`, `get_data_head()`, `PlaygroundProblem`, `PlaygroundGenerator`, `PlaygroundExecutor`, and the `INITIAL_CODE` (a LinearRegression baseline). |
| `compare_era_vs_bon.py` | Single ERA-vs-best-of-N comparison. The "library" the other two scripts import from. |
| `repeat_era_vs_bon.py` | Repeated comparison → JSON/CSV/plots/`summary.md`. Has `--plot-only`. |
| `model_ablation.py` | flash-lite vs flash ablation (not run yet). |
| `plot_progress.py` | Standalone Part-1 progress plotter. |
| `gift_eval_m4_weekly_task.py` | **GIFT-Eval C2 scorer** (run with the GIFT-Eval venv). `forecast(...)` candidate → `reward=-MASE`. See §13. |
| `gift_eval_era_search.py` | **GIFT-Eval C3** ERA/FUTS search (run with the ERA env). Scores candidates via subprocess to the C2 scorer. See §13. |

### Core logic (`implementation/`)
| File | Role |
|------|------|
| `futs.py` | **Unchanged.** Flat UCB Tree Search. `search(problem, initial_solution, initial_score, generate_fn, execute_fn, num_iterations, c_puct)`; dataclasses `Problem`, `Solution`, `Node`; `compute_rank_scores`, `compute_pucts`, `backpropagate_visit`. |
| `llm.py` | `GeminiLLM(api_key, model_name=None)` wrapping `google.genai`. `draw_sample(prompt)` with retry/backoff. `DEFAULT_MODEL = "gemini-2.5-flash"`. |
| `sandbox.py` | `Sandbox(timeout_seconds=60).run(...)` local executor. **Insecure.** |

### Tests
- `implementation/futs_test.py` — absl-based unit tests for `futs.py` (needs `absl-py`, not installed).

### Documentation
- `README.md` (root) — consolidated reproduction report (front page).
- `README_UPSTREAM.md` (root) — original ERA project README.
- `implementation/saved_runs/phase3_summary.md` — Chinese advisor summary.
- `implementation/saved_runs/*/report.{tex,pdf}` — Part 1 & Part 2 formal reports.
- `implementation/saved_runs/repeated_era_vs_bon/summary.md` — Chinese repeated-run summary.
- `Claude recap.md` (root) — **this file**.

### Data / assets
- `implementation/data/playground-series-s3e1/train.csv` — source data (tracked upstream, ~10 MB dir).
- `local_train.csv` / `local_test.csv` — deterministic 80/20 split written by `prepare_data()`.

---

## 5. Important Code Structure

### Data flow (one comparison)
1. `prepare_data()` reads `data/playground-series-s3e1/train.csv`, makes a **deterministic 80/20 split**
   (`local_train.csv`, `local_test.csv`), and returns `y_val` (true targets for scoring).
2. `PlaygroundExecutor(sandbox, y_val).__call__(problem, solution)`:
   injects code that writes the CSVs to a temp dir, appends `wrapper(unused_arg)` that calls
   `train_and_predict(train_path, test_path)`, then `sandbox.run(...)` executes it. Predictions are
   scored as **`-sqrt(mean_squared_error(y_val, preds))`**; failures return `float('-inf')`.
3. `PlaygroundGenerator(llm).__call__(problem, parent_solution, parent_score)` builds a prompt
   (data preview via `.to_markdown()`, the parent code + its RMSE) and calls `llm.draw_sample()`.

### The two methods (equal budget = N LLM calls each)
- **ERA (`run_era`):** calls `futs.search(num_iterations=N)`. Each expansion generates a candidate
  **conditioned on a tree-selected parent** (FUTS picks the node with max PUCT). A `TrackingExecutor`
  records each candidate's raw score.
- **Best-of-N (`run_best_of_n`):** N independent calls, each `generator(problem, initial_solution,
  initial_score)` — **always the same fixed initial prompt**, never conditioning on prior candidates.

### Aggregation / robustness
- `running_best(scores, floor=initial_score)` → monotone best-so-far, floored at the initial score.
- `SafeGenerator` wraps the generator: if an LLM call hard-fails after retries, it returns a sentinel
  `Solution` with no `train_and_predict` → scores `-inf` → the run continues and still saves.
- `_json_safe` converts `-inf` → `null` for valid JSON.
- In `repeat_era_vs_bon.py`: `run_experiment_repeats()` evaluates the (deterministic) initial score
  **once**, then loops `num_repeats`, building per-repeat records; `model_ablation.py` calls it once
  per model.

### Design decisions baked into the code (see also §11)
- The fair comparison reuses one verified core (`compare_era_vs_bon.py`) across all 3 driver scripts.
- `INITIAL_CODE` + `PROBLEM_DESCRIPTION` were **copied** into `compare_era_vs_bon.py` (they live inside
  `run_experiment()` in the original) to avoid editing/breaking `playground_s3e1.py`.

---

## 6. Commands and How to Run

> **Shell note:** in a normal terminal (login zsh), `python`/`python3` resolve to brew 3.12 via
> `~/.zshrc`. In a non-login shell where `/opt/homebrew/bin` isn't on PATH, use the absolute interpreter
> `/opt/homebrew/bin/python3.12`. Same for `/opt/homebrew/bin/tectonic`.

### Install dependencies
```bash
pip install --break-system-packages pandas numpy scikit-learn google-genai matplotlib tabulate
# For running the unit tests:
pip install --break-system-packages absl-py
```

### Set the API key (do this every new terminal — keys are session-only)
```bash
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY              # else the SDK may prefer it
export GEMINI_API_KEY="my_key"
export GEMINI_MODEL=gemini-2.5-flash-lite   # optional; scripts default to flash-lite anyway
```

### Run things
```bash
# Official demo (writes results/futs_progress.json)
python playground_s3e1.py

# ERA vs best-of-N, single (N=20 → 40 Gemini calls)
python compare_era_vs_bon.py --n 20

# Repeated comparison (N=10, 3 repeats → 60 calls)  [ALREADY RUN]
python repeat_era_vs_bon.py --model gemini-2.5-flash-lite --N 10 --num_repeats 3

# Model ablation flash-lite vs flash (N=10, 1 repeat each → 40 calls)  [NEXT TODO — NOT RUN YET]
python model_ablation.py

# Re-plot a repeated run WITHOUT spending API calls
python repeat_era_vs_bon.py --plot-only
```

### Compile a report to PDF
```bash
cd /Users/zhangweikun/era/implementation/saved_runs/era_vs_bon   # or playground_s3e1_iter30
/opt/homebrew/bin/tectonic report.tex
```

### Tests / lint
```bash
cd /Users/zhangweikun/era/implementation
python -m absl.testing... ->  python futs_test.py     # needs absl-py installed first
python -m py_compile *.py                              # quick syntax check (used as smoke check)
```
No linter/formatter is configured in this project.

### Git push (user's repo)
```bash
cd /Users/zhangweikun/era
git add -A && git commit -m "..." && git push repro main   # repro = user's repo; origin = upstream
```

---

## 7. Known Issues / Bugs

1. **`model_ablation.py` not run yet** — `saved_runs/model_ablation/` doesn't exist. Next action.
2. **Gemini `503 UNAVAILABLE` / `429` rate limits.**
   - Message: `503 ... This model is currently experiencing high demand`; and earlier
     `429 RESOURCE_EXHAUSTED ... limit: 0, model: gemini-2.5-flash-preview-image`.
   - Tried: switched off the image model; broadened `llm.py` retries to 5xx; added `SafeGenerator`;
     added `GEMINI_MODEL` override to switch to `flash-lite`.
   - Next: if sustained, wait, lower `N`, or try `gemini-2.0-flash`.
3. **`futs_test.py` won't run** — `ModuleNotFoundError: absl` (package not installed).
   - Next: `pip install --break-system-packages absl-py`, then `python futs_test.py`. `futs.py` is
     unchanged so tests should pass.
4. **LLM-generated candidates often fail to execute** (e.g. `NameError: name 'np_copy' is not defined`,
   `DuplicateError: 'Latitude' 2 times`). **This is expected/healthy** — bad candidates score `-inf`
   and the search continues. flash-lite fails more than flash.
5. **PDF preview unavailable in-tool** — `poppler` not installed (`pdftoppm`/`pdftotext` missing). PDFs
   are valid; open them directly. `brew install poppler` if needed.
6. **Two repeated-comparison code paths** — `compare_era_vs_bon.py --repeats` (mean±std) and the
   canonical `repeat_era_vs_bon.py`. Not a bug, but know which one you're using.
7. **macOS quirks:** `timeout` and `gh` CLI are not installed; old Intel Homebrew still at `/usr/local`
   (harmless); Rosetta not installed.

---

## 8. Remaining TODO

### High priority — GIFT-Eval Stage C: C1–C6B DONE (see §13). Decide next.
- **C6B done = the main POSITIVE GIFT-Eval result** (m4_hourly, naive seed): ERA 11.61→1.1384
  (beats naive AND seasonal-naive 1.1932); ERA beat best-of-N 1.1932 vs 1.3651. README + recap done.
- **Next options (pick one):**
  1. **Consolidate** Stage C (recommended): the story is complete — naive-dominated subset (m4_weekly)
     = tie + process win; seasonal subset (m4_hourly) = ERA beats naive AND best-of-N.
  2. **Optional N=20** ERA-vs-best-of-N on m4_hourly to firm up the BoN gap (small headroom remains;
     mainly strengthens the comparison).
  3. **Secondary dataset** `hospital/M/short` (MA ~0.81 beats naive ~0.97) for a 2nd positive case.
  4. **Optional richer reward:** return quantiles so CRPS (not just MASE) can be optimized.

### Secondary (toy-task track, paused)
- **Run the model ablation** (flash-lite vs flash): `python model_ablation.py` → outputs under
  `saved_runs/model_ablation/`; fold into `README.md` + `phase3_summary.md` and push. Deprioritised
  after the user pivoted to GIFT-Eval.

### Medium priority
- **Tighten the repeated result** — `--num_repeats 5` (or `--N 20`) for a less noisy mean±std, if the
  professor wants stronger evidence. File: `repeat_era_vs_bon.py`.
- **Install `absl-py` and run `futs_test.py`** to confirm `futs.py` integrity. Add a tiny smoke test
  for the new scripts (e.g., the monkeypatched no-network test pattern used during development).
- **Optionally record the model in `compare_era_vs_bon.py`'s `results.json`** (already done in
  `repeat_era_vs_bon.py`) for full reproducibility parity.

### Low priority
- **Migrate beyond the toy demos.** ✅ Done: Stage 2 = sklearn breast-cancer classification (ERA beats
  best-of-N 3/3). Next suggested order: another Kaggle Playground task → GIFT-Eval small subset →
  scRNA 20k-cell → larger paper-level tasks.
- **Replace the insecure sandbox** with real isolation (Docker/firejail/gVisor/VM) before any
  non-toy use. File: `sandbox.py` (keep the same `run()` interface).
- **Consider `.gitignore` for generated `local_train.csv`/`local_test.csv`** if they cause churn.
- **Decide whether to commit `Claude recap.md`** to the repo (currently uncommitted).

---

## 9. Testing Status

- **Existing tests:** `implementation/futs_test.py` (absl) — covers `compute_rank_scores`,
  `compute_pucts`, `backpropagate_visit`, and a mock `search`. **Not run** (absl-py missing); expected
  to pass since `futs.py` is unchanged.
- **Passing:** during development, all new code was smoke-tested with `py_compile` + monkeypatched
  **no-network** end-to-end runs (fake LLM/executor) that validated: JSON schemas, CSV headers, all
  plot files, `--plot-only`, `running_best` monotonicity, `_json_safe` `-inf→null`, `SafeGenerator`
  swallowing a simulated 503, and the transient-error classifier (503/429/500 retry, 400 does not).
- **Failing:** none known (the only "failures" are intentional LLM-candidate `-inf`s).
- **No coverage for:** `llm.py` network paths, `sandbox.py` subprocess execution, and the driver
  `main()` functions (only exercised via real runs / fakes).
- **Suggested next tests:** unit-test `running_best`, `decide_winner`, `count_failures`,
  `aggregate_running_bests`, and a `sandbox.run()` happy-path with a trivial program.

---

## 10. Dependencies and Environment Notes

- **OS:** macOS 26.5.1, **Apple Silicon (arm64)**, deca-core.
- **Python:** **3.12.13** via Homebrew at `/opt/homebrew/bin/python3.12`; `python`/`python3` resolve to
  it through `/opt/homebrew/opt/python@3.12/libexec/bin` (added to PATH in `~/.zshrc`). Packages were
  installed with `--break-system-packages` (Homebrew Python is PEP-668 "externally managed"); no venv.
- **Packages (brew 3.12):** pandas 3.0.4, numpy 2.5.0, scikit-learn 1.9.0, google-genai 2.10.0,
  matplotlib 3.11.0, tabulate 0.10.0. (NOT installed: `absl-py`.)
- **`google-genai` vs `google-generativeai`:** the code uses **`from google import genai`** =
  the **`google-genai`** package. `google-generativeai` was installed early but is unused.
- **Deprecated/old:** a separate CommandLineTools **Python 3.9** still exists with old packages — unused
  now; avoid it (EOL warnings).
- **LaTeX:** Tectonic at `/opt/homebrew/bin/tectonic` (self-contained). MacTeX not installed.
- **Required env vars:** `GEMINI_API_KEY` (required). `unset GOOGLE_API_KEY` first (SDK prefers it if
  both are set). Optional `GEMINI_MODEL` (default `gemini-2.5-flash-lite` in scripts).
- **External service:** Google Gemini API (free-tier key — flaky under load; 429/503 common).
- **Not installed:** `gh`, `timeout`, `poppler`, Rosetta 2.
- **Git remotes:** `origin` → upstream `google-research/era` (no push access); `repro` → user's repo
  `kyleAlexandad/Reproduction-of-AI-system-self-improving` (push via macOS keychain).

---

## 11. Important Decisions and Rationale

- **Native arm64 Homebrew + Python 3.12, not Rosetta.** ML code (numpy/sklearn) is much faster native;
  the broken `python3.10` was an x86_64 leftover from the old Intel Homebrew. Rejected: installing
  Rosetta to keep the Intel toolchain (slow for numerics).
- **`--break-system-packages` into brew Python instead of a venv.** The user wanted a plain `python
  script.py` workflow that "just works" with no activation step. Trade-off: pollutes the brew Python;
  acceptable for a personal toy repo. A venv remains the cleaner option for serious work.
- **Switched model `gemini-2.5-flash-image` → `gemini-2.5-flash` → user runs `gemini-2.5-flash-lite`.**
  The image model has **zero free-tier quota**; flash/flash-lite have real quota and are made for code.
  flash-lite is cheapest for repeated runs; **Pro deliberately avoided** (cost).
- **Reused one verified core across scripts (DRY)** and **copied (not refactored)** `INITIAL_CODE` /
  `PROBLEM_DESCRIPTION` to avoid breaking the original `playground_s3e1.py`.
- **`SafeGenerator` + broadened retries** so a transient 503 never throws away a multi-call run's
  already-paid API spend (we lost ~35 calls to a 503 crash before this).
- **Reports: Part 1/2 in English (LaTeX `article`), Chinese for summaries** (advisor-facing). Tectonic
  over MacTeX for a small, self-contained toolchain.
- **README at repo root + preserve upstream as `README_UPSTREAM.md`** (user-chosen): their reproduction
  is the front page, upstream attribution preserved (Apache-2.0; `LICENSE` intact).
- **Result framing kept honest:** ERA beat best-of-N in the single run and **3/3** repeats, but it's a
  small toy task with a ~0.023 gap — described as "consistent," not a tight statistical claim.

---

## 12. Suggested Next Prompt

Paste this into a fresh Claude Code session:

```
Read "Claude recap.md" in the project root first to load full context, especially §13
(GIFT-Eval Stage C) and the two-environment rule. This is the google-research/era
reproduction; ERA code is in /Users/zhangweikun/era/implementation.

GIFT-Eval Stage C C1–C6B are DONE (see §13). C6B is the main positive result: on m4_hourly from a
weak naive seed (11.61) ERA discovered period-24 seasonality and reached MASE 1.1384 (beats naive
AND seasonal-naive), and beat best-of-N (1.1932 vs 1.3651). Root README + recap updated.

Decide the next step with the user (see §8): (1) consolidate Stage C (recommended), (2) optional
N=20 ERA-vs-BoN on m4_hourly, (3) secondary dataset hospital/M/short, or (4) quantile/CRPS reward.

DOCS POLICY (user, 2026-06-30): do NOT create new per-stage md; only update this recap + root
README.md. NOTE: C5 + C6A + C6B doc/code changes are uncommitted (no push requested yet).

I (the user) will run anything that spends Gemini calls; DO NOT run Gemini/ERA/best-of-N yourself —
prepare code + give me exact commands. Keep N small; do not use gemini-2.5-pro. Run GIFT-Eval in a
normal terminal (native libs segfault under the agent sandbox).

Reminders: GIFT-Eval execution must use /Users/zhangweikun/era/gift-eval/.venv/bin/python and must
run OUTSIDE the agent sandbox (native libs segfault in-sandbox). Keep N small; do not use
gemini-2.5-pro. Secondary/optional TODO: the toy-task model_ablation.py (§8) still hasn't been run.
```

---

## 13. GIFT-Eval reproduction (Stage C) — the current track

> Goal of Stage C: take ERA beyond toy tabular tasks onto a real public benchmark
> (**GIFT-Eval**, time-series forecasting), one careful step at a time: **C1** stand up the
> official benchmark, **C2** wrap one small task as an ERA-scorable reward, **C3** run ERA tree
> search on it, (**C4+** = ERA-vs-best-of-N + more datasets, not started).

### 13.1 The two-environment rule (critical)
ERA needs `google-genai` + numpy 2.x; GIFT-Eval pins `gluonts 0.15.1` + numpy 1.26 → they CANNOT
share an env. So:
- **ERA env** = system brew Python (`/opt/homebrew/opt/python@3.12/libexec/bin/python`, i.e. plain
  `python` in a normal login shell). Runs `futs.py`, `llm.py`, `gift_eval_era_search.py`.
- **GIFT-Eval env** = its own venv `/Users/zhangweikun/era/gift-eval/.venv/bin/python`. Runs the
  C2 scorer `gift_eval_m4_weekly_task.py` and anything importing `gluonts`/`gift_eval`.
- **C3 bridges them via `subprocess`**, never by importing GIFT-Eval into the ERA env.
- The whole `/Users/zhangweikun/era/gift-eval/` clone (repo + `.venv` + downloaded `data/`) is
  **git-ignored** (`/gift-eval/` in the root `.gitignore`) so it never pollutes the ERA repo.

### 13.2 Locations
- Official clone: `/Users/zhangweikun/era/gift-eval` (from `SalesforceAIResearch/gift-eval`).
- GIFT-Eval venv: `/Users/zhangweikun/era/gift-eval/.venv` (Python 3.12.13; `gluonts 0.15.1`,
  `numpy 1.26.4`, `pandas 2.3.3`, `datasets 2.17.1`, `statsforecast 2.0.3`).
- Data: only `m4_weekly` downloaded (~1.5 MB) into `/Users/zhangweikun/era/gift-eval/data`;
  `GIFT_EVAL=...gift-eval/data` set in `gift-eval/.env`. (Full benchmark = 28 dataset families,
  98 configs — deliberately NOT downloaded.)
- The task used everywhere in Stage C: **`m4_weekly/W/short`** (359 univariate weekly series,
  `prediction_length=13`, `windows=1`, `season_length=1`).

### 13.3 C1 — setup + smoke test ✅
- Installed `pip install -e .` (core) + `statsforecast` (skipped heavy `[baseline]` torch/lightning).
- Ran the official **Naive** baseline on `m4_weekly` → reproduced `naive.ipynb` EXACTLY:
  MSE 453525.1459, **MASE 2.7773**, RMSE 673.44, CRPS 0.0609.
- Artifacts: `implementation/saved_runs/gift_eval_c1_setup/` (`README_C1.md`, `setup_notes.md`,
  `commands_used.sh`, `environment_info.txt`, `smoke_test_output.txt`, `all_results.csv`,
  `run_naive_smoke.py`). The smoke script also lives at `gift-eval/run_naive_smoke.py`.

### 13.4 C2 — ERA-scorable task wrapper ✅
- **`implementation/gift_eval_m4_weekly_task.py`** (run with the **GIFT-Eval venv**). Provides
  importable `load_task()`, `evaluate_candidate()`, `load_candidate_fn()`, plus a CLI:
  `--candidates <files...> --out-dir <dir>`.
- **Candidate interface** (what an LLM generates): a `.py` file defining
  `forecast(context, prediction_length, freq, metadata=None) -> 1D np.array[prediction_length]`.
  Only past `context` is given (no leakage). A `FunctionPredictor` adapts the point forecast into a
  degenerate gluonts `QuantileForecast` (all quantiles = the point) so the official `evaluate_model`
  metrics still compute.
- **Reward = `-MASE`** (GIFT-Eval is lower-better; ERA maximises). Invalid candidates (crash, wrong
  length, NaN/inf, no `forecast`) → `valid=False`, `reward=-inf`, with a clear error string.
- Baseline candidates + results on `m4_weekly/W/short`:
  - `candidate_naive` MASE **2.777295** (matches C1 to ~10 s.f.; reward -2.777295)
  - `candidate_moving_average` MASE 3.420637
  - `candidate_seasonal_naive` MASE 9.577987
  - (naive CRPS 0.0634 ≠ C1's 0.0609 — expected: degenerate quantiles collapse CRPS to ND; point
    metrics MASE/RMSE match. We optimise `-MASE`, so this is fine.)
- Artifacts: `implementation/saved_runs/gift_eval_c2_task_wrapper/` (`README_C2.md`,
  `candidate_results.{json,csv}`, `candidate_{naive,seasonal_naive,moving_average}.py`, `logs/`,
  `commands_used.sh`, `environment_info.txt`, `run_output.txt`).

### 13.5 C3 — ERA tree search over GIFT-Eval (engineering ✅; ERA not yet winning) ⏳
- **`implementation/gift_eval_era_search.py`** (run with the **ERA env** `python`). It:
  1. Scores the initial naive candidate locally (no Gemini) → `initial_score = -2.7773`.
  2. Runs the real `futs.search` (`Problem`/`Solution`/PUCT) for `--iterations` expansions.
  3. `generate_fn` builds a prompt (full candidate-interface spec + parent code + parent reward),
     calls `GeminiLLM.draw_sample`, returns a `Solution`. Hard Gemini failure → sentinel invalid
     candidate (SafeGenerator pattern).
  4. `execute_fn` writes the candidate to `candidates/cand_NNN.py` and scores it by
     **subprocess** → `<gift_eval_python> -u <scorer_script> --candidates ... --out-dir ...`,
     then robustly parses `candidate_results.json` (stdout-regex fallback) → reward.
  5. Tracks per-candidate `iteration, candidate_id, parent_id, valid, reward, MASE, CRPS, RMSE,
     error, candidate_path, runtime_s`.
- **CLI:** `--iterations` (5) · `--model` (`$GEMINI_MODEL` or `gemini-2.5-flash`) · `--out_dir`
  (`saved_runs/gift_eval_c3_era_smoke`) · `--gift_eval_python` · `--scorer_script` ·
  `--temperature` (reserved; `llm.py` ignores it) · `--seed` (0) · `--c_puct` (1.0).
- **Verified WITHOUT Gemini:** `python -m py_compile` passes; the cross-env `score_program()`
  plumbing was run on the naive seed → reward `-2.777295` (matches C2). Gemini/ERA NOT executed.
- **Outputs (written by the run):** `results.json`, `progress.csv`, `best_candidate.py`,
  `initial_candidate.py`, `candidates/`, `candidate_logs/`. Static deliverables already in
  `implementation/saved_runs/gift_eval_c3_era_smoke/`: `README_C3.md`, `commands_used.sh`,
  `initial_candidate.py`.
- **How to run it** (user, manual — spends `--iterations` Gemini calls): see §12 / the C3
  `commands_used.sh`. Run in a normal terminal (not the sandbox).

### 13.5b C3.1 — prompt postmortem + fixes (done; next run pending)
- The first **5-iteration smoke ran** (`saved_runs/gift_eval_c3_era_smoke/`) but **ERA did not beat
  naive** (best stayed at the seed, MASE 2.7773; 1 of 5 candidates invalid). Full analysis:
  `saved_runs/gift_eval_c3_era_smoke/postmortem.md`.
- Root causes: open-ended prompt let Gemini drift OFF the strong last-value anchor (it didn't know
  naive=2.7773 is strong, MA=3.42/seasonal=9.58 are worse); `metadata['season_length']` is **1**
  (the *metric* seasonality, NOT the data period) and was misused as a seasonal lag → one
  `IndexError` (`context[len-k+h]`) and degenerate behavior; compounding trend / full-history SES
  drift away from `last`.
- **Fixes (in `gift_eval_era_search.py`, C1/C2 untouched):** new default prompt **`conservative_v2`**
  (states measured baselines, "anchor on naive + small safe edits", warns season_length=1 is not a
  period, adds indexing-safety rule + a worked conservative template); kept old prompt as
  `--prompt_version baseline`; added `--initial_candidate PATH`; both recorded in `results.json`.
- Optional seed `implementation/initial_candidate_conservative.py` (naive + tiny capped damped
  trend) measured **MASE 2.8039** (slightly worse than naive on MASE, slightly better RMSE) → so
  **naive stays the default seed**; conservative seed is opt-in via `--initial_candidate`.
### 13.5c C3 FINAL (done) — ERA tied naive, did not beat it
- Three runs completed: smoke (5, baseline prompt), iter10 + iter20 (conservative_v2). Consolidated
  in **`saved_runs/gift_eval_c3_summary/`** (`README_C3_FINAL.md`, `c3_summary_table.csv`,
  `c3_best_so_far_mase.png`, `build_c3_summary.py`).
- Results (naive MASE = 2.7772950): best **generated** candidate (excl. seed) — smoke 2.7772950
  (degenerate exact-naive copy, 1/5 invalid); iter10 2.7773131 (dist 1.8e-5, 0 invalid); iter20
  **2.7772956 (dist 5.4e-7, 0 invalid)**. Best **including** seed = naive in all runs; ERA never
  beat naive.
- Key behaviour: conservative_v2 made Gemini first copy the worked template (MASE 2.8039), then
  FUTS built an improving chain (iter20 nodes 0→6→7→8→9→10) that shrank the trend weight toward
  naive (best node uses `weight_naive=0.9999`). So ERA refined *toward* naive correctly; the
  heuristic optimum here basically *is* naive.
- **Conclusion:** m4_weekly/W/short is a hard subset for simple hand-written heuristics because
  naive is near-optimal. conservative_v2 also drove invalid candidates to **0** (vs 1/5 baseline).

### 13.6 C4 — fair ERA vs best-of-N (RUN ✅)
- **`implementation/gift_eval_compare_era_vs_bon.py`** (ERA env). Reuses C3's `score_program`,
  `PROMPT_SPECS`, `build_generation_prompt`, `INITIAL_CANDIDATE_CODE` (small backward-compatible
  refactor of `gift_eval_era_search.py`: `score_program` got optional `candidates_dir`/`logs_dir`;
  added module-level `build_generation_prompt`).
- Both methods share model/prompt(conservative_v2)/naive-seed/scorer/reward and spend **N Gemini
  calls each**; only parent selection differs: **ERA** = FUTS tree-selected parent (can refine);
  **best-of-N** = always the same naive seed (independent). Seed scored ONCE, shared.
- **Framing:** naive is near-optimal (per C3), so C4 is a PROCESS comparison, not "beat naive".
- **Results (naive MASE = 2.77729505; "best generated" = excl. seed):**
  - **N=10** (`saved_runs/gift_eval_c4_era_vs_bon_N10/`): ERA best gen **2.77729505** (dist 0.0,
    valid/invalid 10/0) vs best-of-N **2.79541584** (dist 1.81e-2, 10/0). **Winner: ERA.** Neither
    beat naive.
  - **N=20** (`saved_runs/gift_eval_c4_era_vs_bon_N20/`): ERA best gen **2.77729505** (dist 0.0,
    valid/invalid 19/1) vs best-of-N **2.78183119** (dist 4.54e-3, 20/0). **Winner: ERA.** Neither
    beat naive.
  - Takeaway: ERA reached naive exactly; best-of-N stayed measurably above. ERA's tree-search
    refinement > independent sampling even where the baseline is unbeatable by simple heuristics.
- Outputs per run: `results.json`, `summary.csv`, `progress_era.csv`, `progress_bon.csv`,
  `era_candidates/`, `bon_candidates/`, `candidate_logs/`, `era_vs_bon_mase.png`, (`README_C4.md`,
  `commands_used.sh` in N10).

### 13.7 C5 — subset scouting (DONE) → C6 target found
- **`implementation/gift_eval_subset_scout.py`** (run with the **GIFT-Eval venv**; imports the C2
  wrapper). Evaluates 5 cheap baselines (naive, moving_average, seasonal_naive, damped_trend,
  ensemble) across small subsets via the official scorer (reward = -MASE). No Gemini/ERA/BoN.
- Downloaded 3 more small leaf datasets into `gift-eval/data`: `m4_hourly` (414 series, H),
  `hospital` (767, M), `covid_deaths` (266, D). (Skipped m4_daily/monthly/quarterly/yearly — too
  many series / slow.)
- **Findings (ratio = best-non-naive MASE / naive MASE; <1 means a simple method beats naive):**
  - `m4_weekly` ratio **1.002** → naive dominates (consistent with C3/C4). ❌
  - **`m4_hourly` ratio 0.103** (seasonal_naive 1.19 vs naive 11.61) → **strong C6 candidate.** ✅
  - **`hospital` ratio 0.841** (moving_average 0.81 vs naive 0.97) → **strong C6 candidate.** ✅
  - `covid_deaths` ratio 0.983 (damped_trend 46.10 vs 46.91) → marginal/possible. 🟡
- Outputs: `saved_runs/gift_eval_c5_subset_scout/` (`scout_results.{csv,json}`, `README_C5.md`,
  `commands_used.sh`, `environment_info.txt`, `baseline_mase_by_subset.png`, `relative_to_naive.png`,
  `scout_run_output.txt`).
- **Recommendation for C6: primary `m4_hourly/H/short`, secondary `hospital/M/short`.**

### 13.8 C6A — dataset-parametric scorer + ERA scripts (DONE)
- **`implementation/gift_eval_task.py`** = generalized scorer. CLI `--dataset --freq --term
  --candidates --out-dir`. Reuses the C2 machinery (load_task + FunctionPredictor + gluonts
  evaluate_model) from `gift_eval_m4_weekly_task.py` (kept intact for backward-compat). Same
  candidate interface + reward (-MASE); invalid → valid=false/-inf.
- **`gift_eval_era_search.py` + `gift_eval_compare_era_vs_bon.py`** now accept `--dataset/--freq/
  --term`; `score_program` got a `scorer_extra_args` param that forwards these to the scorer
  subprocess; default scorer switched to `gift_eval_task.py`. Prompts are now **dataset-aware**
  (`DATASET_FACTS` per dataset + generic fallback, built via `build_generation_prompt(...,
  dataset, freq, term, horizon)`). Seed is chosen by **`--initial_seed {naive,seasonal_naive}`,
  DEFAULT `naive` for EVERY dataset** (`SEED_LIBRARY` + `resolve_seed()`); `--initial_candidate PATH`
  overrides. (Earlier C6A had m4_hourly defaulting to seasonal-naive — CORRECTED per user: the main
  m4_hourly run must start from the weak naive seed.) `PROMPT_SPECS` dict replaced by
  `PROMPT_VERSIONS` list + builder functions; backward-compat aliases kept (INITIAL_CANDIDATE_CODE,
  DATASET_CONFIG, PREDICTION_LENGTH).
- **Validated (no Gemini):** `gift_eval_task.py` reproduces C5 EXACTLY on m4_hourly
  (naive 11.6077, seasonal_naive 1.1932, MA 11.2784, damped 12.0371, ensemble 11.6902) and on
  m4_weekly (naive 2.7773, seasonal 9.5780, MA 3.4206, ...). All py_compile + --help pass.
- Outputs: `saved_runs/gift_eval_c6a_parametric_scorer/` (`README_C6A.md`, `commands_used.sh`,
  `environment_info.txt`, 5 `candidate_*.py`, `hourly_results/` + `weekly_results/` each with
  `candidate_results.{json,csv}` + `logs/`).

### 13.9 C6B — ERA on m4_hourly from the weak naive seed (DONE; MAIN POSITIVE RESULT)
- **ERA-only, 10 iter, naive seed** (`saved_runs/gift_eval_c6b_era_m4_hourly_naive_iter10/`):
  initial naive MASE **11.6077** → ERA best **1.1384** (best candidate id 2), valid/invalid 11/0.
  The best candidate discovered **daily seasonality (period 24)**: seasonal-naive backbone (tile the
  last 24h) + a small DAMPED seasonal-difference correction (`context[-1]-context[-1-24]` decayed by
  0.95^h) + naive fallback for short series + NaN guard. This even **beats the seasonal-naive
  reference (1.1932)**.
- **ERA vs best-of-N, N=10, naive seed** (`saved_runs/gift_eval_c6b_era_vs_bon_m4_hourly_naive_N10/`):
  ERA best generated **1.1932** (reached pure seasonal-naive via an improving chain
  11.61→2.25→1.51→1.40→1.22→1.19) vs best-of-N **1.3651** (a noisier season-averaging approach that
  plateaued). Winner = **ERA**; both 10/0 valid; both beat the naive seed.
- **ERA vs best-of-N, N=20, naive seed** (`saved_runs/gift_eval_c6b_era_vs_bon_m4_hourly_naive_N20/`):
  SAME outcome with double the budget — ERA **1.1932** vs best-of-N **1.3651**; winner **ERA**; both
  20/0 valid; both beat naive. ERA also explored more distinct solutions (11 unique valid MASE values
  vs best-of-N's 6), i.e. more diverse search, not just luck. The ERA/BoN gap is stable in N.
- Significance: first GIFT-Eval subset where **ERA clears the naive baseline** (not just ties it) AND
  beats best-of-N — because here the strong structure (period-24 seasonality) is *discoverable* and
  naive is weak. Confirms the central mechanism claim on a real benchmark.
- Code cleanup applied: summary banners are now "GIFT-Eval ERA Search Summary" / "GIFT-Eval ERA vs
  Best-of-N Summary"; the compare metric `distance_to_naive_excl_seed`/`beat_naive` were renamed to
  `delta_vs_seed` (NEGATIVE = improvement) / `beat_seed` (+ `either_beat_seed`, `delta_vs_seed_note`).

### 13.6 GIFT-Eval gotchas for the next agent
- Use `hf download ... --include "<dataset>/*"` to add another small dataset (don't pull all 28).
- `huggingface-cli` is now `hf`; unauthenticated download works (rate-limit warning only).
- To score any candidate by hand (no Gemini):
  `/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_m4_weekly_task.py --candidates <f.py> --out-dir /tmp/x`.
- MASE/CRPS/etc. are all **lower-is-better**; ERA reward negates (`-MASE`).

---

## 14. Stage D0 — scRNA-seq batch integration reconnaissance (DONE; planning only)

> D0 = recon/planning ONLY. No Gemini, no ERA, no best-of-N, no dataset download, no new md files.
> Full narrative report was delivered in chat; this is the durable summary.

### 14.1 What exists locally
- **This repo root IS the official `google-research/era` clone** (`origin=google-research/era.git`),
  so all upstream task assets are already here.
- **The scRNA task notebook already exists:** `implementation/notebooks/single_cell_batch_integration.ipynb`
  (27 cells) — a complete ERA task with Overview, Input/Output format, `score()`, Begin/End mutable
  cells, and Validation. Sibling upstream task: `implementation/notebooks/flu-cornell-jhu-hierarchsir.ipynb`
  (a much LIGHTER pure-python probabilistic-forecast task, WIS metric, `pip install emcee pybind11 scipy`).
- **No scRNA data is downloaded** (no `.h5ad` present); no scRNA Python code beyond the notebook.

### 14.2 Task interface (what ERA would generate)
- Implement `eliminate_batch_effect_fn(adata: ad.AnnData, config: dict) -> ad.AnnData`, returning an
  AnnData whose batch-corrected embedding is stored in `obsm['X_emb']`.
- Input = **raw count** AnnData already subset to **2000 highly-variable genes**; batch in
  `obs['batch']`. Constraint: **MUST NOT use `cell_type`**; "minimize specialized single-cell
  packages" — only `scanpy` is allowed, otherwise native `sklearn/numpy/scipy/torch/jax/tf`.
- Expert hint baked into the prompt: a conditional VAE with adversarial (gradient-reversal) batch
  removal.

### 14.3 Reward / metric (opposite sign to GIFT-Eval)
- `score()` returns the **mean of 12 `scib` metrics**, each scaled to [0,1] via provided bounds then
  clipped: ASW-batch, ASW-label, ARI, NMI, graph-connectivity, isolated-label ASW, isolated-label F1,
  kBET, iLISI, cLISI, PCR, cell-cycle-conservation. **HIGHER is better** → ERA reward = score directly
  (no negation, unlike GIFT-Eval's `-MASE`). Reference: gemini-3-pro ERA best ≈ **0.677** in the
  notebook's own results table.

### 14.4 Dataset + small setting
- Data = Kaggle "single-cell-batch-integration" h5ad files expected under
  `./datasets/single-cell-batch-integration/` (train/val/solution + score bounds CSVs). Large (a
  typical matrix is 329,762 cells × 2000 genes).
- **The notebook itself subsamples to 20,000 cells (`target_size=20000, seed=42`)** — this IS the
  "20k-cell" toy setting. There is also a 100-cell dev smoke test (cell 17, TruncatedSVD).

### 14.5 Dependencies + conflicts → SEPARATE env
- Pinned (older): `anndata==0.11.3`, `scanpy==1.10.4`, `scib==1.1.7` (installed from git),
  `anndata2ri==1.3.2`, `rpy2`; plus **R** with `remotes` + **`kBET`** (GitHub). `torch/jax/tensorflow`
  are imported in the mutable cell (likely trimmable).
- **Hard conflict** with the ERA env (numpy **2.5.0**, no R) and with the gift-eval venv (numpy 1.26
  but different pins). scib 1.1.7 wants numpy<2. **R/Rscript are NOT installed on this machine.**
- **Decision: create a THIRD isolated env** `/Users/zhangweikun/era/scRNA-env` (recommend Python
  3.11). Never touch the ERA brew Python or the gift-eval venv. Add `/scRNA-env/` + `datasets/` to
  `.gitignore`.

### 14.6 Cost + risks (heavier than GIFT-Eval)
- Per-candidate scoring on 20k cells = kNN + 7× Leiden + optimal-resolution search + 12 metrics
  (kBET/LISI are slow) → **minutes per candidate**, and a VAE candidate also trains a torch model.
  So an ERA run here is far more expensive than a GIFT-Eval run (which scored in seconds).
- Top risks: (1) **R + rpy2 + kBET + anndata2ri** install on macOS arm64 (kBET is the only R-only
  metric); (2) old `scib 1.1.7` vs modern Python/numpy; (3) slow/memory-heavy scoring; (4) candidate
  torch/jax training time; (5) insecure sandbox executing LLM code; (6) Kaggle auth + large download.

### 14.7 Proposed D1 (de-risked, phased)
1. Create `scRNA-env` (Python 3.11) and install `anndata scanpy scib anndata2ri rpy2` (defer R).
2. Smoke-test the **candidate interface + a REDUCED python-only `score`** (drop kBET/R, and LISI if it
   won't build) on a **tiny SYNTHETIC AnnData** (random counts + fake batch/cell_type) — **no
   download**. Validates env + `X_emb` contract + a trivial baseline (e.g. normalize→log1p→PCA).
3. D2 (later, separate approval): install R+kBET, download the real subsampled Kaggle dataset, run the
   full 12-metric `score` on the 20k-cell subsample; then wire ERA + a scRNA-scorer subprocess bridge
   analogous to the GIFT-Eval two-env pattern.
- **Recommendation:** scRNA is the highest paper-fidelity next stage and worth doing via this phased
  de-risk; if R/kBET proves too painful, the **flu forecasting** notebook is a lighter upstream
  fallback (no R, pure pandas/numpy, and it also has published ERA-vs-BoN numbers).

### 14.8 D1 — scRNA-env + synthetic Python-only smoke test (DONE)
- **Env created:** `/Users/zhangweikun/era/scRNA-env` (Homebrew **Python 3.11.15** venv; installed
  `numpy 1.26.4` (<2), `anndata 0.11.3`, `scanpy 1.10.4`, `scib 1.1.7` + deps). Deferred R/kBET/
  anndata2ri/rpy2. This is the THIRD isolated env (never mix with ERA brew Python or gift-eval venv).
  Note: on this machine `brew` must be the arm64 one — use `/opt/homebrew/bin/brew` (the `/usr/local`
  Intel brew is broken: "Bad CPU type").
- **New script:** `implementation/scrna_synthetic_smoke.py` (run with the scRNA-env python). Fully
  synthetic — NO download, NO R, NO Gemini/ERA/BoN. Builds a 300-cell × 100-gene AnnData with 3
  batches + 3 cell types (cell-type signal AND batch effect injected via Poisson log-rates), runs 4
  candidates through the upstream interface `eliminate_batch_effect_fn(adata, config) -> obsm['X_emb']`
  (the harness pops `cell_type` before calling so candidates can't use it), scores each with a REDUCED
  Python-only proxy, and writes JSON+CSV. No markdown.
- **Reduced proxy (NOT scIB):** `score = bio_score + batch_mixing_score` (higher better), where
  `bio_score=(silhouette(cell_type)+1)/2` and `batch_mixing_score=1-(silhouette(batch)+1)/2`, plus kNN
  cross-checks (cell-type accuracy; batch balanced-accuracy-above-chance inverted). Clearly documented
  as a D1 smoke-test metric only; the official 12-metric scIB score (kBET/LISI/PCR/cell-cycle + R) is
  deferred to D2.
- **Result (verified by running it):** `batch_centered_pca` **1.3024** (bio 0.808, batch_mix 0.494)
  > raw `pca` **1.0661** (bio 0.705, batch_mix 0.361) — the batch correction correctly scores higher;
  both `invalid_missing_x_emb` and `invalid_wrong_shape` were correctly flagged `valid=False` with
  clear errors. **2/4 valid, 2/4 correctly-invalid.** Outputs:
  `saved_runs/scrna_d1_synthetic_smoke/results.{json,csv}`.
- **Command:** `cd /Users/zhangweikun/era/implementation && /Users/zhangweikun/era/scRNA-env/bin/python
  -u scrna_synthetic_smoke.py`.
- `/scRNA-env/` is now in the root `.gitignore` (added in D2A, like `/gift-eval/`).

### 14.9 D2A — synthetic ERA-scorable wrapper + tiny ERA controller (DONE; code-ready, Gemini NOT run)
- **Goal:** turn the D1 synthetic smoke test into the GIFT-Eval-style TWO-ENV pattern — an ERA-env
  controller that drives FUTS + Gemini, scoring each candidate by subprocess into the scRNA-env. Still
  SYNTHETIC ONLY (no real data, no R/kBET, no best-of-N, no large run).
- **New files:**
  - `implementation/scrna_synthetic_task.py` — the **scorer** (runs in scRNA-env). CLI
    `--candidate <f.py> [--candidate ...] --out-dir <dir>`. Reuses D1's deterministic
    `make_synthetic_adata` + `reduced_score` + `validate_output` (imported from
    `scrna_synthetic_smoke`). Loads `eliminate_batch_effect_fn`, pops `cell_type` before calling
    (candidate can't use it), validates `obsm['X_emb']`, scores. Writes `candidate_results.json`
    (`{meta, results:[...]}`, mirroring the GIFT-Eval scorer) + a parseable stdout line.
    **reward = reduced score (HIGHER better — NO negation, unlike GIFT-Eval's -MASE).**
  - `implementation/scrna_era_search.py` — the **ERA controller** (runs in ERA env; imports only
    `futs` + `llm`, NEVER scanpy/anndata). Seed = built-in log-norm **PCA baseline** (`PCA_SEED_CODE`).
    `score_program()` subprocesses `<scRNA-env python> -u scrna_synthetic_task.py --candidate ...
    --out-dir ...`, parses JSON (stdout-regex fallback), maps invalid→reward -inf. Runs real
    `futs.search`. Saves `results.json`, `progress.csv`, `best_candidate.py`, `initial_candidate.py`,
    `candidates/`, `candidate_logs/` under `saved_runs/scrna_d2a_synthetic_era_smoke/`. Prompt states
    all 10 rules (define `eliminate_batch_effect_fn`, set `X_emb`, don't use `cell_type`, may use
    `batch`, finite (n_cells,d), lightweight/no-NN, no downloads, numpy/scipy/scanpy/sklearn only).
- **Non-Gemini validation (all PASS):** py_compile both; controller imports in ERA env with no scanpy
  leak; scorer on 3 hand candidates in scRNA-env → `pca` 1.0699 (bio .726, batch_mix .344) <
  `batch_centered_pca` 1.3408 (bio .853, batch_mix .488), `invalid_missing_x_emb`→valid=False; and the
  **full bridge** (ERA-env `score_program` → scRNA-env subprocess) scored the PCA seed (reward 1.0699)
  and mapped a crashing candidate to reward -inf. (n_comps=10 here vs 20 in D1 → slightly different
  absolute numbers, same ordering.)
- **User runs the 5-iter Gemini smoke manually** (do NOT run it here):
  `cd /Users/zhangweikun/era/implementation && unset GOOGLE_API_KEY && export GEMINI_API_KEY="KEY" &&
  export GEMINI_MODEL=gemini-2.5-flash && python scrna_era_search.py --iterations 5 --model
  gemini-2.5-flash --out_dir saved_runs/scrna_d2a_synthetic_era_smoke`.
- **D2B (next, needs separate go-ahead):** real subsampled Kaggle data + real scIB score (add R/kBET
  incrementally), plus optional ERA-vs-best-of-N on the synthetic task.

### 14.10 D3A — real scRNA data path + tiny real-data smoke (DONE; data not downloaded)
- **Goal:** stand up the REAL-data path and smoke it on a tiny subset — no download, no R/kBET, no
  official scIB, no Gemini/ERA/BoN.
- **Notebook expectations (confirmed from `single_cell_batch_integration.ipynb`):** data dir
  `./datasets/single-cell-batch-integration/` relative to the notebook (=
  `implementation/notebooks/datasets/single-cell-batch-integration/`). Reduced smoke needs only the
  **`ffdaa1f0-b1d1-4135-8774-9fed7bf039ba-train-dataset.h5ad`** (raw counts in **`layers/counts`**,
  `obs['batch']` + `obs['cell_type']`). The `*-train-solution.h5ad` (normalized, in `layers/normalized`)
  + `score_{train,val}.median.bounds.csv` are only for the FULL scIB score (later). Notebook subsample
  = `subsample_adata(target_size=20000, seed=42)` (stratified by batch×cell_type); its dev test uses a
  RANDOM 100-cell subset (cell 17).
- **New file:** `implementation/scrna_realdata_smoke.py` (scRNA env). Locates the train `.h5ad`
  (searches `notebooks/datasets/...`, `implementation/datasets/...`, `./datasets/...`; `--data_file`
  override), loads it (prefers `layers['counts']`), RANDOM-subsamples to `--n_cells` (default 100),
  runs `candidate_pca` + `candidate_batch_centered_pca` via the reused D1 `run_candidate` harness,
  computes the reduced proxy score, writes `saved_runs/scrna_d3a_realdata_smoke/results.{json,csv}`.
  `--batch_key/--label_key` map alternate obs fields. If data is missing it exits(2) with a clear
  message: which file, where to place it, and the Kaggle download command.
- **`reduced_score` hardened** (in `scrna_synthetic_smoke.py`): the secondary kNN cross-check now uses
  adaptive CV and returns `None` if a class is too small for a tiny real subset (silhouette headline
  unchanged). Backward-compatible — D1 synthetic re-run is byte-identical (pca 1.0661, batch_centered
  1.3024).
- **Real dataset is NOT present** locally → running the smoke prints the download instructions and
  exits. The REAL code path (load `layers/counts` → subsample 100 → candidates → score) was validated
  on a FAKE h5ad built with the real structure: loaded 1500×200, subsampled 100 (4 batches, 5 types),
  `pca` 1.0961 < `batch_centered_pca` 1.3465, 2/2 valid. (Note: writing a synthetic h5ad under pandas
  3.0 needs `ad.settings.allow_write_nullable_strings=True`; this only affects *generating* test files,
  not reading the real one.)
- **Commands:** tiny smoke — `cd /Users/zhangweikun/era/implementation && /Users/zhangweikun/era/
  scRNA-env/bin/python -u scrna_realdata_smoke.py --n_cells 100`. Download (Kaggle "Single-Cell
  Biology", needs a Kaggle token): only the `*-train-dataset.h5ad` is required for this smoke; place it
  in `implementation/notebooks/datasets/single-cell-batch-integration/`.
- **Recommendation for D3B:** do the **real 20k-cell REDUCED-score baseline FIRST** (cheap, no R —
  just scale the same reduced proxy up from 100 to 20k cells to confirm memory/runtime), and defer the
  full **R/kBET scIB** setup to a later stage once the reduced real-data path is solid.
