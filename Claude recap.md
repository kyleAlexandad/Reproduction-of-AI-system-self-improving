
# Claude Recap

> Last updated: 2026-06-30. Repo: `/Users/zhangweikun/era` (git root).
> Working subfolder for all reproduction code: `/Users/zhangweikun/era/implementation`.
> Latest milestone: **GIFT-Eval Stage C COMPLETE (C1–C4)** — C1 setup ✅, C2 scorer wrapper ✅,
> C3 ERA tree search ✅ (naive near-optimal, ERA only tied it, best gen within 5.4e-7), **C4 ERA vs
> best-of-N RUN ✅** (N=10 and N=20: neither beat naive, but **ERA's best generated candidate beat
> best-of-N's both times** — ERA reached naive exactly while BoN stayed ≈2.795/2.782). The root
> **README.md was updated** with the full Stage-3 GIFT-Eval section + Final Takeaways. Earlier:
> Stage 1 (regression) + Stage 2 (breast cancer) complete (ERA beat best-of-N 3/3 each).
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

### High priority — GIFT-Eval Stage C is COMPLETE (C1–C4 all done; see §13)
- C1/C2/C3/C4 done; root README updated with the full Stage-3 section. Nothing pending in Stage C.
- **Next dataset (recommended):** pick a small GIFT-Eval config where naive is *weaker* (e.g.
  `bizitobs_l2c/H`, or an `ett*`/`electricity` config) so ERA has real headroom to BEAT naive (not
  just tie it). Reuse the same wrapper+scorer: download via
  `hf download Salesforce/GiftEval --include "<ds>/*" --local-dir gift-eval/data`, then point the
  C2/C3/C4 scripts at it (currently hardcoded to `m4_weekly`; parametrise `DATASET_NAME`/`TERM`).
- **Optional richer reward:** extend the candidate interface to return quantiles so CRPS (not just
  MASE) can be optimized.

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

GIFT-Eval Stage C (C1–C4) is COMPLETE and the root README is updated (see §13). On
m4_weekly/W/short naive is near-optimal: ERA tied it and beat best-of-N both at N=10 and N=20, but
neither beat naive. The recommended next step is a NEW GIFT-Eval dataset where naive is weaker so
ERA can actually beat it (see §8). To add one:

  # download a small config (example), then parametrise DATASET_NAME/TERM in the gift_eval_* scripts
  /Users/zhangweikun/era/gift-eval/.venv/bin/python -m pip --version  # sanity
  cd /Users/zhangweikun/era/gift-eval && source .venv/bin/activate
  hf download Salesforce/GiftEval --repo-type=dataset --include "bizitobs_l2c/*" --local-dir ./data

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

### 13.6 GIFT-Eval gotchas for the next agent
- Use `hf download ... --include "<dataset>/*"` to add another small dataset (don't pull all 28).
- `huggingface-cli` is now `hf`; unauthenticated download works (rate-limit warning only).
- To score any candidate by hand (no Gemini):
  `/Users/zhangweikun/era/gift-eval/.venv/bin/python -u gift_eval_m4_weekly_task.py --candidates <f.py> --out-dir /tmp/x`.
- MASE/CRPS/etc. are all **lower-is-better**; ERA reward negates (`-MASE`).
