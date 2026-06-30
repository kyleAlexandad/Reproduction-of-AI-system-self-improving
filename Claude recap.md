
# Claude Recap

> Last updated: 2026-06-29. Repo: `/Users/zhangweikun/era` (git root).
> Working subfolder for all reproduction code: `/Users/zhangweikun/era/implementation`.
> Latest milestone: **Stage 2 (breast-cancer second benchmark) complete** — ERA beats best-of-N 3/3.
> Baseline commit on `repro/main`: `b14696a`; later commits add Stage 2 (see `git log`).

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

### Partially working / not yet run
- **`model_ablation.py`** (flash-lite vs flash) is written + smoke-tested but **NOT run for real** →
  `saved_runs/model_ablation/` does not exist yet. This is the top TODO.
- **`compare_era_vs_bon.py --repeats`** flag exists (an earlier mean±std version) but the *canonical*
  repeated path is now `repeat_era_vs_bon.py`. Both coexist.

### Broken / uncertain
- **Gemini availability is flaky:** intermittent `429` (rate limit) and `503 UNAVAILABLE`
  ("high demand") errors. Mitigated by retries + `SafeGenerator`, but a sustained outage will still
  make runs slow or all-`-inf`.
- **`futs_test.py` cannot run as-is:** it needs `absl-py`, which is **not installed**.

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

### High priority
- **Run the model ablation** → answers the user's open question (stick with flash-lite or switch to
  flash). `python model_ablation.py`. Files: `model_ablation.py`, outputs under
  `saved_runs/model_ablation/`. Then fold results into `README.md` + `phase3_summary.md` and push.
- **After ablation, update + push docs** like we did for the repeated run (edit `README.md` §5/§6 and
  `saved_runs/phase3_summary.md`, then `git push repro main`).

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
Read "Claude recap.md" in the project root first to load full context. This is the
google-research/era reproduction; all code is in /Users/zhangweikun/era/implementation,
and python = the Homebrew arm64 Python 3.12.

Then continue from the HIGH-priority TODOs in section 8:

1. Run the model ablation to decide flash-lite vs flash:
     cd /Users/zhangweikun/era/implementation
     unset GOOGLE_API_KEY
     export GEMINI_API_KEY="<my key>"
     python model_ablation.py
   (I will run any command that needs my API key; tell me exactly what to run.)

2. When it finishes, inspect saved_runs/model_ablation/{results.json,summary.md} and the plot,
   then fold the real numbers into README.md (section 5/6) and
   implementation/saved_runs/phase3_summary.md, and push to the `repro` remote
   (git add -A && git commit && git push repro main).

Before running anything, re-verify the current files match the recap (especially llm.py,
compare_era_vs_bon.py, repeat_era_vs_bon.py, model_ablation.py) and tell me if anything drifted.
Keep N small (API costs money); do not use gemini-2.5-pro.
```
