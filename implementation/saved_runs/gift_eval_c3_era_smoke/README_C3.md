# C3 — ERA Tree Search over GIFT-Eval forecasting candidates (CODE READY, NOT YET RUN)

Stage C3 connects the C2 GIFT-Eval scorer to the existing ERA Flat-UCB Tree Search
(`futs.search`) and the Gemini generator (`llm.GeminiLLM`). **This stage only delivers the code +
the exact commands to run.** Per instructions, Gemini / ERA search has NOT been executed here.

Task: `m4_weekly/W/short` (weekly univariate forecasting, horizon 13). Reward = **-MASE**
(lower MASE → higher reward). Invalid candidates → **reward = -inf**, recorded cleanly.

## Two-environment architecture (the key design point)
- **`gift_eval_era_search.py` runs in the ERA env** (`python`, with `google-genai`, `futs.py`,
  `llm.py`). It **never imports any GIFT-Eval / gluonts package.**
- **Each candidate is scored in the GIFT-Eval venv via subprocess:**
  ```
  /Users/zhangweikun/era/gift-eval/.venv/bin/python -u \
      /Users/zhangweikun/era/implementation/gift_eval_m4_weekly_task.py \
      --candidates <cand.py> --out-dir <scratch>
  ```
  The reward is read back from the scorer's `candidate_results.json` (robust), with a
  stdout-regex fallback if the JSON is missing.

## ERA loop
Gemini writes/edits a `forecast(context, prediction_length, freq, metadata)` function →
the C2 scorer evaluates it with the official gluonts pipeline → reward = -MASE → FUTS selects a
parent by PUCT and iterates. The initial candidate is the C2 **naive baseline**
(`initial_candidate.py`, reward ≈ **-2.777295**); it is scored locally first (no Gemini call).

## How to run (MANUAL — do this in a normal terminal, not the agent sandbox)
See `commands_used.sh`. Smoke test (5 iterations):
```bash
cd /Users/zhangweikun/era/implementation
unset GOOGLE_API_KEY
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_MODEL=gemini-2.5-flash
python gift_eval_era_search.py --iterations 5 --model gemini-2.5-flash \
    --out_dir saved_runs/gift_eval_c3_era_smoke
```
Then 10 iterations into `saved_runs/gift_eval_c3_era_iter10`.

> Gemini calls = `--iterations` (one per expansion). The native GIFT-Eval libs segfault under the
> Cursor agent sandbox, so this must be run in a normal terminal (which is what you'd do anyway).

## CLI options
`--iterations` (5) · `--model` (`$GEMINI_MODEL` or `gemini-2.5-flash`) · `--out_dir`
(`saved_runs/gift_eval_c3_era_smoke`) · `--gift_eval_python`
(`/Users/zhangweikun/era/gift-eval/.venv/bin/python`) · `--scorer_script`
(`.../gift_eval_m4_weekly_task.py`) · `--temperature` (reserved; current `llm.py` ignores it) ·
`--seed` (0) · `--c_puct` (1.0).

## Outputs you should expect (written by the run into the out_dir)
```
saved_runs/gift_eval_c3_era_smoke/
├── results.json          # summary + per-iteration records
├── progress.csv          # one row per candidate (incl. the initial)
├── best_candidate.py     # best-scoring forecast() found
├── initial_candidate.py  # the naive seed (also written here statically)
├── candidates/           # cand_000.py (=initial), cand_001.py, ...
├── candidate_logs/       # per-candidate scorer out dir (candidate_results.json, stdout.txt, logs/)
├── README_C3.md          # this file
└── commands_used.sh      # the manual commands
```
Per-iteration record fields: `iteration, candidate_id, parent_id, valid, reward, MASE, CRPS, RMSE,
runtime_s, candidate_path, error`. Run-level: `initial_reward, initial_MASE, best_reward, best_MASE,
best_candidate_id, num_candidates_total, num_valid, num_invalid, improvement_over_initial`.

## Invalid-candidate handling
The C2 scorer already classifies crash / wrong-shape / NaN-inf / missing-`forecast` as
`valid=false`. `gift_eval_era_search.py` maps any of these (and subprocess timeout/failure, JSON
parse failure) to **reward = -inf**, logs the error string, and lets FUTS continue. A hard Gemini
failure (after `llm.py`'s retries) becomes a sentinel invalid candidate (`SafeGenerator` pattern),
so a transient outage never throws away the API spend already made.

## C3.1 update (after the first 5-iteration smoke test)
The first smoke test ran fine but ERA did **not** beat naive (best stayed at the naive seed,
MASE 2.7773). Full analysis in **`postmortem.md`**. Key changes made in C3.1:
- **New default prompt `conservative_v2`** — anchors on the measured baselines (naive 2.7773 is
  strong; MA 3.4206 worse; seasonal naive 9.5780 much worse), tells the model to make only small/
  safe edits to naive, warns that `metadata['season_length']` is 1 (metric seasonality, **not** a
  seasonal period — don't use it as a lag), and adds an indexing-safety requirement (the one invalid
  candidate crashed with an `IndexError` from `context[len - k + h]`).
- **New flags:** `--prompt_version {baseline,conservative_v2}` (default `conservative_v2`) and
  `--initial_candidate PATH` (default = built-in naive). Both are recorded in `results.json`.
- **Optional seed** `../../initial_candidate_conservative.py` (naive + tiny capped damped trend);
  measured MASE 2.8039 (slightly worse than naive on MASE), so **naive stays the default seed**.

Next run uses `conservative_v2` automatically (it is the default). Success = at least one valid
candidate with MASE < 2.777295.

## Status
Code is written and passes `python -m py_compile`; the cross-env scoring plumbing and the
conservative seed were verified locally (no Gemini). **Gemini/ERA search NOT executed here.**
Ready for the manual C3 conservative run.
