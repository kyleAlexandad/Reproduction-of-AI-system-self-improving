# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C3: ERA / FUTS tree search over GIFT-Eval forecasting candidates (m4_weekly/W/short).

This is the third GIFT-Eval stage. It connects the C2 scorer
(``gift_eval_m4_weekly_task.py``) to the existing ERA Flat-UCB Tree Search
(``futs.search``) and a Gemini generator (``llm.GeminiLLM``).

ARCHITECTURE (two Python environments, on purpose)
--------------------------------------------------
ERA and GIFT-Eval have *incompatible* Python environments, so:

  * THIS script runs in the ERA implementation env (the one with ``google-genai``,
    ``futs.py`` and ``llm.py``). It NEVER imports any GIFT-Eval / gluonts package.
  * Each candidate is scored by launching the C2 scorer in the GIFT-Eval venv via
    ``subprocess``:
        <gift_eval_python> -u <scorer_script> --candidates <cand.py> --out-dir <scratch>
    and the reward is read back robustly from the scorer's ``candidate_results.json``
    (with a stdout-regex fallback).

ERA loop: Gemini writes/edits a ``forecast(context, prediction_length, freq, metadata)``
function -> the C2 scorer evaluates it with the official gluonts pipeline -> reward = -MASE
-> FUTS selects a parent and iterates.

Reward convention: ``reward = -MASE`` (GIFT-Eval is lower-is-better; ERA/FUTS maximise).
Invalid candidates (crash / wrong shape / NaN / no ``forecast``) get ``reward = -inf`` and
are recorded cleanly so the search continues.

============================ SECURITY WARNING ============================
The C2 scorer executes LLM-generated Python directly (no isolation). Toy use only.
=========================================================================

NOTE: do NOT run this without a GEMINI_API_KEY; it makes real Gemini calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import futs
from llm import GeminiLLM, DEFAULT_MODEL

# --- Defaults (match the C1/C2 setup) ----------------------------------------------------
DEFAULT_GIFT_EVAL_PYTHON = "/Users/zhangweikun/era/gift-eval/.venv/bin/python"
DEFAULT_SCORER_SCRIPT = (
    "/Users/zhangweikun/era/implementation/gift_eval_m4_weekly_task.py"
)
DEFAULT_OUT_DIR = "saved_runs/gift_eval_c3_era_smoke"

DATASET_CONFIG = "m4_weekly/W/short"
PREDICTION_LENGTH = 13
FREQ = "W"

# The C2 naive baseline -- the initial candidate (reward ~ -2.777295).
INITIAL_CANDIDATE_CODE = '''\
import numpy as np


def forecast(context, prediction_length, freq, metadata=None):
    context = np.asarray(context, dtype=float)
    if len(context) == 0:
        return np.zeros(prediction_length)
    return np.repeat(context[-1], prediction_length)
'''

PROBLEM_DESCRIPTION = (
    "Improve a univariate time-series forecasting function for the GIFT-Eval "
    f"`{DATASET_CONFIG}` task (weekly data, forecast horizon {PREDICTION_LENGTH}). "
    "The reward is -MASE (higher is better)."
)

# --- The candidate-interface prompt shown to Gemini --------------------------------------
# Shared interface contract (identical across prompt versions).
BASE_INTERFACE_SPEC = f'''\
TASK: GIFT-Eval `{DATASET_CONFIG}` -- weekly univariate time-series forecasting.
For each series you are given its past values and must forecast the next {PREDICTION_LENGTH} steps
(the horizon is EXACTLY {PREDICTION_LENGTH}; freq is weekly, "{FREQ}").

Write a SINGLE Python function with EXACTLY this signature:

    def forecast(context, prediction_length, freq, metadata=None):
        ...

INPUTS:
  - context: a 1D numpy array of the PAST target values for one series (history only).
  - prediction_length: int, the forecast horizon (will be {PREDICTION_LENGTH}).
  - freq: a frequency string, e.g. "{FREQ}".
  - metadata: an optional dict that may contain item_id, season_length, context_length,
    prediction_length, freq.
    IMPORTANT: metadata['season_length'] is the METRIC seasonality used by the scorer and equals 1
    for this weekly task. It is NOT the data's real seasonal period. Do NOT use it as a seasonal
    lag (doing so, e.g. context[len(context) - season_length + h], overflows the array).

OUTPUT:
  - Return a 1D numpy array of length EXACTLY prediction_length with the point forecasts.

HARD REQUIREMENTS (a violation makes the candidate invalid, reward = -inf):
  1. Define a function named exactly `forecast` with the signature above.
  2. Use ONLY the past `context`. NEVER use or fabricate future label values.
  3. Output length must be EXACTLY prediction_length.
  4. Never output NaN or inf. Clip/guard as needed.
  5. CPU-only and lightweight. No GPU.
  6. Do NOT download data, read files, or call any network/external service.
  7. Do NOT require heavy packages. Standard library + numpy (and pandas if needed) only.
  8. Handle SHORT context arrays robustly (length 0, 1, or a few points) with sane fallbacks.
  9. INDEXING SAFETY: never write patterns like context[len(context) - k + h] that can run past the
     end of the array. Prefer slicing from the END (context[-k:]) and bound-check BOTH ends.
     (A previous candidate crashed with IndexError doing exactly this.)
'''

# v1 ("baseline"): the original open-ended strategy list (kept for A/B comparison).
BASELINE_STRATEGY = '''\
OBJECTIVE: reward = `-MASE` (negated Mean Absolute Scaled Error), so HIGHER reward is better
(you want the SMALLEST MASE). Produce a function expected to beat the parent shown below.

STRATEGY IDEAS you may use or combine (be creative, but keep it robust):
  - last-value naive (strong baseline for weekly random-walk-like series)
  - moving average / weighted moving average of recent points
  - linear or robust trend extrapolation (with damping to avoid runaway forecasts)
  - exponential smoothing / robust smoothing
  - seasonal lag (e.g. weekly seasonal period) IF there is enough history, else fallback
  - small ensembles / blends of the above
  - clipping extreme forecasts to the recent observed range
  - explicit fallback rules for very short series

Return ONLY the Python code for the `forecast` function (plus any imports it needs).
Do not include explanations, examples, tests, or a __main__ block.
'''

# v2 ("conservative_v2"): anchored on the measured baselines; pushes SMALL, SAFE edits to naive.
CONSERVATIVE_STRATEGY = f'''\
OBJECTIVE: reward = `-MASE` (negated Mean Absolute Scaled Error), so HIGHER reward means LOWER MASE.
You want the SMALLEST possible MASE.

CRITICAL BENCHMARK FACTS (measured on THIS exact task -- trust them):
  - This is GIFT-Eval {DATASET_CONFIG}. Prediction horizon = {PREDICTION_LENGTH}.
  - Reward = -MASE, so higher reward means lower MASE.
  - The NAIVE last-value baseline is STRONG here: MASE ~= 2.7773. Beating it is genuinely hard.
  - In baseline tests on this task:
      * moving average was WORSE:       MASE ~= 3.4206
      * seasonal naive was MUCH WORSE:  MASE ~= 9.5780
  - These weekly series behave close to a RANDOM WALK: the last observed value is the single best
    simple predictor. A 52-week seasonal lag and aggressive/long trend extrapolation HURT badly.

THEREFORE -- BE CONSERVATIVE. Do NOT blindly replace naive with moving average, seasonal naive, or
aggressive trend extrapolation. Treat last-value naive as the ANCHOR and make only SMALL, SAFE
corrections on top of it.

RECOMMENDED CONSERVATIVE STRATEGIES:
  - Keep last-value naive as the backbone and the fallback.
  - Add at most a SMALL damped trend, and only when the recent trend is stable and small. Heavily
    damp it (e.g. phi <= 0.7) and CAP the per-step change to a small fraction of the recent scale,
    so forecasts cannot run away from the last value.
  - Use robust clipping: keep forecasts inside a band around recent observations
    (e.g. recent min/max +/- a small margin).
  - Prefer conservative ensembles/blends where NAIVE keeps a HIGH weight (e.g. 0.75-0.9 naive).
  - Do NOT use a 52-week (or any large) seasonal lag for this data.
  - Handle short series safely; never index past the array; never return NaN/inf.

A GOOD CONSERVATIVE TEMPLATE (adapt/improve it, but stay close to naive):

    import numpy as np
    def forecast(context, prediction_length, freq, metadata=None):
        context = np.asarray(context, dtype=float)
        n = len(context)
        if n == 0:
            return np.zeros(prediction_length, dtype=float)
        last = float(context[-1])
        if n < 4:
            return np.full(prediction_length, last, dtype=float)
        recent = context[-4:]
        slope = float(np.mean(np.diff(recent)))
        scale = float(np.std(context[-13:])) if n >= 13 else float(np.std(context))
        slope = float(np.clip(slope, -0.1 * scale, 0.1 * scale))  # tiny correction only
        phi = 0.6
        steps = np.arange(1, prediction_length + 1)
        out = last + slope * np.cumsum(phi ** steps)
        lo = float(np.min(context[-13:])) if n >= 13 else float(np.min(context))
        hi = float(np.max(context[-13:])) if n >= 13 else float(np.max(context))
        span = hi - lo
        out = np.clip(out, lo - 0.25 * span, hi + 0.25 * span)
        return np.nan_to_num(out, nan=last, posinf=hi, neginf=lo).astype(float)

Aim for a SMALL improvement (MASE ~2.70-2.75 would be meaningful). It is better to TIE naive than to
diverge far above it. Return ONLY the Python code for `forecast` (plus imports). No explanations,
examples, tests, or __main__ block.
'''

# Selectable via --prompt_version.
PROMPT_SPECS = {
    "baseline": BASE_INTERFACE_SPEC + "\n" + BASELINE_STRATEGY,
    "conservative_v2": BASE_INTERFACE_SPEC + "\n" + CONSERVATIVE_STRATEGY,
}


def build_generation_prompt(prompt_version: str, parent_program: str, parent_score) -> str:
    """Assemble the full generation prompt for a given parent (shared by C3 search and C4 compare)."""
    if parent_score is not None and math.isfinite(parent_score):
        parent_mase = f"{-parent_score:.6f}"
        parent_reward = f"{parent_score:.6f}"
    else:
        parent_mase, parent_reward = "unknown (invalid)", "-inf"
    spec = PROMPT_SPECS[prompt_version]
    return (
        f"{spec}\n\n"
        f"--- CURRENT (PARENT) CANDIDATE ---\n"
        f"# parent reward (=-MASE): {parent_reward}   (parent MASE: {parent_mase})\n"
        f"{parent_program}\n"
        f"--- END PARENT CANDIDATE ---\n\n"
        f"Write an IMPROVED `forecast` function that should achieve a LOWER MASE "
        f"(higher reward) than the parent above, following the guidance. "
        f"Return ONLY the Python code."
    )


class CandidateRecord:
    """Lightweight per-candidate record (also the row we serialise)."""

    __slots__ = (
        "iteration",
        "candidate_id",
        "parent_id",
        "valid",
        "reward",
        "MASE",
        "CRPS",
        "RMSE",
        "error",
        "candidate_path",
        "runtime_s",
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))

    def as_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


def _finite_or_neg_inf(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("-inf")
    return v if math.isfinite(v) else float("-inf")


def score_program(
    program: str,
    candidate_id: int,
    out_dir: Path,
    gift_eval_python: str,
    scorer_script: str,
    timeout_s: int = 300,
    candidates_dir=None,
    logs_dir=None,
) -> dict:
    """Write a candidate program to disk and score it via the GIFT-Eval venv subprocess.

    Returns a dict with keys: valid, reward, MASE, CRPS, RMSE, error, candidate_path,
    runtime_s. Robust: never raises; any failure -> valid=False, reward=-inf.

    `candidates_dir` / `logs_dir` can be overridden (used by C4 to separate ERA vs best-of-N
    candidate files); defaults preserve the C3 layout (out_dir/candidates, out_dir/candidate_logs).
    """
    candidates_dir = Path(candidates_dir) if candidates_dir is not None else (out_dir / "candidates")
    logs_dir = (
        Path(logs_dir) if logs_dir is not None
        else (out_dir / "candidate_logs" / f"cand_{candidate_id:03d}")
    )
    candidates_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cand_path = candidates_dir / f"cand_{candidate_id:03d}.py"
    cand_path.write_text(program)

    result = {
        "valid": False,
        "reward": float("-inf"),
        "MASE": None,
        "CRPS": None,
        "RMSE": None,
        "error": None,
        "candidate_path": str(cand_path),
        "runtime_s": None,
    }

    cmd = [
        gift_eval_python,
        "-u",
        scorer_script,
        "--candidates",
        str(cand_path),
        "--out-dir",
        str(logs_dir),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        result["error"] = f"TIMEOUT: scorer exceeded {timeout_s}s"
        result["runtime_s"] = round(time.time() - t0, 3)
        (logs_dir / "stdout.txt").write_text("TIMEOUT\n")
        return result
    except Exception as e:  # subprocess could not start
        result["error"] = f"SUBPROCESS_ERROR: {type(e).__name__}: {e}"
        result["runtime_s"] = round(time.time() - t0, 3)
        return result

    result["runtime_s"] = round(time.time() - t0, 3)
    (logs_dir / "stdout.txt").write_text(
        (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    )

    # --- Primary: parse the scorer's JSON file ---
    json_path = logs_dir / "candidate_results.json"
    rec = None
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text())
            results = data.get("results") or []
            if results:
                rec = results[0]
        except Exception as e:
            result["error"] = f"JSON_PARSE_ERROR: {e}"

    # --- Fallback: parse reward/MASE from stdout if JSON unavailable ---
    if rec is None and proc.stdout:
        m_valid = re.search(r"valid=(True|False)", proc.stdout)
        m_mase = re.search(r"MASE=([-+0-9.einf]+)", proc.stdout)
        if m_valid:
            rec = {
                "valid": (m_valid.group(1) == "True"),
                "MASE": float(m_mase.group(1)) if m_mase else None,
                "CRPS": None,
                "RMSE": None,
                "reward": None,
                "error": "parsed-from-stdout (json missing)",
            }

    if rec is None:
        if not result["error"]:
            tail = (proc.stderr or proc.stdout or "")[-500:]
            result["error"] = f"NO_RESULT (returncode={proc.returncode}): {tail}"
        return result

    # Normalise the scorer record into our reward.
    valid = bool(rec.get("valid"))
    mase = rec.get("MASE")
    reward = rec.get("reward")
    if reward is None and mase is not None and math.isfinite(float(mase)):
        reward = -float(mase)
    result.update(
        valid=valid,
        reward=_finite_or_neg_inf(reward) if valid else float("-inf"),
        MASE=mase,
        CRPS=rec.get("CRPS"),
        RMSE=rec.get("RMSE"),
        error=rec.get("error"),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of ERA expansions (Gemini calls). Default 5.")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
                        help="Gemini model. Default: $GEMINI_MODEL or gemini-2.5-flash.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt_version", default="conservative_v2",
                        choices=sorted(PROMPT_SPECS.keys()),
                        help="Prompt strategy. 'conservative_v2' (default) anchors on the measured "
                             "baselines and pushes small/safe edits to naive; 'baseline' is the "
                             "original open-ended prompt (for A/B comparison).")
    parser.add_argument("--initial_candidate", default=None,
                        help="Path to a candidate .py to SEED the search. Default: the built-in "
                             "naive last-value baseline (recommended clean anchor).")
    parser.add_argument("--gift_eval_python", default=DEFAULT_GIFT_EVAL_PYTHON)
    parser.add_argument("--scorer_script", default=DEFAULT_SCORER_SCRIPT)
    parser.add_argument("--temperature", type=float, default=None,
                        help="(Reserved) sampling temperature; current llm.py ignores it.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c_puct", type=float, default=1.0)
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # --- Preconditions ---
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n"
              "  unset GOOGLE_API_KEY\n"
              '  export GEMINI_API_KEY="my_key"\n'
              "then re-run.")
        sys.exit(1)
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] GOOGLE_API_KEY is also set; the SDK may prefer it. "
              "For a clean run, `unset GOOGLE_API_KEY` first.")
    gift_py = Path(args.gift_eval_python)
    if not gift_py.exists():
        print(f"[!] GIFT-Eval python not found: {gift_py}")
        sys.exit(1)
    if not Path(args.scorer_script).exists():
        print(f"[!] Scorer script not found: {args.scorer_script}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(exist_ok=True)
    (out_dir / "candidate_logs").mkdir(exist_ok=True)

    # Resolve the initial (seed) candidate.
    if args.initial_candidate:
        initial_code = Path(args.initial_candidate).read_text()
        initial_name = Path(args.initial_candidate).name
    else:
        initial_code = INITIAL_CANDIDATE_CODE
        initial_name = "naive (built-in last-value)"

    print(f"Out dir:          {out_dir}")
    print(f"Model:            {args.model}")
    print(f"Iterations:       {args.iterations}")
    print(f"Prompt version:   {args.prompt_version}")
    print(f"Initial seed:     {initial_name}")
    print(f"GIFT-Eval python: {args.gift_eval_python}")
    print(f"Scorer script:    {args.scorer_script}")
    print(f"Reward:           -MASE  (higher is better)\n")

    llm = GeminiLLM(api_key, model_name=args.model)

    # --- Tracking state ---
    records: list[CandidateRecord] = []
    # Map id(Solution object) -> candidate_id, so the generator can resolve its parent's id.
    sol_to_id: dict[int, int] = {}
    next_id = {"v": 0}

    # --- 1) Score the initial (seed) candidate ---
    (out_dir / "initial_candidate.py").write_text(initial_code)
    print(f"=== Scoring initial candidate ({initial_name}) ===")
    init_id = next_id["v"]; next_id["v"] += 1  # 0
    init_res = score_program(
        initial_code, init_id, out_dir,
        args.gift_eval_python, args.scorer_script,
    )
    initial_solution = futs.Solution(initial_code)
    sol_to_id[id(initial_solution)] = init_id
    initial_score = init_res["reward"]
    records.append(CandidateRecord(
        iteration=0, candidate_id=init_id, parent_id=None,
        valid=init_res["valid"], reward=init_res["reward"], MASE=init_res["MASE"],
        CRPS=init_res["CRPS"], RMSE=init_res["RMSE"], error=init_res["error"],
        candidate_path=init_res["candidate_path"], runtime_s=init_res["runtime_s"],
    ))
    print(f"  initial: valid={init_res['valid']} reward={initial_score} "
          f"MASE={init_res['MASE']}\n")
    if not init_res["valid"]:
        print("[!] Initial candidate did not score as valid. Check the GIFT-Eval venv / data. "
              "Aborting before spending Gemini calls.")
        _write_outputs(out_dir, records, args, initial_score, initial_name=initial_name,
                       aborted=True)
        sys.exit(1)

    # --- 2) Generator: build prompt from parent, call Gemini, return a Solution ---
    def generate_fn(problem, parent_solution, parent_score):
        parent_id = sol_to_id.get(id(parent_solution))
        prompt = build_generation_prompt(
            args.prompt_version, parent_solution.program, parent_score
        )
        try:
            code = llm.draw_sample(prompt)
        except Exception as e:  # SafeGenerator behaviour: never crash the whole run
            print(f"  [!] Generation failed after retries; recording -inf candidate. "
                  f"({str(e)[:100]})")
            code = "# generation failed - intentionally invalid candidate (no forecast)\n"
        sol = futs.Solution(code)
        cand_id = next_id["v"]; next_id["v"] += 1
        sol_to_id[id(sol)] = cand_id
        # Stash parent for the executor (paired generate->execute in futs.search).
        generate_fn._pending = (cand_id, parent_id)  # type: ignore[attr-defined]
        return sol

    # --- 3) Executor: score the generated Solution via subprocess ---
    def execute_fn(problem, solution):
        cand_id = sol_to_id.get(id(solution))
        parent_id = None
        pending = getattr(generate_fn, "_pending", None)
        if pending and pending[0] == cand_id:
            parent_id = pending[1]
        res = score_program(
            solution.program, cand_id, out_dir,
            args.gift_eval_python, args.scorer_script,
        )
        rec = CandidateRecord(
            iteration=cand_id, candidate_id=cand_id, parent_id=parent_id,
            valid=res["valid"], reward=res["reward"], MASE=res["MASE"],
            CRPS=res["CRPS"], RMSE=res["RMSE"], error=res["error"],
            candidate_path=res["candidate_path"], runtime_s=res["runtime_s"],
        )
        records.append(rec)
        status = "valid" if res["valid"] else f"INVALID ({str(res['error'])[:60]})"
        print(f"  [ERA] iter {cand_id}/{args.iterations}: parent={parent_id} "
              f"reward={res['reward']} MASE={res['MASE']} [{status}]")
        return res["reward"]

    # --- 4) Run FUTS tree search (the real ERA loop) ---
    problem = futs.Problem(PROBLEM_DESCRIPTION)
    print(f"=== ERA / FUTS tree search ({args.iterations} iterations) ===")
    best_solution, best_score = futs.search(
        problem=problem,
        initial_solution=initial_solution,
        initial_score=initial_score,
        generate_fn=generate_fn,
        execute_fn=execute_fn,
        num_iterations=args.iterations,
        c_puct=args.c_puct,
    )

    # Persist the best candidate program.
    (out_dir / "best_candidate.py").write_text(best_solution.program)

    _write_outputs(out_dir, records, args, initial_score, initial_name=initial_name,
                   best_score=best_score)


def _write_outputs(out_dir, records, args, initial_score, initial_name="naive (built-in)",
                   best_score=None, aborted=False):
    """Write results.json, progress.csv and a run summary."""
    valid_records = [r for r in records if r.valid]
    invalid_records = [r for r in records if not r.valid]
    rewards = [r.reward for r in records if r.reward is not None and math.isfinite(r.reward)]
    best_reward = max(rewards) if rewards else float("-inf")
    best_rec = None
    for r in records:
        if r.reward is not None and math.isfinite(r.reward) and r.reward == best_reward:
            best_rec = r
            break

    def _safe(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    # progress.csv
    csv_path = out_dir / "progress.csv"
    fields = ["iteration", "candidate_id", "parent_id", "valid", "reward",
              "MASE", "CRPS", "RMSE", "runtime_s", "candidate_path", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = r.as_dict()
            row["reward"] = "" if (row["reward"] is None or not math.isfinite(row["reward"])) else row["reward"]
            w.writerow(row)

    # results.json
    results = {
        "dataset": DATASET_CONFIG,
        "prediction_length": PREDICTION_LENGTH,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "initial_candidate": initial_name,
        "iterations_requested": args.iterations,
        "reward_definition": "reward = -MASE (lower MASE -> higher reward)",
        "aborted": aborted,
        "initial_reward": _safe(initial_score),
        "initial_MASE": (records[0].MASE if records else None),
        "best_reward": _safe(best_reward),
        "best_MASE": (best_rec.MASE if best_rec else None),
        "best_candidate_id": (best_rec.candidate_id if best_rec else None),
        "num_candidates_total": len(records),
        "num_valid": len(valid_records),
        "num_invalid": len(invalid_records),
        "improvement_over_initial": (
            _safe(best_reward - initial_score)
            if (math.isfinite(best_reward) and math.isfinite(initial_score)) else None
        ),
        "records": [
            {**r.as_dict(), "reward": _safe(r.reward)} for r in records
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    print("\n================ C3 SUMMARY ================")
    print(f"Dataset:            {DATASET_CONFIG}")
    print(f"Model:              {args.model}")
    print(f"Prompt version:     {args.prompt_version}")
    print(f"Initial seed:       {initial_name}")
    print(f"Initial reward:     {initial_score}  (MASE {results['initial_MASE']})")
    print(f"Best reward:        {best_reward}  (MASE {results['best_MASE']})")
    print(f"Best candidate id:  {results['best_candidate_id']}")
    print(f"Valid / invalid:    {len(valid_records)} / {len(invalid_records)}")
    print(f"Wrote: {out_dir/'results.json'}")
    print(f"Wrote: {out_dir/'progress.csv'}")
    if not aborted:
        print(f"Wrote: {out_dir/'best_candidate.py'}")
    print("============================================")


if __name__ == "__main__":
    main()
