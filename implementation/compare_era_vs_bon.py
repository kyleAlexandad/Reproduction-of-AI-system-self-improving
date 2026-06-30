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
"""Small local comparison: ERA tree search vs. best-of-N independent sampling.

Both methods are run on the *same* task, the same local train/test split, and
the same scoring function as ``playground_s3e1.py`` (negative RMSE on the
California-Housing-style regression demo). The point is to reproduce, in
miniature, the paper's claim that ERA's tree search explores more efficiently
than simply drawing N independent samples and keeping the best.

Method A - ERA tree search:
    Runs the existing FUTS / ERA search (``futs.search``) for N expansions.
    Each expansion generates a candidate *conditioned on a tree-selected
    parent* (so later candidates build on earlier successful ones) and scores
    it. We record the running-best score after each candidate.

Method B - best-of-N independent sampling:
    Generates N candidates that are each conditioned on the *same* initial
    solution and initial score. Crucially, best-of-N does NOT condition on any
    previously generated candidate code -- every sample is drawn independently
    from the identical starter prompt, making it a fair independent-sampling
    baseline. We record the running-best score after each sample.

Equal budget: both methods make exactly N LLM calls, so the x-axis ("number of
LLM calls / candidate programs evaluated") is directly comparable.

============================ SECURITY WARNING ============================
This experiment uses the local ``sandbox.py``, which is NOT a secure sandbox.
It executes LLM-generated Python code directly on your machine with your full
user permissions (no isolation). It is only acceptable for this trusted toy
reproduction. For any serious use, run candidates inside real isolation
(Docker, firejail, gVisor, a VM, etc.).
=========================================================================
"""

import argparse
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")  # headless; no display needed
import matplotlib.pyplot as plt

import futs
from sandbox import Sandbox
from llm import GeminiLLM

# Reuse the official demo's data prep, problem, generator and executor as-is.
# Importing playground_s3e1 is side-effect-free: run_experiment() only runs
# under `if __name__ == "__main__"`, which is not triggered on import.
from playground_s3e1 import (
    prepare_data,
    PlaygroundProblem,
    PlaygroundGenerator,
    PlaygroundExecutor,
)

# --- Minimal parts copied from playground_s3e1.run_experiment ----------------
# These two literals live *inside* run_experiment() in the original file (not at
# module scope), so we copy them here rather than refactor/break the original.
PROBLEM_DESCRIPTION = (
    "Improve the regression model for the California Housing dataset."
)

INITIAL_CODE = """
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def train_and_predict(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train.drop('MedHouseVal', axis=1)
    y = train['MedHouseVal']

    model = LinearRegression()
    model.fit(X, y)

    return model.predict(test)
"""

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "saved_runs", "era_vs_bon"
)


class SafeGenerator:
    """Wraps a generator so a hard LLM failure (after llm.py's own retries are
    exhausted, e.g. a persistent 503) becomes a single failing candidate scored
    -inf, instead of crashing the whole experiment and discarding the API spend
    already incurred. The returned sentinel program has no ``train_and_predict``,
    so the executor scores it -inf and the run continues.
    """

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, problem, parent_solution, parent_score):
        try:
            return self.inner(problem, parent_solution, parent_score)
        except Exception as e:  # noqa: BLE001 - we deliberately swallow any API error
            print(f"  [!] Generation failed after retries; recording -inf "
                  f"candidate. ({str(e)[:100]})")
            return futs.Solution("# generation failed - intentionally invalid candidate\n")


def running_best(scores, floor):
    """Running max of `scores`, never dropping below `floor` (initial score)."""
    out = []
    cur = floor
    for s in scores:
        if s > cur:
            cur = s
        out.append(cur)
    return out


def run_era(problem, initial_solution, initial_score, generator, executor, n):
    """Method A: ERA / FUTS tree search for `n` expansions.

    We wrap the executor so we capture the raw score of each generated
    candidate, in order, as the search expands the tree.
    """
    raw_scores = []

    class TrackingExecutor:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, prob, solution):
            score = self.inner(prob, solution)
            raw_scores.append(score)
            return score

    print(f"\n=== Method A: ERA tree search (N={n}) ===")
    futs.search(
        problem=problem,
        initial_solution=initial_solution,
        initial_score=initial_score,
        generate_fn=generator,
        execute_fn=TrackingExecutor(executor),
        num_iterations=n,
        c_puct=1.0,
    )
    for i, s in enumerate(raw_scores, 1):
        print(f"  [ERA]   call {i:2d}/{n}: score={s:.5f}")
    return raw_scores


def run_best_of_n(problem, initial_solution, initial_score, generator, executor, n):
    """Method B: best-of-N independent sampling.

    Every candidate is generated from the SAME initial solution and initial
    score -- it never conditions on previously generated candidate code. This
    is the fair independent-sampling baseline.
    """
    raw_scores = []
    print(f"\n=== Method B: best-of-N independent sampling (N={n}) ===")
    for i in range(1, n + 1):
        # Always sample from the identical starter prompt (initial solution).
        candidate = generator(problem, initial_solution, initial_score)
        score = executor(problem, candidate)
        raw_scores.append(score)
        print(f"  [BoN]   call {i:2d}/{n}: score={score:.5f}")
    return raw_scores


def _json_safe(values):
    """Convert -inf (failed candidates) to None so the JSON stays valid."""
    return [None if (v is None or math.isinf(v)) else v for v in values]


def aggregate_running_bests(runs):
    """Mean and population-std at each step across repeats.

    Args:
        runs: list of R running-best lists, each of length N. Running-best
            values are always finite (floored at the initial score), so the
            statistics are well defined.

    Returns:
        (mean, std): two lists of length N.
    """
    r = len(runs)
    n = len(runs[0])
    mean, std = [0.0] * n, [0.0] * n
    for j in range(n):
        col = [runs[k][j] for k in range(r)]
        mean[j] = sum(col) / r
        std[j] = statistics.pstdev(col) if r > 1 else 0.0
    return mean, std


def make_plot(initial_score, era_mean, era_std, bon_mean, bon_std,
              n, repeats, png_path, pdf_path):
    """Plot running-best score vs. number of candidates evaluated.

    When repeats > 1 the lines are the mean across repeats, with a shaded
    +/-1 std band. When repeats == 1 the std is zero, so it reduces to the
    original single-run plot.
    """
    # Prepend step 0 = the shared (deterministic) initial score; std there is 0.
    xs = list(range(0, n + 1))
    era_y = [initial_score] + era_mean
    bon_y = [initial_score] + bon_mean
    era_lo = [initial_score] + [m - s for m, s in zip(era_mean, era_std)]
    era_hi = [initial_score] + [m + s for m, s in zip(era_mean, era_std)]
    bon_lo = [initial_score] + [m - s for m, s in zip(bon_mean, bon_std)]
    bon_hi = [initial_score] + [m + s for m, s in zip(bon_mean, bon_std)]

    suffix = f" (mean of {repeats})" if repeats > 1 else ""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(xs, era_y, where="post", color="#1f77b4", linewidth=2,
            marker="o", markersize=4, label=f"ERA tree search{suffix}")
    ax.step(xs, bon_y, where="post", color="#ff7f0e", linewidth=2,
            marker="s", markersize=4,
            label=f"Best-of-N independent sampling{suffix}")
    if repeats > 1:
        ax.fill_between(xs, era_lo, era_hi, step="post", color="#1f77b4",
                        alpha=0.18, linewidth=0, label="ERA +/-1 std")
        ax.fill_between(xs, bon_lo, bon_hi, step="post", color="#ff7f0e",
                        alpha=0.18, linewidth=0, label="Best-of-N +/-1 std")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial score ({initial_score:.4f})")

    title = "ERA Tree Search vs. Best-of-N Sampling"
    if repeats > 1:
        title += f"  (N={n}, {repeats} repeats)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of LLM calls / candidate programs evaluated")
    ax.set_ylabel("Running best score (negative RMSE, higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"\nWrote {png_path}")
    print(f"Wrote {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # N small by default: each repeat costs 2*N Gemini calls (N ERA + N BoN).
    parser.add_argument("--n", type=int, default=20,
                        help="Candidates per method (default 20). Keep small.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Independent repeats per method to average "
                             "(default 1). Total Gemini calls = 2*N*repeats.")
    args = parser.parse_args()
    n = args.n
    repeats = max(1, args.repeats)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n"
              "Please run:\n"
              "  unset GOOGLE_API_KEY\n"
              "  export GEMINI_API_KEY=\"my_key\"\n"
              "then re-run: python compare_era_vs_bon.py")
        return
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] Note: GOOGLE_API_KEY is also set; the SDK may prefer it. "
              "For a clean run, `unset GOOGLE_API_KEY` first.")

    total_calls = 2 * n * repeats
    print(f"Total Gemini calls this run: {total_calls} "
          f"(2 x N={n} x repeats={repeats})")

    # Same data split / scoring as playground_s3e1.py.
    y_val = prepare_data()

    llm = GeminiLLM(api_key)
    print(f"Using Gemini model: {llm.model_name} "
          f"(override with: export GEMINI_MODEL=gemini-2.5-flash-lite)")
    sandbox = Sandbox(timeout_seconds=60)
    problem = PlaygroundProblem(PROBLEM_DESCRIPTION)
    initial_solution = futs.Solution(INITIAL_CODE)

    # Wrap the generator so a transient hard failure becomes a -inf candidate
    # rather than crashing the whole run (and wasting the API calls already made).
    generator = SafeGenerator(PlaygroundGenerator(llm))
    executor = PlaygroundExecutor(sandbox, y_val)

    # The initial linear-regression score is deterministic, so evaluate it once
    # and reuse it as the shared starting point for every repeat.
    print("Evaluating initial solution...")
    initial_score = executor(problem, initial_solution)
    print(f"Initial Score (Neg RMSE): {initial_score:.6f}")

    # --- Run both methods for `repeats` independent repetitions --------------
    era_scores_runs, bon_scores_runs = [], []
    era_best_runs, bon_best_runs = [], []
    for r in range(1, repeats + 1):
        if repeats > 1:
            print(f"\n########## Repeat {r}/{repeats} ##########")
        era_scores = run_era(
            problem, initial_solution, initial_score, generator, executor, n)
        bon_scores = run_best_of_n(
            problem, initial_solution, initial_score, generator, executor, n)
        era_scores_runs.append(era_scores)
        bon_scores_runs.append(bon_scores)
        era_best_runs.append(running_best(era_scores, initial_score))
        bon_best_runs.append(running_best(bon_scores, initial_score))

    era_mean, era_std = aggregate_running_bests(era_best_runs)
    bon_mean, bon_std = aggregate_running_bests(bon_best_runs)

    era_final, bon_final = era_mean[-1], bon_mean[-1]
    if abs(era_final - bon_final) < 1e-9:
        winner = "Tie"
    elif era_final > bon_final:
        winner = "ERA tree search"
    else:
        winner = "Best-of-N independent sampling"

    # --- Save results JSON ---------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {
        "N": n,
        "repeats": repeats,
        "model": llm.model_name,
        "initial_score": initial_score,
        # Backward-compatible flat keys: the first repeat (representative run).
        "era_scores": _json_safe(era_scores_runs[0]),
        "era_running_best": _json_safe(era_best_runs[0]),
        "bon_scores": _json_safe(bon_scores_runs[0]),
        "bon_running_best": _json_safe(bon_best_runs[0]),
        # Per-repeat data (length = repeats).
        "era_scores_runs": [_json_safe(s) for s in era_scores_runs],
        "bon_scores_runs": [_json_safe(s) for s in bon_scores_runs],
        "era_running_best_runs": [_json_safe(s) for s in era_best_runs],
        "bon_running_best_runs": [_json_safe(s) for s in bon_best_runs],
        # Aggregates across repeats (length N).
        "era_mean_running_best": era_mean,
        "era_std_running_best": era_std,
        "bon_mean_running_best": bon_mean,
        "bon_std_running_best": bon_std,
        "era_final_best_mean": era_final,
        "era_final_best_std": era_std[-1],
        "bon_final_best_mean": bon_final,
        "bon_final_best_std": bon_std[-1],
        "winner": winner,
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {results_path}")

    # --- Plot ----------------------------------------------------------------
    make_plot(
        initial_score, era_mean, era_std, bon_mean, bon_std, n, repeats,
        os.path.join(OUTPUT_DIR, "era_vs_bon.png"),
        os.path.join(OUTPUT_DIR, "era_vs_bon.pdf"),
    )

    # --- Summary -------------------------------------------------------------
    print("\n================ SUMMARY ================")
    print(f"Model:                      {llm.model_name}")
    print(f"N per method:               {n}")
    print(f"Repeats:                    {repeats}")
    print(f"Initial score:              {initial_score:.6f}")
    print(f"ERA final best:             {era_final:.6f} +/- {era_std[-1]:.6f}")
    print(f"Best-of-N final best:       {bon_final:.6f} +/- {bon_std[-1]:.6f}")
    print(f"Winner (by mean):           {winner}")
    print("=========================================")


if __name__ == "__main__":
    main()
