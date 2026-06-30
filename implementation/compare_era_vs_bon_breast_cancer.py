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
"""ERA tree search vs. best-of-N on the SECOND benchmark (breast cancer).

Mirrors ``compare_era_vs_bon.py`` (the California-Housing version) but wires in
the breast-cancer classification task from ``playground_breast_cancer.py``. The
*task-agnostic* search/sampling core (`run_era`, `run_best_of_n`, `running_best`,
`SafeGenerator`) is reused unchanged from ``compare_era_vs_bon.py`` so both
benchmarks share the exact same, already-verified comparison logic.

Metric is ROC-AUC (higher is better; range 0..1) -- so unlike the RMSE demo the
scores are positive and used directly (no negation).

Fairness: both methods use the same task, same train/val split, same initial
baseline, same scorer, same model, and the same number N of LLM calls. ERA
conditions each candidate on a tree-selected parent; best-of-N samples
independently from the same fixed initial prompt every time.

============================ SECURITY WARNING ============================
Uses the local insecure sandbox.py (runs LLM-generated code with no isolation).
Toy reproduction only.
=========================================================================
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import futs
from sandbox import Sandbox
from llm import GeminiLLM

# Task pieces (breast cancer).
from playground_breast_cancer import (
    prepare_data,
    BreastCancerProblem,
    BreastCancerGenerator,
    BreastCancerExecutor,
    INITIAL_CODE,
)
# Reused, verified, task-agnostic comparison core.
from compare_era_vs_bon import run_era, run_best_of_n, running_best, SafeGenerator, _json_safe
from repeat_era_vs_bon import count_failures, decide_winner

PROBLEM_DESCRIPTION = (
    "Improve the binary classification model for the sklearn Breast Cancer dataset."
)
DEFAULT_MODEL = "gemini-2.5-flash-lite"
SINGLE_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_runs", "breast_cancer_era_vs_bon",
)


def require_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n"
              "  unset GOOGLE_API_KEY\n"
              '  export GEMINI_API_KEY="my_key"\n'
              "then re-run.")
        return None
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] GOOGLE_API_KEY also set; the SDK may prefer it. "
              "`unset GOOGLE_API_KEY` for a clean run.")
    return api_key


def build_task(api_key, model):
    """Set up the breast-cancer task once and evaluate the initial baseline.

    Returns: (model_name, problem, initial_solution, generator, executor, initial_score)
    The same split/baseline/scorer/model is shared by ERA and best-of-N.
    """
    llm = GeminiLLM(api_key, model_name=model)
    if "pro" in llm.model_name.lower():
        print(f"[!] WARNING: '{llm.model_name}' looks like a Pro model (expensive).")
    sandbox = Sandbox(timeout_seconds=60)
    problem = BreastCancerProblem(PROBLEM_DESCRIPTION)
    initial_solution = futs.Solution(INITIAL_CODE)
    generator = SafeGenerator(BreastCancerGenerator(llm))

    y_val = prepare_data()
    executor = BreastCancerExecutor(sandbox, y_val)
    print("Evaluating initial solution...")
    initial_score = executor(problem, initial_solution)
    print(f"Initial Score (ROC-AUC): {initial_score:.6f}")
    return llm.model_name, problem, initial_solution, generator, executor, initial_score


def run_one_comparison(problem, initial_solution, initial_score, generator,
                       executor, n, model_name, repeat_id=0):
    """One ERA run + one best-of-N run (equal budget N). Returns a record dict."""
    era_scores = run_era(
        problem, initial_solution, initial_score, generator, executor, n)
    bon_scores = run_best_of_n(
        problem, initial_solution, initial_score, generator, executor, n)
    era_rb = running_best(era_scores, initial_score)
    bon_rb = running_best(bon_scores, initial_score)
    era_final, bon_final = era_rb[-1], bon_rb[-1]
    return {
        "repeat_id": repeat_id,
        "task": "breast_cancer",
        "metric": "roc_auc",
        "model": model_name,
        "N": n,
        "initial_score": initial_score,
        "era_scores": _json_safe(era_scores),
        "era_running_best": _json_safe(era_rb),
        "bon_scores": _json_safe(bon_scores),
        "bon_running_best": _json_safe(bon_rb),
        "era_final_best": era_final,
        "bon_final_best": bon_final,
        "winner": decide_winner(era_final, bon_final),
        "num_failed_era_candidates": count_failures(era_scores),
        "num_failed_bon_candidates": count_failures(bon_scores),
    }


def make_auc_plot(initial_score, era_rb, bon_rb, n, png_path, pdf_path):
    """Running-best ROC-AUC vs. number of candidates evaluated."""
    xs = list(range(0, n + 1))
    era_y = [initial_score] + list(era_rb)
    bon_y = [initial_score] + list(bon_rb)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(xs, era_y, where="post", color="#1f77b4", linewidth=2,
            marker="o", markersize=4, label="ERA tree search")
    ax.step(xs, bon_y, where="post", color="#ff7f0e", linewidth=2,
            marker="s", markersize=4, label="Best-of-N independent sampling")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial ({initial_score:.4f})")
    ax.set_title("Breast Cancer: ERA vs. Best-of-N", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of LLM calls / candidate programs evaluated")
    ax.set_ylabel("Running best ROC-AUC (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}\nWrote {pdf_path}")
    plt.close(fig)


def write_single_summary(out_dir, record):
    r = record
    n = r["N"]
    lines = [
        "# Breast Cancer — ERA vs. Best-of-N (single run)\n",
        f"- Task: breast cancer binary classification (metric: **ROC-AUC**, higher is better)",
        f"- Model: `{r['model']}`",
        f"- Budget: N = {n} LLM calls per method\n",
        "## Result",
        f"- Initial baseline ROC-AUC: **{r['initial_score']:.6f}**",
        f"- ERA final best: **{r['era_final_best']:.6f}**",
        f"- Best-of-N final best: **{r['bon_final_best']:.6f}**",
        f"- **Winner: {r['winner']}**\n",
        "## Invalid candidates",
        f"- ERA: {r['num_failed_era_candidates']}/{n}",
        f"- Best-of-N: {r['num_failed_bon_candidates']}/{n}\n",
        "> Sandbox is insecure; toy reproduction only.",
    ]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10,
                   help="Candidates per method (default 10; total Gemini calls = 2*N).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Gemini model (default gemini-2.5-flash-lite). Do NOT use Pro.")
    p.add_argument("--out-dir", default=SINGLE_OUTDIR)
    args = p.parse_args()

    api_key = require_api_key()
    if not api_key:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Total Gemini calls this run: {2 * args.n} (N={args.n} ERA + N={args.n} BoN)")

    model_name, problem, initial_solution, generator, executor, initial_score = \
        build_task(api_key, args.model)
    print(f"Using Gemini model: {model_name}")

    record = run_one_comparison(
        problem, initial_solution, initial_score, generator, executor,
        args.n, model_name)

    # Save results.json
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote {os.path.join(args.out_dir, 'results.json')}")

    make_auc_plot(
        initial_score, record["era_running_best"], record["bon_running_best"], args.n,
        os.path.join(args.out_dir, "era_vs_bon_breast_cancer.png"),
        os.path.join(args.out_dir, "era_vs_bon_breast_cancer.pdf"))
    write_single_summary(args.out_dir, record)

    print("\n================ SUMMARY (breast cancer) ================")
    print(f"Model:                 {model_name}")
    print(f"Initial ROC-AUC:       {initial_score:.6f}")
    print(f"ERA final best:        {record['era_final_best']:.6f}")
    print(f"Best-of-N final best:  {record['bon_final_best']:.6f}")
    print(f"Winner:                {record['winner']}")
    print(f"Invalid ERA / BoN:     {record['num_failed_era_candidates']}/{args.n} "
          f"/ {record['num_failed_bon_candidates']}/{args.n}")
    print("========================================================")


if __name__ == "__main__":
    main()
