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
"""Repeated ERA-vs-best-of-N comparison, for a more credible small reproduction.

Runs the fair single comparison from ``compare_era_vs_bon.py`` (same split, same
baseline, same scorer, equal N LLM calls per method) ``num_repeats`` times, then
saves per-repeat results, a summary CSV, plots, and a Chinese summary.

============================ SECURITY WARNING ============================
This uses the local ``sandbox.py``, which is NOT a secure sandbox: it runs
LLM-generated Python code directly on your machine with no isolation. Only
acceptable for this trusted toy reproduction. Use Docker/firejail/a VM for
anything serious.
=========================================================================

Outputs (default --out-dir saved_runs/repeated_era_vs_bon):
    results.json            all repeats + config
    summary.csv             one row per repeat
    repeated_curves.{png,pdf}   running-best curves for every repeat
    final_best_scores.{png,pdf} grouped bar chart of final-best per repeat
    summary.md              short Chinese summary

Examples:
    python repeat_era_vs_bon.py --model gemini-2.5-flash-lite --N 10 --num_repeats 3
    python repeat_era_vs_bon.py --model gemini-2.5-flash --N 10 --num_repeats 1
    python repeat_era_vs_bon.py --plot-only      # regenerate plots/csv/md from results.json
"""

import argparse
import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

import futs
from sandbox import Sandbox
from llm import GeminiLLM
from playground_s3e1 import (
    prepare_data,
    PlaygroundProblem,
    PlaygroundGenerator,
    PlaygroundExecutor,
)
# Reuse the verified, fair single-comparison building blocks. Importing is
# side-effect-free (compare_era_vs_bon's main() is __main__-guarded).
from compare_era_vs_bon import (
    run_era,
    run_best_of_n,
    running_best,
    SafeGenerator,
    PROBLEM_DESCRIPTION,
    INITIAL_CODE,
    _json_safe,
)

DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_runs", "repeated_era_vs_bon",
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def count_failures(raw_scores):
    """Number of candidates that failed to execute (-inf or None)."""
    return sum(
        1 for s in raw_scores
        if s is None or (isinstance(s, float) and math.isinf(s))
    )


def decide_winner(era_final, bon_final, eps=1e-9):
    if abs(era_final - bon_final) < eps:
        return "Tie"
    return "ERA" if era_final > bon_final else "Best-of-N"


# ----------------------------------------------------------------------------
# Core experiment (importable; reused by model_ablation.py)
# ----------------------------------------------------------------------------
def run_experiment_repeats(model, n, num_repeats, api_key):
    """Run the ERA-vs-BoN comparison `num_repeats` times for one model.

    Returns (model_name, initial_score, records) where `records` is a list of
    per-repeat dicts following the documented schema.
    """
    llm = GeminiLLM(api_key, model_name=model)
    if "pro" in llm.model_name.lower():
        print(f"[!] WARNING: '{llm.model_name}' looks like a Pro model -- this is "
              f"expensive. Proceeding because you asked for it explicitly.")
    print(f"\n>>> Model: {llm.model_name} | N={n} | repeats={num_repeats} "
          f"| total Gemini calls = {2 * n * num_repeats}")

    sandbox = Sandbox(timeout_seconds=60)
    problem = PlaygroundProblem(PROBLEM_DESCRIPTION)
    initial_solution = futs.Solution(INITIAL_CODE)
    generator = SafeGenerator(PlaygroundGenerator(llm))

    # Same fixed (deterministic) split + scorer for every repeat and every model.
    y_val = prepare_data()
    executor = PlaygroundExecutor(sandbox, y_val)

    print("Evaluating initial solution...")
    initial_score = executor(problem, initial_solution)
    print(f"Initial Score (Neg RMSE): {initial_score:.6f}")

    records = []
    for rid in range(num_repeats):
        print(f"\n########## [{llm.model_name}] Repeat {rid + 1}/{num_repeats} ##########")
        era_scores = run_era(
            problem, initial_solution, initial_score, generator, executor, n)
        bon_scores = run_best_of_n(
            problem, initial_solution, initial_score, generator, executor, n)

        era_rb = running_best(era_scores, initial_score)
        bon_rb = running_best(bon_scores, initial_score)
        era_final, bon_final = era_rb[-1], bon_rb[-1]

        record = {
            "repeat_id": rid,
            "model": llm.model_name,
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
        print(f"  -> repeat {rid}: ERA={era_final:.5f} | BoN={bon_final:.5f} "
              f"| winner={record['winner']} "
              f"| failures ERA={record['num_failed_era_candidates']}, "
              f"BoN={record['num_failed_bon_candidates']}")
        records.append(record)

    return llm.model_name, initial_score, records


# ----------------------------------------------------------------------------
# Saving + plotting + summary
# ----------------------------------------------------------------------------
def save_results_json(out_dir, model, n, num_repeats, initial_score, records):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": model,
        "N": n,
        "num_repeats": num_repeats,
        "initial_score": initial_score,
        "repeats": records,
    }
    path = os.path.join(out_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}")
    return payload


def save_summary_csv(out_dir, records):
    path = os.path.join(out_dir, "summary.csv")
    cols = ["repeat_id", "model", "N", "initial_score", "era_final_best",
            "bon_final_best", "winner", "era_failures", "bon_failures"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([
                r["repeat_id"], r["model"], r["N"], r["initial_score"],
                r["era_final_best"], r["bon_final_best"], r["winner"],
                r["num_failed_era_candidates"], r["num_failed_bon_candidates"],
            ])
    print(f"Wrote {path}")


def plot_repeated_curves(out_dir, records, initial_score):
    """Running-best curves for every repeat (ERA blue, BoN orange) + means."""
    n = records[0]["N"]
    xs = list(range(0, n + 1))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    era_curves, bon_curves = [], []
    for r in records:
        era_y = [initial_score] + list(r["era_running_best"])
        bon_y = [initial_score] + list(r["bon_running_best"])
        era_curves.append(era_y)
        bon_curves.append(bon_y)
        ax.step(xs, era_y, where="post", color="#1f77b4", alpha=0.35, linewidth=1.2)
        ax.step(xs, bon_y, where="post", color="#ff7f0e", alpha=0.35, linewidth=1.2)

    era_mean = np.mean(era_curves, axis=0)
    bon_mean = np.mean(bon_curves, axis=0)
    ax.step(xs, era_mean, where="post", color="#1f77b4", linewidth=2.5,
            marker="o", markersize=4, label=f"ERA (mean of {len(records)})")
    ax.step(xs, bon_mean, where="post", color="#ff7f0e", linewidth=2.5,
            marker="s", markersize=4, label=f"Best-of-N (mean of {len(records)})")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial ({initial_score:.4f})")

    ax.set_title(f"Repeated ERA vs. Best-of-N  (N={n}, {len(records)} repeats)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of LLM calls / candidate programs evaluated")
    ax.set_ylabel("Running best score (negative RMSE, higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"repeated_curves.{ext}")
        fig.savefig(p)
        print(f"Wrote {p}")
    plt.close(fig)


def plot_final_best_bars(out_dir, records, initial_score):
    """Grouped bar chart: ERA vs BoN final-best per repeat."""
    ids = [r["repeat_id"] for r in records]
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]

    # Anchor bars at a common floor so taller = better (scores are negative).
    floor = min(min(era), min(bon), initial_score) - 0.01
    x = np.arange(len(records))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    b1 = ax.bar(x - w / 2, [v - floor for v in era], w, bottom=floor,
                color="#1f77b4", label="ERA")
    b2 = ax.bar(x + w / 2, [v - floor for v in bon], w, bottom=floor,
                color="#ff7f0e", label="Best-of-N")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial ({initial_score:.4f})")

    for bars, vals in ((b1, era), (b2, bon)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_ylim(bottom=floor)
    ax.set_xticks(x)
    ax.set_xticklabels([f"repeat {i}" for i in ids])
    ax.set_ylabel("Final best score (negative RMSE, higher is better)")
    ax.set_title("Final Best Score per Repeat: ERA vs. Best-of-N",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"final_best_scores.{ext}")
        fig.savefig(p)
        print(f"Wrote {p}")
    plt.close(fig)


def write_summary_md(out_dir, model, n, initial_score, records):
    """Short Chinese summary answering the required questions."""
    R = len(records)
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
    avg_era = sum(era) / R
    avg_bon = sum(bon) / R
    diff = avg_era - avg_bon
    era_wins = sum(1 for r in records if r["winner"] == "ERA")
    bon_wins = sum(1 for r in records if r["winner"] == "Best-of-N")
    ties = sum(1 for r in records if r["winner"] == "Tie")
    era_fail = sum(r["num_failed_era_candidates"] for r in records)
    bon_fail = sum(r["num_failed_bon_candidates"] for r in records)
    total_cand = R * n

    # Simple, data-driven judgement of significance.
    if abs(diff) < 0.005:
        sig = ("两者平均分数差异很小（小于 0.005），在如此小的重复次数下"
               "**不能认为差异显著**，更可能是采样噪声。")
    elif diff > 0:
        sig = (f"ERA 平均比 best-of-N 高约 **{diff:.4f}**（负 RMSE，越高越好），"
               f"在本玩具任务上**倾向于 ERA 更好**，但重复次数仍然偏少，"
               f"建议增大 N 或重复次数以确认。")
    else:
        sig = (f"best-of-N 平均反而高约 **{-diff:.4f}**，本次重复中**未能复现 "
               f"ERA 的优势**，需要更多重复或更大 N 进一步检验。")

    fail_line = (f"ERA 失败 {era_fail}/{total_cand}（{100*era_fail/total_cand:.0f}%），"
                 f"best-of-N 失败 {bon_fail}/{total_cand}（{100*bon_fail/total_cand:.0f}%）。")
    if bon_fail > era_fail * 1.5 and bon_fail > 0:
        fail_comment = ("best-of-N 的失败率明显更高：每次都从弱的线性基线"
                        "从零生成复杂方案，更容易写出无法运行的代码；"
                        "而 ERA 在已经能运行的父代码上做改进，更稳定。")
    else:
        fail_comment = "两种方法的失败率相近，没有明显差异。"

    lines = []
    lines.append("# ERA vs Best-of-N 重复实验小结\n")
    lines.append("## 实验配置")
    lines.append(f"- 模型: `{model}`")
    lines.append(f"- 每个方法候选数 N: {n}")
    lines.append(f"- 重复次数: {R}")
    lines.append(f"- 初始基线分数（负 RMSE）: {initial_score:.6f}\n")

    lines.append("## 每次重复的获胜方")
    lines.append("| repeat | ERA 最终最佳 | Best-of-N 最终最佳 | 获胜方 |")
    lines.append("|---|---|---|---|")
    for r in records:
        lines.append(f"| {r['repeat_id']} | {r['era_final_best']:.6f} | "
                     f"{r['bon_final_best']:.6f} | {r['winner']} |")
    lines.append(f"\n统计：ERA 获胜 {era_wins} 次，best-of-N 获胜 {bon_wins} 次，"
                 f"平局 {ties} 次。\n")

    lines.append("## 平均最终最佳分数")
    lines.append(f"- ERA 平均: **{avg_era:.6f}**")
    lines.append(f"- Best-of-N 平均: **{avg_bon:.6f}**")
    lines.append(f"- 差异（ERA − Best-of-N）: **{diff:+.6f}**\n")

    lines.append("## 差异是否显著")
    lines.append(sig + "\n")

    lines.append("## 失败的候选程序")
    lines.append(fail_line)
    lines.append(fail_comment + "\n")

    lines.append("## 对下一步的启示")
    lines.append("- 这是搜索空间很小的玩具任务，分数很快进入平台期，绝对差距有限。")
    lines.append("- 建议增大重复次数（如 5）或 N（如 20）来降低噪声、得到更可信的结论。")
    lines.append("- 可用 `gemini-2.5-flash` 做一次小型模型消融，观察更强模型是否减少失败、提升分数。")
    lines.append("- 之后再迁移到更难的小型基准任务（另一个 Kaggle Playground、GIFT-Eval 子集、scRNA 等）。")

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def generate_all_outputs(out_dir, model, n, num_repeats, initial_score, records):
    save_results_json(out_dir, model, n, num_repeats, initial_score, records)
    save_summary_csv(out_dir, records)
    plot_repeated_curves(out_dir, records, initial_score)
    plot_final_best_bars(out_dir, records, initial_score)
    write_summary_md(out_dir, model, n, initial_score, records)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="gemini-2.5-flash-lite",
                        help="Gemini model (default gemini-2.5-flash-lite). "
                             "Do NOT use Pro by default; pass it explicitly only "
                             "if you really mean to.")
    parser.add_argument("--N", type=int, default=10,
                        help="Candidates per method per repeat (default 10).")
    parser.add_argument("--num_repeats", type=int, default=3,
                        help="Number of repeats (default 3). "
                             "Total Gemini calls = 2*N*num_repeats.")
    parser.add_argument("--out-dir", default=DEFAULT_OUTDIR,
                        help="Output directory.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip the API experiment; regenerate CSV/plots/md "
                             "from an existing results.json in --out-dir.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot_only:
        results_path = os.path.join(args.out_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"--plot-only: {results_path} not found. Run the experiment first.")
            return
        with open(results_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload["repeats"]
        save_summary_csv(args.out_dir, records)
        plot_repeated_curves(args.out_dir, records, payload["initial_score"])
        plot_final_best_bars(args.out_dir, records, payload["initial_score"])
        write_summary_md(args.out_dir, payload["model"], payload["N"],
                         payload["initial_score"], records)
        print("\nplot-only: regenerated outputs from existing results.json")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.\n"
              "Please run:\n"
              "  unset GOOGLE_API_KEY\n"
              "  export GEMINI_API_KEY=\"my_key\"\n"
              "then re-run this script.")
        return
    if os.environ.get("GOOGLE_API_KEY"):
        print("[!] Note: GOOGLE_API_KEY is also set; the SDK may prefer it. "
              "For a clean run, `unset GOOGLE_API_KEY` first.")

    model_name, initial_score, records = run_experiment_repeats(
        args.model, args.N, args.num_repeats, api_key)

    generate_all_outputs(args.out_dir, model_name, args.N, args.num_repeats,
                         initial_score, records)

    # Console summary
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
    print("\n================ REPEATED SUMMARY ================")
    print(f"Model:                 {model_name}")
    print(f"N / repeats:           {args.N} / {args.num_repeats}")
    print(f"Initial score:         {initial_score:.6f}")
    print(f"ERA avg final best:    {sum(era)/len(era):.6f}")
    print(f"Best-of-N avg final:   {sum(bon)/len(bon):.6f}")
    print(f"ERA wins:              {sum(1 for r in records if r['winner']=='ERA')}/{len(records)}")
    print("==================================================")


if __name__ == "__main__":
    main()
