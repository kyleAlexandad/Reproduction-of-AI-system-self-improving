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
"""Repeated ERA-vs-best-of-N on the breast-cancer benchmark (reliability check).

Runs the single comparison (from ``compare_era_vs_bon_breast_cancer.py``)
``num_repeats`` times and aggregates, mirroring ``repeat_era_vs_bon.py`` but for
the ROC-AUC classification task. Reuses ``build_task`` / ``run_one_comparison``
and ``repeat_era_vs_bon.save_summary_csv`` so logic is shared, not duplicated.

Uses the local insecure sandbox.py -- toy reproduction only.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from compare_era_vs_bon_breast_cancer import (
    build_task, run_one_comparison, require_api_key, DEFAULT_MODEL,
)
from repeat_era_vs_bon import save_summary_csv  # task-agnostic CSV writer

REPEATED_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_runs", "repeated_breast_cancer_era_vs_bon",
)


def plot_repeated_curves_auc(out_dir, records, initial_score):
    n = records[0]["N"]
    xs = list(range(0, n + 1))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    era_curves, bon_curves = [], []
    for r in records:
        ey = [initial_score] + list(r["era_running_best"])
        by = [initial_score] + list(r["bon_running_best"])
        era_curves.append(ey)
        bon_curves.append(by)
        ax.step(xs, ey, where="post", color="#1f77b4", alpha=0.35, linewidth=1.2)
        ax.step(xs, by, where="post", color="#ff7f0e", alpha=0.35, linewidth=1.2)
    em = np.mean(era_curves, axis=0)
    bm = np.mean(bon_curves, axis=0)
    ax.step(xs, em, where="post", color="#1f77b4", linewidth=2.5, marker="o",
            markersize=4, label=f"ERA (mean of {len(records)})")
    ax.step(xs, bm, where="post", color="#ff7f0e", linewidth=2.5, marker="s",
            markersize=4, label=f"Best-of-N (mean of {len(records)})")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial ({initial_score:.4f})")
    ax.set_title(f"Breast Cancer: Repeated ERA vs Best-of-N "
                 f"(N={n}, {len(records)} repeats)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of LLM calls / candidate programs evaluated")
    ax.set_ylabel("Running best ROC-AUC (higher is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"repeated_curves.{ext}")
        fig.savefig(p)
        print(f"Wrote {p}")
    plt.close(fig)


def plot_final_best_bars_auc(out_dir, records, initial_score):
    ids = [r["repeat_id"] for r in records]
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
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
    ax.set_ylabel("Final best ROC-AUC (higher is better)")
    ax.set_title("Breast Cancer: Final Best per Repeat", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"final_best_scores.{ext}")
        fig.savefig(p)
        print(f"Wrote {p}")
    plt.close(fig)


def write_repeated_summary_md(out_dir, model, n, initial_score, records):
    """Chinese summary (matches repeat_era_vs_bon.py style), adapted for ROC-AUC."""
    R = len(records)
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
    avg_era, avg_bon = sum(era) / R, sum(bon) / R
    diff = avg_era - avg_bon
    ew = sum(1 for r in records if r["winner"] == "ERA")
    bw = sum(1 for r in records if r["winner"] == "Best-of-N")
    ties = sum(1 for r in records if r["winner"] == "Tie")
    ef = sum(r["num_failed_era_candidates"] for r in records)
    bf = sum(r["num_failed_bon_candidates"] for r in records)
    total = R * n

    if abs(diff) < 0.005:
        sig = "两者平均 ROC-AUC 差异很小（<0.005），在少量重复下**不能认为差异显著**。"
    elif diff > 0:
        sig = (f"ERA 平均比 best-of-N 高约 **{diff:.4f}**（ROC-AUC，越高越好），"
               f"在第二个任务上**仍然倾向于 ERA 更好**。")
    else:
        sig = (f"best-of-N 平均反而高约 **{-diff:.4f}**，本次重复中**未能复现 ERA 优势**，"
               f"需要更多重复确认。")
    if bf > ef * 1.5 and bf > 0:
        failcmt = "best-of-N 失败率明显更高：每次从弱基线从零生成更易出错；ERA 在可运行父代码上改进更稳。"
    else:
        failcmt = "两种方法失败率相近。"

    lines = [
        "# 乳腺癌任务：ERA vs Best-of-N 重复实验小结\n",
        "## 实验配置",
        f"- 任务: sklearn 乳腺癌二分类（指标 **ROC-AUC**，越高越好）",
        f"- 模型: `{model}`",
        f"- 每个方法候选数 N: {n}",
        f"- 重复次数: {R}",
        f"- 初始基线 ROC-AUC: {initial_score:.6f}\n",
        "## 每次重复的获胜方",
        "| repeat | ERA 最终最佳 | Best-of-N 最终最佳 | 获胜方 |",
        "|---|---|---|---|",
    ]
    for r in records:
        lines.append(f"| {r['repeat_id']} | {r['era_final_best']:.6f} | "
                     f"{r['bon_final_best']:.6f} | {r['winner']} |")
    lines += [
        f"\n统计：ERA 获胜 {ew} 次，best-of-N 获胜 {bw} 次，平局 {ties} 次。\n",
        "## 平均最终最佳 ROC-AUC",
        f"- ERA 平均: **{avg_era:.6f}**",
        f"- Best-of-N 平均: **{avg_bon:.6f}**",
        f"- 差异（ERA − Best-of-N）: **{diff:+.6f}**\n",
        "## 差异是否显著",
        sig + "\n",
        "## 失败的候选程序",
        f"ERA 失败 {ef}/{total}（{100*ef/total:.0f}%），best-of-N 失败 {bf}/{total}（{100*bf/total:.0f}%）。",
        failcmt + "\n",
        "## 对下一步的启示",
        "- 第二个玩具任务上结果与第一个任务一致即说明 ERA 的优势具有一定迁移性。",
        "- 仍是小规模玩具任务，必要时增大 N 或重复次数收紧结论。",
    ]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10,
                   help="Candidates per method per repeat (default 10).")
    p.add_argument("--num_repeats", type=int, default=3,
                   help="Repeats (default 3). Total Gemini calls = 2*N*num_repeats.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Gemini model (default gemini-2.5-flash-lite). Do NOT use Pro.")
    p.add_argument("--out-dir", default=REPEATED_OUTDIR)
    args = p.parse_args()

    api_key = require_api_key()
    if not api_key:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Total Gemini calls this run: {2 * args.n * args.num_repeats} "
          f"(2 x N={args.n} x repeats={args.num_repeats})")

    # Build the task ONCE (deterministic split + baseline shared across repeats).
    model_name, problem, initial_solution, generator, executor, initial_score = \
        build_task(api_key, args.model)
    print(f"Using Gemini model: {model_name}")

    records = []
    for rid in range(args.num_repeats):
        print(f"\n########## [breast cancer | {model_name}] "
              f"Repeat {rid + 1}/{args.num_repeats} ##########")
        rec = run_one_comparison(
            problem, initial_solution, initial_score, generator, executor,
            args.n, model_name, repeat_id=rid)
        print(f"  -> repeat {rid}: ERA={rec['era_final_best']:.5f} | "
              f"BoN={rec['bon_final_best']:.5f} | winner={rec['winner']} | "
              f"fail ERA={rec['num_failed_era_candidates']}, BoN={rec['num_failed_bon_candidates']}")
        records.append(rec)

    # Save results.json
    payload = {
        "task": "breast_cancer",
        "metric": "roc_auc",
        "model": model_name,
        "N": args.n,
        "num_repeats": args.num_repeats,
        "initial_score": initial_score,
        "repeats": records,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {os.path.join(args.out_dir, 'results.json')}")

    save_summary_csv(args.out_dir, records)  # reused, task-agnostic
    plot_repeated_curves_auc(args.out_dir, records, initial_score)
    plot_final_best_bars_auc(args.out_dir, records, initial_score)
    write_repeated_summary_md(args.out_dir, model_name, args.n, initial_score, records)

    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
    print("\n========== REPEATED SUMMARY (breast cancer) ==========")
    print(f"Model:               {model_name}")
    print(f"N / repeats:         {args.n} / {args.num_repeats}")
    print(f"Initial ROC-AUC:     {initial_score:.6f}")
    print(f"ERA avg final:       {sum(era)/len(era):.6f}")
    print(f"Best-of-N avg final: {sum(bon)/len(bon):.6f}")
    print(f"ERA wins:            {sum(1 for r in records if r['winner']=='ERA')}/{len(records)}")
    print("======================================================")


if __name__ == "__main__":
    main()
