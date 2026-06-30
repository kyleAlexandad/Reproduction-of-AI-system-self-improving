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
"""Small model ablation: gemini-2.5-flash-lite vs gemini-2.5-flash.

For each model it runs the same ERA-vs-best-of-N comparison (default N=10,
num_repeats=1) and reports whether the stronger `flash` produces noticeably
better code / search results than the cheaper `flash-lite` on this toy task.

Gemini Pro is intentionally NOT included by default (too expensive for this
stage). You can override the list with --models if you really want to.

Outputs (default --out-dir saved_runs/model_ablation):
    results.json
    model_ablation.{png,pdf}
    summary.md   (Chinese)

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
import numpy as np

from repeat_era_vs_bon import run_experiment_repeats

DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "saved_runs", "model_ablation",
)
DEFAULT_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]


def _avg(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_model(records):
    """Aggregate per-model stats from its repeat records."""
    era = [r["era_final_best"] for r in records]
    bon = [r["bon_final_best"] for r in records]
    era_fail = sum(r["num_failed_era_candidates"] for r in records)
    bon_fail = sum(r["num_failed_bon_candidates"] for r in records)
    n = records[0]["N"]
    total = len(records) * n
    return {
        "era_avg": _avg(era),
        "bon_avg": _avg(bon),
        "era_fail": era_fail,
        "bon_fail": bon_fail,
        "total_candidates": total,
    }


def plot_ablation(out_dir, models, per_model_stats, initial_score):
    """Grouped bars: ERA-avg and BoN-avg final best per model."""
    era = [per_model_stats[m]["era_avg"] for m in models]
    bon = [per_model_stats[m]["bon_avg"] for m in models]
    floor = min(min(era), min(bon), initial_score) - 0.01

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    b1 = ax.bar(x - w / 2, [v - floor for v in era], w, bottom=floor,
                color="#1f77b4", label="ERA (avg final best)")
    b2 = ax.bar(x + w / 2, [v - floor for v in bon], w, bottom=floor,
                color="#ff7f0e", label="Best-of-N (avg final best)")
    ax.axhline(initial_score, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"Initial ({initial_score:.4f})")

    for bars, vals in ((b1, era), (b2, bon)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylim(bottom=floor)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Final best score (negative RMSE, higher is better)")
    ax.set_title("Model Ablation: flash-lite vs flash", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"model_ablation.{ext}")
        fig.savefig(p)
        print(f"Wrote {p}")
    plt.close(fig)


def write_summary_md(out_dir, models, per_model_stats, n, num_repeats, initial_score):
    lite = "gemini-2.5-flash-lite"
    flash = "gemini-2.5-flash"
    lines = []
    lines.append("# 模型消融小结：flash-lite vs flash\n")
    lines.append("## 实验配置")
    lines.append(f"- 对比模型: {', '.join('`'+m+'`' for m in models)}")
    lines.append(f"- 每个方法候选数 N: {n}")
    lines.append(f"- 每个模型重复次数: {num_repeats}")
    lines.append(f"- 初始基线分数（负 RMSE）: {initial_score:.6f}\n")

    lines.append("## 各模型结果")
    lines.append("| 模型 | ERA 平均最终最佳 | Best-of-N 平均最终最佳 | ERA 失败 | BoN 失败 |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        s = per_model_stats[m]
        lines.append(f"| `{m}` | {s['era_avg']:.6f} | {s['bon_avg']:.6f} | "
                     f"{s['era_fail']}/{s['total_candidates']} | "
                     f"{s['bon_fail']}/{s['total_candidates']} |")
    lines.append("")

    lines.append("## 结论：flash 是否明显优于 flash-lite")
    if lite in per_model_stats and flash in per_model_stats:
        d = per_model_stats[flash]["era_avg"] - per_model_stats[lite]["era_avg"]
        df = per_model_stats[lite]["era_fail"] - per_model_stats[flash]["era_fail"]
        if d > 0.005:
            quality = (f"在 ERA 最终最佳分数上，`flash` 比 `flash-lite` 高约 "
                       f"**{d:.4f}**（负 RMSE，越高越好），**有可见提升**。")
        elif d < -0.005:
            quality = (f"本次实验中 `flash` 反而比 `flash-lite` 低约 **{-d:.4f}**，"
                       f"**未显示优势**（可能受单次重复的噪声影响）。")
        else:
            quality = (f"两者 ERA 最终分数差异很小（{d:+.4f}），在本玩具任务上"
                       f"**没有明显差别**。")
        if df > 0:
            failcmt = f"此外，`flash` 的失败候选更少（少 {df} 个），代码更可靠。"
        elif df < 0:
            failcmt = f"不过 `flash` 的失败候选反而多 {-df} 个，差异可能是噪声。"
        else:
            failcmt = "两者失败候选数量相同。"
        lines.append(quality)
        lines.append(failcmt)
    lines.append("")
    lines.append("## 建议")
    lines.append("- 若 `flash` 仅有微小提升：继续用便宜的 `flash-lite` 做大量重复实验。")
    lines.append("- 若 `flash` 明显更好/失败更少：在关键的最终评测上改用 `flash`。")
    lines.append("- 任务变难或便宜模型频繁失败前，**不要**使用 `gemini-2.5-pro`。")
    lines.append("- 注意：单次重复噪声较大，必要时把 num_repeats 调大再下结论。")

    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to compare (default: flash-lite flash). "
                             "Pro is intentionally excluded by default.")
    parser.add_argument("--N", type=int, default=10,
                        help="Candidates per method per repeat (default 10).")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Repeats per model (default 1).")
    parser.add_argument("--out-dir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    total_calls = sum(2 * args.N * args.num_repeats for _ in args.models)
    print(f"Model ablation over {args.models}")
    print(f"Total Gemini calls this run: {total_calls}")

    per_model_records = {}
    per_model_stats = {}
    initial_score = None
    for m in args.models:
        model_name, init_score, records = run_experiment_repeats(
            m, args.N, args.num_repeats, api_key)
        per_model_records[model_name] = records
        per_model_stats[model_name] = summarize_model(records)
        initial_score = init_score  # deterministic; same for all models

    models_in_order = list(per_model_records.keys())

    # Save results.json
    payload = {
        "N": args.N,
        "num_repeats": args.num_repeats,
        "initial_score": initial_score,
        "models": models_in_order,
        "results": per_model_records,
        "stats": per_model_stats,
    }
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {results_path}")

    plot_ablation(args.out_dir, models_in_order, per_model_stats, initial_score)
    write_summary_md(args.out_dir, models_in_order, per_model_stats,
                     args.N, args.num_repeats, initial_score)

    print("\n================ ABLATION SUMMARY ================")
    for m in models_in_order:
        s = per_model_stats[m]
        print(f"{m:28s} ERA_avg={s['era_avg']:.6f}  BoN_avg={s['bon_avg']:.6f}  "
              f"ERA_fail={s['era_fail']}/{s['total_candidates']}")
    print("==================================================")


if __name__ == "__main__":
    main()
