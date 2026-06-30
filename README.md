# 🔬 Reproduction of ERA — *An AI System to Help Scientists Write Expert-Level Empirical Software*

A multi-stage reproduction (and small extension) of the official
[`google-research/era`](https://github.com/google-research/era) code for the Nature paper
*An AI system to help scientists write expert-level empirical software*.

This repo runs the official minimal demo, then asks the sharper research question —
**is ERA's tree search actually better than plain best-of-N sampling?** — and adds
**repeated-run** and **model-ablation** experiments to make the answer credible.

> 📄 The original ERA project README is preserved at **[`README_UPSTREAM.md`](README_UPSTREAM.md)**.

---

## 📑 Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Part 1 — Stage-1 Reproduction (official demo)](#3-part-1--stage-1-reproduction-official-demo)
4. [Part 2 — ERA Tree Search vs. Best-of-N](#4-part-2--era-tree-search-vs-best-of-n)
5. [Reliability — Repeated Runs & Model Ablation](#5-reliability--repeated-runs--model-ablation)
6. [Phase-3 Summary (中文 / 给导师)](#6-phase-3-summary-中文--给导师)
7. [How to Run](#7-how-to-run)
8. [Repository Structure](#8-repository-structure)
9. [Security Note & Attribution](#9-security-note--attribution)

---

## 1. Overview

- **Task:** the official ERA demo `playground_s3e1.py` — a Kaggle / California-Housing-style **regression** problem.
- **Metric:** **negative RMSE** (`-RMSE`), so **higher (closer to 0) is better**.
- **The ERA loop:** an LLM writes/edits Python code → a sandbox executes it → it is auto-scored → **Flat UCB Tree Search (FUTS)** selects and iterates.

**What this reproduction adds on top of the demo:**

| Stage | Deliverable |
|------|-------------|
| Part 1 | Stage-1 reproduction report + search-progress figure |
| Part 2 | A *fair* ERA-vs-best-of-N comparison harness |
| Reliability | Repeated-run evaluation (mean ± std, CSV, plots) |
| Ablation | A small `flash-lite` vs `flash` model comparison |

---

## 2. Environment Setup

- **Python 3.12** (native arm64 / Apple Silicon via Homebrew).
- Packages: `pandas numpy scikit-learn google-genai matplotlib tabulate`.
- A Google **Gemini API key**.

```bash
pip install pandas numpy scikit-learn google-genai matplotlib tabulate

# Use the Gemini key (avoid the SDK preferring GOOGLE_API_KEY):
unset GOOGLE_API_KEY
export GEMINI_API_KEY="your_key"

# Optional model override (scripts default to gemini-2.5-flash-lite):
export GEMINI_MODEL=gemini-2.5-flash-lite
```

> ⚠️ **Security:** the local `implementation/sandbox.py` is **NOT a secure sandbox** — it runs
> LLM-generated code directly on your machine with no isolation. Use it **only** for this trusted
> toy reproduction; use Docker / firejail / a VM for anything real.

---

## 3. Part 1 — Stage-1 Reproduction (official demo)

Confirms the full ERA loop runs and **self-improves** on the official demo. (This is the official
*minimal demo*, **not** the full scRNA-seq / COVID experiment from the paper.)

**Key results**

| Stage | Best score (neg RMSE, higher is better) |
|-------|------------------------------------------|
| Initial baseline | **−0.7339** |
| 10-iteration best | ≈ **−0.5785** |
| 30-iteration best | **−0.5776** |

**Search progress**

![Search progress](implementation/saved_runs/playground_s3e1_iter30/progress.png)

**Interpretation:** the search jumps off the linear-regression baseline immediately at step 1, then
makes small steady gains on gradient-boosting + feature-engineering solutions and **quickly plateaus**
— 30 iterations barely beat 10, indicating a small toy-task search space.

📎 Full report: [`implementation/saved_runs/playground_s3e1_iter30/report.pdf`](implementation/saved_runs/playground_s3e1_iter30/report.pdf)

---

## 4. Part 2 — ERA Tree Search vs. Best-of-N

**Question:** is ERA more than best-of-N sampling? Both methods get an **equal budget of N = 20 LLM
calls**, the **same** train/val split, the **same** initial baseline, and the **same** scorer.
The only difference: **ERA** conditions each new candidate on a *tree-selected parent*, while
**best-of-N** samples independently from the *same fixed initial prompt* every time.

**Result (single run, `gemini-2.5-flash-lite`)**

| Method | Final best score (neg RMSE) |
|--------|------------------------------|
| Initial baseline | −0.7339 |
| **ERA tree search** | **−0.5735** ✅ |
| Best-of-N sampling | −0.6149 |
| **Winner** | **ERA** |

![ERA vs Best-of-N](implementation/saved_runs/era_vs_bon/era_vs_bon.png)

**Two findings:**
1. **Higher final score** — ERA keeps climbing while best-of-N flatlines after ~5 calls.
2. **More reliable** — ERA produced only **3 / 20 invalid** candidates vs best-of-N's **13 / 20**,
   because ERA edits already-working parent code instead of rewriting complex models from scratch
   off the weak linear baseline.

**Caveat:** this is a *single stochastic run* on a toy task — confirmed/strengthened by the repeated
runs below.

📎 Full report: [`implementation/saved_runs/era_vs_bon/report.pdf`](implementation/saved_runs/era_vs_bon/report.pdf)

---

## 5. Reliability — Repeated Runs & Model Ablation

To turn a suggestive single run into a defensible result:

- **`repeat_era_vs_bon.py`** — repeats the fair comparison `num_repeats` times and writes:
  `results.json` (per-repeat records), `summary.csv`, `repeated_curves.{png,pdf}`,
  `final_best_scores.{png,pdf}`, and a Chinese `summary.md`.
  Supports `--model`, `--N`, `--num_repeats`, and `--plot-only` (regenerate plots with **zero** API calls).
- **`model_ablation.py`** — compares `gemini-2.5-flash-lite` vs `gemini-2.5-flash` (N = 10, 1 repeat)
  and writes `results.json`, `model_ablation.{png,pdf}`, and a Chinese `summary.md`.
  *Gemini Pro is intentionally never the default.*

### Repeated-run result (`gemini-2.5-flash-lite`, N = 10, 3 repeats)

**ERA won all 3 repeats** — the single-run advantage holds up.

| Repeat | ERA final | Best-of-N final | Winner |
|:------:|:---------:|:---------------:|:------:|
| 0 | −0.5810 | −0.6017 | **ERA** |
| 1 | −0.5910 | −0.6084 | **ERA** |
| 2 | −0.5750 | −0.6071 | **ERA** |
| **Average** | **−0.5823** | **−0.6058** | **ERA (3/3)** |

- **Average gap:** ERA is **+0.0234** higher (neg RMSE) than best-of-N.
- **Failed candidates:** ERA **6/30 (20%)** vs best-of-N **10/30 (33%)** — best-of-N fails more, consistent with the single-run finding.

![Repeated running-best curves](implementation/saved_runs/repeated_era_vs_bon/repeated_curves.png)
![Final best per repeat](implementation/saved_runs/repeated_era_vs_bon/final_best_scores.png)

> The `model_ablation.py` outputs appear under `implementation/saved_runs/model_ablation/` once you run it.
> Full Chinese write-up: [`implementation/saved_runs/repeated_era_vs_bon/summary.md`](implementation/saved_runs/repeated_era_vs_bon/summary.md).

---

## 6. Phase-3 Summary (中文 / 给导师)

### 已完成的工作
- **成功复现官方 ERA 最小 Demo**：打通 LLM 生成/改写代码 → 本地沙箱执行 → 自动打分（负 RMSE）→ 树搜索迭代 的完整闭环。
- **绘制搜索进度图**：初始 `-0.7339`，10 次迭代约 `-0.5785`，30 次迭代 `-0.5776`。
- **实现 ERA vs best-of-N 对照实验**（相同预算 N=20，`flash-lite`）：ERA 最终 `-0.5735`、best-of-N 最终 `-0.6149`，**ERA 获胜**；且 ERA 仅 3/20 个失败候选，best-of-N 有 13/20 个，**ERA 明显更稳定**。
- **运行了重复评估实验**（`flash-lite`，N=10，3 次）：**ERA 3/3 全胜**，平均 `-0.5823` vs best-of-N `-0.6058`；失败率 20% vs 33%。

### 当前结论
- ERA 的核心闭环已被验证，系统能稳定运行并自我改进。
- 玩具任务**很快进入平台期**（10→30 次迭代提升极小），绝对差距有限。
- **ERA 稳定优于 best-of-N**：重复 3 次中 ERA 全胜（3/3），平均分数更高、失败更少，说明该优势不是单次运行的偶然；玩具任务绝对差距有限，后续可增大 N / 重复次数进一步确认。

### 模型建议
- **继续用 `gemini-2.5-flash-lite`** 做便宜的重复实验（脚本默认）。
- **用 `gemini-2.5-flash` 做一次小型消融**，更强模型可能写出更好的代码、减少失败。
- **暂不使用 `gemini-2.5-pro`**：成本过高，仅在任务明显变难或便宜模型频繁失败时再考虑。

### 下一步
从官方玩具 Demo 迁移到更难的小型基准，建议顺序：
1. 另一个 **Kaggle Playground** 风格任务 → 2. **GIFT-Eval** 小子集 → 3. **scRNA 20k-cell** 搜索设置 → 4. 之后再做论文级别的更大任务。

📎 Standalone copy: [`implementation/saved_runs/phase3_summary.md`](implementation/saved_runs/phase3_summary.md)

---

## 7. How to Run

```bash
cd implementation

unset GOOGLE_API_KEY
export GEMINI_API_KEY="your_key"

# 1) Official ERA demo (produces results/futs_progress.json)
python playground_s3e1.py

# 2) ERA vs best-of-N — single comparison (N=20 → 40 Gemini calls)
python compare_era_vs_bon.py --n 20

# 3) Repeated reliability run (N=10, 3 repeats → 60 calls)
python repeat_era_vs_bon.py --model gemini-2.5-flash-lite --N 10 --num_repeats 3

# 4) Model ablation: flash-lite vs flash (N=10, 1 repeat each → 40 calls)
python model_ablation.py
```

> 💡 Cost scales as **2 × N × repeats** Gemini calls per comparison run. Keep `N` small.

---

## 8. Repository Structure

```
implementation/
├── playground_s3e1.py     # official ERA demo (only change: iterations 10 → 30)
├── futs.py                # Flat UCB Tree Search (FUTS) — unchanged
├── llm.py                 # Gemini wrapper: transient-error retries + GEMINI_MODEL override
├── sandbox.py             # local executor (INSECURE — toy reproduction only)
├── compare_era_vs_bon.py  # ERA vs best-of-N — single fair comparison
├── repeat_era_vs_bon.py   # repeated comparison + CSV + plots + 中文 summary
├── model_ablation.py      # flash-lite vs flash ablation
├── plot_progress.py       # robust progress-figure plotter
└── saved_runs/
    ├── playground_s3e1_iter30/   # Part-1 report (.tex/.pdf) + progress figures
    ├── era_vs_bon/               # Part-2 report (.tex/.pdf) + figures + results.json
    ├── repeated_era_vs_bon/      # (generated by repeat_era_vs_bon.py)
    ├── model_ablation/           # (generated by model_ablation.py)
    └── phase3_summary.md         # advisor summary (中文)
```

---

## 9. Security Note & Attribution

- ⚠️ **`sandbox.py` is not a secure sandbox.** It executes LLM-generated Python directly with your
  user permissions. Toy reproduction only — use real isolation (Docker / firejail / gVisor / a VM)
  for anything serious.
- 🙏 Built on top of [`google-research/era`](https://github.com/google-research/era) (Apache 2.0).
  The original project README is at [`README_UPSTREAM.md`](README_UPSTREAM.md); the license is in
  [`LICENSE`](LICENSE).
