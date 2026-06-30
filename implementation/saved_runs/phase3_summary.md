# ERA 复现 · 第三阶段小结（给导师）

> 项目：复现 Nature 论文 *An AI system to help scientists write expert-level
> empirical software* 对应的官方代码 `google-research/era`。
> 任务：官方最小示例 `playground_s3e1.py`（Kaggle / California Housing 风格回归）。
> 评分：负 RMSE（negative RMSE），**越高越好**。

---

## 1. 已完成的工作

1. **成功复现官方 ERA 最小 Demo**（`playground_s3e1.py`），打通整条流水线：
   LLM 生成/改写代码 → 本地沙箱执行 → 自动打分（负 RMSE）→ 树搜索迭代。
2. **绘制了搜索进度图**（`saved_runs/playground_s3e1_iter30/`）：
   - 初始基线：`-0.7339`
   - 10 次迭代最佳：约 `-0.5785`
   - 30 次迭代最佳：`-0.5776`
3. **实现了 ERA 树搜索 vs best-of-N 独立采样 的对照实验**
   （`compare_era_vs_bon.py`，`saved_runs/era_vs_bon/`）。
   单次运行（`gemini-2.5-flash-lite`，N=20，预算相同）：
   - ERA 最终最佳：`-0.5735`
   - best-of-N 最终最佳：`-0.6149`
   - **ERA 获胜**；并且 ERA 仅有 3/20 个失败候选，best-of-N 有 13/20 个失败候选，
     **ERA 明显更稳定**。
4. **新增并运行了重复评估实验**（`repeat_era_vs_bon.py`，`flash-lite`，N=10，重复 3 次）：
   - **ERA 3 次全胜（3/3）**，best-of-N 0 胜。
   - ERA 平均最终最佳：**-0.5823**；best-of-N 平均：**-0.6058**；平均差异 **+0.0234**。
   - 失败候选：ERA 6/30（20%），best-of-N 10/30（33%），best-of-N 失败率更高。
   - 另提供 `model_ablation.py`（`flash-lite` vs `flash` 小型消融，待运行）。

---

## 2. 当前结论

- **核心 ERA 闭环已被验证**：LLM 改写代码 → 沙箱执行 → 自动打分 → 树搜索，
  系统能稳定运行并自我改进。
- **玩具任务很快进入平台期**：迭代从 10 增加到 30，分数仅从约 `-0.5785`
  提升到 `-0.5776`，提升非常小，说明该 Demo 搜索空间小、收敛快。
- **ERA 优于 best-of-N，且该结论在重复实验中保持稳定**：在相同 LLM 调用预算下，
  ERA 不仅最终分数更高，而且失败候选更少（因为 ERA 在"已经能跑通的父代码"上改进，
  而 best-of-N 每次都从弱基线从零生成复杂方案）。
  **重复实验（N=10，3 次）已确认：ERA 3/3 全胜，平均 -0.5823 vs best-of-N -0.6058，
  失败率 20% vs 33%**，说明这一优势不是单次运行的偶然。
  注意：玩具任务绝对差距有限，后续可增大 N 或重复次数进一步收紧置信度。

---

## 3. 模型建议

- **继续使用 `gemini-2.5-flash-lite`** 做便宜的重复实验（脚本默认模型）。
- **用 `gemini-2.5-flash` 做一次小型消融**：更强的模型可能写出更好的代码、降低失败率；
  通过 `model_ablation.py` 比较二者在本任务上的差异。
- **暂不使用 `gemini-2.5-pro`**：成本过高。仅当任务明显变难、或便宜模型频繁失败时再考虑，
  且只作为可选项手动开启，不设为默认。

---

## 4. 下一步计划

从官方玩具 Demo 逐步迁移到更有挑战性的小型基准，建议顺序：

1. 另一个 **Kaggle Playground** 风格任务（验证流程可迁移性）；
2. **GIFT-Eval** 小子集；
3. **scRNA 20k-cell** 搜索设置；
4. 之后再推进到论文级别的更大任务。

在每一步都先用 `flash-lite` + 小 N + 多次重复跑通并评估稳定性，再决定是否升级模型或扩大规模。

---

## 附：相关产出文件

| 路径 | 内容 |
|---|---|
| `saved_runs/playground_s3e1_iter30/` | 第一阶段复现报告与搜索进度图 |
| `saved_runs/era_vs_bon/` | ERA vs best-of-N 单次对照与报告 |
| `saved_runs/repeated_era_vs_bon/` | 重复实验结果（已生成：ERA 3/3 全胜，含曲线图与柱状图） |
| `saved_runs/model_ablation/` | 模型消融结果（运行 `model_ablation.py` 后生成） |

> 注：本阶段所用的本地 `sandbox.py` **不是安全沙箱**，它会直接在本机执行 LLM 生成的代码，
> 仅用于可信的玩具复现；正式实验应改用 Docker / firejail / 虚拟机等隔离环境。
