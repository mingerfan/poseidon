# R8a3：修正小模型的训练配方，跑完 R8a

## 0. 诊断（Claude 于 2026-10-21 复核）

- R8a2 的 ATen 前端已经完成：Claude 重跑 test_r8a*，51 项全部通过；ResNet20、SiLU、CNN 三个模型的规则基线也编译通过。
- **MLP 在 reference 阶段失败，原因是训练发散，不是前端或 reference 的问题。** R8a spec §4 的配方（SGD，lr 0.05，momentum 0.9）用在没有归一化层、输入 3072 维的 MLP 上，loss 的变化是：

  | epoch | 1 | 2 | 3 | 4 | 20 |
  |---|---|---|---|---|---|
  | loss | 3.31 | 1488 | 1.0e7 | 7.2e9 | 2.9e6 |

  测试 top-1 只有 17%。权重和 logits 因此非常大，float64 下的绝对差就达到了 6.7e-8。这是配方的错误，属于 Claude 的设计问题。
- **新配方 v2** = 原配方 + 梯度范数裁剪（max_norm=1.0）。Claude 用同样的种子和数据做了原型，**只看训练 loss**，没有用测试集选择：loss 从 1.699 单调下降到 0.558，最大 |logit| 约为 21。
- **通用化**：v2 对两个小模型统一使用，并且加入发散检查。CNN 也按 v2 重新训练，保证两个模型的配方一致。旧权重保留，不覆盖。

## 1. 修改

### 1.1 `pt_agent/models/train_small.py`

- 新增模块常量：
  ```python
  RECIPE = "v2-clip1.0"
  CLIP_NORM = 1.0
  DIVERGENCE_FACTOR = 1.5
  ```
- 每个 batch 在 `loss.backward()` 之后、`optimizer.step()` 之前，执行 `torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)`。两个模型都这样做。
- 新增纯函数，并在每个 epoch 结束、写完 row 之后调用：
  ```python
  def check_divergence(losses: list[float]) -> None:
      """losses 是按 epoch 排列的平均 loss。最后一个值不是有限数，或者大于 DIVERGENCE_FACTOR*losses[0]，
      就抛出 RuntimeError(f"training diverged at epoch {len(losses)}: loss {losses[-1]} vs first {losses[0]}")。"""
  ```
  抛出异常后，沿用现有的 except 路径：写入 error，exit_code=1，complete 保持 False。
- report 中增加 `recipe=RECIPE` 和 `clip_norm=CLIP_NORM`。
- 其余设置不变：数据、batch、优化器参数、余弦退火、epoch 数、数据增强、种子。

### 1.2 `pt_agent/experiments/r8a_dev.py`

- `--models-dir` 的默认值改为 `root/'models-v2'`。旧目录 `/home/lhohy/pt_agent_results/models` 及其中的权重保持原样，不删除，也不覆盖。
- REPORT 顶层增加 `train_recipe=RECIPE`（从 train_small 导入）。

### 1.3 新测试 `pt_agent/tests/test_r8a3_train.py`

1. `check_divergence([1.7, 1.5, 1.4])` 不抛异常；`[3.3, 1488.0]` 抛出 RuntimeError，消息中含 "epoch 2"；`[1.0, float('nan')]` 和 `[1.0, float('inf')]` 也抛出。
2. `RECIPE == "v2-clip1.0"`，`CLIP_NORM == 1.0`，`DIVERGENCE_FACTOR == 1.5`。
3. **小规模训练冒烟**：在 train_small 中抽出一个函数 `run_epochs(model, images, labels, epochs, rng, augment_fn=None, report=None)`，train() 改为调用它。测试中用随机生成的 512 张 3072 维输入和 10 类标签训练 Mlp 2 个 epoch，要求不抛异常，并且每个 batch 都经过了裁剪：用 mock 包装 `clip_grad_norm_`，断言调用次数等于 batch 数（512/128 × 2 = 8）。

## 2. 运行

1. R1–R8a2 的测试和 test_r8a3 全部通过。
2. 用新的 out 目录完整运行 `python3 -m pt_agent.experiments.r8a_dev`（默认 models-v2）。内容是：归一化选择、两个模型按 v2 重新训练、4 个模型的 reference 和规则基线。
3. 如果训练触发发散检查，或者 reference 仍然超过 1e-9：照实记录，停下来写进阻塞，不调学习率，也不放宽容差。

## 3. 汇报

内容同 R8a §7/§9：
- 两个模型每个 epoch 的 loss 和训练准确率，以及测试 top-1（只用于报告）；
- 4 个模型的 nodes、coverage、ref_diff、规则程序的行数、boots、编译耗时和峰值内存；
- 注明 `frontend: "aten"` 和 `train_recipe: "v2-clip1.0"`。

STATUS.md：
- R8a、R8a2 改为完成；
- 新增 R8a3 一行；
- 删除"阻塞"栏中 R8a2 的条目；
- 在 R8a 一行注明：旧权重（v1）保留在 `pt_agent_results/models`，不再使用。

## 4. 不要做

- 不修改 ATen 前端、reference、容差和规则规划器。
- 不用测试集选择配方或提前停止。
- 不删除旧权重和旧 REPORT。
- 不调用 LLM API。
