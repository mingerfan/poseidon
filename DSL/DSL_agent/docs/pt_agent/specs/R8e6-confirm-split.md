# R8e6：独立确认集，冻结时做一次确认；对 R8e5 的候选离线补做确认

## 0. 诊断与决定（Claude 于 2026-10-22 复核；用户已决定）

- R8e5 中 ResNet20 冻结的程序（精调后，189 次自举）在留出集上 top-1 为 92.7%，**高于**精确网络的 92.1%；但在报告集上只有 90.29%，比 torch 低 **1.4pp**。其余 3 个模型都通过了。
- Claude 在报告集前 1000 张上诊断：只有 1 张图越出定义域；把定义域放宽到覆盖校准范围后，精度反而更差（86.5%）。所以**主因不是定义域覆盖，而是选择偏差**：在同一批 1000 张留出图上，用 LLM 和精调对几百个候选反复比较，选出的程序恰好在这 1000 张上"运气好"。1000 张图上的准确率差，估计误差约为 ±0.5pp。
- **用户决定（2026-10-22）**：增加独立的**确认集**，即测试集第 8000–8999 张，只在冻结时使用：
  - 候选按预测延迟从低到高依次确认，取第一个"准确率下降 ≤0.5pp、max_u ≤0.90、不发散"的程序冻结；
  - 循环内仍然使用留出集（第 9000–9999 张）；
  - **报告集改为第 0–7999 张**。M1 第 5 条的 40 张加密图（索引 0–67）仍在报告集中。

  AGENTS.md 已经同步。
- R8e5 的 agent_trace 完整保存了每一轮的 LLM 程序（assistant 消息中的 program）和精调后的源码（`tuned.source`），所以本轮**不调用 LLM、不重新精调**，直接从轨迹中取出候选并做确认。

## 1. 修改

路径相对 `DSL/DSL_agent/scripts/`。

### 1.1 `pt_agent/data/cifar.py`

```python
CONFIRM = range(8000, 9000)
REPORT = range(0, 8000)

def confirm_set(normalization=(MEAN, STD)):
    """测试集第 8000–8999 张及其标签。"""
```

`report_test_set` 改为返回第 0–7999 张。HELDOUT 保持不变。

### 1.2 `pt_agent/planners/confirm.py`（新）

```python
MAX_CONFIRM_EVALS = 6

@dataclass
class Candidate:
    source: str
    origin: str              # "llm" 或 "tuned"
    iteration: int
    latency_pred_us: float
    boots_pred: int

def build_confirm_context(ctx) -> AgentContext:
    """与 ctx 共用 graph、weights、calibration 等字段；val_images、val_labels 换成 confirm_set
    （归一化方式与 ctx 相同），val_exact_logits 和 val_exact_top1 在确认集上重新计算，
    val_split="test[8000:9000]"。probe 等其他字段沿用 ctx。"""

def confirm_select(candidates: list[Candidate], ctx, *, max_evals=MAX_CONFIRM_EVALS) -> dict:
    """按 (latency_pred_us, iteration, origin) 升序排序，并按 source 字符串去重。依次对每个候选执行
    validate(parse(c.source), confirm_ctx)，判定与留出集上的 feasible 相同：未发散、max_u ≤ 0.90、
    acc_drop_pp ≤ 0.5、自举输入上界满足。取第一个通过的候选；最多评估 max_evals 个。
    返回 dict(selected=Candidate 或 None, evals=[每次评估的 iteration、origin、latency_pred_us、
    boots_pred、top1、exact_top1、acc_drop_pp、max_u、diverged、passed])。"""
```

### 1.3 `pt_agent/planners/agent.py`

- 循环中，所有在留出集上 feasible 的候选都放进候选池，包括 LLM 原程序和精调结果；
- 循环结束后，调用 confirm_select 选出冻结程序。`best_program` 为被选中的源码；如果没有候选通过，status 为 `"confirm_failed"`，best_program 为 None；
- 确认集的结果**不进入**任何 LLM 消息（此时循环已经结束）；
- 返回值中增加 `confirm`（即 confirm_select 的返回）和 `pool_size`。

### 1.4 `pt_agent/run.py`

- test 阶段使用 `report_test_set`（8000 张）。REPORT 记录 `test_split="test[0:8000]"`，`torch_top1_report` 在这 8000 张上计算。
- M1 第 3 条的判定不变：`top1_test >= torch_top1_report - 1.0`。
- agent 的 status 为 confirm_failed 时，不编译也不测试，照实记录。

### 1.5 `pt_agent/experiments/r8e6_reconfirm.py`（新，不调用 API）

对 R8e5 的 4 个模型（run_dir 为 `pt_result("r8e5-dev-20261022T041646")`）：

1. 从 `<model>/agent_trace.jsonl` 取出候选：
   - 每一轮的 LLM 程序：从 assistant 消息的 JSON 中取 program，仅保留该轮 fb.feasible 为 True 的；
   - 每一轮的精调结果：`tuned.source`，仅保留 tuned.feasible 为 True 的。

   每个候选的 latency_pred_us 和 boots_pred 用当前的 cost.estimate 重新计算，并与轨迹中记录的值对照，写入 REPORT。数值不一致时只记录，不停止。
2. 用 `build_context`（留出集）和 `build_confirm_context` 构造 ctx，执行 confirm_select。
3. 对被选中的程序，执行 `run.py` 的 `check,mock,count,emit,compile,test` 阶段（使用新的报告集）。程序源码原样使用。
4. 两个基线（R6c、R7c B）在确认集和新报告集上各评估一次，用于对照。

REPORT 中，每个模型记录：
- 候选池大小、每次确认评估的结果、选中候选的来源、iteration、boots 和预测延迟；
- 编译 plan 的 boots；报告集 top1、torch_top1_report、下降值、n_diverged；
- 与 R8e5 原冻结程序的对照。

## 2. 测试（test_r8e6_confirm.py，用 nixrun）

1. confirm_set、heldout_set、report_test_set 两两不相交，并集为 0–9999；encrypted_test 的 40 个索引都在 REPORT 中。
2. **confirm_select 的顺序与停止**：用 patch 替换 validate，让前两个最快的候选不通过、第三个通过，断言选中第三个、evals 长度为 3；全部不通过时 selected 为 None，并且 evals 长度不超过 MAX_CONFIRM_EVALS；source 相同的候选只评估一次。
3. **agent 循环**：用 FakeClient 跑 2 轮，并 patch confirm_select，断言候选池包含所有 feasible 候选，并且 best_program 等于被选中候选的源码；confirm_failed 分支下 best_program 为 None。
4. **轨迹重建**：在 R8e5 的 resnet20 轨迹上，重建得到的候选数等于"feasible 的 LLM 程序数 + feasible 的 tuned 数"，每个源码都能通过 parse。
5. R1–R8e5 的全部测试通过。报告集从 9000 张改为 8000 张，旧测试中仅涉及这个张数的断言，按 AGENTS 的兼容适配例外同步修改，并在汇报中列出。

## 3. 运行

1. 测试全部通过。
2. 运行 `r8e6_reconfirm`（不调用 API）。

## 4. 汇报

- 测试输出和 §1.5 的对照表。
- STATUS.md：新增 R8e6 一行；M1 第 1–4 条以本轮各模型的结果为准，测试口径为报告集 8000 张。

## 5. 不要做

- 不调用 LLM，不重新精调，不修改候选源码。
- 确认集只在 confirm_select 中使用，报告集只在 test 阶段使用，两者的结果都不回传、不影响候选生成。
- 不放宽 0.5pp 的门槛，也不调整 MAX_CONFIRM_EVALS 以凑出结果。

## 6. 修订（2026-10-22，回应 STATUS 中的 R8e6 实现前核对）

以下 3 项都是本轮改动的直接结果，全部允许。其他算法、门槛和断言保持不变。

1. **`pt_agent/planners/agent_tools.py` 纳入修改范围**：AgentContext 增加字段 `normalization: tuple = (MEAN, STD)`，build_context 把传入的 normalization 原样保存进去。build_confirm_context 用 `ctx.normalization` 读取确认集；确认集图像按 graph 的输入形状适配（与 heldout 的处理相同），probe 等字段沿用原 ctx。
2. **`tests/test_r8e4_split.py`**：
   - 第 25–26 行的"两集合互斥且并集为全集"，改为三集合 HELDOUT、CONFIRM、REPORT 两两互斥，且并集为 `set(range(10000))`。这是本轮划分的新不变量。
   - 第 32–33 行的 9000 和 8999，改为 8000 和 7999。
   - 第 101 行合成报告元数据中的 test_split，改为 `"test[0:8000]"`。
3. **旧的 agent 编排测试**（`tests/test_r7_agent.py:100` 附近的 test_loop_selects_and_stops，`tests/test_r8e_tuner.py:175` 附近的 test_05_agent_tuner_on_off）：在原有的 patch 之外，再 patch `pt_agent.planners.agent.confirm_select`，替换为下面这个假函数。它返回候选池中预测延迟最低的候选，也就是确认集出现之前的选择规则，所以这两项测试原有的全部断言（best_iteration、best_program、llm_calls 等）保持不变：
   ```python
   def fake_confirm(candidates, ctx, **kwargs):
       ordered = sorted(candidates, key=lambda c: (c.latency_pred_us, c.iteration, c.origin))
       return dict(selected=ordered[0] if ordered else None, evals=[])
   ```

以后凡是 spec 修改数据划分、选择流程这类全局约定，我会在 spec 中直接列出受影响的旧测试和改法。
