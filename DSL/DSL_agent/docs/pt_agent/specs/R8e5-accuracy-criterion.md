# R8e5：精度可行条件改为"留出集准确率下降 ≤0.5pp"，验证批量减小，修正依赖数据的旧测试，然后跑完 R8e4

## 0. 诊断与决定（Claude 于 2026-10-22 复核；用户已决定）

- 两个基线在 1000 张留出图上的表现（Claude 实测）。精确网络的 top-1 为 92.1%：

  | 程序 | 与精确网络一致 | top-1 | max_u |
  |---|---|---|---|
  | R6c 规则（全部 cheb 127） | 999 | 92.1% | 0.83 |
  | R7c B（全部 cheb 63） | **986** | **92.2%** | 0.98 |

  在网络没见过的数据上，"与精确网络一致率 ≥99%"比"准确率不下降"严格得多：R7c B 的准确率完全没掉，一致率却只有 98.6%。
- **用户决定（2026-10-22）**：精度可行条件改为"留出集 top-1 不低于精确网络在同一批图像上的 top-1 减 0.5 个百分点"。精调的确认门槛更严：不超过 0.3 个百分点。安全条件（max_u ≤ 0.90、不发散、自举输入上界）不变。这与 M1（报告集上 ≤1pp）的口径一致，并且留出了余量。
- **内存**：127 阶的程序按 50 张一批做 Mock 时，会超出本机内存，被系统杀掉（Claude 复现过：退出码 137）。R8e 中 SiLU 运行"执行会话丢失"，很可能也是这个原因。批量统一改为 20。
- R8e4 卡住的 R7 测试，以及新出现的 R8e test_02，都是**检验流程的测试却依赖真实数据的统计量**。改为脚本化的反馈，或者只做条件断言。

## 1. 修改

路径相对 `DSL/DSL_agent/scripts/`。

### 1.1 `pt_agent/planners/agent_tools.py`

- 新增常量：
  ```python
  VAL_BATCH = 20
  MAX_ACC_DROP_PP = 0.5
  ```
  `build_context` 和 `validate` 中的批量 50 一律改为 VAL_BATCH。
- AgentContext 增加两个字段：`val_labels: np.ndarray`（来自 heldout_set 的标签）和 `val_exact_top1: float`（精确网络在 val_images 上的 top-1，百分比）。build_context 负责填写。
- `validate(program, ctx, n=None)` 的返回中增加三个字段：
  ```
  top1          程序在这 n 张图上的 top-1（百分比），发散的批按错误计
  exact_top1    精确网络在同样 n 张图上的 top-1
  acc_drop_pp   exact_top1 - top1
  ```
  agree 继续计算和报告，但不再参与可行性判定。可行条件改为：未发散；`max_u <= FEASIBLE_MAX_U`；`acc_drop_pp <= MAX_ACC_DROP_PP`；自举输入上界为 None 或未超出。
- evaluate 的 feasible 判定同步修改（原来的 `agree >= .99*total` 这一条删除），validation 中增加同样的三个字段。

### 1.2 `pt_agent/planners/tuner.py`

- 删除 `TUNER_AGREE_MIN`，新增 `TUNER_MAX_ACC_DROP_PP = 0.3`。
- `_strict(result)` 改为 `result['feasible'] and result['acc_drop_pp'] <= TUNER_MAX_ACC_DROP_PP`，粗筛和全量确认都用它。

### 1.3 `pt_agent/run.py`

mock、validation、test 阶段中按 50 张分批的地方，一律改为 `agent_tools.VAL_BATCH`。

### 1.4 `pt_agent/planners/prompts.py`：r8-v4

`PROMPT_VERSION = "r8-v4"`。TEMPLATE 中只替换这一句：

```text
and the predicted class agrees with the exact network for at least 99% of the images.
```

替换为：

```text
and the top-1 accuracy on these images is at most 0.5 percentage points below that of the exact network.
```

feedback 的 validation 中自动带上 top1、exact_top1 和 acc_drop_pp（来自 1.1）。

### 1.5 测试修正

1. **`tests/test_r7_agent.py::test_loop_selects_and_stops`**：用 patch 替换 `pt_agent.planners.agent.evaluate`，让它按迭代顺序返回以下脚本化的反馈：
   - 第 1 轮：`dict(stage='response', ...)`，由 response 解析失败自然产生，不经过 evaluate；
   - 第 2 轮：parse 失败；
   - 第 3 轮：可行，`latency_pred_us=200`；
   - 第 4 轮：可行，`latency_pred_us=100`；
   - 第 5、6 轮：可行，`latency_pred_us=100`。

   反馈的字段齐全，与 evaluate 的返回结构相同：stage、feasible、cost、validation 等。原有断言全部保留（6 行 trace、first_feasible=3、best=4、llm_calls=6、best_program=self.low、token 计数），使这项测试只检验循环的选择和停止逻辑。
2. **`tests/test_r8e_tuner.py::test_02_real_resnet_monotonic_and_feasible`**：仍然在真实的 ResNet20 和留出数据上运行（ctx 用 n_val=200，`N_COARSE=100`）。断言改为：
   - 结构保持：只有 act 的 deg 和 domain 可能改变，其余逐项相同（保留原有的逐项比较）；
   - changes 与差异一致（保留原断言）；
   - `result.feasible` 为 True 时，`validate(result.program, ctx)['feasible']` 也为 True，并且 cost_after ≤ cost_before；
   - `result.feasible` 为 False 时，result.program 与输入的 canonical 相同。

   删除无条件的 `assertTrue(result.feasible)`。
3. 新增 `tests/test_r8e5_criterion.py`：
   - **validate 的判定**：mock 掉内部的 Mock 运行，构造 exact_top1=92.0、top1=91.6、max_u=0.8 的情况 → feasible；top1=91.4 → 不可行；max_u=0.95 → 不可行；
   - `_strict`：acc_drop_pp=0.3 → True，0.4 → False；
   - `VAL_BATCH == 20`；PROMPT_VERSION 为 "r8-v4"；TEMPLATE 中包含 "0.5 percentage points"，不再包含 "at least 99%"。

## 2. 然后

1. R1–R8e4 的测试全部通过（包括修正后的两项）。
2. 按 R8e4 §3 执行：运行 `r8e4_baselines`（REPORT 中增加留出集 top1、exact_top1、acc_drop_pp），然后用新的 run_dir 运行 `r8b_dev --round R8e5 --tuner on`，4 个开发模型，LLM 调用合计不超过 24 次。中断时用 `--resume` 接续。
3. 汇报内容同 R8e4 §3，每个模型增加"留出集 acc_drop_pp"一项。STATUS.md：新增 R8e5 一行，R8e4 标为完成。

## 3. 不要做

- 不修改 TARGET_U、LADDER 和 N_COARSE；不使用报告测试集（第 0–8999 张）的任何结果。
- 除 §1.5 列出的两项和 §3.1 的兼容适配外，不修改其他旧测试。

### 3.1 允许的兼容适配（2026-10-22 修订，回应 STATUS 中的 R8e5 阻塞）

以下修改只是让旧测试跟上本轮的接口和版本变化，不改变任何断言的含义和期望数值，全部允许：

1. **版本号断言**：`tests/test_r7c_feedback.py:49`、`tests/test_r8b_generic.py:171`、`tests/test_r8c_feedback.py:111`、`tests/test_r8e_tuner.py:182` 中的 `"r8-v3"` 改为 `"r8-v4"`，只改这个字符串。
2. **FakeValidate 补字段**（`tests/test_r8e_tuner.py` 中的 test_03、test_04）：返回的 dict 中增加 `top1`、`exact_top1`、`acc_drop_pp`：
   - test_03：全量验证可行时（所有 deg ≥ 63）返回 `top1=92.0, exact_top1=92.0, acc_drop_pp=0.0`；不可行时返回 `top1=90.0, exact_top1=92.0, acc_drop_pp=2.0`；粗筛（n 不为 None）返回 `acc_drop_pp=0.0`。原有的 agree、feasible 和回退预期（保留 2 项、撤销 6 项）不变。
   - test_04：一律返回 `top1=92.0, exact_top1=92.0, acc_drop_pp=0.0`。
3. **R8c 的合成 AgentContext**（`tests/test_r8c_feedback.py:39`）：补上
   ```python
   val_labels = val_exact_logits.argmax(axis=1)
   val_exact_top1 = 100.0
   ```
   也就是说，这个合成夹具的"真实标签"取精确网络的预测，所以准确率下降就等于与精确网络的不一致率。原测试的含义保持不变。
4. **`experiments/r8b_dev.py:92`**：顶层的 `prompt_version` 改为引用 `planners.prompts.PROMPT_VERSION`，不再写死字符串。

### 3.2 以后同类情况的处理

AGENTS.md 已经补充：当 spec 修改接口、新增必填字段或升级版本号时，旧测试中只为适配而做的机械修改（版本字符串、给夹具补新增字段的中性值、元数据改为引用常量）可以直接做，并在汇报中逐条列出。改变断言含义或期望数值的修改，仍然必须停下。
