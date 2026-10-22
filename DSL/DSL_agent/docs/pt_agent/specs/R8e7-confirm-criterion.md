# R8e7：确认阶段按结果判定；验证中的发散逐张计算；重做 R8e6 的离线确认

## 0. 诊断（Claude 于 2026-10-22 复核 R8e6）

R8e6 中 ResNet20 和 SiLU 的 6 个候选都没有通过确认，两个基线也都没有通过。逐项分析如下：

| 候选 | 确认集准确率下降 | max_u | 发散 | 原判定 |
|---|---|---|---|---|
| SiLU 的 6 个（160–171 次自举） | 0.0 / −0.3pp | 0.94 | 无 | 只因 max_u > 0.90 被拒 |
| 规则基线 R6c（383 次，cheb 127，余量 1.5） | 0.3pp | 0.906 | 无 | 只因 max_u > 0.90 被拒 |
| ResNet20 的 6 个候选、R7c B | 7.4–8.4pp | 极大 | 有 | 被拒 |

存在三个问题：

1. **确认阶段不应要求 max_u ≤ 0.90。** 这个门槛是循环内的安全余量。max_u 是一批图上的最大值，换一批新图，它天然会变大：定义域最宽的规则基线也有 0.906。确认阶段应当按**结果**判定，即准确率下降（发散的图计为错误）。用户选择方案时，选项写的也是"取第一个准确率下降 ≤0.5pp 的"；max_u 和"不能有任何发散"是 Claude 写 spec 时额外加上的，本轮去掉。
2. **validate 按整批计错**：一批 20 张中只要 1 张发散，整批都变成非有限值，20 张全部记为错误。
3. **validate 遇到发散就跳出循环**：剩余的图不再计算，却仍然计入分母。

ResNet20 的 7–8pp 下降，主要来自第 2、3 点。报告集的 test 阶段已经有逐张重算，validate 没有。

## 1. 修改

### 1.1 `pt_agent/planners/agent_tools.py`：validate

签名改为 `validate(program, ctx, n=None, *, boot_nodes=None, stop_on_divergence=True)`。

- 某一批抛出 BackendError（非有限值）时，对这一批的每张图单独重算（MockBackend batch=1）：
  - 单张仍然出错的，记为发散，计入 `diverged_indices`（以 ctx.val_images 中的下标记录），按分类错误计；
  - 单张正常的，照常统计 correct、agree，以及 max_u、obs_min、obs_max。
- `stop_on_divergence=True`（循环内的默认值）：出现第一张发散的图后停止，`diverged=True`，`top1` 和 `acc_drop_pp` 记为 None，feasible 为 False。这与现在的循环内行为一致，并且不再报告一个错误的准确率。
- `stop_on_divergence=False`：处理完全部图像。`top1` 中发散的图按错误计；`diverged = len(diverged_indices) > 0`。
- 返回值中增加 `n_diverged` 和 `diverged_indices`。
- 循环内 feasible 的判定不变：未发散；max_u ≤ 0.90；acc_drop_pp ≤ 0.5；自举输入上界满足。

### 1.2 `pt_agent/planners/confirm.py`

- 调用 `validate(..., stop_on_divergence=False)`。
- 判定改为：
  ```python
  passed = result["acc_drop_pp"] is not None and result["acc_drop_pp"] <= MAX_ACC_DROP_PP and boot_ok
  ```
  其中 boot_ok 表示自举输入上界为 None 或未超出。max_u 和 n_diverged 只记录、不参与判定。发散的图已经按错误计入准确率。
- `MAX_CONFIRM_EVALS` 保持为 6，排序方式不变。

### 1.3 `pt_agent/experiments/r8e6_reconfirm.py`

新增 `--round` 参数（默认 "R8e6"），本轮用 "R8e7"。基线评估同样改用 `stop_on_divergence=False`。

## 2. 测试（test_r8e7_criterion.py，用 nixrun）

1. **逐张重算**：用一个假的 lower 或 backend，让第 2 批中只有第 3 张图产生非有限值，断言：
   - 只有这 1 张计入 diverged_indices；
   - 其余 19 张正常计入；
   - `stop_on_divergence=False` 时，所有批次都被处理，`top1` 的分母为全部图像。
2. **stop_on_divergence=True**：遇到发散后，`top1` 和 `acc_drop_pp` 为 None，feasible 为 False。
3. **确认判定**：patch validate，max_u=0.95、acc_drop_pp=0.3、n_diverged=1 → passed；acc_drop_pp=0.6 → 不通过；acc_drop_pp=None → 不通过。
4. R1–R8e6 的测试全部通过。旧测试中如有断言"确认时 max_u > 0.90 必须不通过"，按本轮新判定同步修改，并在汇报中列出。

## 3. 运行（不调用 API）

`python3 -m pt_agent.experiments.r8e6_reconfirm --round R8e7`：对 R8e5 的 4 个模型的候选池和两个基线，重新确认；被选中的程序编译，并在报告集（8000 张）上测试。

## 4. 汇报

- 每个模型每次确认的 acc_drop_pp、n_diverged、max_u 和是否通过；
- 选中候选的 boots，以及报告集 top1 / torch_top1_report / 下降 / n_diverged；
- 两个基线的同样指标。

STATUS.md：新增 R8e7 一行；M1 第 1–4 条以本轮 resnet20 的结果为准。

## 5. 不要做

- 不修改循环内的可行性条件和 MAX_CONFIRM_EVALS；不调用 LLM；不修改候选源码。
- 报告集结果不参与选择。
