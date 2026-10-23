# R8f2：修复 flat 布局激活后尾槽非零的 kernel 缺陷，然后继续 R8f

前置：R8f 已实现张量引擎和三层缓存，344/344 测试通过；§3.1 在 `e1/mlp/random/051` 处停止（结果目录 `pt_result("r8f-20261023T092453")`）。本轮不调用 LLM。读取报告集的规则与 R8f 相同（AGENTS.md 数据划分的例外）。

## 0. 诊断（Claude 于 2026-10-23 复核 R8f 的停止项）

`e1/mlp/random/051`（fc1 → relu cheb deg 255，domain=(-36.70, 24.37) → fc2）在 100 张图上，Mock 与张量引擎的 logits 差：

- 每个类别是一个**与输入无关的常数**，不同图像之间的差异只有 2e-14；
- 10 个类别的偏移**逐个等于** `p(0) · fc2.weight.sum(axis=1)`，其中 `p(0)=0.011018203906415093` 是这个多项式在 0 处的值。

原因如下：

1. 在 flat 布局中，`size` 以后的槽位应该是 0。`pack_flat`，以及 `linear_flat` 和 `mean_mux` 的输出掩码，都保证了这一点。
2. `act` 对**所有**槽位计算多项式，所以尾槽变成 `p(0)`。Chebyshev 对 ReLU 等函数的近似一般有 `p(0) ≠ 0`；只有 minimax_comp 的最后一步是 `x*(s+0.5)`，在 0 处恰好为 0。
3. 下一层 `linear_flat` 先做 `xr = x + x.rotate(-Ip)`，把槽位 `[n-Ip, n)` 的值叠加到 `[0, Ip)`。这一步假设尾槽为 0，于是相当于输入被整体加了 `p(0)`，logits 被加上 `p(0)·W.sum(1)`。

Mock 与真实 CKKS 的逐槽语义相同，所以**编译出的加密程序确实带有这个偏移**；张量引擎算的才是预期的数学结果。这是可信 kernel 组合的正确性缺陷，不是张量引擎的错误。

影响范围：只有"flat 布局上 `p(0) ≠ 0` 的 act，后面接 linear"的程序，即使用 Chebyshev 激活的 MLP 类程序。mux 布局下，激活后的消费者（卷积、下采样、mean）都先乘掩码，或者只在网格槽位上求和；R8f 已执行的 37 个 CNN/ResNet 程序都一致到 1e-14。

冻结的 4 个程序都不受影响：mlp 使用 minimax_comp，`p(0)=0`；其余 3 个是 mux 布局。

## 1. 修复（通用）

**不变量**：FlatLayout 中 `size` 以后的槽位必须为 0。在 `kernels/layout.py` 中 `FlatLayout` 的 docstring 里写明这条不变量，只加注释。

修改 `lower.py` 的 act 分支：layout 为 FlatLayout 时，在 act 计算完之后执行修正。

1. 用与 kernel 完全相同的数值路径计算尾值 `t`：用一个长度为 1 的 numpy 零向量调用同一个 act 函数，即 `act_cheb(None, np.zeros(1), fn, deg, domain)`、`act_minimax(None, np.zeros(1), degrees, domain)`，square 为 0。这样得到的 `t` 与 Mock 在尾槽上算出的值逐位相同。
2. 如果 `t != 0`：`value = value + b.const(v)`，其中 `v[:size] = 0`，`v[size:] = -t`。这是一次明文加法，不消耗 level，不增加乘法。
3. 如果 `t == 0`：什么都不做。这保证冻结程序和所有 mux 程序发出的运算序列与修复前**逐字相同**。

同步修改：

- P2 包模板中 `ops.py` 的 act 函数，做同样的修正，保持与 lower 逐条一致；
- `tensor_sim.py` 不需要修改：张量层没有尾槽。

act 之后紧跟 act 的情况，每一次 act 都各自修正，不变量依然成立。不改动 `linear_flat`：在 linear 里给输入乘掩码会多消耗一个 level。

## 2. 测试（新增到 `tests/test_r8f_tensor_validate.py`，或新建 `tests/test_r8f2_flat_tail.py`）

1. **不变量**：对 flat 布局的每种 act（cheb，域对称和不对称各一个；minimax；square），在 Mock 中 act 之后 `size` 以后的槽位全部 `== 0.0`，严格相等。
2. **e1/mlp/random/051**：修复后 Mock 与张量引擎的 logits 满足 R8f §3.1 的容差。
3. **冻结程序不变**：对 P2 的 7 个程序，修复后 `EmitBackend.render()` 的文本与常量，与修复前（用 git 中 fb6d7f3c 版本的 lower 生成，或者测试中保存的修复前文本）完全相同。也可以断言这些程序中没有任何 act 触发修正，并与 P2 的包导出结果比较。
4. **P2 一致性**：P2 的 12 项测试全部通过；包的 forward 与修复后的 lower 发出完全相同的文本。
5. **代价**：对一个触发修正的 MLP 程序，`estimate` 的 boots 与修复前相同，只多出一次 add_cp。
6. 既有 344 项测试全部通过。

## 3. 继续 R8f

修复并通过测试之后，按 R8f spec 从头完整执行 §3.1（46 个程序全部执行，包括已通过的 37 个），然后执行 §3.2、§3.3，并按条件执行 §4。

§3.2 重放有一项调整：R8e5/R8e7 轨迹中，对于含有触发修正的 act 的程序（MLP 类的 Chebyshev 候选），trace 里记录的 Mock 结果是带缺陷的语义算出来的，不能作为比较基准。处理方式：

- 对这些条目，用修复后的 `engine="mock"` 重新运行（mlp 的 Mock 很快），把这次结果作为基准，再与 `engine="tensor"` 比较；
- 单独列出这些条目：原记录与修复后 Mock 的差异，以及 feasible 或 accepted 是否因修复而改变；
- 其余条目仍然与 trace 中的记录比较。

判定规则与 R8f 相同：除上述受影响条目之外，任何不一致都停下来汇报。

## 4. 受影响的历史数据（只重算，不覆盖）

R9a 的 E1、E4，以及 R9a2 的 e1confirm 中，mlp 的 Chebyshev 配置都是用带缺陷的语义得到的。修复后在新目录 `pt_result("r8f2-mlp-recheck-<时间戳>")` 中重算这些行：

- 只重算 mlp 中含触发修正的 act 的配置，图像范围与原实验相同；
- 使用 `engine="tensor"`，前提是 §3 已通过；否则使用 mock。

汇报：
- 重算的行数；
- feasible_like 和 robust_feasible 的变化；
- E1 中 mlp 的延迟范围和 Pareto 前沿是否改变。

原 R9a、R9a2 目录保持不变。

## 5. 汇报

- 修复的 diff 摘要；
- §2 的测试结果；
- R8f §3.1、§3.2、§3.3 的完整结果（按 R8f §6 的格式），以及是否执行了 §4；
- §4 的重算结果；
- STATUS.md 中把 R8f 一行更新为完成或阻塞，并新增 R8f2 一行。

## 6. 不要做

- 不修改 `linear_flat` 和其他 kernel 的计算，`layout.py` 只加 docstring。
- 不修改 `lower.py` act 分支以外的部分。
- 不调整 R8f 的任何容差。
- 不改动 tuner 参数。
- 不调用 LLM。

## 7. 补充（Claude 于 2026-10-23 复核默认切换后的停止项）

R8f 的 §3.1–§3.3 已全部通过：46/46 数值一致；重放 18 个候选、13 次精调、224 条 history 全部一致，另有 6 次不开缓存的重放；9 组计时完成。按 R8f §4 把默认引擎切到 tensor 后，`tests/test_r8e_tuner.py:54`（`test_01_validate_matches_evaluate`）的 max_u 出现 4.44e-16 的差异，导致严格相等失败。

**判定**：这不是缺陷。这个测试的本意是"在同一引擎下，validate 与 evaluate 的结果相同"。切换后，`validate` 默认仍是 mock，而 `evaluate` 默认是 tensor，两边比较的已经是不同引擎；两种引擎的差异在 R8f §3.1 中已经证明不超过 1e-9。

### 7.1 旧测试的口径修订（本 spec 明确授权）

`test_01_validate_matches_evaluate`：两边都显式传入 `engine="mock"`，保留全部严格相等断言不变。再在同一个测试类中新增 `test_01b`：两边都用 `engine="tensor"`，同样严格相等。

除此之外，其余旧测试和断言一律不改。如果在其他入口回归中出现同类"比较双方默认引擎不同"的严格相等失败，用同样的方式（双方显式传入同一 engine）处理，并在 STATUS 中逐项列出；如果是其他性质的失败，停下来汇报。

### 7.2 exact 模式下的尾槽（同一类缺陷，一并修复）

当前修复在 `act_mode == "exact"` 时把尾值固定为 0。但 exact 模式通过 `MockBackend.apply_exact` 对所有槽位计算 `EXACT[fn]`，对于 `sigmoid(0)=0.5`、`exp(0)=1`，尾槽同样非零。flat 布局上接这类激活再接 linear 时，`_exact_check` 会把这个偏移误报为程序与参考网络不一致。

修订：exact 模式下 `tail = EXACT[fn](np.zeros(1))[0]`（square 为 0），按同样的方式用明文加法清零。cheb/minimax 分支不变。

新增测试：flat 布局上的 sigmoid 和 exp，在 exact 模式下 act 之后尾槽严格为 0；并且一个"linear → sigmoid → linear"的小 MLP 通过 `_exact_check`（logit 差 ≤1e-9）。冻结程序都不含 flat 上的 sigmoid/exp，运算序列不受影响；P2 的 7 个程序发出的文本仍须与修复前相同。

### 7.3 继续

完成 7.1、7.2 后：

1. 重跑全部测试，包括切换后的入口回归，即 STATUS 中余下的 3 项；
2. 执行 §4 的 155 行 MLP 历史重算；
3. 按 §5 汇报，STATUS 中把 R8f 和 R8f2 标记为完成。

不需要重跑 R8f §3 的数值对照、重放和计时，除非 7.2 改动了任何非 exact 模式的运算序列，此时必须重跑。

## 8. 补充二（Claude 于 2026-10-23 复核 `test_r8c_feedback.py:73` 停止项）

**判定**：这次停得对，暴露的是 R8f §4 的设计疏漏，旧测试本身没有问题。

`planners/prompts.feedback_message` 把 `evaluate` 的完整结果 `fb` 序列化后发给 LLM，其中包括 `validation.boot_input_max_abs`、max_u、rows 等。R8e5 ResNet20 的 6 条反馈消息中都含有这个字段。如果 `evaluate` 默认使用 tensor：

- `boot_input_max_abs` 从数值变成 null；
- max_u、obs_min/obs_max 等数值会有 1e-16 级的末位差异。

**LLM 看到的文本就会改变**，冻结 Agent 的 LLM 交互不再逐字相同。R8f §3.2 的重放只覆盖了验证结果和精调决策，没有覆盖面向 LLM 的反馈文本。

### 8.1 修订 R8f §4 的切换范围

- **`agent_tools.evaluate` 的默认 engine 改回 `"mock"`**。它是面向 LLM 的路径，每次迭代只调用一次。`agent.run` 调用 evaluate 时不传 engine。
- **`tuner.tune` 的默认 engine 保持 `"tensor"`**。精调的结果中只有 `latency_pred_us`、`boots_pred` 和 `changes` 进入 LLM 反馈；R8f §3.2 已证明这些逐条一致，并且精调占了主要耗时。
- `validate`、`confirm_select`、`_exact_check` 和报告集测试保持 mock。
- 旧测试 `test_r8c_feedback.py` 一律不改。§7.1 对 `test_01` 的修订保留：它比较的是 validate 与 evaluate 在同一 engine 下的一致性，仍然成立。

### 8.2 新增测试

1. **反馈文本不变**：对 R8e5 ResNet20 和 mlp 的 agent_trace 中每一条已发送的 user 反馈消息，取对应迭代的 LLM 回复，用当前代码执行 `evaluate`（默认参数）和 `feedback_message`，生成的文本必须与 trace 中的原文**逐字相同**。
   - tuned 部分取 trace 中记录的 tuned 结果（精调已在 R8f 中证明一致）。
   - 受 flat 尾槽修复影响的 mlp 候选（R8f2 §3 中的 `mlp/iteration-2`）预期会不同。单独列出这一条与原文的差异，并验证差异只来自修复本身：改用修复前的 lower 生成时文本相同，或者差异字段只限于该 act 下游的统计量。
2. `agent.run` 中：evaluate 实际使用 mock，tune 实际使用 tensor（通过 patch 记录调用参数）。

### 8.3 继续

完成 8.1、8.2 后，按 §7.3 继续：

1. 重跑全部测试，包括余下的 105 项；
2. 执行 155 行 MLP 历史重算；
3. 汇报，并在 STATUS 中把 R8f、R8f2 标记为完成。

PLAN 和 README 中关于默认引擎的说明，同步改为"精调用张量引擎，面向 LLM 的 evaluate、冻结确认和报告用 Mock"。如果再出现其他性质的失败，停下来汇报。

## 9. 补充三（Claude 于 2026-10-23 复核 §8 核对中的两项停止）

### 9.1 比较基准改为冻结 tag 的代码，不再用 R8e5 的旧 trace

`n_diverged` 和 `diverged_indices` 在冻结 tag `pt-agent-freeze-v1`（048f737c）的 `validate` 中就已经存在，R8e5 的 trace 早于冻结。因此"反馈文本不变"的含义是：**当前代码与冻结 tag 的代码，对同一批 LLM 回复生成的 feedback 文本逐字相同**，而不是与 R8e5 trace 中的原文相同。

做法：
1. 用 `git worktree` 检出 048f737c 到临时目录，与 R8f2 的 flat 尾槽修复组合：在该 worktree 中只应用 lower.py act 分支的修复，可以从当前 lower.py 的 diff 中取出这一段。
2. 以 R8e5 四个模型 trace 中的 LLM 回复为输入，依次重演 `agent.run` 的每一轮：
   - evaluate；
   - 需要精调时执行 tune，两边各自用自己的代码；
   - 更新 best；
   - 生成 feedback_message。
3. 比较两边每一轮生成的 user 消息文本，必须逐字相同。

不调用 LLM，回复全部取自 trace。只有生成反馈所需的计算，才在两边各自执行。

### 9.2 精调结果中进入 best 的统计改为最终 Mock 复核

`agent.run` 把精调最终验证的 `agree`、`max_u`、`boot_input_max_abs`、`worst_acts` 写入 `best_feasible_so_far`，并发给 LLM。tensor 下 `boot_input_max_abs` 为 None，max_u 末位也可能不同。

修订 `tuner.tune`：
- 当 `engine="tensor"` 时，在 `finish` 确定最终程序和 feasible 之后，对最终程序在**同一个 ctx**（完整验证集，`stop_on_divergence` 默认）上执行一次 `validate(..., engine="mock", boot_nodes=最终程序的 estimate 结果)`；
- 用这次的 Mock 结果替换 history 末条 `'result'` 记录中的统计字段，以及 TuneResult 对外暴露的验证统计；
- 所有决策（接受、回滚、最终程序、feasible）仍然来自 tensor 的过程，不因复核结果改变。

一致性检查：Mock 复核得到的 feasible、agree，必须与 tensor 最终结果中的对应值相同（R8f 已证明两者一致）。如果不同，抛出明确的异常并停止，不静默采用任何一方。

原来的 mock 精调中，最终统计本来就来自 mock 的 full 验证，验证的是同一个程序、同一批图像，所以替换后的字段与冻结代码逐位相同。代价是每次精调多一次 1000 张图的 Mock 验证（ResNet20 约 133 s），相比节省的十几分钟可以接受。

新增测试：
- tensor 精调后，`result.history[-1]` 的统计字段与 `engine="mock"` 精调的末条记录逐项相等，包括 `boot_input_max_abs` 和 worst_acts；
- 构造 Mock 复核与 tensor 结论不一致的情形（patch 其中一方），会抛出异常。

### 9.3 继续

1. 完成 9.1、9.2；
2. 执行 9.1 的逐轮反馈文本比较：四个模型全部轮次。预期唯一的差异来源是 flat 尾槽修复，而 9.1 中两边都已带上修复，所以应当**全部逐字相同**。任何差异都停下来汇报；
3. 然后按 §7.3 重跑全部测试，执行 155 行 MLP 重算，汇报，把 R8f、R8f2 标记为完成。

§8.2 第 1 条中"与 trace 原文逐字相同"作废，由 9.1 取代；§8.2 第 2 条保留。

**9.1 的运行时间**：冻结 tag 一侧，如果每次都用 mock 重新执行 13 次精调，大约需要 4–5 小时。允许复用 R8e5 trace 中已记录的 `tuned` 结果作为 tag 一侧的精调输出，前提是同时满足：

1. 用 `git diff` 确认，R8e5 运行时的源码版本与 048f737c 之间，`planners/tuner.py` 以及 `validate`、`estimate` 的代码路径没有影响精调结果的差异。R8e5 运行时的源码版本见其 REPORT 中的 `git_rev`；如果没有记录，就不能复用。
2. R8f §3.2 已证明这 13 次精调的 history 与 trace 逐条一致。

不满足时，在 tag 一侧真实执行精调。当前代码一侧必须真实执行（tensor 加 Mock 复核）。

## 10. 补充四：用轻量证明替换 §9.1 的整轮重放（Claude 于 2026-10-23，用户反映太慢）

**现状**：§9.1 的逐轮重放在冻结 tag 一侧不满足复用条件，正在用 Mock 重新执行 13 次精调（每次 16–24 分钟），加上两侧各 18 次 evaluate，总计约 6 小时，已运行 75 分钟。

**判定**：没有必要整轮重放。"当前代码与冻结 tag 生成的反馈逐字相同"可以拆成三个已经证明或可以廉价验证的部分：

- **A. evaluate 路径**：当前代码中 evaluate 默认使用 mock。从 evaluate 到反馈文本，经过的 validate（mock 分支）、lower、kernels、backends、`prompts.feedback_message` 和 `agent.run` 中拼接反馈的代码，与"冻结 tag 加尾槽修复"相比，只多了 engine 和 cache 参数的传递，以及 boot_nodes 的复用。
- **B. 精调决策**：R8f §3.2 已证明，当前 tensor 精调对 13 次精调、224 条 history 与 trace 逐条一致，最终程序也相同。
- **C. 精调对外统计**：§9.2 中，当前代码对最终程序做一次 Mock 复核，得到对外的统计。冻结代码的对外统计，就是它最后一次 Mock 全量验证的结果：同一个程序、同一个 ctx、同一个 validate 函数，结果是确定的。

### 10.1 立即停止当前的 §9.1 重放

终止 `/tmp/r8f2_s9_feedback_outer.py` 及其 Nix 子进程，已经产生的结果目录保留。

### 10.2 替代验证（预计 1 小时以内）

1. **A 的静态证明**：对上面列出的文件，生成"冻结 tag 加尾槽修复"与当前代码之间的 diff。逐段确认，除 engine/cache 参数传递、boot_nodes 复用、§9.2 的 Mock 复核以及尾槽修复之外，没有其他改动，并把 diff 保存到结果目录。如果发现其他改动，停下来汇报。
2. **A 的动态抽查**：
   - cnn_plain 和 mlp 全部迭代：两侧各运行 evaluate（mock），比较 fb 字典逐项相等。这两个模型很快。
   - ResNet20 和 SiLU：各只取**第 1 次迭代**做同样的比较。
   - 其中受尾槽修复影响的 `mlp/iteration-2`，按 §8.2 的方式单独说明。
3. **C 的验证**：对 13 次精调的最终程序（取 trace 中 tuned 的 source），用当前代码执行一次 `validate(..., engine="mock")`，即 §9.2 的复核调用。将得到的 agree、total、max_u、acc_drop_pp、feasible、boot_input_max_abs、worst_acts，与 trace 中该次精调 history 末条记录的对应字段比较，必须逐项相等。
   - trace 由冻结之前的代码生成，如果某字段因冻结前后的已知差异（例如 n_diverged）在 trace 中不存在，只比较两边都有的字段，并在报告中列出。
4. **反馈文本**：用第 2、3 步得到的 fb 和 tuned 统计，分别调用当前代码和冻结 tag 代码中的 `feedback_message`，以及 `agent.run` 中构造 best 的那段逻辑（可以抽成函数调用，也可以按源码等价重现）。两侧生成的文本逐字比较。这一步是纯函数计算，不需要 Nix 以外的重计算。
5. 第 1–4 步全部通过，即视为 §9.1 已满足，然后按 §9.3 继续：重跑全部测试，执行 155 行 MLP 历史重算，并把 R8f 和 R8f2 标记为完成。

任何一步发现不一致，都停下来汇报。

### 10.3 补充：P1 路径解耦改动的处理（Claude 于 2026-10-23 复核 §10 静态核对的停止项）

静态 diff 中，`kernels/poly.py` 的 `MINIMAX_DATA` 从 `<repo>/third_party/dacapo/...` 改为 `DACAPO_ROOT/...`。这是 P1 可移植性改造的已提交改动（e4583b24，用户 10/23 批准），不属于 R8f/R8f2 的改动。

Claude 已核对两个位置的 `sgn151527.txt`：内容完全相同（`diff` 无差异，均为 1538 字节）。因此 `minimax_coeffs` 读到的系数相同，计算结果不变；P1 当时也验证过 cnn_plain minimax 仍然编译出 72 次自举。

**处理口径**：把"P1 路径解耦（提交 72efa58b、e4583b24、883c9b14 中只改变文件读取位置、import 来源或路径常量的改动）"列为 §10.2.1 允许的第五类变更。

条件是：改动前后读到的数据内容相同。由 codex 对每处这类改动所涉及的数据文件做内容比较（`diff` 或数值读取后 `np.array_equal`），把结果写入报告。如果内容不同，或者这类改动之外还有其他未列出的变更，仍然停下来汇报。

完成后按 §10.2 的第 2–5 步继续，不需要重新执行已完成的静态核对，只补上这一类的确认即可。

### 10.4 补充：测试组时限（Claude 于 2026-10-23 复核完整回归超时）

§10.3 的全部证明都已通过：静态核对、P1 数据内容比较、9/9 动态抽查、13/13 最终 Mock 复核、18/18 轮反馈文本逐字一致；完整回归中已有 341 项通过。张量测试组因编排脚本沿用 1200 秒时限而退出 124。原因是 §9.2 给每次精调增加了一次完整的 Mock 复核，而 6a 的缓存对照要在 ResNet20 和 SiLU 上各执行完整精调。这是时限设置问题，没有任何断言失败的证据。

授权：
- 只把张量测试组的墙钟时限提高到 3600 秒，其他资源限制（内存、磁盘、trusted 运行器）不变；
- 只重跑**未完成的 16 项**，已有结果的 341 项不重跑；
- 然后执行 155 行 MLP 历史重算；
- 全部完成后，把 R8f 和 R8f2 标记为完成，并按 §5 汇报。

如果在 3600 秒内仍未完成，或者出现任何断言失败，停下来汇报。
