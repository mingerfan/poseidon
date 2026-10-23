# R8f：张量级快速验证，加速精调和 Agent 循环

用户要求（2026-10-23）：精调效率太低，设法优化。本轮不调用 LLM；新增的代码、实验和测试不读取报告集（既有回归测试按 AGENTS.md 的例外照常执行，见 §5.7）。目标是在决策完全不变的前提下，把验证提速一到两个数量级。

## 0. 诊断（Claude 于 2026-10-23 根据 R8e5 运行记录）

- ResNet20 一次 Agent 运行共 6247 秒，其中 3 次精调占 3117 秒（50%）；SiLU 占 65%。
- 每次精调做 18 次粗筛（200 张）和 3 次全量验证（1000 张），合计约 6600 次单图前向，每张约 0.15 秒。
- 时间花在 `agent_tools.validate` 上：
  - 它用 `MockBackend` 在 16384 个槽位上逐条模拟每次旋转、乘法和加法，ResNet20 每次前向约 1.9 万步；
  - 每次调用都会因为 `boot_nodes=None` 重新执行一遍 `estimate`（代价模型），只为了得到 `boot_nodes`，用于观察自举输入的幅度。
- 槽级模拟算的是数学上相同的张量函数：mux 卷积就是零填充的 same 卷积，下采样就是 `x[:, :, ::2, ::2]` 加通道补零，mean 是先求和再乘 `1/(H·W)`，linear 是 `W@x+b`，多项式在逐元素地计算。在张量层直接计算，可以去掉全部旋转和掩码运算。
- 当前配置下 `boot_input_bound=None`、`boot_abs_error=None`，所以 `boot_nodes` 只影响记录项 `boot_input_max_abs`，不影响任何可行性判定。

## 1. 新增：张量参考解释器 `pt_agent/tensor_sim.py`

`run_tensor(program, weights, images, *, observer=None) -> logits`：在张量层逐个执行 PT-DSL 运算，全程 float64，使用 torch（CPU）或 numpy。

| 运算 | 张量层实现 |
|---|---|
| pack | 不做变换（输入已是 `(N,C,H,W)` 或 `(N,F)`） |
| conv2d | 用 `kernels.conv.fold_bn` 得到 scale/shift，然后 `F.conv2d`（stride、padding 取自属性），再乘 scale、加 shift；无 BN 时加 bias。属性检查与 `lower.py` 相同，不满足时报同样的错误 |
| act | cheb：用与 `act_cheb` 相同的表达式 `u = x*a + c`（`a=2/(hi-lo)`、`c=-(hi+lo)/(hi-lo)`），系数取自 `cheb_fit`，再调用 `kernels.poly.cheb_eval(None, u, coeffs)`，它只用到算术，可以直接作用于数组；minimax_comp：与 `act_minimax` 相同的表达式；exact square：`x*x` |
| add | 逐元素相加 |
| slice/slice/pad 链 | 识别条件与 lower 相同；`x[:, :, ::2, ::2]` 后在通道维前后补零 |
| mean | 先对 H、W 求和，再乘 `1/(H*W)` |
| relayout | flatten |
| linear | `x @ W.T`，有 bias 时加 bias |
| refresh | 不做变换 |

- 只复用 kernels 中的纯数学函数（`fold_bn`、`cheb_fit`、`cheb_eval`、`minimax_coeffs`），不修改它们。
- 浮点异常的处理与 `MockBackend` 相同：在 `np.errstate(over="raise", invalid="raise")` 下计算，torch 部分出现非有限值时抛出 `BackendError`，使"发散"的判定口径与 Mock 一致。
- observer 的调用方式与 lower 相同，传入 `(name, value_array, cfg)`，值是有效元素组成的张量。mux 的复制副本与原值相同，所以 min/max 也相同。

## 1.5 原型实测（Claude 于 2026-10-23 在 Nix 中测得，留出集 20 张、batch 20）

| 模型 | Mock 加 lower | 张量原型 | 加速 | logits 最大相对差 | argmax |
|---|---|---|---|---|---|
| cnn_plain | 0.55 s | 0.039 s | 14× | 1.2e-15 | 全部一致 |
| resnet20 | 2.47 s | 0.099 s | 25× | 1.3e-14 | 全部一致 |

另外两项测量：
- 在 200 张图上，从网络中间的 act 开始重算后半段只需 0.35 s，整网需要 2.72 s：前半段宽层的卷积和高阶多项式占了大部分时间。
- `estimate`（代价模型）对 ResNet20 每次只需 0.56 s，不是瓶颈。

batch 200 时每张图的速度反而比 batch 20 慢，所以张量引擎沿用 `VAL_BATCH=20`，失败时逐张重试的语义也随之保持。原型脚本：`/home/lhohy/pt_agent_results/r8f-prototype-20261023/proto_tensor.py`，仅供参考，实现以本 spec 为准。

## 1.6 缓存（张量引擎内，三层）

所有缓存都必须做到**与不开缓存的张量引擎逐位相同**：同一份输入、同一 batch 划分、同一运算顺序，只是跳过重复计算。缓存只存在于进程内，不落盘，可以用参数关闭（`cache=False`）。

1. **结果缓存**：键为（`print_program` 规范化后的程序文本、图像集合标识、n、stop_on_divergence、engine），值为整个 validation 字典的深拷贝，命中时返回深拷贝。
   - 典型命中：`evaluate` 已在 1000 张图上验证过 LLM 候选，`tune` 随后对同一程序做的 `full(cur)` 是完全相同的计算；
   - 另外，二分回滚可能重复访问同一个前缀程序。
2. **结构化前缀激活缓存**：为每个运算计算一个结构键，由该运算及其全部祖先运算的（kind、规范化属性、输入的结构键）递归组成。键只依赖运算本身，与变量名和行号无关，所以残差等 DAG 结构也适用。
   - 缓存每个 act 运算的**输入张量**，键为（结构键、图像集合标识、batch 起始下标、batch 大小）。
   - 精调每次只改一层，这一层之前的所有输入都能命中；不同 LLM 候选之间共享的前几层也能命中。
   - 命中时同时恢复前缀部分的观测统计（rows 中的 obs_min、obs_max、max_u），保证 rows 与不开缓存时相同。
   - 采用 LRU 淘汰，内存上限默认 1.5 GiB（按数组 nbytes 统计），可以配置。
   - 某个 batch 在前缀中发生浮点异常时，不缓存这个 batch，按无缓存路径处理。
3. **小函数缓存**：`cheb_fit` 的结果按（fn、deg、lo、hi）缓存，`minimax_coeffs` 已有缓存；`estimate` 按程序文本缓存（只在 mock 路径需要 boot_nodes 时，以及 tuner 计算 cost 时使用），值返回深拷贝。
   - 这些都在新模块或调用方实现，**不修改 kernels/poly.py 和 cost.py 本身**。可以在 tensor_sim 中包一层 `functools.lru_cache`，或在 tuner 内建一个字典。

## 2. 修改 `validate`（`planners/agent_tools.py`）

增加参数 `engine: str = "mock"`。

- `engine="tensor"`：用 `run_tensor` 代替 `MockBackend` 加 `lower`，其余逻辑完全不变：分批、失败时逐张重试、rows 统计、worst_acts、agree、acc_drop、feasible。
  - 此时不调用 `estimate`，`boot_input_max_abs` 记为 `None`。
  - 如果 profile 的 `boot_input_bound` 或 `boot_abs_error` 不是 None，`engine="tensor"` 直接报错，必须使用 mock。这一条留给以后接入 GPU 自举约束时用。
- `engine="mock"` 的行为逐字节保持原样。
- 另外，mock 路径下 `boot_nodes` 由调用方传入时不再重算（现在已是如此）。tuner 在 mock 模式下把自己已经算过的 `estimate` 结果的 `boot_nodes` 传给 validate，避免重复调用代价模型。这一项只改性能，不改结果。

`tuner.tune` 和 `agent_tools.evaluate` 增加 `engine` 参数，并逐层传递下去。`confirm.confirm_select`、`_exact_check` 和报告集测试**不改**，始终使用 Mock，作为最终裁判。

## 3. 等价性验证（本轮的核心）

### 3.1 数值一致

对以下程序，在留出集 test[9000:9100] 上比较 `run_tensor` 与 Mock 加 lower：

- R8e7 的 4 个冻结程序；
- R6c、R7c-B；
- R9a E1 `rows.jsonl` 中每个模型随机抽取 10 个配置（固定种子 20261023），共 40 个，必须包含至少一个 minimax_comp 程序和一个阶数为 7 的程序。

要求：
- logits 逐元素满足 `|Δ| ≤ 1e-9·(1+|ref|)`；
- 每个 act 的 obs_min、obs_max、max_u 之差不超过 1e-9；
- 发散的图像集合完全相同。

报告每个程序的最大差值。

### 3.2 决策重放（验收标准）

用 R8e5 和 R8e7 运行目录中的 `agent_trace.jsonl`（4 个模型的全部迭代）做两件事：

1. 对每次迭代中 LLM 候选的验证结果（`fb.validation`），用 `engine="tensor"` 重新验证，比较 agree、total、n_diverged、diverged、feasible、max_u（容差 1e-9）和 acc_drop_pp（容差 1e-9）。
2. 对每次调用了精调的迭代，从同一个起点程序出发，用 `engine="tensor"` 重新运行 `tune(..., budget_s=原值, max_sweeps=1)`，逐条比较 history：phase、accepted、agree、total、max_u（容差 1e-9）、latency_pred_us（必须完全相等），并比较最终程序文本。

   如果原运行曾因 budget 提前停止（检查 history 和 seconds），在报告中注明，并用足以完成同样步骤的预算重放，不改变比较口径。

起点程序的提取方式与 `planners/agent.py` 相同，codex 自行定位。

**判定**：如果全部一致，执行 §4。如果有任何不一致，停下来汇报：列出不一致的条目、相关图像的 logits 差值和 top-2 间隔，**不调整容差，也不改代码去迁就**。

### 3.3 计时

记录以下几项的耗时，分别在三种配置下测量：mock、tensor 不开缓存、tensor 开缓存。

- ResNet20 冻结程序 `validate(n=1000)`；
- 一次完整重放的 `tune`；
- R8e5 ResNet20 第一次精调。

同时记录三层缓存各自的命中次数。

§3.2 的决策重放使用 tensor 开缓存的配置（默认配置）；另外对 ResNet20 的全部精调调用，再用 tensor 不开缓存的配置重放一次，确认两者相同。

## 4. 切换默认值（只有 §3.2 全部一致时才执行）

- `tuner.tune` 和 `agent_tools.evaluate` 的默认 `engine` 改为 `"tensor"`，`agent.run` 随之使用。
- `confirm_select`、`_exact_check` 和报告集测试仍然使用 Mock。
- `PLAN.md` 和 `pt_agent/README.md` 中说明：循环内的验证用张量参考解释器，冻结确认和最终报告用槽级 Mock。
- 不打新 tag。是否打 `pt-agent-freeze-v1.1`，由用户在看到汇报后决定。

## 5. 测试（新增 `tests/test_r8f_tensor_validate.py`）

1. 每种运算各一个小程序（conv 有无 BN、stride 1 和 2、cheb、minimax、square、下采样链、mean、linear 的 bsgs 和 diag），`run_tensor` 与 Mock 的 logits 满足 §3.1 的容差。
2. 构造一个定义域过窄、会发散的程序，两种 engine 报告的发散图像下标相同。
3. `engine="tensor"` 时不调用 `estimate`：patch 后被调用就报错。
4. profile 设置了 `boot_input_bound` 时，`engine="tensor"` 报错。
5. `engine="mock"` 的结果与修改前完全相同：取一个固定小程序，与修改前记录的 validation 字典逐项相等。
6. `confirm_select` 不接受 engine 参数，并且内部始终使用 Mock。
6a. **缓存逐位相同**：对 R8e7 的 ResNet20 和 SiLU 程序，用 `cache=True` 和 `cache=False` 分别执行一次完整的 `tune`（engine="tensor"），history 和最终程序完全相同，每次 validation 的 logits 用 `np.array_equal` 比较。
6b. **前缀命中**：构造两个只在最后一层 act 阶数上不同的程序，验证第二个程序的前缀全部命中，并且结果与不开缓存时 `np.array_equal`。结构键与变量名无关：把程序中的变量全部改名后，仍然命中同一批缓存。
6c. **LRU**：把内存上限设得很小时，缓存会淘汰旧条目，且结果不变；超出上限的单个数组不缓存。
6d. **结果缓存隔离**：命中后修改返回的字典，不影响缓存中的值。
7. 既有 327 项测试全部通过。其中 3 项历史回归测试会读取报告集：`test_r3_resnet20` 的 `test_first_500_argmax`、`test_frozen_data`，以及 `test_r8e4_split` 的 `test_01_disjoint_complete_and_first_record`。它们检查的是精确模型和数据划分本身，按 AGENTS.md 数据划分的例外照常执行，不修改、不跳过。

## 6. 运行与汇报

结果目录：`pt_result("r8f-<时间戳>")`。汇报内容：

- §3.1 的数值一致性表；
- §3.2 的重放结果：一致的条目数和总条目数，以及不一致项（如有）；
- §3.3 的计时和加速比；
- 测试总数和通过数；
- STATUS.md 新增 R8f 一行。

## 7. 不要做

- 不修改 `lower.py`、kernels、backends、cost、placement、confirm、dsl 和 intake。
- 只修改 `planners/agent_tools.py` 和 `planners/tuner.py` 的 engine 传递，以及 §2 所说的 boot_nodes 复用；`planners/agent.py` 最多只增加 engine 的传递。
- 不改 LADDER、阈值、`max_sweeps`、预算等任何精调参数。提速后是否增加 `max_sweeps` 是另一项决定，会改变 Agent 的行为，本轮不做。
- 不调用 LLM。本轮新增的代码、实验和新增测试不读取报告集；§5.7 所列 3 项既有回归测试除外（见 AGENTS.md 数据划分的例外）。
