# R9b：三组对照（只用 LLM / 只用搜索 / LLM + 搜索）、LLM 直接写代码的基线、专家 DSL 对照、消融

前置：Agent 已冻结（`pt-agent-freeze-v1`）；R9a 已完成或已经开始运行。本轮的数据约定与 R8e7 相同：
- 留出集（第 9000–9999 张）在循环内使用；
- 确认集（第 8000–8999 张）只在冻结候选时使用；
- 报告集（第 0–7999 张）只用于最终报告。

LLM 调用预算：本轮合计不超过 **140 次**（累计约 185/300），分配见 §9。目标日期：2026-10-24～28。

## 0. 目的

论文的核心对照（PLAN §6.5 RQ4）：在**同一套旋钮、同一评估口径**下，比较三种方法：

| 组 | 方法 | 回答的问题 |
|---|---|---|
| G1 LLM + 搜索 | 完整的冻结 Agent | 我们的方法 |
| G2 只用 LLM | 冻结 Agent，但关闭精调 | 搜索带来了什么 |
| G3 只用搜索 | 不调用 LLM：从规则程序出发，做方法切换、阶数阶梯、定义域、线性算法的坐标搜索，评估预算与 G1 相同 | LLM 带来了什么 |

同时完成以下基线：
- B1：LLM 直接写 Hecate 代码；
- B1+：写完后按测试结果迭代修复（FHECrafter 式）；
- B4：Dacapo 专家 DSL（resnet20_silu）；
- B3：规则规划器，已有结果，直接复用。

另外，在 resnet20 上做 5 项消融。

## 1. 冻结行为保护（必须先做）

- 本轮需要给 `planners/prompts.py`、`planners/agent.py` 和 run.py 增加消融开关（§6）。**默认参数下，行为必须与冻结版本逐字节相同。**
- 第一步：在修改任何代码之前，用当前代码（即 048f737c 的行为）为 4 个开发模型生成两个"金标准"文件，保存到 `pt_agent/tests/fixtures/r9b_frozen/<model>/`：
  - `system_prompt.txt`（`build_system_prompt(ctx)`，ctx 的 n_val=50）；
  - `feedback.json`：对 R7c B 程序在 n_val=50 的 ctx 上执行 `evaluate`，再调用 `feedback_message(fb, 1, None)` 得到的字符串。
- 测试：修改完成后，默认参数下的输出与金标准**完全相等**。

## 2. G1 / G2：冻结 Agent 的重复运行

- 使用 `experiments/r8b_dev.py`（不修改其默认行为），新增参数 `--tag <name>`，只用于给运行目录命名。
- **G1（精调开启）**：4 个模型各新跑 1 次，`--tuner on`。R8e5 的那次运行作为 G1 的第 1 次，它的确认与报告结果取自 R8e7。
- **G2（精调关闭）**：`--tuner off`。resnet20 和 resnet20_silu 各跑 2 次，cnn_plain 和 mlp 各跑 1 次。
- 每次运行都走完冻结流程：循环 → confirm_select → 编译 → 报告集测试。
- 每次运行记录：
  - **评估量 FE**（full-equivalent，以图像数 / 1000 计）：循环中 evaluate 的验证量、精调粗筛和全量验证量分开统计，确认集的评估另列；
  - LLM 调用次数、token 数、墙钟时间；
  - 选中候选的来源、自举数（预测和实测）、确认集下降、报告集下降、发散数。

## 3. G3：只用搜索（`planners/search.py`，新增的基线模块，不属于冻结的 Agent）

```python
def search(graph, calib, ctx, *, fe_budget: float, rng_seed=20261024) -> dict
```

1. **起点**：
   - `rule.plan(graph, calib, deg=127, margin=1.5)`（cheb）；
   - 模型含 relu 时，再加一个 `rule.plan(..., act_method="minimax_comp", margin=1.5)`。

   两个起点各自先用 `validate`（留出集 1000 张）检查，不可行的丢弃。
2. **方法切换**（只对含 relu 的模型）：按 cost.estimate 的 hotspots 顺序，对每个 relu act 尝试切换方法：
   - cheb → minimax_comp，定义域为 ±1.5·max(|min|,|max|)；
   - minimax_comp → cheb 63，定义域为校准范围 × 1.5。

   预测延迟没有降低的直接跳过；否则在 `validate(n=200)` 上检查，满足精调的严格条件（acc_drop_pp ≤ 0.3、max_u ≤ 0.90、无发散）就接受。
3. **线性算法**：对每个 linear op 尝试 algo ∈ {bsgs, diag}，只比较预测延迟，降低就接受（不影响精度）。
4. **数值精调**：对当前最好的程序调用冻结的 `tuner.tune`（阶段 A 加阶数阶梯），`budget_s` 取剩余预算对应的时间。
5. **预算**：FE 的计算方式与 §2 相同。`fe_budget` 取 G1 两次运行中，该模型循环内 FE 的**中位数**，从 G1 结果读取。预算用完就停止，进入下一步。
6. **候选池与冻结**：每个阶段接受后的程序都进入候选池，然后调用冻结的 `confirm_select` 选出程序，编译，并在报告集上测试。

REPORT 记录每个阶段的接受次数、FE 用量、最终程序的自举数、确认集和报告集的下降。

## 3.9 编译路径（P2 之后，2026-10-23 补充）

G1/G2/G3 和消融中所有"编译"都使用 P2 之后的 run.py 路径：export 阶段生成结构化包，compile 阶段编译这个包。不要再使用旧的 emit→trace 路径。P2 已证明两条路径的编译结果相同（7 个程序的自举数和运算计数一致），因此与 R8e7 的历史数字可以直接比较。每个最终程序的包保存在对应的运行目录中。

## 4. B1 / B1+：LLM 直接写 Hecate 代码

> ✅ **用户已批准（2026-10-23，见 STATUS"用户批准记录"）**：B1 和 B1+ 可以执行 LLM 写的代码，作为 AGENTS.md "LLM 输出只解析，永不执行"的例外，仅限本节。条件：只在 `poseidon_candidate_sandbox`（`trusted=False`）中执行，执行前必须通过 §4.3 第 1 级的 AST 白名单检查；不得在可信运行器或 pt_agent 进程内执行 LLM 代码。

### 4.1 任务约定（写进 B1 提示词 `planners/b1_prompt.py`，新增，不属于冻结的 Agent）

- 目标：写一个 Hecate Python 程序 `@hc.func("c") def model(x)`，可用的 API 只有：`+`、`-`、`*`、取负、`x.rotate(k)`、`hc.Plain(vec)`、`hc.bootstrap(x)`。
- **输入**：图像按 CHW 顺序展平后放在前 3072 个槽位，其余槽位为 0；slots=16384。
- **输出**：logits 位于前 10 个槽位（mlp 也是 10 类）。
- **权重**：程序中可以使用 `W[name]`，即 numpy 数组，名字与 ATen 图中的权重名一致，由我们的执行环境提供。
- 提示词包含：ATen 图摘要（与 Agent 的 initial_message 相同的节点表）、上述 API 和约定、一个完整的小例子（2 层 MLP 的 Hecate 程序，权重名与目标模型无关）。不包含 PT-DSL、kernel 库和 lowering 的任何信息。

### 4.2 运行

- 模型：mlp、cnn_plain、resnet20。
- B1：每个模型 3 次独立调用，单轮完成。
- B1+：每个模型 2 次运行，每次最多 6 轮。每轮把 §4.3 的评估结果反馈给 LLM，包括出错的级别、错误信息、探针误差、200 张图上的准确率。这是该基线方法本身的设定，所以允许使用编译结果。
- LLM 设置与冻结 Agent 相同（模型、思考模式、max_tokens）。

### 4.3 分级评估（每个程序都评到失败为止）

1. 回复中包含可解析的 Python，AST 白名单检查通过（只允许上述 API，以及 numpy 下标、for、range）；
2. 在现有的候选沙箱（`poseidon_candidate_sandbox`）中，用真实的 hecate 前端完成 trace，限时 600 s，内存 4 GiB；
3. `dacapo-depth-dp` 编译成功；
4. 用 numpy 假前端（`tests/fixtures/fake_hecate_expr.py`）在沙箱中执行：4 张探针图的 logits 与精确网络的 top-1 一致；
5. heldout_set(200) 上的准确率下降 ≤ 1pp。

记录每个程序达到的最高级别和失败原因（分类为：超长或截断、API 违规、trace 失败、编译失败、数值错误、精度不足）。

## 5. B4：Dacapo 专家 DSL（resnet20_silu）

- 程序：`third_party/dacapo/examples/benchmarks/ResNet.py`，它加载 `third_party/dacapo/examples/data/resnet20.silu.model`，属于可信代码，不是 LLM 的输出。
- 在可信运行器中执行，PYTHONPATH 包含 `third_party/dacapo/python/poly` 和 hecate 前端，得到 traced 的 MLIR。然后分别用 `dacapo-depth-dp` 和 `pars`（保留专家写的 bootstrap）各编译一次，记录编译耗时、峰值 RSS、自举数和 plan_latency。
- 如果缺少 torchvision 等依赖导致无法 import，停下来记录，**不安装**。
- 精度（可选）：如果能用 numpy 假前端加上 poly 库自带的打包函数（MPCB 中的 MultParPack）执行专家程序，就在 heldout_set(200) 上报告准确率下降；否则只报告编译侧指标，并在 REPORT 中注明。

## 6. 消融（只在 resnet20 上做，每项 1 次 G1 运行，精调开启）

| 编号 | 开关（新增 CLI 参数，默认关闭） | 含义 |
|---|---|---|
| A1 | `--ablate-cost` | 反馈中去掉 cost 一节（延迟、自举、深度、hotspots） |
| A2 | `--ablate-hotspots` | 只去掉 hotspots |
| A3 | `--ablate-validation` | validation 只保留 feasible 一个字段，不给 max_u、worst_acts、准确率 |
| A4 | `--llm-reasoning none` | 关闭思考模式（已有参数） |
| A5 | `--anonymize` | initial_message 中的 network 和 model 字段改为 "net"，各节点、张量和权重名保持不变 |

- 开关只影响发给 LLM 的文本。可行性判定、精调、确认流程都不变。
- 默认参数下与金标准逐字节相同（§1）。

## 7. 汇总（`experiments/r9b_summary.py`，新增）

生成 `r9b-summary/table.csv` 和 `summary.json`。每一行是一次运行或一个基线：

```
group, model, run, llm_calls, fe_loop, fe_confirm, wall_s,
boots_plan, latency_plan_s, confirm_drop_pp, report_drop_pp, n_diverged, frozen_source
```

另外：
- G1、G2、G3 每个模型的均值和极差；
- B1 / B1+ 的分级通过率表；
- B4 与 G1（resnet20_silu）的对照；
- 消融相对 G1 的变化。

## 8. 测试（test_r9b_*.py，离线，不调用 API）

1. **金标准**：4 个模型默认参数下的系统提示词和反馈字符串与 fixtures 完全相等。
2. **消融开关**：A1–A3、A5 各自只改动对应字段（解析 JSON 后比较字段集合），其余字段与默认相同。
3. **search**：
   - 用 patch 替换 validate 和 estimate：方法切换只在延迟下降并且严格条件满足时接受；
   - FE 计数正确（粗筛 200 张计 0.2）；
   - 预算用完时停止，并且仍然调用 confirm_select。
4. **B1 静态检查**：白名单 API 通过；`import os`、`open(`、`__import__`、属性访问 `__class__` 等被拒绝。
5. **FE 统计**：从 R8e5 的 resnet20 轨迹中统计 FE，结果等于 evaluate 次数 × 1.0，加上 tuner 的 coarse_evals × 0.2 与 full_evals × 1.0 之和。
6. R1–R9a 的全部测试通过。

## 9. 预算与顺序

| 项目 | LLM 调用上限 |
|---|---|
| G1 新增 1 次 × 4 个模型 | 24 |
| G2：2 + 2 + 1 + 1 次 | 36 |
| B1：3 个模型 × 3 次 | 9 |
| B1+：3 个模型 × 2 次 × 6 轮 | 36 |
| 消融 A1–A5 | 30 |
| **合计** | **135**（上限 140） |

- **顺序**：§1 金标准 → 测试 → G1 → G3（依赖 G1 的 FE）→ G2 → 消融 → B4 → B1/B1+（已批准）→ 汇总。
- 每一项都用 `--resume` 方式接续，单项失败照实记录，然后继续下一项。
- **Nix 占用**：G1、G2 和消融每次约 1.5–2 小时（resnet20），全部约 25–30 小时。运行期间不要并行启动其他 Nix 任务。

## 10. 不要做

- 不修改冻结 Agent 的默认行为（§1）：提示词文本、可行性条件、精调与确认参数一律不变。
- 报告集只用于最终测试；确认集只在 confirm_select 中使用。
- B1/B1+ 的 LLM 代码只在候选沙箱中执行，且必须先通过 AST 白名单；不在可信运行器中执行。
- 不重试失败的 LLM 调用；不为了改善某一组的结果而调整预算或参数。
