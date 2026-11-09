# E5c：Agent 循环使用程序层放置（2026-11-06，接 E5b）

## 0　目的

E5b 的程序层放置（`placement="program-plan"`）让 6 个冻结程序编译后的自举比编译器 depth-dp 少 3.3%–15.4%。但这些程序的
近似设置是在 depth-dp 代价下由 LLM 写出、由精调器调出的。动机二要说的是：Agent 生成程序时，近似和自举一起决定。
本轮让 Agent 循环（反馈和精调）直接用 program-plan 的代价，并记录循环各部分的耗时。

分三部分：
- **A（API 0，本轮执行）**：系统侧的放置开关；在 program-plan 代价下重算已有 BERT、Qwen 会话的反馈，回放这两个会话
  （LLM 输出不变，精调重新进行），冻结后用 pars 编译验收。
- **B（需要用户另行批准后才执行）**：提示词 r8-v11 和反馈中的放置信息，以及新的 LLM 会话。本轮只实现并离线测试，不发请求。
- **C（随 A 记录）**：Agent 循环的耗时拆分，供论文与编译器耗时对比。

精调规则 v8-6.24、判据 a1-6.20、提示词 r8-v10 都不改；开关默认关闭时，所有已有行为不变。

## 1　放置开关

- `AgentContext` 增加字段 `placement: str | None = None`。取 `None` 时一切照旧；取 `"program-plan"` 时启用下面的改写。
- `planners/agent_tools.py` 新增 `apply_placement(source, ctx) -> str`：
  - `ctx.placement` 为 `None` 时原样返回；
  - 否则只改写唯一的 `@fhe.program(` 那一行：`refresh` 设为 `"agent"`，`placement` 设为 `ctx.placement`，补上 `l_max=5`、
    `boot_out_level=5`，`slots` 和其他关键字保持原值。只改这一行，所以反馈里的行号不变。
  - 找不到这一行或这一行无法解析时原样返回，由后续的 parse 照常报错。对已改写过的源码再调用一次，结果不变。
- 调用位置：`planners/agent.py` 两个循环里取出 `source = response["program"]`、写完原始响应文件之后，立即改写；之后的反馈、
  候选池、精调、确认和冻结都用改写后的源码。原始响应文件（`sources/iteration-N.ptdsl`）保持 LLM 原文。trace 记录里增加
  `placement` 字段。

## 2　代价调用与耗时记录

- `cost.estimate` 增加关键字参数 `verify_plan=True`。为 `False` 时跳过 E5a 的 CountBackend 重放核对，其余结果相同。
  在 `cost.py` 模块级定义 `estimate_fast` 和 `estimate_normalized_fast`（`verify_plan=False`），让 `cached_estimate` 的缓存键保持稳定。
- `ctx.placement` 非空时，Agent 循环的四处代价调用改用 fast 版本：`agent_tools.evaluate` 的 cost 阶段、`validate` 取 boot_nodes、
  `planners/tuner.py` 和 `planners/transformer_tuner.py` 的 `cost`。导出（`export_package`）照旧核对。
- 把 `evaluate` 中 cost 阶段生成 `fb["cost"]`（含 hotspots）的代码提成函数 `cost_feedback(program, ctx, estimator, line_text)`，
  `evaluate` 和 §3 的重算共用。
- 耗时记录：`tensor_sim.cached_estimate` 在未命中时累计 `estimate` 的秒数；`agent_tools.validate` 累计验证秒数。提供
  `timing_snapshot()` 和 `reset_timing()`。回放每一步记录这两项的增量和该步总耗时。

## 3　A 部分：重算反馈、回放、验收

结果目录 `/home/lhohy/pt_agent_results/E5c-agent-loop-placement-<开跑日期>/`，每个模型一个子目录。

### 3.1　重算反馈（新 part `log-recost`，放在 `experiments/a1_v10_replay.py`）

- 输入为 A1-a624 回放用过的 REEVALUATION.json：
  - BERT：`/home/lhohy/pt_agent_results/A1-a620-20261030/regrade/bert/REEVALUATION.json`
  - Qwen：`/home/lhohy/pt_agent_results/A1-r121window-20261030/regrade/qwen/REEVALUATION.json`
- 对每个 case：读原 feedback；若 stage 为 complete，用 `apply_placement` 改写源码，以 `cost_feedback` 在
  `estimate_normalized_fast` 下重算 `fb["cost"]`。validation 保持原值，因为 qwen48 的 `boot_input_bound` 为 `None`，
  验证结果与放置无关（代码里断言这一点）。资源限制按 evaluate 的 resources 阶段重查。
- 写出新的 `call-NNN-feedback.json` 和 REEVALUATION.json（cases 的 `feedback` 指向新文件），并记录每个 case 的新旧
  `boots_pred`、`latency_pred_us`。

### 3.2　回放（`log-step` 增加参数 `--placement`）

- 命令与 `A1-a624-20261102/replay.sh` 相同：`--prompt-version r8-v10`、`--tune-rule v8-6.24`、相同的模型目录和原会话
  （BERT：`A1-bert-session-20261029T055416-4780`；Qwen：replay.sh 中的 5 个会话依次回放，冻结即停），只是
  `--response-file` 换成 §3.1 的新文件，并加上 `--placement program-plan`（写入 ctx 和 SESSION.json）。
- 回放中 §6.18 的 3.8 GiB 精调内存检查保持；超出时停止并记录。

### 3.3　验收

- report：与 A1 相同的 `--part evaluate`（report100），程序为新冻结程序。
- cost：`--part cost`（会走 program-plan，带重放核对）。
- compiler：`u4_relative_acceptance.compiler` 解析 depth-dp 的编译日志，program-plan 程序改用 E5b driver 的导出、trace、
  pars 编译流程，记录预测、导出、编译三个自举数。preflight 和 managed 只适用于编译器放置，记为跳过并写明原因。

### 3.4　对照与汇总（E5c.md、E5c.json）

| 模型 | A1-a624 冻结：depth-dp 编译 | 同一程序 program-plan（E5b） | E5c 冻结程序：program-plan 编译 | report100（一致率 / 误差） | 冻结来源（llm/tuned、调用号） |

另列每个模型的耗时拆分（§3.5）和重算前后各 case 的预测变化。E5c 冻结程序与 A1-a624 冻结程序不同时，列出改动的语句和字段。

### 3.5　耗时拆分（C 部分）

每个模型汇总：LLM 原始响应耗时（原会话 trace 的 `seconds`，回放不调用）、反馈中的代价估计、验证、精调（其中代价与验证各多少）、
放置计划（plan 的 lower_s、dp_s、candidates_s、map_s 之和）、确认，以及 pars 编译时间。与 E4c/M7 的编译器放置时间并列。

## 4　B 部分：r8-v11 与新会话（需另行批准）

本轮实现并做离线测试，但不发 API 请求；运行要等用户批准 API 额度（当前 244/450）和解除冻结。

- 提示词 r8-v11 = r8-v10，加两处：程序头写 `refresh="agent", placement="program-plan", l_max=5, boot_out_level=5`；
  说明反馈中新增的放置字段。
- 反馈的 cost 增加：`placement_variant`；每个 hotspot 行增加 `levels_left_at_refresh`（该行产生、被自举的值在自举时
  还剩的层数的平均值，由选中计划的模拟算出）。
- 运行设想（批准后另写 spec 细节）：BERT、Qwen 各一个新会话，各不超过 10 次调用；其余规则与 A1 相同。

## 5　测试

- `apply_placement`：改写正确、行号不变、重复调用结果不变、`placement=None` 原样返回、其他关键字保留。
- `ctx.placement=None` 时，MLP 夹具上 `evaluate` 的反馈和 `tune` 的结果与改动前逐项相同（与改动前保存的输出比较）。
- `verify_plan=False` 与默认值的结果只差 `refresh_plan.replay`；`estimate_fast` 在缓存中命中。
- 耗时计数：两次相同的 `cached_estimate` 只累计一次秒数。
- `log-recost` 在一个两次调用的小会话夹具上运行，validation 不变，cost 换成 program-plan。
- r8-v11 的提示词和反馈字段有快照测试，不发请求。
- E5a、E5b 和已有测试照常通过。

## 6　不要做

- 不调用 API；不改 r8-v10 提示词、v8-6.24 精调规则、a1-6.20 判据、能力注册表；`placement=None` 时行为不变。
- 不改编译器、上游脚本；不提高限制；失败不重试；不删历史结果；不提交、不推送。

## 7　完成标准

- §1、§2 的实现和 §5 的测试通过；B 部分代码和快照测试就位但未运行会话。
- BERT、Qwen 各有重算后的 REEVALUATION.json、回放会话目录、冻结程序（或写明未冻结的原因）、report/cost/compiler 验收记录。
- E5c.md、E5c.json 齐全，含 §3.4 对照表和 §3.5 耗时拆分；registry 中 round 为 `E5c`；最后一次残留进程检查为 0。
- STATUS.md 新增 E5c 一行，"与 spec 不一致之处"逐条列出。Nix 独占，串行。
