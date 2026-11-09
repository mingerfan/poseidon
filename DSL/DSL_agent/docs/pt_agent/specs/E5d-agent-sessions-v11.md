# E5d：r8-v11 真实会话（2026-11-06，接 E5c；用户已批准本轮真实调用）

## 0　目的与预算

E5c 在 program-plan 代价下回放了已保存的响应，LLM 输出固定。本轮让 LLM 在 program-plan 代价和放置反馈下重新生成程序：
BERT、Qwen 各开一个 r8-v11 新会话，每个模型最多 10 次调用，合计最多 20 次，账本从 244/450 到至多 264/450。

用户已批准这 20 次真实调用。精调规则 v8-6.24、判据 a1-6.20 不变；r8-v10 及更早的提示词文本不变；精调预算 2400 秒、
每步进程 3600 秒、精调 3.8 GiB 检查保持。

## 1　调用前的修正（API 0，先做完并通过测试）

### 1.1　`levels_left_at_refresh` 扣除自举输入层

`stage_plan.simulate` 中值的层数下限是 `boot_in_level`（qwen48 为 1），所以一个层都没剩就被自举的值，现在报出来是 1。
`cost.py` 生成 `levels_left_at_refresh` 时改为 `level[nid] - model.boot_in_level`，即自举时还没用掉的层数。
快照测试同步：加一个断言，toy 程序中恰好用尽层数后才自举的值报 0。

### 1.2　r8-v11 提示词与程序层放置一致

r8-v11 现在沿用 r8-v10 正文，正文有两处说自举由编译器放置。只在 r8-v11 的派生中替换，每个原串必须恰好出现一次，
否则抛错：

| r8-v10 原文 | r8-v11 |
|---|---|
| `an unmodified compiler then inserts\nrescale/modswitch/bootstrap.` | `a placement planner inserts bootstraps and\nan unmodified compiler then inserts rescale/modswitch.` |
| `bootstraps, computed with the same bootstrap placement as the compiler, and` | `bootstraps, computed with the program-level placement used for compilation, and` |

`PLACEMENT_EXPLANATION` 中 `levels_left_at_refresh` 的说明改为：
`(the average number of levels still unused when values produced on that line are refreshed; null if none)`。
r8-v11 提示词快照测试按新文本更新；r8-v10 快照保持。

### 1.3　真实会话入口接上 r8-v11

E5c 的 B 部分只做了提示词和反馈字段，真实会话入口还不接受 r8-v11：

- `experiments/a1_transformer_agent.py`：`--prompt-version` 增加 `r8-v11`；`session_step` 的版本检查、`session` /
  `split_session` 中按版本元组决定字段的几处同步包含 r8-v11。
- `planners/transformer_session.py`：`output_tokens('r8-v11')` 取 r8-v8 的 65536（与 r8-v9 的别名方式相同）；
  `model_call_offset` 对 r8-v11 为 0。
- r8-v11 必须配合 `--placement program-plan`：`session_step` 在构建上下文后设置 `ctx.placement`，SESSION.json、
  AGENT.json、trace 记录 placement；r8-v11 缺少该参数，或其他版本带上该参数进入真实会话时，报错退出，不发请求。
- 调用上限：命名空间 `[model, 'r8-v11']` 每模型 10 次，`max_iters=min(10, 10-used)`，`remaining_calls` 按 10 计。
  r8-v8 至 r8-v10 保持 `min(8, 40-used)` 和 40。
- provider 与 A1 相同：`LLMClient(model='deepseek-flash', reasoning_effort='high', max_total_calls=450)`，凭证只经既有
  加载器读取。

### 1.4　离线测试

- 1.1、1.2 的快照测试。
- 桩 client 走一遍 `split_session` 的第一步 evaluate（桩返回 E5c 已保存的 BERT 第 35 次响应，不联网）：
  请求的 system prompt 等于 `build_system_prompt(ctx)`（r8-v11）；调用记录写到 `call_dir/bert/r8-v11/`；SESSION.json 中
  `max_iters=10`、`placement="program-plan"`、`prompt_version="r8-v11"`；反馈含 `placement_variant` 和
  `levels_left_at_refresh`。
- r8-v11 缺 `--placement` 时报错且请求数为 0。
- E5a、E5b、E5c 和已有相关测试照常通过。

## 2　真实会话

- 结果目录 `/home/lhohy/pt_agent_results/E5d-agent-sessions-<开跑日期>/`，每个模型一个子目录；调用账本仍为
  `/home/lhohy/pt_agent_results/llm_calls.jsonl`。
- 模型目录与 E5c 相同：BERT `/home/lhohy/pt_agent_results/A1-bert-prepare-v2-a620-20261030/model`，
  Qwen `/home/lhohy/pt_agent_results/A1-qwen-prepare-v2-20261028T135355-6529/model`。
- 命令：`--part session --prompt-version r8-v11 --tune-rule v8-6.24 --placement program-plan`，走 `split_session`
  的逐步进程。先 BERT，后 Qwen。
- 每个模型一个会话。会话在冻结、用满 10 次或确认失败时结束；结束后不再为该模型开新会话。
- 停止规则：
  - provider 错误：该模型停止并记录，失败的请求不重试；另一个模型照常进行。
  - 代码错误（非 provider）：整轮停止，不开始下一个模型，记录到 STATUS 后汇报。
  - 精调 3.8 GiB 检查或进程 3600 秒触发：按 A1 现有规则停止该模型并记录。
- 每步前后记录账本条数；本轮结束时账本增量应等于两个命名空间的调用数之和。

## 3　验收（与 E5c §3.3 相同）

对每个冻结程序：
- report：`--part evaluate`（report100，tensor 与 Mock）。report 结果只做验收，不参与程序选择。
- cost：`--part cost`，program-plan，带重放核对。
- compiler：沿用 E5b driver（`/home/lhohy/pt_agent_results/E5b-program-placement-20261106/driver.py`）的导出、trace、
  pars 编译、plan_stats、prune；记录预测、导出、编译三个自举数。preflight 和 managed 只适用于编译器放置，记为跳过。

未冻结的模型不做验收，写明结束原因和最后一次反馈。

## 4　汇总（E5d.md、E5d.json）

### 4.1　对照表

| 模型 | A1-a624 depth-dp 编译 | E5b（A1 程序，program-plan） | E5c（回放，program-plan） | E5d 预测／导出／编译 | report100 一致率／误差 | 冻结来源（llm/tuned、调用号） | 调用数 | tokens（输入／输出） |

另列每个模型的预测延迟，以及 E5d 冻结程序相对 A1-a624 冻结程序的语句字段变化（格式同 E5c 的 statement_changes）。

### 4.2　耗时拆分

格式同 E5c §3.5，LLM 一栏换成本轮真实响应的秒数之和；另列精调的 sweeps、converged 和 coarse/full 评估次数。

### 4.3　放置反馈的使用情况

只做描述性统计，不改变任何选择：
- 每次 complete 反馈之后的下一次响应：相对上一个程序改动了哪些语句和字段；这些语句是否出现在上一次反馈的 hotspots 中，
  以及当时的 `latency_share` 和 `levels_left_at_refresh`。
- LLM 原文程序头需要 `apply_placement` 改写的次数（r8-v11 下应为 0）。

## 5　不要做

- 不超过每模型 10 次、合计 20 次调用；失败不重试。
- 不改 r8-v10 及更早提示词、v8-6.24、a1-6.20、能力注册表、编译器和上游脚本；不提高限制。
- 不读 report 集参与选择；不删历史结果；不提交、不推送。

## 6　完成标准

- §1 的修正和测试通过后再发第一个请求。
- BERT、Qwen 各有一个会话目录：已冻结的模型有 report、cost、compiler 验收；未冻结的写明原因。
- E5d.md、E5d.json 含 §4 的三部分；registry 中 round 为 `E5d`；账本前后条数与调用数一致；最后一次残留进程检查为 0。
- STATUS.md 新增 E5d 一行和一节，"与 spec 不一致之处"逐条列出。Nix 独占，串行。
