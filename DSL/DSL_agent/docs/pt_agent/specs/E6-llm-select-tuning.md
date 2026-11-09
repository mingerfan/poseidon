# E6：LLM 选步精调（2026-11-07，接 E5d；用户批准放宽调用预算）

## 0　目的

E5d 中精调占 Agent 循环时长的 79%（BERT）和 89%（Qwen）。6 次精调共做了约 338 次完整代价估计，每次 18–42 秒，
接受 23 步（约 7%）；其中 3 次精调接受 0 步，各花 13–16 分钟。精调器逐点逐档试，每一步都要整份程序估计一次。

本轮新增精调搜索方式 `llm-select`：定义域对齐、合法步生成、接受规则都沿用 v8-6.24；把逐点扫描换成由 LLM 从合法步
菜单中挑选候选，系统只估计和验证 LLM 选中的候选。分两个实验：
- **E6a**：回放 E5d 已保存的主循环响应，只把精调换成 llm-select，与 E5d 在同一输入程序上逐次对比精调。
- **E6b**：新会话，每个模型 2 个，r8-v11 主循环加 llm-select 精调，与 E5d 对比端到端结果和时长。

## 1　调用预算与限制

| 部分 | 每个单位的上限 | 合计上限 |
|---|---|---:|
| E6a | 每个模型的精调调用最多 12 次（每次精调最多 3 轮） | 24 |
| E6b | 每个会话：主循环最多 10 次，精调调用最多 20 次 | 4 个会话，120 |

账本从 259/450 到至多 403/450，剩余 47 次留给之后的重跑，本轮不动用。预计实际用量约 70–90 次。
精调预算 2400 秒（含精调内 LLM 调用的时间）、每步进程 3600 秒、精调 3.8 GiB 检查、全局 450 上限保持。

## 2　实现（API 0，先做完并通过测试）

### 2.1　开关与命名空间

- `AgentContext` 和会话状态增加 `tune_search`：`"sweep"`（默认，现有行为）或 `"llm-select"`。命令行 `--tune-search`。
  `llm-select` 只允许与 r8-v11、`placement="program-plan"` 同时使用，其他组合报错退出，不发请求。SESSION.json、
  AGENT.json、trace 记录 `tune_search`。
- `ModelBudgetClient` 的命名空间增加可选的会话标签：主循环 `[model, 'r8-v11', label]`，上限 10；精调
  `[model, 't1', label]`，上限按 §1。E5d 用过的 `[model, 'r8-v11']`（已用 7/8 次）保持原样，新会话不占用它。
- `session_step` 在 tune 阶段、`tune_search="llm-select"` 时创建精调 client（provider 设置同主循环：deepseek-flash、
  reasoning_effort high；`max_tokens=16384`；ledger 的 purpose 为 `tune-select-iter<N>-round<R>`）。sweep 的 tune 阶段
  照旧不创建 client。

### 2.2　llm-select 的精调流程（`planners/transformer_tuner.tune`）

初始测量、exact 检查、定义域对齐（含 refit 和 refit floor）与现在完全相同。对齐之后，用至多 3 轮选步代替扫描：

1. **菜单**：从当前程序生成，与 sweep 会去估计的步完全一致：
   - softmax：`softmax_moves(current, ctx, op_index)` 产出的每一步；去掉 attrs 为 None 的、`program_failures` 报错的、
     被 `dominated_approximation` 支配的。
   - 其他近似点：`_new_settings` 和 cheb 的 LADDER 给出的设置；去掉 `below_floor` 的和 attrs 为 None 的。
   - 每项记录：编号（`m1`、`m2`……）、行号和宏调用行、所属点（op id 与 prefix/role）、变化字段的旧值和新值、
     局部深度与局部运算数的变化（softmax 近似步用 `local_approximation_cost`，其他点用 CountBackend 计算该点的深度）。
2. **上下文**（与菜单一起发给 LLM）：
   - 带行号的当前程序（对齐后）。
   - 当前代价：`latency_pred_us`、`boots_pred`、放置方案，hotspots 的 `latency_share` 与 `levels_left_at_refresh`。
   - 当前精度：agree/total、logit_rel_err、max_u_calibrated、max_u_analytic；每个点取验证结果 `rows` 中的 max_u（没有则
     为 null），以及该点的 refit floor；接受条件（calibrated ≤0.90、analytic ≤1、`move_logit_limit(ctx, rule)`）。
   - 本次精调前几轮的候选及其结果。
3. **提示词 t1**（`planners/prompts.py` 新增 `build_tune_prompt`，有快照测试）：
   - 从菜单中选出至多 8 个候选；每个候选含 1–4 个菜单编号，同一个点至多一个编号。
   - 候选按预计的预测延迟降幅排序。系统只接受预测延迟更低、并且精度满足接受条件的候选。
   - 输出仅为 JSON：`{"candidates":[{"moves":["m3","m7"],"note":"<=30 words"}]}`。
4. **解析**：取响应中的第一个 JSON 对象。编号不存在，或同一点出现两个编号时，丢弃该候选并记录原因；没有合法 JSON 或
   `finish_reason` 为 length 时，该轮候选为 0。都不重试。
5. **评估**：按 LLM 给出的顺序逐个处理：
   - 若候选中某一步所在点的设置已被本轮先前接受的候选改动，跳过并记为 `stale`。
   - 否则在当前程序的副本上一次应用候选的全部步，依次执行：`require_margin` → `program_failures` → 代价估计（fast），
     预测延迟不低于当前时记为 `cost` 拒绝 → N_COARSE 粗验证 → `accept` → 全验证 → `accept` → 更新当前程序。
   - 接受规则、粗验证样本数、验证引擎与 sweep 完全相同。
   - 每个候选开始前检查精调预算。
6. **停止**：某轮接受 0 个候选、满 3 轮、精调调用用满或精调预算用完时停止，然后执行原有的 `finish()`（最终 Mock 核对）。

记录：
- TuneResult 增加 `search`、`llm_calls`、`llm_s`、`estimates`。`estimates` 计未命中缓存的代价估计次数，sweep 也计数，
  计数不改变其行为。
- history 增加 `llm_round` 行：菜单大小、tokens、LLM 秒数、finish_reason、各候选的编号、结果、预测延迟和粗／全验证结果。

### 2.3　离线测试

- 菜单：在已有的 token-softmax 夹具上，菜单的步集合与 sweep 在同一程序上会去估计的步集合相同。
- 桩 LLM：候选按序评估；stale 跳过；同一点两个编号的候选被丢弃；非法 JSON 时该轮为 0 且不重试；三种停止条件；
  调用计数与命名空间；history 字段。
- `tune_search="sweep"`（默认）时，精调结果与改动前保存的输出逐项相同（排除墙钟字段，做法同 E5c 的默认路径测试）。
- t1 提示词快照；`llm-select` 搭配非 r8-v11 时报错，请求数为 0。
- E5a、E5b、E5c、E5d 和已有相关测试照常通过。

## 3　E6a：固定主循环响应，只换精调

- 结果目录 `/home/lhohy/pt_agent_results/E6-llm-select-<开跑日期>/e6a/`，模型目录与 E5d 相同。
- 输入为 E5d 已保存的主循环响应：BERT 7 次、Qwen 8 次，位于
  `/home/lhohy/pt_agent_results/E5d-agent-sessions-20261106/calls/<model>/r8-v11/`。主循环调用 0 次。
- 回放方式同 E5c（`--response-file` 逐步回放，r8-v11、v8-6.24、program-plan），加 `--tune-search llm-select`；精调调用
  真实发出，命名空间 `[model, 't1', 'E6a']`。
- 精调触发规则不变（new_best 只看原文程序），所以 E6a 与 E5d 在同样的输入程序上精调：BERT 第 4、5 次，
  Qwen 第 4、5、6、8 次；若触发次序与此不同，如实记录。
- 冻结后按 E5d §3 验收（report100、cost、pars）。
- 记为开发回放：后面的主循环响应是 LLM 看到 E5d 精调结果之后写的。
- E6a 出现代码错误（非 provider）时整轮停止，不开始 E6b。

## 4　E6b：新会话

- 结果目录同上的 `e6b/`。每个模型 2 个会话，标签 `s1`、`s2`；顺序 BERT s1 → Qwen s1 → BERT s2 → Qwen s2。
- 每个会话：`--part session --prompt-version r8-v11 --tune-rule v8-6.24 --placement program-plan --tune-search llm-select`，
  主循环最多 10 次，精调调用最多 20 次。
- 结束条件和停止规则同 E5d §2：冻结、主循环用满或确认失败时结束；provider 错误时该会话停止，不重试；代码错误时整轮停止。
- 冻结后按 E5d §3 验收。

## 5　汇总（E6.md、E6.json）

### 5.1　E6a 逐次精调对比

| 模型／调用 | 原文自举 | E5d sweep：秒、估计次数、接受步、精调后自举 | E6a llm-select：秒（其中 LLM 秒）、精调调用、菜单大小、估计次数、接受候选、精调后自举 |

另列 E6a 冻结程序与 E5d 冻结程序的自举数、延迟、report100 和冻结来源。

### 5.2　E6b 端到端

| 会话 | 预测／导出／编译自举 | 预测延迟 | report100 | 冻结来源 | 主循环调用 | 精调调用 | tokens | 循环总时长 |

耗时拆分：主循环 LLM、精调 LLM、代价估计、验证、精调合计、确认、pars；E5d 的对应数字并列。

### 5.3　LLM 选步质量

每轮：提出的候选数、合法数、评估数、接受数、stale 数；接受率；被接受的候选中含 2 步以上的个数；与 E5d sweep 接受过的
步相比，有多少同样被选中。只做描述，不参与选择。

## 6　不要做

- 不超过 §1 的调用上限，失败不重试；剩余 47 次不动用。
- 不改 r8-v11 主循环提示词、v8-6.24（定义域对齐、refit、floor、接受规则）、a1-6.20、sweep 路径、能力注册表、编译器和
  上游脚本；不提高限制。
- 不读 report 集参与选择；不删历史结果；不提交、不推送。

## 7　完成标准

- §2 的实现和测试通过后再发第一个请求。
- E6a 两个模型、E6b 四个会话各有会话目录；已冻结的有 report、cost、compiler 验收，未冻结的写明原因。
- E6.md、E6.json 含 §5 的三部分；registry 中 round 为 `E6`；账本前后条数与各命名空间调用数之和一致；最后一次残留进程
  检查为 0。
- STATUS.md 新增 E6 一行和一节，"与 spec 不一致之处"逐条列出。Nix 独占，串行。
