# E7：确认阶段加尾部误差检查（2026-11-07，接 E6r2）

## 0　背景

E6r2-b Qwen p2 冻结了精调后第 1 份程序（预测、导出、编译都是 20705 自举）。它在确认集 20 个样本上的 logit_rel_err 是 0.0038，
通过了确认；在 report 集 100 个样本上是 0.0172，超过 Qwen 的任务上限 0.01，冻结后验收失败。

原因在指标的定义：`logit_rel_err` 是所有样本最后一个 token 的 max|exact−approx| / max|exact|（`metrics.py:53`），取的是全部样本里的
最坏值。确认集有 20 个样本，report 集有 100 个，同一个程序在样本更多的集合上更容易遇到尾部样本。

Claude 的诊断（只读已有输出，不参与任何选择）：以 E5d Qwen 的 report 输出为参照（它自身的误差为 0.0014，所以下面每个数有 ±0.0014
的不确定度），Qwen p2 的 100 个 report 样本中只有 1 个的相对误差超过 0.01（约 0.0158），第二高约 0.0058，中位数约 0.0011。
假如这类样本约占 1%，20 个样本的确认集碰到其中一个的概率约 18%，123 个样本约 71%。

精调器接受一步时，logit 误差上限是任务上限的一半（`move_logit_limit`，`planners/transformer_tuner.py:56`，v8-6.24 的系数为 0.5），
验证用 heldout 的 20 个样本。精调后的程序最容易贴近这个上限，尾部风险也最大。Qwen p2 的候选池只有 4 项：
- llm 1（31745）；
- tuned 1（20705）；
- llm 2（20416，确认集 logit 0.0112，未通过）；
- tuned 2（与 tuned 1 的程序相同）。

所以，如果只加尾部检查而拒掉 tuned 1，能选的就只剩 31745。

本轮的两处改动都只作用于确认阶段：
- **C1 尾部检查**：确认时再用全部非 report 样本（123 个），按任务指标检查一次。
- **C2 精调中间状态入池**：精调中每轮有接受时的程序都进入确认候选池。最终程序被拒时，可以退回到精调中较早的状态。

接着做两部分：
- **E7a**：API 0，回溯全部已有会话；
- **E7b**：跑两个新的 Qwen 会话（用户已批准调用），E7a 完成后开始。

Qwen p1 的失败（确认集上第 32 行 rsqrt 的校准 u 为 0.944 > 0.90）已被现有确认规则拦下，不在本轮范围内。

## 1　调用预算

| 部分 | 上限 |
|---|---|
| E7a | API 0 |
| E7b | 沿用 E6 已批准的 144 次额度中剩余的 38 次，账本从 365/450 到至多 403/450。2 个 Qwen 会话，每个会话主循环最多 10 次、精调调用最多 9 次，重试计入上限 |

用户已于 2026-11-07 在聊天中批准 E7b 的 38 次调用。E7b 仍要等 §2 的测试和 E7a 都完成且没有代码错误后才开始。

## 2　实现（API 0，先做完并通过测试）

### 2.1　确认规则开关

- 会话状态、AGENT.json、trace 增加 `confirm_rule` 字段：`"c1"`（现有规则，默认）或 `"c2"`（本轮）。命令行参数为 `--confirm-rule`。
  旧会话和未指定时都用 c1，行为与改动前逐项相同。
- c2 对所有带 model_info 的模型启用（transformer 路径）。CIFAR 旧路径（`test[8000:9000]`）保持 c1，记录为不适用。

### 2.2　C1：非 report 尾部检查（`planners/confirm.py:50` 的 `confirm_select`）

- **非 report 上下文**：按 calibration → heldout → confirm 的固定顺序拼接三份样本（split-v2 为 83+20+20=123 个），精确输出用
  `reference_forward` 计算，做法与 `build_confirm_context`（`confirm.py:27`）相同。请求 report 样本时报错。
- **评估顺序**：候选仍按预测延迟从低到高处理，`MAX_CONFIRM_EVALS=6` 不变。
- **每个候选的判定**：
  1. 先做现有的全部检查（确认集）。
  2. 通过后，在非 report 上下文上执行 `validate`（引擎与确认相同，`stop_on_divergence=False`）。通过条件：
     `passes(result, info.threshold)`（agreement_pct、logit_rel_err 等任务上限），并且没有发散。
  3. u 和定义域条件只在确认集上判定。非 report 样本中的 calibration 样本决定了校准定义域，在它们上面算出的 u 偏乐观。
  4. 两项都通过才选中。
- **记录**：
  - 每个候选的尾部检查：total、agree、agreement_pct、logit_rel_err、发散数、是否通过、用时；
  - 误差最大的 5 个样本：所属分区、分区内序号、该样本的 max|误差| / 全体 max|exact|；
  - 达到评估次数上限时记 `max_evals_reached`。

### 2.3　C2：精调中间状态进入候选池

- **记录中间状态**：TuneResult 增加 `accepted_states`。每一轮结束时（llm-select 是一轮选步，sweep 是一遍坐标扫描），如果本轮有接受，
  就记录当时的程序正文、代价估计（沿用已算出的 fast 估计）和轮次；最后一项就是最终程序。sweep 和 llm-select 都记录，记录本身不改变
  精调的任何决策。
- **加入候选池**：`confirm_rule="c2"` 时，`_finish_tune`（`planners/agent.py:292`）把 `accepted_states` 中的每一项作为候选加入候选池，
  origin 记为 `tuned-r<轮次>`，iteration 用所在精调的 iteration；最终程序仍以 origin `tuned` 加入。去重沿用现有做法：同一程序正文只保留
  预测延迟最低的一项。
- **不受影响的部分**：会话的 best、反馈、stale 计数和精调触发都只看最终程序，与现在相同。
- **检查范围**：中间状态没有经过 `finish()` 的最终 Mock 核对，只在确认阶段检查（引擎与确认相同）。冻结后照常做 report100 双引擎验收。

### 2.4　附带修正（E6r2 收尾遗留）

- **汇总数字**：E6r2.json 的 `e6ra.*.replayed_responses` 改为读取各回放会话 AGENT.json 的 `llm_calls`（BERT 6、Qwen 8），不再写死为 7/8
  （`experiments/e6r2_resume_after_outage.py:220`）。只重写汇总，原 REPORT、trace 不动。
- **STATUS 补记**：在 E6r2 的"与 spec 不一致之处"补一条：BERT 回放第 6 次迭代后，按 `agent.py:268–270` 的 stale 规则转入确认，
  第 7 份保存回复没有使用；E5d 第 7 份回复对应的程序同样是 7175。

### 2.5　离线测试

- **c1**：`confirm_select` 的结果与改动前逐项相同（排除墙钟字段）；已有会话的 SESSION.json 在 c1 下确认结果不变。
- **C1**：
  - 桩程序在确认集上通过、但非 report 中有一个样本超过 logit 上限时被拒，系统改选下一个候选；
  - 非 report 上下文的样本顺序和数量正确；
  - 请求 report 样本时报错；
  - 尾部检查记录的字段齐全。
- **C2**：
  - sweep 和 llm-select 的 `accepted_states` 与 history 中的接受记录一致；
  - c2 下候选池中出现 `tuned-r<k>`，c1 下不出现；
  - 去重正确；best 和 stale 不受影响；
  - sweep 的 TuneResult 除新增字段外逐项相同。
- **精调重放客户端（§3.3）**：请求与保存的请求逐项相同时返回保存的响应；不同时停止并记录，不发请求。
- E5a–E6r2 和已有相关测试照常通过。

## 3　E7a：回溯（API 0）

结果目录：`/home/lhohy/pt_agent_results/E7-confirm-tail-<开跑日期>/e7a/`。

### 3.1　已冻结程序的尾部检查（只做描述）

对以下已冻结的程序做 C1 尾部检查，并与已有的 report100 结果并列：
- E5b 的 BERT、Qwen；
- E5c、E5d、E6a、E6 s1、E6r2-a 的 BERT、Qwen；
- E6r2-b 的 BERT p1、BERT p2、Qwen p2。

只读程序和已有报告，不重算 report。

### 3.2　按 c2 重新确认已有候选池

- **范围**：E5c、E5d、E6a、E6 s1、E6r2-a 的 BERT 和 Qwen 会话，以及 E6r2-b 的 4 个会话。
- **做法**：用保存的候选池（SESSION.json 的 `pool`）按 C1 规则重新确认。这些候选池里没有中间状态，C2 不适用。
- **记录**：选择是否变化。

### 3.3　重放精调，得到中间状态

对 §3.2 中选择发生变化或确认失败、并且候选池含 tuned 候选的会话：重新执行这些 tuned 候选对应的精调，得到 `accepted_states`，再按完整的
c2（C1+C2）重新确认。
- **sweep 精调**：直接重跑（API 0）。
- **llm-select 精调**：用保存的 t1 响应重放（`calls/<model>/t1/<label>/`）。
  - 每次请求先与保存的请求逐项比较，做法同 E6r2 的 PAYLOAD_COMPARISON；
  - 相同才使用保存的响应；
  - 不同就停止该会话的重放并记录差异，不发请求。
- **核对**：重放得到的最终程序应与原 tuned 候选的程序正文相同。如果不同（例如精调预算的墙钟截断点不同），如实记录，并以重放结果为准做确认。

### 3.4　验收

§3.2、§3.3 中选出新程序的会话，按 E5d §3 验收（report100 双引擎、cost、pars）。选择没有变化的会话沿用原来的验收结果。

## 4　E7b：新的 Qwen 会话（用户已批准）

- **结果目录**：同上的 `e7b/`。标签 `q1`、`q2`，顺序 q1 → q2。
- **参数**：`--part session --prompt-version r8-v11 --tune-rule v8-6.24 --placement program-plan --tune-search llm-select --confirm-rule c2`。
- **上限**：主循环最多 10 次，精调调用最多 9 次。
- **provider 故障**：按 E6r2 §2 处理。
- **结束与验收**：结束条件和停止规则同 E5d §2；冻结后按 E5d §3 验收。
- **前提**：E7a 出现代码错误时不开始 E7b。

## 5　汇总（E7.md、E7.json）

### 5.1　尾部检查

| 程序 | 确认集 logit | 非 report logit / agree | 尾部检查通过 | 误差最大的样本（分区:序号, 值） | 已有 report100 logit / feasible |

### 5.2　重新确认

| 会话 | c1 选中（来源/自举） | c2 选中（来源/自举） | 被尾部检查拒绝的候选 | 中间状态数 | 新选中程序的 report100 / 三方自举 |

### 5.3　端到端

E5d、E6 s1、E6r2-b、E7b 的会话各占一行（格式同 E6r2），增加 `confirm_rule` 列。

### 5.4　确认用时

每次确认的评估数、尾部检查秒数。

以上只做描述。

## 6　不要做

- 不读 report 集参与选择；尾部检查只用 calibration、heldout、confirm 三个分区。
- 以下全部不改：
  - v8-6.24（定义域对齐、refit、floor、接受规则）、a1-6.20；
  - r8-v11 主循环提示词、t1 提示词；
  - llm-select 和 sweep 的搜索逻辑；
  - 编译器；
  - split-v2。
- 不新增样本，不下载，不提高限制。
- 不改历史会话目录中的文件，回溯结果写入 E7 目录；不删历史结果；不提交、不推送。
- 不超过 §1 的上限；只按 E6r2 §2 重试。

## 7　完成标准

- §2 的实现和测试通过后，才开始 E7a。
- E7a：
  - §3.1 的全部程序都有尾部检查记录；
  - §3.2 的全部会话都有 c2 确认记录；
  - §3.3 适用的会话有重放记录或停止原因；
  - 新选中的程序都有验收。
- E7b：两个会话各有目录；已冻结的有验收，未冻结的写明原因。
- E7.md、E7.json 齐全；registry 中 round 为 `E7`；账本与各命名空间调用数一致；最后一次残留进程检查为 0。
- STATUS.md 新增 E7 一行和一节，"与 spec 不一致之处"逐条列出。Nix 独占，串行执行。
