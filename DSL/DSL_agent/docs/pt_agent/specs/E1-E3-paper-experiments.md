# E1–E3：论文补充实验（2026-11-03，用户"Start with E1–E3"）

目的：让论文主结果使用同一个编译器（R121），给出 Agent 开销，并用同一套评估回答"LLM 和精调器各贡献了什么"。
全部在本机执行，不改 pt_agent 库、提示词、判据或编译器；驱动脚本放在结果目录。Nix 独占，三项按 E1 → G3 → G2 串行。

## E1　CNN／MLP 规则基线在 R121 下重跑

- 对象：resnet20、resnet20_silu、cnn_plain、mlp；权重 `models-v2`（与 R8e7 相同）。
- 流程：`pt_agent.run --planner rule`，阶段 validate,plan,check,mock,count,export,compile,test（`--final`）。validate 是 R6 的规则：在 2000 张 train 划分验证图上对 deg∈{63,127} × margin∈{1.4,1.5,1.75} 选择；compile 用默认 R121 构建、depth-dp；test 是 report test[0:8000]，判据与 R8e7 相同（top-1 ≥ torch − 1pp）。
- 限制：地址空间 4 GiB 软／6 GiB 硬，与 r8e6_reconfirm 相同。API 0。
- Agent 一侧沿用 R8e7 冻结程序：R121 下 depth-dp 自举取 M7b（13／40／114／176），report 准确率取 R8e7（程序未变，Mock 与编译器无关）。
- 结果目录：`pt_agent_results/E1-cnn-rule-r121-20261103/`（run.sh、RUNS.txt 指向各 `run-<model>-rule-*`）。

## E2　Agent 开销

- 只读现有记录：全局账本 `llm_calls.jsonl`、A1 会话与 A1-a624 重放、R8e5 会话。`collect.py` 生成 E2.md／E2.json，并核对账本行与重放 token 合计一致。
- 结果目录：`pt_agent_results/E2-agent-cost-20261103/`。

## E3　三组对照（Transformer：BERT-tiny、Qwen2-tiny）

| 组 | 方法 | 来源 |
|---|---|---|
| G1 LLM + 精调 | A1 冻结程序（v8-6.24，事后重评） | A1-a624（已有） |
| G2 只用 LLM | 新的 r8-v10 会话，`new_agent_state(tuner=False)`，每个模型 1 个会话、最多 8 次调用 | `E3-llm-only-20261103/` |
| G3 只用搜索 | 同一套 Agent 步骤，唯一的"回复"是规则程序 R0（无 API），之后 evaluate → tune（v8-6.24、new_best、2400 s、收敛为止）→ confirm | `E3-search-only-20261103/` |

- 三组共用：r8-v10 判据、tensor 引擎、split-v2、heldout20 内部评估、confirm 选择；冻结后按 A1 验收（report100、cost、compiler、preflight、managed）。
- G2 的 LLM 调用经已批准的 Provider 加载器，记入全局账本，单独计数在 `E3-llm-only-20261103/calls/<model>/r8-v10`。本轮上限 16 次（BERT 8、Qwen 8），在用户为 E3 认可的约 40 次之内。
- 已知不对称：r8-v10 系统提示词写明"每个可行程序之后有确定性精调器"。G2 保持提示词不变，所以 G2 衡量的是"从系统中去掉精调器"，而不是"专门为无精调器写的提示词"。论文中照此表述。
- 已知局限：G1 是事后重评（Agent 当时收到 r8-v9 反馈），report100 参与过规则诊断；G2、G3 的程序第一次接触 report100。每组只有 1 个会话，LLM 的随机性未覆盖。

## 不要做

- 不改提示词、判据、精调规则或编译器；不为某一组调参数或重试失败的 LLM 调用。
- report 只在冻结后的验收中使用；confirm 只在 confirm 步骤中使用。
- 不提交、不推送（另行审批）。
