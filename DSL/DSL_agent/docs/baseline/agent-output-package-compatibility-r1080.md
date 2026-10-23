# 输出接口：真实程序包兼容检查

只读检查三份历史 Poseidon 数值通过包：`function_sections_0`、`function_sections_1`、`function_sections_2`。原模型、候选、权重、包和历史记录保持。

| 检查 | 结果 |
|---|---|
| 原 task/program 文件 | 三份均逐字节保持 |
| SDK、Poseidon 和编译工具身份 | 与历史和当前待验收配置一致 |
| 当前 Application 交付 | 三份均返回 `cached_dependency_mismatch` |
| 当前程序包完整验收 | 三份均在 `Trusted reference identity` 拒绝 |
| 当前输出收据 | 0 份 |
| 新编译／密态／解密／API | 0 |

失败发生在当前来源和软件身份核验。原接受以来，捕获、规范化、来源、包检查及阶段目录等代码已变化；每份包的编译/执行/验证身份均有13项来源差异。SDK本体保持。此次没有进入编译或数值比较，不能将这三项记为新的算子、编译器或数值失败，也不能把旧包改签为当前通过。

## 后续验收

这三份历史包不在当前 r1076 的151项案例中，不追加到其冻结范围。

获批后先执行原定151项和25包复用。从已规划的 `channel_broadcast_mix_0`、`indexed_batched_matmul_1`、`public_weight_1` 选择实际通过的当前包，检查输出收据及JSON重载；若原批次已产生实际执行报告，再核验命名输出的结构恢复及其证据绑定。该检查可以复用原批次已有执行证据，额外编译、执行或生成需单列范围。

当前默认生产捕获、任务、预算和执行代码保持；新输出适配器的正向真实包交付仍未验收。原r878历史结论及本次负向版本检查分别保留。

[输出接口](agent-output-delivery-v1.md) · [151项提案](agent-common-pytorch-offline-proposal-r1076.md) · [机器记录](agent-output-package-compatibility-r1080.json)

绑定：fbb3580ccdb5855638dc13ce01b4ae55b0df617e7cc610dd5641971a238eeb72
