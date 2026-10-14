# 三个小型整模型的真实 Agent 结果

当前 Poseidon CPU 链路完成一次已批准的有界生成批次：2 passed，1 failed，0 skipped／not_run。

| 模型 | 生成次数 | 修复次数 | 结果 |
|---|---:|---:|---|
| 4→2→2 平方激活 MLP | 1 | 0 | 首次通过 |
| 4 元素 RMSNorm | 2 | 1 | 修复后通过 |
| 小型 Attention decode | 4 | 3 | 四次均编译失败，修复额度耗尽 |

成功包含真实 Hecate tracing、Dacapo-modified 编译、RuntimePlan／OperatorSpec／常量 bundle 检查、Poseidon CPU 密态执行、解密、四组输入逐项比较及包重载。首次比较共 24 个值，最大误差 1.9241565937799798e-8；重载 24 值，最大误差 5.011467837934269e-9。

实际 7 次生成、7 次 HTTP、0 次传输重试；原批准上限为 12 次生成／48 次 HTTP。未用额度不用于重启已耗尽单例修复次数的 Attention。货币费用未从服务商账单核验。

Attention 四次都通过前端并在 Earth mul 类型推导处失败，当前表达的消耗层／scale 超出固定编译预算。该数学模型的规则基线已通过；本结果不证明所有表达均不可执行。没有人工替换 Provider 程序。

开发工具两处来源审计／导出参数转发错误保留在 r450/r451；已完成的 MLP 未重新生成。r452 是最终批次绑定，r453 是独立审计。原响应、原源码、Provider 调用记录及旧报告保持原样。

后续修正了 Agent 端对当前 Poseidon 编译配置的诊断识别；原 Attention 候选离线重放仍编译失败，反馈正确指出 720 的编译类型预算。这不是实际模数容量。对应 [诊断验收](poseidon-compiler-feedback-acceptance-r456.json)。

本报告绑定生成时源码；后续两份成功源码的当前绑定重编译、数值及重载复核见 [r473](poseidon-stage3-pytorch-acceptance-r473.md)。后续复核不增加新生成计数。

[机器报告](poseidon-whole-models-agent-acceptance-r453.json)。数值门限保持 1e-5 + 1e-4*abs(reference)。这些是小型显式近似模型，完整网络精度和 ResNet20 分别验收。

2026-10-02：两份原成功Agent源码按当前601执行文件重新编译、密态和重载2/2通过（r506）。未再次生成，原源码、模型、测试输入、权重与reference保持。详见[r508](poseidon-pytorch-broadcast-acceptance-r508.md)。
