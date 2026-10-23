# MHA v12 两项真实 Agent 评测方案

验证通用逻辑轴/分组指导对剩余 self 数值失败的改善，并为 cross 流异常保存受控诊断。两例数学模型、公开权重、布局、packing与P4/w45保持，只更新生成规则与当前阶段身份。

当前已完成[规则与后端验收](agent-tensor-axis-guidance-r1332.md)和[完整上下文预算修复](agent-tensor-axis-context-budget-r1334.md)。历史成功源码重放与规则程序不计入本次 Agent 新生成成功。

| 案例 | 新请求哈希 | 公开上下文字节 |
|---|---|---:|
| self_post_linear_residual__declared_shallow_v1 | 5a4136d4ccdab5dd1ee478fe7b4fe71f727ec7a5f5386ace41e1c488c10185b2 | 125428 |
| linear_into_separate_cross_none__declared_shallow_v1 | 8d51d457fab5f907b287daaa7a62a933e1156cdba3de340d099de5d811a6c15a | 131069 |

## 范围和预算

- DeepSeek deepseek-flash/high/stream；explicit-v12。
- 每例最多3轮语义修复，每次生成最多3次传输重试；两例合计最多8次生成、32次HTTP。
- 单次1200秒、384000 tokens；API与密态串行，编译≤2/链接≤1。
- 每例4小时、全批8小时；两例新增产物合计≤4GiB、全局≤32GiB、磁盘余量≥4GiB。
- native 4GiB/CPU150秒保持。
- 外发公开模型、权重或描述符、布局、DSL规则和受控诊断。
- 人工/规则源码、私有测试输入、reference、凭据及环境信息不外发。
- 未设置明确货币上限；超时重试可能重复计费。
- 成功要求新的真实Agent程序通过原数值门限及程序包交付/重载/复用验收；不放宽模型或约束。

r1330额度已经关闭，新方案为awaiting_user_approval，当前已执行付费调用0。批准前不建立live transport。输入不支持、编译/数值失败、Provider错误、重复候选及预算耗尽按既有终态分别记录。

[完整冻结方案](agent-mha-tensor-axis-live-proposal-r1333.json)。

方案绑定：475391a7ef31efef047900e89fe39db9bbc70d63ea63a900e1e88c0f82ae2d1f
