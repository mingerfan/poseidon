# MHA v16 真实 Agent 测试申请

本批验证正式来源接入后的自由生成及有界修复；此前规则候选结果保留，独立计分。

| 案例 | 内容 |
|---|---|
| self_post_linear_residual__declared_shallow_v1 | self Attention → 输出 Linear → 残差 |
| linear_into_separate_cross_none__declared_shallow_v1 | 输入投影 → 独立 Q/K/V cross Attention → 命名输出及 None |
| boolean_padding_residual__declared_shallow_v1 | bool mask、padding、残差与输出 |

公开模型、权重、P4、explicit-v11、w45 及原近似保持；精确请求、源码和依赖身份见同名 JSON。

- DeepSeek deepseek-flash／high／stream，API 和密态串行。
- 每例最多 3 轮修复、每次生成最多 3 次传输重试；全批最多 12 次生成、48 次 HTTP。
- 单次 1200 秒、384000 tokens；每例 4 小时、全批 12 小时。
- 新批次产物合计最多 4 GiB、全局 32 GiB、磁盘余量至少 4 GiB；编译≤2、链接≤1，native 原 4 GiB／CPU 150 秒限制保持。
- 外发公开图、权重／描述符、布局、DSL 规则与受控诊断。规则 DSL、历史候选、私有测试数组、reference 和凭据留在本地。
- 没有明确货币上限；超时重试可能重复计费。仅验收声明的近似计算，原函数应用精度单列。
- 当前付费调用为 0，状态为等待授权。r1306 额度已经执行完，本批需要新的具体批准。

[冻结范围](agent-mha-v16-live-proposal-r1313.json)
