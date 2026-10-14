# V12 输入与 explicit-v11 真实 Agent 评测

待批准，API调用为0。三例已在当前代码下完成规则基线密态验收，本批检验新接口下Agent的生成和自动修复质量。

| 案例 | 内容 | 冻结 request_id |
|---|---|---|
| linear | Linear、局部视图写入、三输出快照 | 976a4c98ad4f0efdc44ea9be5e4e9831468e71e10fdeb30d91f53a9ffe282670 |
| norm | LayerNorm近似、原地更新、双输出 | 2dc4e43c3f6e72bdb8d4b286a53879e5bf46db05eaa6d5247983260e5b5efc89 |
| split_weight | 公开权重split、Linear、输出组合 | f781a798e419af8db6a86b6c98fd1e3120c02b199c15a12a34fb3fb727a3bb98 |

DeepSeek deepseek-flash/high/stream；每例最多3轮修复、每次生成最多3次传输重试，共最多12次生成／48次HTTP。单次1200秒、384000 tokens；API和密态串行，编译2／链接1。整批12小时、每例4小时／4GiB、总产物32GiB、磁盘余量4GiB。

只外发公开模型、公开权重或描述符、布局、DSL规则与受控反馈。规则DSL答案、来源包／reference报告、测试数组、reference输出、凭据和私钥不外发。无明确货币上限，超时重试可能重复计费；总空间边界触及时停止并保留现场。

模型、源码、reference、参数和门限保持；通过须有真实Agent来源、真实编译、Poseidon密态比较及包重载／新输入。原76项不重开，不转移其未用额度。冻结细节见同名JSON与 scripts/baseline/benchmarks/plans/v12-application-live-r1232。
