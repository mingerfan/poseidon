# Poseidon 三个小模型真实 Agent 评测方案 r243

状态：待批准；当前付费调用 0。

| 案例 | 输入 | 输出 |
|---|---|---|
| 平方激活 MLP | 4 元素 | 2 元素；结构 4→2→2 |
| RMSNorm | 4 元素 | 4 元素；原三轮倒平方根近似 |
| Attention decode | 当前 token、cached key、cached value 各 [1,2] | 预测 [1,2]、key/value 缓存各 [2,2]；原六阶 exp、三轮倒数近似 |

模型来自原验收案例，没有换权重或近似。每例 16 组双参考一致性检查通过；三个 CLI --prepare 请求与组件请求一致。新文件是原模型副本，不增加 1,200 个 Benchmark 的模型计数。

生成使用 deepseek-flash/high/stream、explicit-v10 和 Poseidon w45。官方当前将 deepseek-flash 映射到 DeepSeek-V4.1-Flash，详见 [官方模型说明](https://api-docs.deepseek.com/quick_start/pricing/)。每例首次生成加最多 3 轮修复，每次生成最多 3 次传输重试；合计最多 12 次生成、48 次 HTTP。API 和密态串行，编译最多2、链接最多1；单次1200秒／384000 tokens，每例4小时、整批12小时；单例4GiB／总产物32GiB，磁盘余量4GiB。无明确货币上限，传输超时重试可能重复计费。

外发公开计算图、固定权重、布局、DSL规则和受限诊断；人工通过程序、规则 DSL、私有数组、reference 答案和凭据不外发。成功要求真实 Agent 输出、真实 tracing/编译、产物检查、Poseidon 密态执行、解密与四组逐项比较，并通过程序包重载。误差门限保持原值，编译失败及预算耗尽分别保留；不退回 SEAL，不自动重启或替换为人工答案。

精确案例、请求、命令、预算和绑定见 [JSON](poseidon-whole-models-live-proposal-r243.json)。bootstrap 参数试验 r235 是另一项独立授权。
