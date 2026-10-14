# Poseidon 三模型真实 Agent 提案

状态：待批准。替代未执行的 r256；此次仅更新已验收的 bundle 适配器与测试源码绑定。模型、公开请求、固定参数和后端二进制均相同。

| 模型 | 输入 | 输出 |
|---|---|---|
| 平方激活 MLP | 4 元素 | 2 元素 |
| RMSNorm | 4 元素 | 4 元素 |
| Attention decode | x、cached_k、cached_v 各2元素 | 2元素结果、4元素keys、4元素values |

DeepSeek deepseek-flash/high/stream，explicit-v10，Poseidon w45。每例最多3轮修复、每次生成最多3次传输重试；总上限12次生成/48次HTTP。单次1200秒、384000 tokens，整批12小时、每例4小时；API/密态串行、编译≤2、链接≤1。单例4GiB、全局产物32GiB、磁盘余量4GiB。

仅发送公开模型、权重、布局、DSL规则及受限诊断；不发送人工表达、测试输入或reference。无明确货币上限，超时重试可能重复计费。使用现有依赖，下载与安装为0。

验收包括真实frontend、RuntimePlan/OperatorSpec/bundle、Poseidon CPU执行、四组输入逐项解密比较、程序包重载。冻结门限保持1e-5+1e-4*abs(reference)。失败分层记录，人工候选不计入Agent成功。

新binding：6c1479394207dfed7fa4f071f2b61ea4c9108125c0b4f2b0cce4d49bcf82a6b8。精确请求、源码哈希、命令及预算见 [JSON](poseidon-whole-models-live-proposal-r268.json)。准备结果 results/poseidon-whole-models-prepare-r268a；48组双参考通过，API调用0。
