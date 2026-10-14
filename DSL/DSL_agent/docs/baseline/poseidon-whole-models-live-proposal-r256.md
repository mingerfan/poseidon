# 三个小模型的真实 Agent 提案

案例为 MLP 4→2→2、4 元素 RMSNorm、三输入三输出 Attention decode，模型文件与 r243 逐字节相同。三例均已在新入口完成离线数值验收、程序包重载和缓存复用；48 组双参考与真实 CLI prepare 通过。

新绑定：`15f13faa2f010accae98c9378a91c3589269c1f71dfc5e9bae1991181fa30e33`。r243 未执行，由本提案替代；原文件保留。

DeepSeek deepseek-flash/high/stream，explicit-v10，Poseidon w45。最多 3 轮修复、每次生成最多 3 次传输重试，总上限 12 次生成、48 次 HTTP。API/密态串行，编译≤2、链接≤1；单次 1200 秒、384000 tokens，单例 4 小时、整批 12 小时。单例 4 GiB、总产物 32 GiB、磁盘余量 4 GiB。

外发仅公开图、权重、布局、DSL 规则和受限反馈。人工源码、私有输入和 reference 保留本地。无明确货币上限，重试可能重复计费。当前 paid_calls=0；批准前保持准备状态。

准确请求、预算和模型哈希见 [JSON](poseidon-whole-models-live-proposal-r256.json)。
