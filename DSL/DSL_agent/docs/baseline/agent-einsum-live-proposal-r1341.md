# einsum 真实 Agent 测试方案 r1341

本方案等待新的付费批准，当前 API 调用为 0。
八份数学请求来自当前来源版本 v18 的已验收任务，模型、权重及来源保持。
数据冻结没有重新捕获来源、编译或执行密态计算。

## 案例

| 案例 | 主要内容 |
|---|---|
| einsum_matrix | 双矩阵收缩 |
| einsum_heads | 批次与多头张量收缩 |
| einsum_contraction_broadcast | 长度 1 收缩轴广播 |
| einsum_three | 三张量收缩 |
| einsum_four | 四张量收缩 |
| einsum_public_weight | 固定非重复矩阵权重 |
| einsum_view_mutation | 共享转置视图写入与快照 |
| einsum_partition_composition | 投影、chunk/unbind 和多输出 |

八例规则候选均已通过真实编译、Poseidon CPU 数值和程序包新输入验收。
本批检验实际生成、自动修复和交付，规则候选源码不进入模型请求。

## 预算

- DeepSeek `deepseek-flash`／high／stream，`explicit-v13`，原 P4/w45。
- 每例最多 3 轮修复：最多 4 次生成、16 次 HTTP 尝试。
- 全批最多 **32 次生成、128 次 HTTP 尝试**；每次生成最多 3 次传输重试。
- 单次 1200 秒、384000 tokens；每例 4 小时，全批 12 小时。
- API 和密态串行；编译最多 2 jobs，链接 1。
- 新产物合计最多 4 GiB；每例最多 4 GiB；全局 32 GiB；磁盘余量至少 4 GiB。
- native 原有 4 GiB／CPU 150 秒限制保持。
- **无明确货币上限**；超时和传输重试可能重复计费。

外发为公开数学图、权重或描述符、布局、DSL 规则和受控反馈。
私有输入、reference 答案、规则 DSL、凭据和内部来源记录保留在执行端。
首次通过、修复通过、编译／数值失败、预算耗尽和环境故障分别计数。

## 绑定与执行要求

SDK：`ee05605763e8f26e106c8c48167afb7017cee7c6fa946093626dc9c35bf29995`。

提案绑定：`f6a8b322dbcbf083e8ce7e914940d0d5ca4f60cfb114a8b59c0fe5eaf287083d`。

完整请求、公有上下文、模型／权重／来源清单和四阶段源码身份见
[机器方案](agent-einsum-live-proposal-r1341.json)。
八份公有上下文为 40512–50864 字节，均在原 131072 字节上限内。

执行需创建新任务和账本，再次核验当前运行依赖、来源捕获结果和冻结数学请求。
已结束的规则任务、旧付费额度与未结算的历史 HTTP 均不复用。
当前只完成数据冻结和计划完整性检查，运行批准和实际执行另行记录。

只读查看：

~~~bash
python3 -B scripts/baseline/benchmarks/tools/einsum_live_v1/scope.py --plan
~~~

15 项计划控制测试通过，failed/errors/skipped 为 0。
首轮工具把压缩传输表示直接用于逻辑哈希，完整性检查拒绝；
随后通过已有 codec 恢复表示再校验，保留首轮工具和失败记录，
没有修改请求、哈希规则或额度。

六项 v17 拆分输入与两项 MHA 不在本批，仍待新的来源准备和付费方案。
容量、应用精度、bootstrap、CUDA、密文交接及整网络目标保持原计划。
