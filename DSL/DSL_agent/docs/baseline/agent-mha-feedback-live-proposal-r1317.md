# MHA 两项反馈重测方案

当前状态：待批准；新增 API 调用 0。

案例保持原模型、公开权重、request、reference 和 P4/w45 配置：

1. self Attention → Linear → 残差：`self_post_linear_residual__declared_shallow_v1`
2. Linear → cross Attention：`linear_into_separate_cross_none__declared_shallow_v1`

本轮验证 Agent 层新增工作量／hc.func 签名反馈及两个未通过组合的自由生成。布尔 mask 模型已有原 Agent 源码离线交付结果，不再次付费生成。原失败和调用记录保留。

使用 DeepSeek `deepseek-flash`、high、stream、explicit-v11。每例最多 3 轮修复、4 次生成；每次生成最多 3 次传输重试、4 次 HTTP。全批最多 **8 次生成／32 次 HTTP**。API 和密态串行，编译最多 2、链接最多 1；native 内存 4 GiB、CPU 150 秒保持。

单次请求最多 1200 秒、384000 tokens；每例 4 小时、全批 8 小时。新增产物合计 4 GiB，全部 results 32 GiB，磁盘余量至少 4 GiB。达到上限即停止并保留证据。

只外发公开模型、权重或描述符、layout、DSL 规则和受控反馈；不发送人工答案、私有输入、reference 或凭据。无明确货币上限，超时重试可能重复计费。

新测试独立绑定当前源码与依赖，不沿用已关闭批次的批准，不复写旧报告。新增付费调用须先批准该范围，执行前再次核对哈希、进程和硬预算。

授权依据：Ubuntu `docs/baseline/agent-mha-feedback-live-proposal-r1317.json`。

方案绑定：`509391f78f2bc75e6f5fb749e9c048ce9ea049b9aa0e51f5d7c62521b8fd4d69`。


## 校验表修正

r1316 保留。r1317 仅将重复校验表中的 candidate_check_diagnostic.py、decorated_functions.py、
unified_graph_contract.py 三个旧哈希改为原方案四阶段身份中已经记录的当前哈希。

模型、完整请求、四阶段身份、Provider、外发内容、数值门限和预算均与 r1316 相同。
修正依据见 agent-mha-proposal-correction-r1317.json；批次入口逐次核验该等价关系，
不会把变更后的执行范围套用到原批准上。原两项批准问题继续对应相同执行范围。
当前尚未取得这两项的批准，没有创建正式授权记录或调用 API。

批次入口：scripts/baseline/benchmarks/tools/mha_feedback_live_v3/run.py。
只读计划可在 Ubuntu checkout 执行：

```sh
python3 -B scripts/baseline/benchmarks/tools/mha_feedback_live_v3/run.py --plan
```
