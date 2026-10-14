# MHA 两项有界重测 r1330

案例：self Attention＋Linear＋残差；投影后的 cross Attention。沿用原两项的模型、公开权重、请求及 P4/w45，代码和 worker 策略按 r1329 验收后的状态冻结。

配置：DeepSeek deepseek-flash／high／stream，explicit-v11。每例最多 3 轮修复；每次生成最多 3 次传输重试。全批最多 8 次生成／32 次 HTTP；单次 1200 秒／384000 tokens。

API、密态均串行；编译最多 2、链接 1。每例 4 小时，全批 8 小时；新增产物合计最多 4 GiB、全局最多 32 GiB，磁盘余量至少 4 GiB。native 原 4 GiB／CPU 150 秒保持。

只外发公开模型、权重或描述符、布局、DSL 规则及受控反馈。规则答案、私有输入、reference 和凭据留在执行端。无明确货币上限，超时重试可能重复计费；原函数应用精度另行认证。

目的：检查剩余两项的当前真实生成、预算内自动修复及数值交付。r1313 三项已结算；r1317 未执行，旧批准不能授权当前新绑定。

runner 54 项控制测试和只读 `--plan` 已通过，当前 approved=false、付费调用 0。新授权必须对应 r1330 的精确案例、源码和额度。

程序绑定：`c9932a659a0ac6a0c34ed4038b067b1ee7129930df0a5e581b0240d1f312c9e5`。

[精确方案](agent-mha-feedback-live-proposal-r1330.json)

批准后入口：`python3 -B scripts/baseline/benchmarks/tools/mha_feedback_live_v4/run.py`，由工具校验授权记录后启动；当前仅支持只读计划。
