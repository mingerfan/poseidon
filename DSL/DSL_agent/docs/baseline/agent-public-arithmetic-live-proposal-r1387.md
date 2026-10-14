# v25 公开常量算术：真实 Agent 测试方案

状态：待用户批准。源码 SDK `8e88db16925fad029bdeaea53ebb2ed927ebdb9aece66dd7a252923ccb711809`，ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。

## 验收目的与案例

验证显式 v25 来源经过正式应用入口、真实 DeepSeek 生成与预算内修复后，能够完成 Poseidon CPU 数值验收和程序包复用。15 个新上下文已通过独立参考和规则候选密态验收，使用相同模型、公开权重、布局、参数、reference 和门限。三个旧兼容例不重跑。

| 案例 | 输入 shape | request_id 前缀 |
|---|---|---|
| add_c0 | [2, 3] | dd1ab54776b09496 |
| add_c1 | [2, 3] | 515b0986caa82b75 |
| add_c2 | [2, 3] | 844c8244045e99cc |
| subtract_c0 | [2, 3] | 302d14877b0f4df7 |
| subtract_c1 | [2, 3] | 6a8e372e608b5344 |
| subtract_c2 | [3, 2] | 82bcb6ad46190d89 |
| multiply_c0 | [2, 3] | 28cb9be6b78834aa |
| multiply_c1 | [2, 3] | 9ede4c0594e7f2c5 |
| multiply_c2 | [1, 3] | 894913b5e3041c4b |
| divide_c0 | [2, 3] | fe7ac9f6401089d7 |
| divide_c1 | [2, 3] | 99c16eb1a56b2a64 |
| divide_c2 | [2, 3] | e582bd9a6a501784 |
| negate_c0 | [2, 3] | a62d03c94188755b |
| negate_c1 | [2, 3] | 7bfe00ace91551bd |
| negate_c2 | [1, 3] | 96fe2f984bfb3852 |

公开加、减、乘、除、取负各三例，覆盖均值／标准差预处理、两公开张量广播、布尔 mask 转浮点、alpha、倒数与除法链、计算 Linear 权重、转置和 reshape。精确源码、权重、请求和依赖身份见 [机器方案](agent-public-arithmetic-live-proposal-r1387.json)。

## 有界调用与资源

| 项目 | 上限 |
|---|---|
| Provider | DeepSeek，deepseek-flash，high，stream |
| 单次生成 | 1,200 秒，384,000 tokens |
| 每例 | 初次生成＋最多 3 轮修复；每次生成最多 3 次传输重试 |
| 全批 | 最多 60 次生成／240 次 HTTP |
| 并发 | API 1，密态 1；编译最多 2，链接最多 1 |
| 时间 | 每例 4 小时，全批 12 小时 |
| 产物 | 每例最多 4 GiB，全批新增最多 4 GiB，结果总量最多 32 GiB，磁盘余量至少 4 GiB |
| native | 地址空间 4 GiB，CPU 150 秒 |
| 验收 | 逐项门限 `1e-5 + 1e-4*abs(reference)` |

若每轮均进入后端，上界为 60 次编译与 90 次密态调用，其中 30 次为程序包追加输入／重复执行。实际失败、静态拒绝、重复候选或预算边界会减少调用；未运行项保留，不记为通过。

当前没有明确货币上限，传输超时重试可能重复计费。超过任何既有时间、空间或完整性边界时停止相应批次；不自动扩额，不重试历史未知调用。

[官方 Chat Completions 参数](https://api-docs.deepseek.com/api/create-chat-completion/) 与 [模型列表接口](https://api-docs.deepseek.com/api/list-models/) 在本次准备时核对；请求 token 上限保持 384,000。此核验未使用凭据、未调用生成接口。

## 外发与结果区分

仅发送公开模型／分解图、权重或描述符、布局、DSL 规则及受控反馈。规则候选、私有输入、reference 输出、凭据与私钥不进入生成请求。

正常工作流程复用现有 Application、Provider、自动修复、验证适配器和程序包接口。候选必须由真实 Agent 原样产生；人工修复不计入 Agent 成功。每项检查初次／修复后结果、实际 tracing／编译／密态数值、包重载、三组新输入、一组重复输入和缓存复用。原函数应用精度另计，本批为声明的 float64 数值语义。

仅新增开发测试驱动、清单和文档，生产 SDK、后端、网络、依赖、参数及默认入口保持。34 项无收费控制检查通过，0 failed／skipped；首轮旧批次的硬编码计数上限已在本开发驱动中改为本批冻结预算，原日志保留。`--plan` 已核验，当前 API、编译、密态调用均为 0。

计划绑定：`093d8de29da92f57680066c97925680fbc524fa9232b1acc00ebd46d0b9c9755`。
运行器清单绑定：`d27e623f98b4e754a6ed3ee3281c2c807f584103df0e28cb31d08fbad5f50fb2`。
原离线 [r1386 结果](agent-public-arithmetic-offline-results-r1386.md) 保持原绑定。

唯一启动入口，在获得相应审批记录后运行：
`python3 -B scripts/baseline/benchmarks/tools/public_arithmetic_live_v1/run.py --execute /home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-arithmetic-live-v1 --approval-record docs/baseline/agent-public-arithmetic-live-runner-approval-r1387.json`
