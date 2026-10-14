# GroupNorm / InstanceNorm 源码校准

v3 源码准备接口增加分组和通道统计。GroupNorm 绑定分组数、通道数、eps；InstanceNorm 绑定维度、通道数、eps、training 和 running-statistics 模式。旧 v1/v2 行为保持。

## 统计

group_norm_variance 计算每批、每组的总体方差加 eps；instance_norm_variance 计算每批、每通道的空间总体方差加 eps。参数从原始模块或静态函数调用提取，观测保留原生 float32 / float64。

输入统计与固定 running statistics 分开处理。冻结配置使用已观测的方差区间核验 rsqrt，不用原输入区间替代，也不自动扩大区间。

## 本批结果

| 验收 | 状态 |
|---|---:|
| 控制检查及旧接口回归 | 118/118 |
| 隔离校准、冻结与原生捕获 | 6/6 |
| 原规则基线 | 0/6；编译 scale750–795 超过固定预算720 |
| 人工等价表达 | 6/6；真实编译、密态数值、包复用通过 |
| 总编译尝试 | 12；6成功、6失败保留 |
| 密态调用 | 18 |
| API调用 | 0 |

每个案例合并三个相邻的精确二进制公开因子，模型、权重、reference、布局和参数均保持。原规则源、人工源及数学差分见证分开保存；人工源不计为 Agent 生成，也不外发给模型。

规则源和等价源的独立数学差分约2.22e-16以内。人工表达的资格验收为24组输入、192值，最大绝对误差 1.30829e-07；复用同为24组、192值，最大误差 1.85171e-07。冻结门限仍为1e-5+1e-4*abs(reference)。

184个包文件、172个产物文件哈希复核。原六项编译失败保留，不归因为模型无法表达或编译器bug。原函数应用精度未认证。

## 真实 Agent 测试

GroupNorm三项（模块、函数输入变换、批量模块）及InstanceNorm三项（1d无batch、函数输入变换、2d无batch）已冻结。

采用当前计划内统一授权：DeepSeek deepseek-flash/high/stream、explicit-v21、P4/w45；每例最多3轮修复、每次最多3次传输重试；全批24生成/96HTTP，API与密态串行，编译2/链接1。单次1200秒/384000tokens，全批12小时/每例4小时，新增512MiB/总48GiB/余量4GiB，native4GiB/CPU150秒。货币费用无明确上限，重试可能重复计费。

只提供公开数学图、权重或描述符、布局、DSL规则和受控反馈。规则与人工答案、私有输入、reference和凭据留在本地。新生成结果另计。

SDK 6b94fa16f195f543839fdf83e4f2e4fbd224cc1cf8c525746e588d0df920c429
审计 97bd4796bef91669fb0e1562b2428ea0a13d07971de3cd4ac3c7b66b4eff2f60
累计结果 39.715/48GiB，余量 29.549GiB。

本批保持Poseidon、Runtime、Dacapo、安全参数及网络；无安装下载、commit、push或PR。后续继续精度、容量、算子组合、bootstrap、CUDA及整模型工作。

[机器可读报告](agent-normalization-source-calibration-r1692.json) · [真实测试范围](agent-normalization-source-live-scope-r1693.json)
