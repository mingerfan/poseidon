# 静态张量操作验收

JSON 图 v5 和显式 PyTorch v30 已接入 repeat/tile、repeat_interleave、flip、roll。次数、维度及轴采用公开静态参数；repeat_interleave 首版使用统一正整数次数。各操作与线性层、归约、残差和多输出组合验收。

| 项目 | 结果 |
|---|---|
| 控制测试 | 140 passed，0 failed/errors/skipped |
| 执行与调用账本控制 | 12 passed |
| 隔离捕获、独立明文参考 | 18 来源，288 组／1,120 值 |
| Poseidon 规则候选 | 18/18 |
| 当前真实 Agent | 15/15，首次 14/15，修复 1 |
| 真实生成／HTTP | 16／16 |
| 规则轨编译／密态调用 | 18／54 |
| Agent 轨编译／密态调用 | 16／46 |
| 规则轨逐项比较 | 144 组／560 值，maxabs 3.417705238e-07 |
| Agent 轨逐项比较 | 120 组／448 值，maxabs 8.488301066e-07 |

15 个新来源上下文包含 13 份不同的公开生成请求；3 个旧 v29 来源作为兼容回归单列。两轨使用相同数学模型，数量分别计分。程序包通过重载、新输入、重复执行和零调用缓存复用；Earth/CKKS IR、RuntimePlan、OperatorSpec、常量 bundle、参数绑定和 Poseidon CPU 数值均已独立审计。

冻结门限为 `1e-5 + 1e-4*abs(reference)`。明文参考采用独立 NumPy 与 PyTorch，源规范化保留复制和别名语义。旧身份、初始环境加载失败、身份操作边界及漏报权重的拒绝均保留；新增绑定对应当前源码。

SDK：`6f5c77e2c8ec43ee08cb1fe4e2a7aa292f3a4beccfa0e0f3fee5abf55bed9e2c`。原 x86 配置、后端库、依赖、参数和网络保持，Git 无提交或推送。产物总上限 48 GiB，每批 4 GiB、磁盘余量 4 GiB。

默认源码契约 v18，latest 为 v28，本批 v30 显式选择。动态次数、零元素张量、任意 Python、全部组合与大布局范围另行登记。bootstrap、CUDA、应用精度和完整 ResNet20 继续独立推进。

下一条只读命令：`python3 -B -m json.tool docs/baseline/agent-static-tensor-transforms-acceptance-r1454.json`。
