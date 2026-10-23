# PyTorch v25 公开参数算术：真实 Agent 验收

2026-10-11。15项全部首次生成通过，修复0、失败0、跳过0、未运行0。采用当前真实 Agent 返回的原始 DSL，候选源码逐字节与原始响应一致。

| 公开参数运算 | 上下文数 | 首次通过 | 修复后通过 | 失败 |
|---|---:|---:|---:|---:|
| 加 | 3 | 3 | 0 | 0 |
| 减 | 3 | 3 | 0 | 0 |
| 乘 | 3 | 3 | 0 | 0 |
| 除 | 3 | 3 | 0 | 0 |
| 取负 | 3 | 3 | 0 | 0 |

输入来自 `poseidon-trusted-pytorch-v25`，公开参数和 buffer 的运算先由可信导入工具计算并绑定；涉及加密输入的计算保留在图中。本批包含公开广播、类型与形状转换、张量算术及接入 Linear 的组合。生成指导为 `explicit-v13`，后端为 Poseidon CPU，P4/w45。

通过链路：Hecate tracing → Dacapo-modified → Earth/CKKS IR → RuntimePlan、OperatorSpec 和 plaintext bundle → CKKS Runtime → Poseidon CPU → 解密和独立 reference 逐项比较。

| 验收项 | 结果 |
|---|---:|
| 真实生成 / HTTP | 15 / 15 |
| 编译 / 密态调用 | 15 / 45 |
| 输入组次 / 输出值 | 120 / 656 |
| 最大绝对误差 | 1.93615496136e-07 |
| MAE | 4.39776182765e-09 |
| 非零 reference 最大相对误差 | 6.04714670768e-06 |
| 最小有定义 cosine similarity | 0.999999999998946 |

数值门限：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。120组次由60组初次资格输入、45组新输入和15组重复输入构成；重复组不作为新增独立样本。

15份程序包全部通过导出、重载、3组新输入、1组重复输入和缓存复用。重载执行与缓存命中未增加生成或编译调用。执行器只获得评估密钥。

最终只读审计复核1323个产物文件，重新计算两套参考和逐项误差，核对原始响应、返回源码、请求、公开权重、编译产物、参数、沙箱与执行绑定。审计新增API、编译和密态调用均为0。冻结执行器的34项控制检查通过，failed/errors/skipped均为0。

后续只读复核工具首轮误用了要求存活PID的运行中校验，修正为结束态的计划、授权、已退出PID及保留锁声明校验后通过；原错误日志和工具草稿保留。15项后端结果及其绑定保持，未追加API、编译或密态执行。

服务返回用量共238,361 tokens，其中prompt 156,787、completion 81,574。所有调用已结算；本地记录未提供实际美元账单。

从启动到最终只读审计47.29分钟；新增产物205.23 MiB，全局29.08 GiB，可用磁盘40.60 GiB。原时间、空间、并发、native资源和调用上限保持。

执行 checkout：`/home/lhohy/Code/Poseidon`，分支`lhy-agent-dsl`，HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。
SDK身份：`8e88db16925fad029bdeaea53ebb2ed927ebdb9aece66dd7a252923ccb711809`。编译、执行和验证各阶段身份保存在JSON；本轮SDK、后端库、参数、reference、门限及x86配置保持。网络设置、安装、下载、commit、push均为0。

程序包和原始证据位于 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-public-arithmetic-live-v1`，每例的 `tasks/<task_id>/program` 保存可复用程序包。原始响应、失败诊断槽位、IR、解密结果、调用账本和旧报告保留。

[机器记录](agent-public-arithmetic-live-acceptance-r1388.json) · [原冻结方案](agent-public-arithmetic-live-proposal-r1387.json) · [授权记录](agent-public-arithmetic-live-authorization-r1387.json) · [使用说明](operator-decomposition-and-python.md#pytorch-v25-公开常量算术)

本次验收覆盖这15个输入写法上下文。v25继续显式选择，默认v18/latest=v21保持；更大模型、更多组合、bootstrap和CUDA使用各自验收状态。
