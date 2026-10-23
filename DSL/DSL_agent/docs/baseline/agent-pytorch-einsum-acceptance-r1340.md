# PyTorch einsum 接入与验收 r1340

可信 PyTorch 输入新增 `poseidon-trusted-pytorch-v18`。静态 `torch.einsum`
在 Agent 捕获层分解为现有 permute、reshape、matmul、广播乘法和 sum，
继续使用 Dacapo → RuntimePlan → CKKS Runtime → Poseidon CPU。

开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。
变更限于 Agent 输入适配、来源契约、能力登记、测试和文档。
后端库、数学 reference、参数和数值门限保持。

## 输入范围

支持 1–4 个张量操作数、静态字符串方程、显式或隐式输出、ellipsis、
标签广播和静态归约；沿用 v17 的 chunk/unbind、有序视图、公开权重及输出结构。
Tensor 操作数 rank 1–32，仍受所选资源配置的元素、轴和展开节点上限约束。
标量归约输出可沿用现有输出绑定；标量 Tensor 操作数、同一操作数内的重复标签、
整数标签接口和动态方程返回明确能力拒绝。

双输入优先形成分组矩阵乘，避免不必要的外积工作区；
其余合法情形形成有界广播乘法和归约。
传输检查重新计算计划、形状及逐节点映射，保留视图别名和原写入顺序。
旧版本来源、JSON、CLI 和请求规则保持。[输入格式](operator-decomposition-and-python.md#pytorch-v18-静态-einsum)。

## 验收结果

| 范围 | passed | failed / errors / skipped |
|---|---:|---:|
| 新方程规划、独立参考、来源与传输检查 | 36 | 0 / 0 / 0 |
| chunk/unbind 与旧输入、能力、MHA、GELU、捕获兼容 | 120 | 0 / 0 / 0 |
| 八模型真实编译、密态数值与程序包新输入 | 8 | 0 / 0 / 0 |

每例通过隔离来源捕获和独立参考检查，再执行真实 Hecate tracing、
Earth/CKKS 编译、RuntimePlan／OperatorSpec／plaintext bundle 检查、
Poseidon CPU 密态计算及解密比较。程序包重载后执行四组新输入，
复用时生成和编译调用均为 0。

| 模型 | 输入 / 组合 | 结果 |
|---|---|---|
| einsum_matrix | 双矩阵 `ij,jk->ik` | 通过 |
| einsum_heads | 四维批次与分头收缩 | 通过 |
| einsum_contraction_broadcast | 收缩轴长度 1 的广播 | 通过 |
| einsum_three | 三张量收缩 | 通过 |
| einsum_four | 四张量收缩 | 通过 |
| einsum_public_weight | 非重复公开矩阵权重 | 通过 |
| einsum_view_mutation | 转置视图写入与 clone 快照 | 通过 |
| einsum_partition_composition | 公开投影、chunk/unbind、多输出 | 通过 |

合计 **8 次编译、16 次密态调用、64 组输入、304 个输出值**，
最大绝对误差 **5.834266505861253e-7**。
逐项门限为 `1e-5 + 1e-4*abs(reference)`。

八例均为规则候选，当前真实 Agent 新生成通过数为 **0**。
高 rank 与拒绝边界来自静态测试；本批密态测试限于表列小型形状。

SDK 身份：`ee05605763e8f26e106c8c48167afb7017cee7c6fa946093626dc9c35bf29995`。
阶段源数量：生成 631，编译／执行／验证各 625。
654 项后端和数学参考文件、两份历史付费请求保持原字节。
x86 配置保留，本次实际执行为 ARM CPU。

## 保留记录与后续

最初工具缺少命令级的既有 C++ 动态库路径；首轮八任务又因预算对象缺少字段、
HTTP 额度为 0，在来源捕获前被拒绝。纠正工具后创建新任务，保留旧终态、
失败报告及原截止时间；没有重置时间窗口。

另一份 16 源码准备工具在 Nix 元数据进程中停止，尚未进入来源 worker。
原始具体原因未确认；之后显式 ARM 配置的只读元数据探针成功。
该准备窗口已结束，记录保留，不重启或复用它的额度。
本批八份当前来源请求可直接用于新付费方案的数据冻结。

六例拆分输入与两例 MHA 的真实 Agent 测试仍待新的来源绑定。
规模、应用精度、密文交接、bootstrap、CUDA 和整网络目标继续。

本批 API、安装、下载、后端及参数修改、commit/push 为 0。
原生验收约 579.36 秒，新结果约 66.07 MiB，
全局结果约 25.86 GiB，磁盘余量约 42.51 GiB。

[机器报告](agent-pytorch-einsum-acceptance-r1340.json)。
