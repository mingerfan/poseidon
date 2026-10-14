# MHA 组件接入验收（r1312）

正式入口已接入可信 PyTorch v16；mixed packed／replicated-cell 布局已提取到 Agent 自有组件。默认规则路径和旧请求格式保持。三个原固定低阶模型完成当前正式组件绑定下的真实编译、Poseidon CPU 密态数值验收、程序包重载、新输入和实际重复执行。

## 修改

- 激活已审查的 MHA 捕获、公开权重来源、近似选择和 None 输出结构检查，来源清单显式选择 v16。旧 v1–v15 保留各自规则。
- 新增 `unified_mixed_layout.py`、`replicated_cell_emitter.py`、`unified_layout_strategies.py`；运行组件不再导入实验工具。增加 foreign emitter／storage 拒绝。
- `candidate_source(request, strategy="mixed-packed-cells-v1")` 返回原有 source／record 二元组。默认行为保持；Provider 候选原样验证。
- 能力清单登记 v16；各配置仍分别进行编译与密态资格检查。三个公开示例及简短输入说明已加入仓库。
- 修改前的 Agent 文件逐字节保存在 `scripts/baseline/benchmarks/tools/mha_component_cutover_v1/source-before`，原 staged 源码、模型、响应和结果保持。

## 验收

| 项目 | 结果 |
|---|---|
| 组件与兼容测试 | 161 passed，0 failed／error／skipped |
| 正式来源隔离捕获 | 3 模型，48 组／672 值；CPU float64 双参考 |
| tracing／Dacapo／RuntimePlan／Poseidon CPU | 3/3 |
| 原四组密态资格 | 3/3 |
| 程序包重载后的三组新输入 | 3/3 |
| 实际第二次执行一组输入 | 3/3 |
| 密态比较总量 | 24 组／336 值，包含 3 组重复 |
| 最大绝对误差 | 6.62685212399e-08 |
| 数值门限 | `1e-5 + 1e-4*abs(reference)` |
| 原请求与规则 DSL | 逐项绑定相同，源码逐字节相同 |

三个上下文为 self Attention＋输出 Linear＋残差、投影后 cross Attention＋嵌套 None 输出、布尔 mask／padding＋残差。沿用原 P4、权重、布局、w45 参数和 reference；本批是规则候选，不计为真实 Agent 新生成。

固定 exp 一阶／reciprocal 一次近似的原函数误差分别约为 0.2845、0.006420、0.7659。CKKS 误差比较的是声明近似图；这些低阶配置尚未取得应用精度认证。更高阶原案例仍保留其编译／深度限制，不由本批替代。

## 保留的初轮问题

验收驱动最初的未使用 import 在测试前失败，测试数量为 0。组件测试随后全部通过。

第一轮正式捕获三例全部通过；验收驱动错误地比较内存绑定 list 与磁盘压缩 dict，在真实编译前停止。既有 `request_file_bytes` 序列化后的三个请求与原文件完全一致，模型、权重、布局、参数、请求 ID 不变。首轮 0/3 工作流报告保留；修正仅在驱动内完成，复用已通过的正式来源捕获，不重写旧结果。

## 身份与资源

四种应用阶段的实际源码闭包均为 691 项，来源捕获为 113 项；保护的 654 项无变化。程序包使用当前完整身份和依赖绑定，旧包、旧报告保留原绑定。原请求哈希规则和版本保持。

累计验收约 308.3 秒，低于原 1200 秒窗口。编译≤2、链接≤1、密态串行；原 native 4 GiB／CPU 150 秒限制保持。总产物约 25.08 GiB，余量约 43.06 GiB。实际执行通过；峰值内存、密钥及 evaluator 工作区仍无完整测量。

本批付费 API、HTTP、安装、下载、后端库及参数修改、commit／push／PR 均为 0。ARM 真实执行；x86 配置保留，未进行本批 x86 执行。Bootstrap 和 CUDA 不在本批执行范围。

## 继续工作

1. 以模型类型、shape 和公开绑定选择布局指导；真实 Agent 不接收规则 DSL 答案。
2. 冻结 v16 的真实 Agent 案例、请求、源码和调用预算，再申请新批次授权。
3. 继续通用算子、容量、近似精度、bootstrap 和整模型目标。当前身份仍存在较宽的共享依赖闭包，后续按实际依赖边界完善。

只读复核：

~~~sh
python3 -B scripts/baseline/benchmarks/tools/mha_component_cutover_v1/inspect_current.py
~~~

[机器报告](agent-mha-component-cutover-results-r1312.json) · [输入用法](operator-decomposition-and-python.md#pytorch-v16-multiheadattention)
