# Agent 接口与模块

## 当前来源选择

CLI 默认使用 `poseidon-trusted-pytorch-v18`；`--source-format latest` 选择 v34。
清单保存解析后的固定版本，旧显式来源和请求按原规则重建。
v29–v34 共用多文件源码包策略，目录、文件及工厂入口均参与来源绑定。

v34 接入公开与加密 Tensor 的 cat/concat/concatenate/stack，以及闭合公开拼接求值。
公开 dtype、静态轴、shape、派生状态及额外密文运算分别检查并绑定。
规则轨21/21、真实Agent18/18与多文件CLI三例已完成真实编译、Poseidon数值及程序包复用。
[拼接验收](agent-public-join-acceptance-r1480.md) · [CLI验收](agent-source-cli-acceptance-r1482.md)。

统一图 v2 可选 `output_binding_policy="exact-values-v1"`，按同一图值合并物理输出块；命名、shape、逻辑元素与结构保持。CLI 为 `--output-binding exact-values-v1`，`explicit-v15` 提供对应指导。规则12/12、当前真实Agent9/9首轮通过，并完成新输入和包复用；默认布局与旧请求保持。[输出共享验收](agent-output-alias-acceptance-r1486.md)。

以下版本小节保留兼容契约及各历史批次配置，验收结果归属对应来源绑定。

用户入口见 [Agent README](../../scripts/README.md)，模型格式见[输入说明](supported-inputs-models-configurations.md)。

## 调用链

```text
模型 → 类型/形状/布局检查 → 生成请求 → Provider
     → DSL 检查 → Hecate tracing → dacapo-modified → RuntimePlan / plaintext bundle
     → CKKS Runtime → Poseidon CPU → 解密比较 → 程序包
```

输入是静态图和公开权重。Provider 只接收公开模型、布局、DSL 规则和受控反馈，不接收参考答案、测试数组、凭据或私钥。候选响应只包含 schema、request_id、hecate_source。

## 已有 DSL 验收

生成与后端验收是两个边界。模型、公开权重和候选源码固定后，可直接通过 `component_backend.qualify()` 进入当前编译与数值链路。旧候选需要新后端请求绑定，原响应和原请求保留；不再调用 Provider。

已验收的程序包由 `candidate_bundle` 重载，可重复执行或按相同身份复用。 宿主可通过[编译资源接口](application-agent-component-v1.md#编译程序包的资源分析)先分析当前产物；该分析不改变原验收等级。结果分别标明历史真实 Agent、规则基线、人工表达或当前新生成来源。后端通过不改变原生成来源，也不代表当前生成质量评测通过。历史来源封装及迁移结果见 [迁移验收](poseidon-migration-status-r373.md)。

## 模块

| 模块 | 职责 |
|---|---|
| application_component.py | 任务状态、预算、取消、终态、复用 |
| application_policy.py / unified_poseidon_guidance.py | 能力描述、按公开图选择 Poseidon 上下文和修复决策 |
| component_contract.py | 请求准备、候选检查、生成接口 |
| run_candidate.py / run_feedback_loop | 单例 CLI 与生成修复循环 |
| validation_adapter.py | 静态、编译、产物和数值验收 |
| component_backend.py | 对接 Poseidon 验证运行器 |
| deepseek_provider.py | DeepSeek 协议和受控传输重试 |
| candidate_bundle.py | DSL 程序包导出、完整性和重放 |
| application_planning/compiled_resources.py | 已编译程序包的资源事实、存储下界与可选预检 |
| benchmark_graph.py | 通用图类型与 shape 检查 |
| model_decomposition.py / restricted_model_python.py | 高层算子分解、受限 Python 导入 |

文件均在 [scripts/baseline](../../scripts/baseline/) 下。`benchmark_graph` 不读取实验批次，模型准入不依赖模型名称。

后端版本及验收范围见 [Poseidon Runtime](poseidon-runtime-migration.md)。

## 运行约定

默认要求真实密态执行并逐项比较；只有应用组件的显式 `compiled` 等级允许只编译。数值门限为 `1e-5 + 1e-4*abs(reference)`。

模型、权重、布局、规则与配置参与请求身份；候选不能更改它们。生成和修复共用同一 Provider，额度由宿主控制。重复失败、不可自动修复错误、取消和超时均返回终态。

应用 worker 与实验工具分开。宿主接口详见[应用组件](application-agent-component-v1.md)。应用与CLI共用QualificationSession直接验收候选；SDK及工作路径由独立worker绑定。当前编译／执行／验证源码身份已脱离Provider和实验运行器，仍保守绑定共享模型语义。

## DSL 配置

统一图默认 `native` 构造；`--unified-profile public-v1` 选择公开函数、容器和控制流的受限归一化。两者并非完整 Python，也不能任意混用旧构造开关。

上游 helper 使用 `--unified-helpers` 显式选择受信适配器；定向构造和 helper 组合由契约检查。配置清单见 [DSL 选项](dsl-options.md)。可用值以 `candidate -- --help` 为准，绑定实现在 [upstream_candidate_helpers.py](../../scripts/baseline/upstream_candidate_helpers.py)。

对象数组的一个 Expr 单元代表完整密文表达式，不是一个 slot。公开循环只展开公开计算，不能根据密文数据分支。当前 Runtime CPU API 尚未接入真实 bootstrap。

Benchmark、人工探针和批次修复入口见[开发测试说明](semantic-benchmark-v1.md)，不属于应用请求流程。

统一图 CLI 可选择 `--unified-guidance explicit-v10`；应用组件对相关图自动选择。新上下文区分 Earth 消耗深度、RuntimePlan 模数索引和实际 Poseidon 验收，旧指导版本保留原请求哈希。见 [应用上下文验收](poseidon-application-context-r241.md)。

## 版本化容量

prepare_task同时接受v1和显式 `poseidon-model-graph-v2`。v2的resource_profile属于模型身份，使用 `unified-capacity-chunked-inputs-v2`；旧默认与请求绑定保留。capabilities返回版本和状态。

unified_capacity_layout负责逻辑/物理I/O、尾部和selector。unified_capacity_constants从公开模型生成张量与Linear绑定，与DSL生成和reference分开；capacity-public-tensors-v1/v2/v3/v4分别校验，旧请求按原版本重建。缺少capacity descriptor的历史请求按v3重建。v3绑定局部行宽重排与Linear映射，v4增加静态concat/stack/轴归约映射；参数只来自公开形状、轴和权重。新统一绑定由纯Python索引计算生成，不调用DSL转换器或reference。unified_capacity_lowering提供受预算约束的逐行归约与跨块旋转加权求和基线，不是请求准备的前提。基线的分块Linear typed helper已通过三个上下文；其他helper与Agent定向组合待验收。容量与Agent生成分别计分，见[r400](poseidon-capacity-linear-acceptance-r400.md)。

当前基线路径为capacity-chunks-v5。连续旋转窗口及tiled行映射复用已开放的rotation和typed helper；内部节点名避开用户节点及其他虚拟节点。公开张量常量参与的连接仍可提交，当前跨块密文基线不将其冒充已验收范围。旧v1–v3绑定与原失败保留。矩形Linear见[r430](poseidon-capacity-tiled-linear-r430.md)，最新连接/归约见[r441](poseidon-capacity-tensor-acceptance-r441.md)。


## 跨密文空间算子

medium-v1已验收跨块Conv1d/2d（含分组/深度卷积）、AvgPool1d/2d及代表组合。新容量公开张量版本为v5；v5文件编码可无损压缩重复向量，展开后按原逻辑请求及公开绑定验证。旧版本按原规则重建。[范围与结果](poseidon-capacity-spatial-acceptance-r449.md)。当前结果为规则基线，真实Agent验收另计。

## PyTorch v5 来源配置

`pytorch_approximation.py` 对版本化选择器作确定性匹配：FX 节点优先于所属模块，模块优先于算子默认值。同级冲突和无效选择器在捕获阶段拒绝，惰性的程序包校验重新检查选择记录、配置和原图。该过程复用原算子注册和 reference，不调用生成模型；可信来源包通过已有 `submit_import`、程序包和执行接口进入应用流程。[接口](operator-decomposition-and-python.md#逐节点与模块近似) · [验收](agent-pytorch-approximation-acceptance-r721.md)。

## PyTorch v7 矩阵来源

pytorch_matrix_plan 对静态向量提升、批次广播、派生公开权重和输出形状作确定性规范化；matrix_capture重新核验节点、来源与权重。统一计算图、生成模型、验收服务和执行接口继续复用。v7容量导入采用v9公共路由，旧格式及显式选项保持。[输入说明](operator-decomposition-and-python.md#pytorch-矩阵运算) · [验收](agent-pytorch-matrix-acceptance-r815.md)。

v7 同时接入 InstanceNorm1d／2d 与 F.instance_norm，能力标识为 `poseidon-pytorch-instance-normalization-v1`。非 batch 模块输入经显式 reshape 规范化；当前统计量使用声明的 rsqrt，冻结统计量复用公开 BatchNorm 推理分解。完整 NPY 状态、近似选择和输出结构参与来源绑定；数学请求与生成／执行身份分别校验。[输入说明](operator-decomposition-and-python.md#pytorch-instancenorm) · [验收](agent-pytorch-instance-acceptance-r847.md)。

## 版本化 JSON 矩阵接入

高层图增加显式 `poseidon-operator-graph-v2`，接入静态向量、矩阵、批次广播和固定公开矩阵两侧乘法。旧 v1 分解与请求规则保留。新格式的规范化版本、实现哈希、原图与节点来源参与程序包绑定；标量矩阵输出按 `[1]` 张量声明。

用户层仍通过导入目录及 `submit_import` 接入。36例已完成导入、独立参考和静态任务准备，当前新格式的真实编译、密态和 Agent 验收待运行。[接口与示例](operator-decomposition-and-python.md#json-矩阵计算图)。

## PyTorch v10 算子结果接口

可信清单使用 `poseidon-trusted-pytorch-v10`，沿用工厂、NPY权重、dtype、输入/输出与资源配置。结果形状查询扩展到AdaptiveAvgPool、InstanceNorm和padding；函数式AdaptiveAvgPool映射到现有分解。旧v1–v9行为保持。新来源、参考和来源包具有独立版本与哈希绑定。

[范围与用法](operator-decomposition-and-python.md#pytorch-v10-算子结果与形状查询) · [接入检查](agent-pytorch-operator-shape-v10-acceptance-r994.md)。本次没有扩大资源、算子数学、参数或后端执行范围。

## PyTorch v22 公开张量算术

显式 v22 将经原始 NPY 绑定的公开 Tensor 除法与 bool mask 乘法接入既有可信来源、应用和程序包接口。
可信规范化分别记录原始状态、有效状态与派生倒数／0-1 常量，保留完整类型和返回依赖检查；
float32 转换通过既有 `convert-float64-v1` 显式声明。旧 schema、来源版本、CLI 默认 v18 和 latest=v21 保持。
v22 的 5 个代表组合已完成真实 Agent 生成、Poseidon CPU 密态数值验收和程序包复用；4 项首轮成功，1 项修复一次成功。
[接口](operator-decomposition-and-python.md#pytorch-v22-公开张量算术) · [真实 Agent 验收](agent-public-tensor-live-acceptance-r1366.md) · [离线基线](agent-public-tensor-activation-acceptance-r1363.md)。


## PyTorch v23 公开张量表达式

显式 `poseidon-trusted-pytorch-v23` 将来自已绑定公开参数／buffer 的闭合张量表达式规范化为公开常量。登记范围为 reshape/view、permute/transpose/t/T、静态基础切片、unsqueeze/squeeze/flatten、contiguous 及 bool→float32/float64。密文运算顺序和静态输出结构沿用有序来源接口。

原始数组、逻辑 dtype、物理 float64 数组、来源节点及变换参数分别绑定。程序包核验通过声明式记录与原始 NPY 重算派生状态；首次可信源码捕获仍在原隔离 worker 内执行。计算得到的公开除数沿用 v22 的非零、广播和幅值约束；模型输入相关的数据仍不能充当公开常量。

253 项控制与兼容检查通过，18 份来源配置完成独立参考，其中 6 份完成隔离源码捕获。原批次 18 份均通过编译和四组密态数值验收，完整流程 17 通过、1 项返回结构恢复失败。修复后的 32 项相关回归通过；原 named 案例在新执行身份下完成 1 次编译、3 次密态调用、8 组／96 值的验收，程序包重载、新输入和重复执行全部通过。原批次记录与绑定保持，未重跑其余 17 项；v23 真实 Agent 新生成另计。默认 v18、latest=v21、旧格式哈希规则、资源配置及后端保持。
[用法](operator-decomposition-and-python.md#pytorch-v23-公开张量表达式) · [当前检查](agent-public-expression-component-checkpoint-r1368.md) · [离线方案](agent-public-expression-offline-proposal-r1369.md)。

[离线结果](agent-public-expression-offline-results-r1370.md) · [返回结构修复](agent-public-expression-output-repair-r1371.md) · [补验范围](agent-public-expression-output-offline-proposal-r1372.md) · [补验结果](agent-public-expression-output-results-r1373.md)。

## PyTorch v24 公开参数类型转换

显式 `poseidon-trusted-pytorch-v24` 使用 public-value v2 策略。已绑定公开参数/buffer 的 float32、float64、int32、int64、bool，可通过 `float()`、`double()`、`to(dtype=torch.float32/float64)` 转浮点；公开表达式中保留实际舍入，原始状态、逻辑 dtype 和物理数组分别绑定。已有原始值、类型、资源、静态形状与广播约束保持。

v1 和 v23 保留原转换范围与记录格式；生产默认 v18、latest=v21。161 项控制测试和 33 份真实隔离捕获通过，独立参考 528 组/4,224 值。36 项配置（12 个去重请求）完成规则候选编译、Poseidon 密态数值和三份程序包复用，156 组次／1,272 值，最大绝对误差 3.0748e-8。真实 Agent 新生成另计；既有结果保持原身份。

[用法](operator-decomposition-and-python.md#pytorch-v24-公开参数类型转换) · [当前检查](agent-public-dtype-component-checkpoint-r1374.md) · [集中离线验收](agent-public-dtype-offline-results-r1380.md)。

## 应用接入与资源控制

`Application` 的旧任务、来源与程序包接口保留。显式 `admission_policy="structured-v1"` 返回基于公开约束的接入原因；可选 `poseidon-operational-limits-v1` 在同一任务内监控来源导入、生成/修复、资格验证和程序包的保留产物峰值与磁盘余量。预算和接入策略不参与模型语义复用键。

`execute` 可选择独立产物窗口。计量为 `monitored-v1`，不是硬文件系统配额；生成、HTTP、时间和 native 资源控制继续独立执行。199 项组件检查通过；三个原规则候选已完成新身份的编译、Poseidon CPU 数值、程序包重载、新输入、重复及缓存回归。r1376/r1380 数值结果保持历史绑定。

[接口](application-agent-component-v1.md#接入诊断与产物预算) · [组件检查](agent-application-admission-budget-checkpoint-r1381.md) · [CPU 回归](agent-application-control-offline-results-r1383.md)。

## PyTorch 源码包

显式契约 `poseidon-trusted-pytorch-v29` 将源码组织与数学图规则分开。模型来源可使用常规 Python 包、相对导入、嵌套模块及工厂函数重导出；v28 的算子、形状、公开权重和输入绑定规则继续适用。

接入依次经过 CLI 文件清单、应用源码快照、bubblewrap 独立捕获、原图及归一化图参考比较、公开请求准备和来源归档。源文件按原始字节绑定 SHA-256。所有包初始化文件纳入清单，输入组织最多 8 文件、64 KiB 和 8 层路径。

Agent 生成结果沿用单份 DSL 源码加可信依赖的程序包。v29 已完成控制、三个来源的真实捕获及明文验收；新身份的编译、密态及真实生成结果待执行。默认 v18、latest=v28。[接入结果](agent-source-package-intake-acceptance-r1437.md)。

## PyTorch v33 与产物输出绑定

显式来源 `poseidon-trusted-pytorch-v33` 接入固定公开 bool 条件的 where 与 masked_fill，保留有序别名、输入工作副本、公开逻辑 dtype 和静态输出结构。默认来源 v18、latest=v28；旧来源和请求规则保持。

`poseidon-artifact-gate-v2` 检查 RuntimePlan 最终输出是否为不同的已定义密文 Value ID。重复绑定在执行前返回 `compiled_duplicate_final_output`（artifact_gate 层）、公开输出索引及受控上下文。相同输出位置不构成相同表达式的证明，重复失败判断仍保留候选内容校验。

新增可显式选择的生成指导 `explicit-v14` 继承 v13 并提供输出绑定规则与修复指导；v13 请求字节保持。数值等级继续要求真实 Poseidon 执行与独立 reference 比较。

[接入与测试范围](operator-decomposition-and-python.md#pytorch-v33-公开条件选择) · [当前验收](agent-public-mask-output-binding-acceptance-r1476.md)。原 18 上下文的通过证据分别来自两个版本，不声明全部历史能力已在当前 SDK 复验。
