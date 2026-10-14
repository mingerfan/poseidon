# 文档

## 使用 Agent

- [快速开始](../scripts/README.md)：环境配置、构建、生成与程序包。
- [模型输入](baseline/supported-inputs-models-configurations.md)：JSON 格式、算子和尺寸。
- [Python / 高层算子](baseline/operator-decomposition-and-python.md)：静态导入和近似计算。
- [安装与依赖](baseline/agent-setup.md) · [Linux 平台](baseline/linux-platforms.md)。

## 接入和开发

- [应用接口](baseline/application-agent-component-v1.md)：任务、状态、预算和验收等级。
- [模块与契约](baseline/agent-current-contract.md)。
- [DSL 构造与 helper 选项](baseline/dsl-options.md)。
- [编译配置](baseline/compiler-configuration.md)。
- [Poseidon Runtime 与产物](baseline/poseidon-runtime-migration.md)。
- [迁移状态与当前容量](baseline/poseidon-capacity-tensor-acceptance-r441.md)。
- [近似误差与执行误差](baseline/approximation-error-decomposition.md)。
- [Benchmark](baseline/semantic-benchmark-v1.md)。
- [本机开发接续](baseline/mac-migration-handoff.md)。

- [JSON 矩阵输入](baseline/agent-json-matrix-static-acceptance-r913.md)：36例接入、206项测试；新格式后端验收待运行。

## 验收记录

- [PyTorch源码包准备层](baseline/agent-source-package-preparation-r1436.md)：67项静态与快照检查通过，应用捕获和后端接入待完成。
- [JSON 空间计算 CPU 验收](baseline/agent-spatial-offline-results-r1403.md)：45 项编译、数值和程序包重载通过。
- [PyTorch 空间入口 CPU 验收](baseline/agent-spatial-source-offline-results-r1406.md)：18 项编译、数值、程序包重载通过，3 包新输入与缓存复用通过。
- [PyTorch 空间接入快照](baseline/agent-spatial-source-component-checkpoint-r1404.md)：来源、独立参考和 128 项检查。

- [应用控制层 CPU 回归](baseline/agent-application-control-offline-results-r1383.md)：3 个模型、包重载与复用。

这些是指定版本的记录，不是安装或运行指南。JSON 报告、模型与执行产物按原绑定保留。

- [历史 SEAL 产物检查](baseline/seal-artifact-gate-v2.md)：原后端记录与检查约束。
- [应用组件 r216](baseline/application-component-acceptance-r216.md)。
- [第二阶段 r152](baseline/stage2-acceptance-r152.md)。
- [剩余问题 r211](baseline/stage2-remaining-closure-r211.md)。
- [验证适配器 r171](baseline/validation-adapter-acceptance-r171.md)。
- [旧 worker 接口快照](baseline/agent-component-v1.md)：作为验收输入按原文保留；新集成使用上面的应用接口。

旧 Provider、迁移过程、阶段启动说明和已被替代的重复文档已删除。仍被原始记录引用的少量报告保留原文件名，不代表推荐旧流程。

## Poseidon GPU

以下文档属于 GPU 后端，与 Agent 当前的 Poseidon CPU 验收分开：

- [Modswitch 与 level 约定](baseline/poseidon-modswitch-level-contract.md)。
- [GPU modswitch 验证记录](baseline/poseidon-gpu-modswitch-primitive.md)。
- [GPU drop schedule 验证记录](baseline/poseidon-gpu-drop-schedule.md)。
- [Multi-GPU 模块](../src/poseidon/mgpu/README.md)。

## 最新Poseidon CPU验收

- [历史Agent helper与分块](baseline/poseidon-historical-helper-acceptance-r377.md)。
- [1024元素容量与迁移状态](baseline/poseidon-capacity-acceptance-r386.md)。

- [CPU bootstrap ratio128试验](baseline/poseidon-bootstrap-ratio128-result-r388.md)：原生执行与重复刷新数值失败。

- [公开权重与跨块Linear容量](baseline/poseidon-capacity-linear-acceptance-r400.md)：1024元素输出、旧请求兼容及矩形batch阻塞。

- [分块Linear最终组件复核](baseline/poseidon-capacity-final-review-r415.md)：114项测试、CLI重载与绑定兼容。

- [矩形批量Linear与程序包验收](baseline/poseidon-capacity-tiled-linear-r430.md)：解除原两个容量阻塞，123项测试、三个上下文密态与重载通过。

- [跨块连接与指定轴归约](baseline/poseidon-capacity-tensor-acceptance-r441.md)：concat、stack、sum、mean 与命名多输出，132项测试通过。

- [跨密文卷积与平均池化验收](baseline/poseidon-capacity-spatial-acceptance-r449.md)

- [真实Agent三个整模型](baseline/poseidon-whole-models-agent-acceptance-r453.md)
- [PyTorch张量输入与中型程序包](baseline/poseidon-stage3-pytorch-acceptance-r473.md)

- [PyTorch矩阵输入、分块广播与当前程序包](baseline/poseidon-pytorch-broadcast-acceptance-r508.md)：263项静态、24项密态与重载；新增生成0。

- [4096元素物理布局验收](baseline/poseidon-extended-capacity-acceptance-r518.md)

- [4096元素PyTorch与ResNet20源模型](baseline/poseidon-pytorch-extended-acceptance-r535.md)：五例密态／重载、原模型明文对照及stem路由阻塞。

- [通道布局与 BatchNorm](baseline/poseidon-channel-layout-acceptance-r547.md)：v8布局、静态/密态验收和原尺寸ResNet首层成本。

## Agent 通用化

- [通用接入与资格审查](baseline/agent-generic-qualification-audit-r1378.md)：当前能力范围、历史证据索引和控制层改造草案。

- [阶段报告](baseline/agent-generalization-acceptance-r622.md)：能力与反馈控制、参数化 PyTorch、权重描述符、程序包新输入执行。
- [真实 Agent 验收](baseline/agent-generalization-live-acceptance-r626.md)：14/14，12 项首次通过；程序包重载及三项新输入执行通过。
- [原冻结评测范围](baseline/agent-generalization-live-proposal-r620.md)：已按该范围执行完毕。

- [PyTorch 外部权重接入](baseline/agent-pytorch-state-weights-acceptance-r636.md)：v3 NPY 状态入口，六类 Poseidon CPU 模型及程序包验收。
- [原PyTorch外部权重提案](baseline/agent-pytorch-state-live-proposal-r635.md)：六项未执行历史绑定，后续范围改用r675。

- [确定性布局准备](baseline/agent-layout-acceptance-r646.md)：15 项组件测试、8 项真实后端与 3 项新输入重复执行。

- [固定工作上下文验收](baseline/agent-context-acceptance-r655.md)：请求绑定、程序包重载、新输入和缓存复用。

- [资源预检与万参数 Linear](baseline/agent-resource-acceptance-r665.md)：可选接口、预算筛选、数值验收与程序包复用。

- [PyTorch 输入写法审计与离线工具调整](baseline/agent-input-form-audit-r685.md)
- [PyTorch函数式输入验收](baseline/agent-pytorch-functional-acceptance-r678.md)：254项测试、9项规则候选密态与程序包复用、20项请求兼容。
- [新入口真实Agent提案](baseline/agent-pytorch-functional-live-proposal-r675.md)：6项外部状态＋9项函数式输入，15项未执行历史绑定，当前源码已变化。

- [PyTorch 调用形式与离线续验](baseline/agent-pytorch-spelling-acceptance-r695.md)：24 模型数值通过、三份程序包复用、301 项静态测试；新增付费 0。

- [新输入区间与 split](baseline/agent-execution-domain-acceptance-r706.md)：172 项测试、五个规则模型数值通过；Attention 一项编译失败保留。

- [PyTorch 静态形状与标量输出验收](baseline/agent-pytorch-static-acceptance-r710.md)：六个来源包及程序包复用。

- [PyTorch 逐节点与模块近似](baseline/agent-pytorch-approximation-acceptance-r721.md)：v5 配置、规则／人工表达的密态验收及原编译失败。

- [身份分析缓存与组件回归](baseline/agent-identity-cache-acceptance-r730.md)。

- [归一化规则表达与原失败修复](baseline/agent-normalization-lowering-acceptance-r746.md)：105 项测试、六例密态及程序包复用。
- [PyTorch 静态索引与重复输出验收](baseline/agent-pytorch-index-acceptance-r761.md)：184 项测试、15 个索引模型及两个回归，原失败记录与两批绑定保留。

- [组件身份目录与真实依赖审计](baseline/agent-identity-catalog-acceptance-r766.md)：92 项测试、三个 CPU 回归和当前阶段耦合。

- [生成来源核验与公共读取](baseline/agent-receipt-acceptance-r789.md)：公共接口、历史来源、程序包与Poseidon CPU验收。
- [独立资格验收会话](baseline/agent-qualification-acceptance-r776.md)：应用／CLI 共用验证、232 项通过、3 条历史检查跳过和真实 CPU 回归。

- [Poseidon沙箱清单与helper回归](baseline/agent-sandbox-profile-acceptance-r797.md)：313项测试及五份历史Agent程序的真实CPU验收。

- [可信PyTorch捕获依赖验收](baseline/agent-capture-boundary-acceptance-r805.md)：364项测试，生成服务／实验工具边界及三个真实CPU模型。

- [PyTorch 矩阵入口](baseline/agent-pytorch-matrix-acceptance-r815.md)：v7向量、多维批次、公开权重及30例CPU验收。

- [编译程序包资源分析](baseline/agent-compiled-resources-acceptance-r828.md)：70项测试、30份产物审计及三份程序包真实CPU重载。

- [PyTorch InstanceNorm 接入](baseline/agent-pytorch-instance-acceptance-r847.md)：146 项测试、30 个来源上下文真实 CPU 数值、三份程序包复用。
- [历史 PyTorch 真实 Agent 提案](baseline/agent-pytorch-general-input-live-proposal-r848.md)：60 个上下文，未执行；源码已变化，当前付费评测需刷新绑定。

- [PyTorch 算子组合验收](baseline/agent-pytorch-composition-acceptance-r864.md)：24 个数学上下文、21 个来源捕获、16 个规则／人工等价图密态通过；保留 split 映射和八项编译容量缺口。

- [PyTorch 静态 split 接入快照](baseline/agent-pytorch-split-static-acceptance-r872.md)：196 项测试、18 个来源上下文和36份旧捕获兼容；后续密态结果见 r878。

- [PyTorch 静态 split 数值验收](baseline/agent-pytorch-split-acceptance-r878.md)：24/24，含原三项失败来源、rank32、公开权重；三份程序包重载和复用。

- [容量规则排程静态验收](baseline/agent-capacity-scheduling-static-acceptance-r885.md)：67测试与768组参考检查，原八项编译失败重测待执行。
- [容量规则表达离线验收提案](baseline/agent-capacity-neutral-offline-proposal-r892.md)：八个原失败图、四个兼容回归与三包复用，待批准。

- [矩阵模型跨入口审计](baseline/agent-matrix-input-parity-audit-r900.md)：24 个隔离捕获通过，15 个相同 JSON 图存在接入差异；运行源码和 r892 待批准绑定保持。

- [高层算子版本参考回归](baseline/agent-operator-version-reference-acceptance-r923.md)：34个登记算子、110上下文；明文参考与分解结果。

[公开常量运算接入审计](baseline/agent-public-operand-intake-acceptance-r934.md)：JSON/PyTorch 运算写法的接入差异和规范化方向。

- [PyTorch 幂运算与公开除法](baseline/agent-pytorch-exact-arithmetic-acceptance-r1348.md)：v21 固定幂、公开标量与共享视图，Poseidon CPU 数值及程序包复用。

- [PyTorch 来源配置与真实 Agent 验收准备](baseline/agent-source-expansions-readiness-r1349.md)：latest 清单、36 项隔离捕获及有界测试方案。


- [公开张量除法开发版](baseline/agent-public-tensor-division-draft-r1350.md)：52 项控制检查、三个等价图 Poseidon CPU 数值和程序包复用；活动入口保持 v21。

- [公开参数类型转换](baseline/agent-public-dtype-component-checkpoint-r1374.md)：v24、161 项测试、33 份隔离捕获；[36 项 Poseidon CPU 数值及包复用](baseline/agent-public-dtype-offline-results-r1380.md)通过。

- [应用接入诊断与产物预算](baseline/agent-application-admission-budget-checkpoint-r1381.md)：199 个组件和固定环境适配器检查；[接口示例](baseline/application-agent-component-v1.md#接入诊断与产物预算)。

- [PyTorch 静态广播与最新入口](baseline/agent-static-broadcast-phase-summary-r1433.md)：v28，三个数学请求、六个来源写法完成真实生成或原样重放、CPU 数值及程序包复用。

- [应用来源能力清单](baseline/agent-source-capabilities-acceptance-r1435.md)：实际版本门禁、静态广播、168项组件控制与旧请求兼容；本轮新增后端执行为0。

- [PyTorch 源码包接入](baseline/agent-source-package-intake-acceptance-r1437.md)：显式 v29，301 项控制及三个来源的隔离捕获、明文参考和归档验证通过。

- [空间图近似区间检查](baseline/agent-spatial-domain-acceptance-r1439.md)：v4 参考分派、169 项控制、六种组合的边界与失败定位。
