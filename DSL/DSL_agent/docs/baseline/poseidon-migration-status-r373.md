# Poseidon 执行链迁移验收

2026-10-01，ARM Ubuntu `/home/lhohy/Code/Poseidon`，`lhy-agent-dsl`。HEAD `7588e48093631eaf8385506e3db058e7ca3cfe7e`。

**核心执行链已迁移到 Poseidon CPU。** 当前 CLI 和应用组件使用 Hecate → dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU，完成真实编译、密态执行、解密、独立数值比较及程序包重载。历史 Agent 源码可独立验收，重新生成不是执行后端迁移的条件。

兼容性结论限于已验收范围。完整历史能力、bootstrap、大张量和整网目标分别保留状态。

## 核心链路

| 检查项 | 当前结果 |
|---|---|
| 活动执行入口 | CLI、组件和验证适配器使用 Poseidon CPU；旧 SEAL 执行配置在进入执行前拒绝 |
| tracing 与编译 | 原候选重新进入真实 Hecate frontend、Earth/CKKS lowering 和 RuntimePlan 导出 |
| 产物检查 | RuntimePlan、OperatorSpec、常量 bundle、输入输出布局、实际模数、level/scale 和密钥需求参与检查与绑定 |
| 密态与数值 | Poseidon CPU 加密计算、解密，逐项与独立 reference 比较 |
| 程序包 | v4 绑定实际后端、编译器、库、配置和产物；直接重载重复执行 |
| 新旧身份 | 原响应、DSL、旧报告和旧请求保留；当前后端另建请求及程序包身份 |

CLI 的执行配置集合与 `component_backend.qualify()` 的入口检查共同限制后端；`validation_adapter` 的数值路径只调用 Poseidon 验证器。r372 中的 CLI 和组件反例验证了旧执行配置被拒绝。历史配置仍用于请求重建与记录核对。

依赖来源保持：Poseidon API `62dbe22559667199487edb8ecdcb06624f66e6d7`，Runtime `23c8b94202a14d3e322f87c50434253d56acb9f7`，dacapo-modified `edcebf5de955dd2a5cd1b87726b92355276e4390`。批准的语义补丁和构建补丁未变。LLVM/MLIR 18.1.2、GCC 13.2.0、GMP 6.3.0、Python 3.10.14、NumPy 1.25.2、ARM Torch 2.0.1。实际二进制、构建配置与源码 SHA-256 见机器可读报告。

## 来源分别计分

| 来源 | Poseidon 结果 | 记录 |
|---|---|---|
| 历史真实 Agent 原源码 | 14/14；14 份包重载通过 | 本次 r371/r372 |
| 当前规则基线 | 15/15 默认模型；三个包重载通过 | [r369](poseidon-attention-default-acceptance-r369.md) |
| 人工等价表达 | RMSNorm、Attention、归一化表达已有通过，按原版本保留 | [r240](poseidon-expression-acceptance-r240.md)、[r242](poseidon-attention-acceptance-r242.md)、[r322](poseidon-normalization-matmul-acceptance-r322.md) |
| 当前新生成真实 Agent | 0 项；尚未启动 | 已批准的 r368 三模型批次留作生成与自动修复能力验收 |

这些批次有重叠，不能相加为唯一模型数。人工等价表达不能记入历史原候选的通过数。

本次审核历史 DeepSeek 响应、传输账本、原候选、原请求及旧产物哈希；属于保留记录的来源核验，不是服务商签名或账单证明。进入当前后端的 `hecate_source` 与原响应逐字节相同。新响应封装只重绑定 `request_id`。模型、公开权重、常量来源、布局、测试输入、reference 的 dtype/shape/值均与旧证据一致。

## 历史源码兼容回归

从缺少相同源码 Poseidon 证据的历史程序中，按能力特征选择以下 14 例。选择时没有依据 Poseidon 成功结果筛选。每例均重新 tracing、编译、密态执行，并重新导出和执行程序包。

| 内容 | 原模型 | 首次 Poseidon 最大绝对误差 | 重载 |
|---|---|---:|---|
| Linear | bench_primitive_0006 | 3.51757e-09 | 通过 |
| 真实 Linear → square → Linear | bench_composition_0013 | 4.82018e-10 | 通过 |
| 3 输入、2 输出及融合 | bench_graph_0204 | 5.27638e-08 | 通过 |
| P8、6 元素尾块 | semantic_context_e591cb631c9addfea3bd3fc6 | 1.1405e-06 | 通过 |
| P16、15 元素尾块 | bench_boundary_0024 | 7.8836e-09 | 通过 |
| P64、rank 4 输入 | bench_boundary_0102 | 3.57385e-11 | 通过 |
| P256、rank 4 输入 | bench_boundary_0108 | 9.78241e-12 | 通过 |
| 公开循环和增强赋值 | bench_composition_0105 | 5.77689e-09 | 通过 |
| 函数返回对象矩阵 | bench_composition_0171 | 2.74486e-09 | 通过 |
| 空返回结构 | bench_composition_0167 | 1.26878e-08 | 通过 |
| 矩阵展开参数 | bench_composition_0215 | 2.57884e-09 | 通过 |
| 标量增强赋值 | bench_composition_0183 | 4.80431e-09 | 通过 |
| 公开别名、复制和修改 | bench_composition_0233 | 1.86052e-09 | 通过 |
| 声明的 SiLU 多项式 | bench_helper_0115 | 1.98515e-09 | 通过 |

首次执行：56 组输入、200 个输出值，最大绝对误差 `1.140500886e-06`。14 包重载：同样的 56 组输入、200 个输出值，最大绝对误差 `6.391500691e-07`，重载生成与编译均为 0。两次执行计数分别保留，模型数仍为 14。

逐项门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。记录含 MAE、最大绝对误差、非零 reference 相对误差、适用时的 cosine similarity 和逐项结果。SiLU 案例验证声明的多项式，不代表原始函数精度验收；空返回作为结构项计分，不声明其具有数值贡献。

来源及安全反例：**10 passed / 0 failed / 0 skipped**，14 份交付包来源审计全部通过。覆盖源码篡改、缺失原响应、来源误标、符号链接、不安全证据路径、旧执行后端拒绝和新旧语义绑定。该组检查没有重新编译或密态执行。

代表性兼容性通过不等于全覆盖。历史真实 Agent 的上游 helper-profile、分块布局、P32/P128 以及全部 schema/语义组合尚未在本批逐项重放。既有规则/helper/构造证据继续按原绑定计分，包括 [r233](poseidon-runtime-migration-r233.json)、[r245](poseidon-operator-migration-r245.md)、[r261](poseidon-dsl-migration-r261.md) 与 [r267](poseidon-dsl-migration-r267.md)。

## 程序包与来源保存

每例目录包括：

- `historical/`：原报告、请求、模型、原响应与原 DSL 的保留副本。
- `program/`：当前 v4 数值等级程序包。
- `delivery.json`：连接原生成来源与当前 Poseidon 验收的独立绑定。

旧 results 未改写。v4 包中的 `agent_generated=false` 表示本次验收没有新生成；历史 Agent 来源由 `delivery.json` 的 `origin_category=historical_agent_source`、原响应和传输账本表达。来源审计工具同时核验这个封装和标准包，避免用当前调用字段伪造历史生成。

## 后端缺口与原计划

| 项目 | 状态及下一行动 |
|---|---|
| 原 12 个深层多项式模型 | 原失败和容量阻塞保留；[r312](poseidon-polynomial-depth-audit-r312.md) 有深度与安全容量依据 |
| 真实 bootstrap | 独立 CPU 刷新已执行；有符号误差约 1.04e-4 未满足冻结门限，当前 Runtime/Agent 尚未开放。r235 新参数方案未批准、未执行 |
| 中型/ResNet 物理布局 | 逻辑资源配置已有；跨密文卷积、归约、残差和大张量容量继续实现、验收 |
| 通用算子与 PyTorch | 注册、分解与受限导入已有部分实现；按新增分区补独立参考、后端及真实 Agent 验收 |
| MLP/RMSNorm/Attention | 默认规则小模型已通过；真实 Agent 功能验收、原函数近似精度和更大尺寸分别推进 |
| ResNet20 | 权重/数据冻结、完整网络近似精度、整网密态和程序包尚未完成 |
| x86 原生、空白安装、CUDA | 保留配置，暂缓实机验收 |

下一批先核验尚缺的历史源码代表项，只重放缺口；随后继续算子、物理布局与 bootstrap 原计划。生成契约、输入扩展和自动修复能力用真实 Agent 单独验收。r368 三模型已获批准、尚未启动，原额度没有消耗；此次迁移重放付费调用为 0。

## 执行与复现

本批在源码冻结及无活动批次的边界开始，串行执行；没有修改应用核心、Dacapo、SEAL、Runtime 核心、库、模型、reference、安全参数或门限。增加三个开发工具及文档。执行源码 575 文件、应用核心 518 文件与 r369/r368 一致；精确工具源码和修改前文档另存恢复快照。

原始证据：

- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-agent-replay-r371`：计划、14 例来源与新验收记录。
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-integrity-r372`：10 项静态测试及 14 份包来源审计。
- `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-migration-status-r373`：工具快照、修改前文档和 Git 状态。

工具位于 `scripts/baseline/benchmarks/tools/`，不进入应用请求流程。现有包重复执行：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay   --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-agent-replay-r371/linear_square_linear/program --execute
~~~

新一轮历史源码验收需要明确选择清单和全新输出目录：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/baseline/benchmarks/tools/requalify_historical_agent_sources.py   --selection /absolute/path/to/selection.json --output /absolute/path/to/new-result
~~~

[机器可读报告](poseidon-migration-status-r373.json) 记录每例原/新请求、源码、产物、包与实际后端身份。本批安装、下载、付费调用、提交和推送均为 0。核心迁移完成不代替原计划完成。
