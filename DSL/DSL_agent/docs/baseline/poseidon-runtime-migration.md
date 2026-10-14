# Poseidon Runtime 后端

Agent 的核心执行链已迁移到 Hecate → dacapo-modified → RuntimePlan / OperatorSpec / plaintext bundle → CKKS Runtime → Poseidon CPU。当前 ARM CPU 支持范围已通过真实编译、密态、数值与程序包验收。历史真实 Agent 原源码可独立重放；重新生成不作为后端迁移条件。

最新状态见 [核心迁移与基础容量](poseidon-capacity-acceptance-r386.md)及[权重与Linear容量](poseidon-capacity-linear-acceptance-r400.md)。历史 SEAL 请求和报告保留原绑定。

## 版本

| 部分 | 来源 |
|---|---|
| Poseidon CPU API | integration/runtime-gpu，62dbe22559667199487edb8ecdcb06624f66e6d7 |
| CKKS Runtime | 23c8b94202a14d3e322f87c50434253d56acb9f7 |
| dacapo-modified | edcebf5de955dd2a5cd1b87726b92355276e4390 |
| Poseidon CPU 库 | 当前 checkout 的既有 CPU 构建；程序包另绑定实际库 SHA-256 |
| LLVM / MLIR、GCC | 18.1.2、13.2.0 |
| Python、NumPy、Torch | 3.10.14、1.25.2；ARM 2.0.1，x86 2.0.1+cpu |

采用选择性导入，未合并或切换分支。固定 Runtime 和 Dacapo 源码保存在 `third_party/ckks-runtime-agent`；原有独立 `third_party/ckks-runtime` 和旧 Dacapo checkout 保留。

7 个语义/常量补丁按批准顺序移植，另保留 3 个构建补丁。文件和 SHA-256 见 `scripts/baseline/poseidon-runtime-sources.lock.json`。Runtime 核心与导入的 Poseidon API 没有改运算语义。

## 构建配置

平台选择仍使用 `POSEIDON_PLATFORM=x86_64-linux` 或 `aarch64-linux`，工作根目录使用 `POSEIDON_WORK_ROOT`。

本批复用已安装的隔离 SDK 与 Poseidon CPU 库。平台内的 `deps/poseidon-runtime-v1.json` 记录编译器、执行器和 Poseidon 库构建目录，运行器校验目录所属平台，程序包绑定实际二进制。

已有 SDK 与 CPU 库时，先查看构建计划：

```sh
python3 -B scripts/baseline/build_poseidon_agent.py \
  --poseidon-build <当前平台内的Poseidon构建目录>
```

审阅后追加 `--apply` 构建 RuntimePlan 编译器、frontend 和 CPU 执行器。已有不同工具链清单时会停止，保留原目录。新入口尚未进行空白机器安装验收；旧 `setup_agent.py --apply` 已停止执行 SEAL 安装流程。现有安装器的计划输出仅用于查询历史依赖锁。

本机配置：

- 源码：`/home/lhohy/Code/Poseidon`。
- work root：`/home/lhohy/poseidon-work`。
- 编译器：`platforms/aarch64-linux/build-dacapo/runtime-plan-r232`。
- 执行器：`platforms/aarch64-linux/build-poseidon/agent-runtime-r232`。
- CPU 库：`platforms/aarch64-linux/build-poseidon/bootstrap-cpu-r217/poseidon_build`。

## 运行

```sh
python3 -B scripts/agent.py --backend local doctor
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/linear-example.json --self-test
python3 -B scripts/agent.py --backend local candidate -- \
  --case scripts/baseline/cases/unified-two-input-two-output.json --self-test
```

默认编译配置为 `poseidon-cpu-eva-w45-v1`，另提供显式 `poseidon-cpu-eva-w40-v1`。w40 用于本批 SiLU 和 Poly_Default 等固定 helper 的验收；程序不会自动切换配置。CUDA 配置用于独立后端身份登记，当前执行入口明确返回未验收。历史 SEAL 配置可重建请求及核对哈希，但不能执行。

## 参数与布局

沿用原批准参数：degree 32768、13 个数据模数、1 个特殊模数、tc128；w45 输入 scale 为 2^45，w40 为 2^40。实际素数由执行器输出，并与检查器逐项核对。

Poseidon level 从 0 开始，level 12 对应 13 个数据模数。编译配置上界由旧 HEVM 计数 13 映射为 12，模数与安全级别保持原值。产物检查器验证指令类型、引用顺序、模数容量、scale 传播、旋转需求及输出声明。

Hecate 的周期常量在可信布局适配层展开为完整 slot 向量，通过标准 plaintext bundle 传给 Runtime。原始编译产物保留；展开规则、原始哈希、bundle 哈希单独绑定。此过程直接处理 RuntimePlan，没有 HEVM 转换层。

## 程序包

新格式为 `poseidon-dsl-bundle-v4`，包含 DSL、模型和请求、Earth/CKKS IR、原始 RuntimePlan、绑定后的 RuntimePlan、OperatorSpec、常量 bundle、布局记录及依赖身份。

```sh
python3 -B scripts/dsl_bundle.py export --evidence <验收目录> --output <新程序包目录>
python3 -B scripts/dsl_bundle.py replay --bundle <程序包目录>
python3 -B scripts/dsl_bundle.py replay --bundle <程序包目录> --execute
```

重放直接加载已导出的 RuntimePlan，重新加密约定输入、执行并比较；生成次数和编译次数均为 0。私钥只存在于本次验收进程；Runtime API 接收评估密钥。

宿主应用使用 `application_backend.PoseidonBackend`。任务状态、取消、调用账本、预算及验证等级接口保持原协议；后端身份、请求哈希和程序包版本已更新。

## 验收范围

下表为 r233 时的 ARM Ubuntu 结果；后续原失败模型的更新见本文末尾。候选来自规则转换或人工 fixture，API 调用为 0。

| 范围 | 结果 |
|---|---|
| helper 构造回归 | 84/84；10 个配置组，含真实上游调用和输出贡献检查 |
| 输入/算子回归 | 28/28；schema 1–5、统一图、分块、packing、旋转及高层分解算子 |
| 组件和兼容测试 | 199 通过，0 失败，3 项历史 SEAL 证据测试跳过 |
| 应用程序包 | compiled / numerical 两个等级均通过导出、重载和缓存复用 |
| 小型平方激活 MLP | 真实编译、密态执行、四组输入比较通过 |
| RMSNorm、Attention decode | w40、w45 均在 Earth 累计 level/scale 检查失败 |
| 固定参数 bootstrap 探针 | 构建成功，CoeffToSlot 完成，EvalMod 中报 scale out of bounds |

helper 包括 BN、Concat、卷积、池化、下采样、Linear、SiLU、分块适配及固定多项式。84 是任务数，不是不同模型或上游 API 数量。Poly_Default 的 3 条 w45 失败记录保留；同一案例在显式 w40 下通过。最大绝对误差为 1.8592e-6，仍使用冻结的逐项门限。

应用重载直接使用编译产物，生成次数和编译次数均为 0。数值等级重新加密、执行、解密和比较；编译等级只复核产物。两种等级的缓存分开。

详细路径和绑定见 [r233 报告](poseidon-runtime-migration-r233.json)；较早的 Linear、mlp4x4x2 和 native 构造结果见 [r232 报告](poseidon-runtime-migration-r232.json)。两个版本分别保留，不改写历史绑定。

## 当前状态与剩余工作

| 类别 | 状态 |
|---|---|
| 核心 Poseidon CPU 链路 | 已迁移；CLI、应用、产物检查、真实密态与数值验收、v4 重载已接通 |
| 历史真实 Agent 原源码兼容 | 代表性 14/14 通过，源码与原权重/reference 不变，14 包重载通过 |
| 默认规则整模型 | MLP、RMSNorm、Attention decode 及归一化、批量 matmul 共 15/15 通过 |
| 新真实 Agent 生成 | r368 三例已批准，未启动；用于生成和自动修复能力验收 |
| 全历史兼容 | 尚未逐项完成；按缺口补验，不重复全量 |
| 深度容量与 bootstrap | 原 12 例失败保留；bootstrap 有符号精度未达标，当前入口未开放 |
| 大张量与整网目标 | 跨密文布局、完整近似精度与 ResNet20 继续推进 |

本表为当前状态，下文 r232–r369 是各自源码绑定下的历史过程。完整数值与范围见 [r373](poseidon-migration-status-r373.md)。真实 bootstrap 的独立参数试验及其失败见 [r234](poseidon-native-bootstrap-result-r234.md)；数学诊断与未批准的新参数方案见 [r235](poseidon-bootstrap-math-audit-r235.md)。

x86 原生执行、空白机器安装和 CUDA 暂缓。核心迁移验收与通用算子、bootstrap、真实 Agent 功能和整网目标分别报告。

## 启动与依赖清理 r237

Poseidon CPU 使用独立 Nix shell，依赖现有 LLVM/MLIR、GCC、Python 和 GMP；活动 shell 不再要求 SEAL 包，Dacapo 路径从新源码与工具链清单解析。数值比较和沙箱策略已从旧执行模块抽离，门限和资源限制保持原值。

84 项相关测试通过；独立进程中的应用验收、真实密态比较、程序包重载和缓存复用通过。另修正测试导入次序，6 项回归通过。见 [r237 记录](poseidon-migration-cleanup-r237.json)。这不扩大已验证模型或 bootstrap 范围。

## 程序包身份 v2

后端身份现在绑定 Nix 环境文件、平台与依赖锁、三项实际 CMake 构建配置和二进制。配置变化需要重新验收；旧程序包保留历史身份，不自动改写。58 项相关测试及新绑定下的 Linear、平方激活 MLP、MLP 包重载、应用验收和缓存复用通过。详见 [r238](poseidon-migration-binding-r238.json)。

## 原失败模型的等价表达验收

RMSNorm 已在相同模型、近似配置和参数下获得通过程序包。42 项相关测试通过；Attention 的原数学语义保留，四项等价表达配置均编译失败。详见 [r240](poseidon-expression-acceptance-r240.md)。旧报告保留原源码绑定，不改写为新通过结果。

## Attention 等价表达验收 r242

相同投影与 RoPE 的当前 q/k 点积可精确改写为带公共系数的平方和。原 Attention decode 在 w45、w40 下均通过真实编译和四组输入、40 值的 CPU 密态比较，最大误差分别为 5.20717e-9、1.91664e-7。应用接口自动选择 explicit-v10，数值任务、v4 包重载和缓存复用通过；没有使用 bootstrap。候选仍为人工程序。详见 [r242](poseidon-attention-acceptance-r242.md)。

## 第三阶段算子补测 r245

原45例扩展算子中，缺少Poseidon证据的30例已通过CPU真实数值验收，最大误差5.00092e-7；原15例保持r233绑定。广播、公开索引、平均池化、激活、LogSoftmax、三种padding以及比较/最大池化共15个分区，各有3模型证据。近似误差单独记录，最高0.5；不是整网精度结论。候选为规则程序，真实Agent计数0，详见 [r245](poseidon-operator-migration-r245.md)。

## RuntimePlan 入口与构造验收 r257

CPU编译配置现在为schema3：真实frontend保留原始IR与构造记录；无符号引用的非入口函数标为private，再由现有symbol-dce清理。EVA、验证器、RuntimePlan导出及Runtime算法保持原样。旧schema1/2请求可精确重建，新请求和程序包另有绑定。

native129项及public48项已取得CPU数值证据，原9个未使用helper定义引发的编译失败已解决。应用数值/编译等级、入口记录重载与缓存均验收通过。各来源绑定与结构/数值计分见[r257](poseidon-dsl-migration-r257.md)。

## 定向构造与协议迁移 r261

冻结627项构造任务已有CPU数值证据，209分区各三个上下文。新增450项全部通过，最大绝对误差8.31713e-7；规则候选与真实Agent分开计分。486项静态正反例和五项RuntimePlan产物协议完成映射。60项原helper任务保留历史身份和源码绑定，产物及共同二进制哈希核验一致。

Poly_Default三个补充原例仍编译失败，未进入Runtime。后续源码核验修正了原因：Earth Mul 并不要求两输入 scale 相等，实际受消耗层数/scale 容量检查约束；详见 [r267](poseidon-dsl-migration-r267.md)。r261 的原诊断与源码绑定保留。


## 数学上下文与无常量程序包 r267

冻结数学上下文已有 268/280 个模型、442/459 个任务的 CPU 数值证据；146/153 个数学分区完成三个原定上下文。剩余 12 模型中，两个高阶多项式最大池化模型编译失败，十个仍受规则基线 AST/工作量限制，未据此认定 Agent 不支持。

修正布局适配器的 bundle 声明：仅在有 bundle Encode 时声明 plaintext_bundle；无常量程序的空辅助 manifest 仍可随包保存。5 个原运行失败全部解决。110 项测试、6 项应用/程序包回归通过；数值与编译等级分别记录，包重载无生成或编译，缓存无 Provider 调用。

候选均为规则或人工等价表达。新付费三整模型提案 [r268](poseidon-whole-models-live-proposal-r268.json) 已冻结，尚未执行；独立 bootstrap 试验 r235 仍待批准。完整分层结果见 [r267](poseidon-dsl-migration-r267.md)，旧模型、账本、失败和绑定保持原样。

## 批量入口迁移 r273

通用 Benchmark baseline/free/directed 和 check_operator_regression 已使用 Poseidon CPU Nix 环境与配置，冻结数据和当前执行源码分别绑定。28 静态测试通过、3 历史测试 skipped；9 个规则模型真实密态及独立数值审计通过，两份包重载通过。历史 helper-directed/HEVM 产物审计入口仍待迁移。见 [批量使用与结果](poseidon-benchmark-migration-r273.md)。r274 取代未执行的 r268 付费提案，仍需批准。

## helper 与产物审计 r277

公开 Benchmark 的 helper-directed/compiler-evidence 已接入 Poseidon CPU。33 项测试通过；SiLU、BatchNorm、Concat 三例真实调用、返回贡献及密态数值通过，12 份 RuntimePlan 产物审计通过。审计不新增密态执行，也没有逐指令 native trace。旧 HEVM 审计模块仅保留历史用途；静态 rejection CLI 仍待迁移。见 [入口与结果](poseidon-helper-compiler-migration-r277.md)。r274 应用核心绑定保持一致。


## Benchmark 静态拒绝入口

`scripts/benchmark.py rejections` 已接入 Poseidon 请求和 RuntimePlan 静态检查，原48任务身份保留。终态、控制产物和中断保存见 [r280](poseidon-static-rejection-migration-r280.md)。

## Python 导入和来源程序包 r288

model-import 的两个 Nix 重入路径已显式选择 Poseidon CPU。可信 PyTorch 捕获追加原模块独立参考、精确运行时校验及读取前文件预算；来源清单随 DSL 包保存，重载直接运行 RuntimePlan。

124 passed / 0 failed / 3 历史 skipped。两个原 MLP 导入示例首次密态与包重载均通过，数值门限保持不变；self-test 候选、真实 Agent 0、付费 0。[验收与用法](poseidon-python-import-migration-r288.md)。

最新真实 Agent 提案为 [r286](poseidon-whole-models-live-proposal-r286.json)，514 应用核心文件绑定，取代未执行的 r274/r284；案例、公开请求、后端与预算保持一致。旧证据不重新标记。剩余 12 编译容量失败及 bootstrap 精度尚未解决。

## 剩余多项式表达检查 r295

原12模型的等价表达检查保留了编译容量和工作量阻塞，未增加通过数。诊断及具体原模型见 [r295](poseidon-lane-schedule-diagnostics-r295.md)。当前应用核心仍匹配 r286，三个整模型真实 Agent 与 r235 bootstrap 试验分别待批准。

## 剩余多项式深度与容量 r312

原 12 个失败模型保持未通过。精确符号检查证明一个输出含非零 6076 次项，至少需要 13 层密文乘法；5 项检查通过。代表模型的 4 种不同人工候选共 5 次编译尝试均失败，1 次为重复候选，密态执行与 Agent 成功均为 0。当前参数 Q×P 840 位，N=32768/tc128 表上限 881，不能直接再加 60 位 Q。详见 [r312](poseidon-polynomial-depth-audit-r312.md)。

当前应用核心 514 文件仍匹配 r286；该三整模型真实 Agent 提案及 r235 bootstrap 比例试验分别待批准。本批没有修改应用、编译器、Runtime、参数或门限，没有安装、下载、付费、提交或推送。下一步为获准后的上述有界验收，及现有依赖下的算子／物理布局扩展。

## 归一化与批量矩阵乘 r322

LayerNorm、GroupNorm、InstanceNorm、批量 matmul 各三个上下文通过 Poseidon CPU，共 48 组密态输入、336 个输出，最大误差 7.56172e-7；两个包直接重载通过。归一化使用人工提前增益/打包表达，默认规则的九项编译失败保留，真实 Agent 新增为 0。两轮 Newton 相对原函数最大误差 0.632650，尚不是网络精度验收。

新增补充模型位于 scripts/baseline/cases/operator-extension-v5，冻结 1200 集未变。应用核心仍匹配 r286；原 12 项多项式编译问题、真实 bootstrap、扩大物理布局及付费 Agent 分开处理。本批安装、下载、API、commit/push 为 0。见 [r322](poseidon-normalization-matmul-acceptance-r322.md)。

## 2026-10-01：默认规则生成验收 r355

新增通用整周期归约、广播标量、平衡幂次及有界 DAG 调度。九个归一化案例和三个批量 matmul 的默认 self-test 全部通过，原 RMSNorm 默认路径也已打通。规则候选和 Provider 候选分别处理；公开模型/request/reference/参数/门限未变。

最终 112 passed / 0 failed / 3 历史 skipped；15 个默认案例 14 通过、Attention 1 编译失败。56 组密态输入、360 个值，最大误差 9.95399e-7；两个包重载另 40 值、最大误差 6.81978e-8，生成及编译调用 0。详见 [r355](poseidon-default-scheduling-acceptance-r355.md)。

应用核心 516 文件、完整执行源码 573 文件分别绑定。r354 取代未执行 r286，保留旧提案；三模型、公开请求、二进制和 12 生成/48 HTTP 预算不变，仍待批准。r235 独立 bootstrap 配置仍待批准。当前批次已结束，安装、下载、API、commit/push 为 0。

唯一开发路径仍为 /home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。下一开发项：保持原 Attention 模型的通用 packed 点积/分组多项式生成；原 12 深层多项式、真实 bootstrap、物理容量扩展继续保留，x86 原生/空白安装/CUDA 暂缓。

下一条无 API 的已有包重载命令：
~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-scheduled-packages-r352/rmsnorm/program --execute
~~~

## 2026-10-01：Attention 默认路径验收 r369

ARM Ubuntu 唯一 checkout、lhy-agent-dsl 与 HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e 保持不变。无活动批次时保存旧 516 文件到 results/poseidon-quadratic-boundary-r361/core-source.zip，新增有界 packed 平方和／旋转相关式点积、完整共享密文因子提取和分组 power 多项式生成；不重写 Provider 候选。

最终默认 15/15 数值通过，包括原 MLP/RMSNorm/Attention、九个归一化及三个批量 matmul。60 组密态输入、400 个值，最大误差 1.01947e-6；Attention 40 值最大 3.16146e-9。三个包重载 80 值最大 7.20333e-8，零生成/编译。静态、参考、组件测试去重为 120 passed / 0 failed / 3 历史 skipped。详见 [r369](poseidon-attention-default-acceptance-r369.md)。

应用核心 518、执行源码 575 文件绑定。r368 替代未执行 r354；公开三模型 request、后端及 12 生成/48 HTTP 预算一致，仍待批准。旧失败/提案未改写。r235 bootstrap 配置仍待批准，原 12 深层多项式失败和 bootstrap 精度问题未解决；新 Poseidon 真实 Agent 成功数 0。x86 原生、空白安装、CUDA 暂缓。API/安装/下载/commit/push 为 0。

下一条无 API 命令（既有环境自动进入）：
~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-scheduled-packages-r365/attention_decode/program --execute
~~~

后续优先获准后验收原三个整模型的真实 Agent 生成；继续工作前核实进程、518 源码绑定和空间预算。大张量物理布局及完整网络近似精度仍需独立完成。

## 历史源码与容量补验

[r377](poseidon-historical-helper-acceptance-r377.md)补验13份历史Agent原DSL，helper/分块/P32/P128全部编译、数值和重载通过，累计27份程序；11helper补有限输出干预审计，匹配原真实tracing。

[r386](poseidon-capacity-acceptance-r386.md)新增v2/medium物理布局：输入/输出各1024元素、4块。18新模型、1自动布局变体、2旧回归全部数值通过，6包重载通过。新容量Agent生成及bootstrap/深度独立保留状态。

## CPU bootstrap最新试验

用户批准的ratio128独立试验已执行。零组完成重复刷新，有符号第二刷新15值超门限，随机/边界按首次失败即停未运行。bootstrap未完成精度验收，当前Agent Runtime尚未开放；核心CPU迁移结论保持。[r388结果与下一步](poseidon-bootstrap-ratio128-result-r388.md)

## 公开权重与 Linear 容量 r400

公开张量绑定与跨块Linear新增12个预算内数学模型，含512/576/1024元素batched输出；各分区三个上下文完成独立参考和真实Poseidon CPU。112项静态/组件测试通过，24次密态执行覆盖18个不同模型、20,972个值，4包重载通过。两个矩形batch保留请求准备层容量阻塞；详细分母、源码版本与生成来源见[r400](poseidon-capacity-linear-acceptance-r400.md)。当前执行584/应用527文件绑定。新Agent生成、bootstrap与剩余大张量算子分别验收。

## 当前矩形Linear容量

[r430](poseidon-capacity-tiled-linear-r430.md)记录原两个容量阻塞的解除：三个原/新上下文完成真实编译、密态数值与包重载。公开行宽重排、typed helper复用和v3绑定保持既有安全及资源上限；旧v1/v2请求与历史失败保留。123项相关测试通过，新Agent生成单列。


## 跨块张量容量 r441

concat、stack、sum、mean 已增加静态索引和跨密文路由。15个补充模型覆盖非对齐尾部、内轴、多个归约轴及命名多输出，真实编译、Poseidon CPU和逐项比较通过。最新源码只回归受影响的命名冲突、较大stack和旧Linear/MLP；旧报告按原绑定保留。132项测试通过，本阶段新Agent生成0。完整范围、程序包和剩余Conv/Pool等缺口见[r441](poseidon-capacity-tensor-acceptance-r441.md)。


## 跨密文空间算子

medium-v1已验收跨块Conv1d/2d（含分组/深度卷积）、AvgPool1d/2d及代表组合。新容量公开张量版本为v5；v5文件编码可无损压缩重复向量，展开后按原逻辑请求及公开绑定验证。旧版本按原规则重建。[范围与结果](poseidon-capacity-spatial-acceptance-r449.md)。当前结果为规则基线，真实Agent验收另计。
