# 第三阶段扩展进度

开发位置：ARM Ubuntu `/home/lhohy/Code/Poseidon`，分支 `lhy-agent-dsl`。

## 已实现

- 统一legacy-v1、medium-v1、resnet20-v1资源配置。medium的1024元素／4块和resnet20-v1的4096元素／16块IO已有JSON与可信PyTorch代表性数值验收；大中间张量与整网布局继续扩展。
- 当前分解注册表为 34 项，早期扩展到 21 项，新增方差、LayerNorm、GroupNorm、InstanceNorm、全局平均池化、推理 Dropout、sqrt，以及显式多项式 ReLU/GELU/SiLU/Sigmoid/Tanh/log；matmul 增加受限批量广播。
- 权重包采用绑定大小、形状、dtype 和 SHA-256 的 float64 NPY；拒绝 pickle、符号链接及超容量文件。
- 可信本地 PyTorch FX 入口使用明确文件清单和 bubblewrap 隔离，输出高层图、分解图及来源绑定。当前支持范围由捕获器和测试限定。
- Poseidon CPU 库已在 ARM 编译并加载。ARM 借位操作使用原有通用实现；x86 intrinsic 路径保留。

## 早期验收记录（r220）

记录根目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results`。

| 范围 | 结果 | 证据目录 |
| --- | --- | --- |
| 分解、资源、权重、PyTorch 捕获 | 51 项通过，0 失败/跳过；捕获 MLP 的 16 组独立参考比较通过 | stage3-extensions-r220 |
| 组件与旧接口 | 249 项通过，0 失败/跳过；2208 个历史请求精确重建 | stage3-compatibility-r220 |
| Linear、捕获 MLP、方差、全局平均池化 | 4 项脚本化 self-test 完成真实 SEAL 编译、密态执行和数值比较 | stage3-extensions-r217b、stage3-extensions-r218 |
| Poseidon CPU 链接、GMP、ARM 借位及参数拒绝 | CTest 2/2 通过 | build-poseidon/bootstrap-cpu-r217/Testing/Temporary/LastTest.log |

这些新增后端结果是脚本化候选验收，真实 Agent 新批次尚未运行。各目录保留各自源码绑定，后续源码变化不改写旧报告。冻结 Benchmark 未重新生成。旧 x86 配置保留，本批没有 x86 实机验收。

## GMP 依赖

用户批准的 GMP C++ 6.3.0 来自既有固定 Nix revision，使用独立 `poseidon-cpu-shell.nix`。实际下载约 0.23 MiB、解包约 0.85 MiB，低于批准的 32 MiB / 256 MiB。derivation、平台和安装记录见 `results/poseidon-gmp-r219`。SEAL SDK 默认 shell 保持原配置。

## 早期 Bootstrap 与 CUDA 审计（r220）

原 bootstrap 测试的 Q×P 模数总位数下界为 1314；实际 Poseidon tc128 表在 N=16384、32768 时分别允许 438、881 位，N=65536 无条目。原测试使用 sec_level_type::none，不能作为当前安全要求下的验收。现阶段通过的是构建、动态加载和参数拒绝测试，尚未执行真实 bootstrap。

下一步先提出可复现的安全参数方案和所需源码范围，经单独批准后再接入执行。CUDA adapter 的 Modswitch/Upscale 映射及 BootstrapFallback 仍有缺口；本机没有 NVIDIA 执行环境。

## 未完成

1. 扩展物理密文布局及跨块运算，让中型和 ResNet20 资源配置进入实际 Agent 链路。
2. 补齐能力清单中的高阶算子、PyTorch 映射和每项至少三个真实后端上下文。
3. 对新增能力冻结案例、请求及调用预算，申请真实 Agent 测试。
4. 真实 CPU bootstrap、完整 RMSNorm/Attention、ResNet20 和 CUDA 实机验收。

r220 批次没有下载 CIFAR-10 或训练权重，也没有付费调用、提交或推送。后续执行见下方分批记录。

机器可读汇总：[r220 验收记录](stage3-extension-acceptance-r220.json)。

## 张量与激活扩展（r226）

本批新增 6 项分解，注册表共 27 项。每项有 3 个离线密态样例，合计 18/18 通过、72 组输入、448 个输出值；最大 CKKS 绝对误差 2.4234e-7。这里比较的是明确声明的近似计算 reference。

| 输入算子 | 当前范围 | PyTorch 入口 |
| --- | --- | --- |
| broadcast | rank 1–4，静态 shape，按尾轴广播，受资源配置约束 | Tensor.expand、torch.broadcast_to |
| public_gather | 一个轴上的公开一维整数索引；支持重复和重排，索引需非负且在界内 | index_select |
| adaptive_avg_pool | N,C,L 或 N,C,H,W；输出空间维度为正且不大于输入；允许窗口重叠 | AdaptiveAvgPool1d/2d |
| leaky_relu | 公开有限斜率和显式 ReLU 多项式近似 | nn.LeakyReLU |
| prelu | 公开标量或逐通道斜率和显式 ReLU 多项式近似 | nn.PReLU |
| logsoftmax | 显式轴、exp 近似和正区间 log 多项式 | nn.LogSoftmax |

LeakyReLU/PReLU 本批回归多项式在 16 组探针上的原函数最大误差为 0.15；LogSoftmax 约为 1.69e-4。这些配置用于验证分解与生成链路，尚未完成目标模型的近似精度校准。

自适应池化的一例原先超过确定性基线的 AST 限制。新增 singleton-v1 回退：将提取值保留在第零槽位，对公开加数加掩码，输出时旋转到各自位置。仅在原有两种生成路径超过资源限制后启用，旧的合法输出保持不变。原失败记录保留，修复后同一模型通过；另有两个非线性样例强制走该路径并通过密态验收。

离线检查：64 项通过，0 失败/跳过。兼容回归：249 项通过，2208 个旧请求精确重建。机器记录见 [r226](stage3-operator-offline-r226.json)。

## 真实 Agent 批次

r227 的 18 项自由生成已全部通过独立审计，18 项均首次成功。实际 18 次生成、18 次 HTTP，0 次修复、0 次传输重试；耗时约 292 秒。4 组输入/例、448 个输出值，最大 CKKS 绝对误差 1.0793e-6，门限保持 1e-5 + 1e-4*abs(reference)。运行与独立审计保存在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage3-operator-live-r227*`。批次已结束；原冻结绑定与证据保留。汇总见 [真实 Agent r227](stage3-operator-agent-r227.json)。

截至 r227 仍未完成：padding、max/min/MaxPool 等分解、Attention 通用映射、更大物理布局、真实 CPU bootstrap 和 ResNet20。Bootstrap 参数与 GPU 实机状态维持原记录。

## Padding 与近似比较（r228–r230）

新增 7 项分解，注册表共 34 项。冻结模型集位于 `scripts/baseline/cases/operator-extension-v4`，包含 27 个模型；旧模型集及报告保留。

| 输入算子 | 范围 | PyTorch 映射 |
| --- | --- | --- |
| padding | rank 1–4，逐轴非负静态 padding；constant、reflect、replicate | ConstantPad1d/2d、ZeroPad2d、ReflectionPad1d/2d、ReplicationPad1d/2d、functional.pad |
| maximum / minimum | 两个输入，按已有广播规则逐项比较 | torch.maximum / minimum |
| max / min | 静态唯一轴列表、keepdims；标量结果表示为 [1] | torch.amax / amin、Tensor.amax / amin |
| max_pool1d / max_pool2d | N,C,L 或 N,C,H,W；公开 kernel、stride、padding、dilation；floor 输出大小 | nn.MaxPool1d/2d |

比较算子必须提供 abs(lhs-rhs) 的多项式与有效区间。归约采用固定的相邻平衡树；MaxPool 只比较窗口内有效位置。区间检查包含比较中间值。返回索引、ceil 模式、负 padding 和 circular padding 尚未开放。

本批使用 `0.5 + 0.25*x²`、区间 [-4,4] 验证生成链路。在 16 组探针上，相对原函数的最大误差为 0.5；这份配置用于回归测试，目标网络还需要单独校准近似精度。

验收结果：

- 74 项单元测试通过，0 失败、0 跳过；可信 PyTorch MLP 的隔离捕获及 16 组独立参考比较通过（r228）。
- 27/27 脚本化候选通过真实 Hecate tracing、Earth/CKKS IR、HEVM/CST、SEAL CPU 执行、解密和逐项比较（r229）。
- 108 组密态输入、544 个输出值，最大 CKKS 绝对误差 4.0792e-7；冻结门限保持 1e-5 + 1e-4*abs(reference)。
- 249 项兼容测试通过，0 失败、0 跳过；2208 个历史请求精确重建（r230）。
- 本批付费调用 0。截至 r230，27 个新模型的真实 Agent 生成待批准；前一批 r227 的 18/18 Agent 结果仍保留原绑定。

r229 执行后更新了能力清单的分类说明；执行记录仍绑定原源码，r230 兼容回归绑定更新后源码。验收与逐例证据见 [r229](stage3-extrema-offline-r229.json)，新付费范围见 [r230 申请](stage3-extrema-live-proposal-r230.json)。

离线重现（使用尚不存在的输出目录）：

```sh
POSEIDON_PLATFORM=aarch64-linux python3 -B scripts/baseline/check_operator_regression.py \
  --manifest scripts/baseline/cases/operator-extension-v4/manifest.json \
  --output /home/lhohy/poseidon-work/platforms/aarch64-linux/results/extrema-next \
  --run-local
```

下一批开发重点：Attention 组合、跨密文布局与容量。真实 bootstrap 仍需先解决安全参数方案；CUDA 等待 NVIDIA 环境。

## 真实 Agent：padding 与比较（r231）

经本轮单独批准，27 项自由生成全部通过独立审计：

| 范围 | 通过 |
| --- | ---: |
| padding：constant、reflect、replicate | 9/9 |
| maximum、minimum | 各 3/3 |
| max、min 归约 | 各 3/3 |
| MaxPool1d、MaxPool2d | 各 3/3 |

25 项首次通过，2 项修复后通过。padding_8 首次槽位搬移错误导致数值失败，修复一轮后通过；min_1 先出现循环覆盖保留绑定的静态错误，再出现数值失败，两轮修复后通过。失败响应、诊断及产物保留，未使用人工答案替换。

实际 30 次生成、30 次 HTTP、0 次传输重试，低于批准的 108/432 上限。耗时 571.72 秒；108 组输入、544 个输出值，最大 CKKS 绝对误差 7.8919e-7。所有成功均包含真实 tracing、编译、SEAL 执行、解密和独立 reference 比较。比较多项式相对原函数的误差仍按 r229 单列，最高 0.5；网络精度尚待校准。

证据：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage3-extrema-live-r231*`；[机器可读验收报告](stage3-extrema-agent-r231.json)。本批已结束，无自动续跑；既有 r227、r229 绑定保留。未修改编译器、runtime、安全参数或门限，未安装、提交或推送。

下一步继续 Attention 组合与物理布局容量；bootstrap 安全参数方案、整模型和 CUDA 分别验收。

## Poseidon Python 输入验收 r288

受限多文件 Python 与可信本地 PyTorch `nn.Module` 平方 MLP 均完成当前 Poseidon CPU 路线验收。可信捕获在隔离进程中比较原模块、高层图及分解图，包重载保持来源信息并直接执行 RuntimePlan。

124 项测试通过，3 项历史测试 skipped；两个示例编译、密态、解密及逐项比较通过，两个包重载通过，最大绝对误差含重载为 1.55691e-7。候选为 self-test，真实 Agent 计数 0。见 [r288](poseidon-python-import-migration-r288.md)。

本节是当前 Poseidon 输入结果；此前 SEAL Agent 批次继续按其原后端、代码和请求绑定记录。真实 Agent 三整模型最新提案为 r286，尚未执行；剩余 12 编译容量失败、bootstrap 精度、大模型物理布局分别保留。

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

## 2026-10-01：核心迁移与历史 Agent 原源码验收 r373

按用户新口径，后端验收独立于重新生成。核心 Poseidon CPU 链路已迁移；完整历史兼容和原计划分别推进。历史响应、原源码、模型、权重、reference 与旧报告保持不变，当前依赖/产物/后端另绑定。

r371 代表性历史 Agent 14/14，真实编译、Poseidon CPU 密态、解密与逐项比较均通过；56 组/200 值，最大误差 1.1405008855e-6。14 包重载同样 56 组/200 值，最大误差 6.3915006909e-7，生成/编译为 0。r372 10 静态来源/拒绝测试 passed，14 包来源审计通过，failed/skipped 0。结构性空返回单列；SiLU 是声明的多项式参考。见 [当前迁移状态](poseidon-migration-status-r373.md)。

原始记录在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-agent-replay-r371`、`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-integrity-r372`；工具与修改前文档快照在 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-migration-status-r373`。唯一开发路径、分支与 HEAD 不变；保留所有已有修改/未跟踪。575 执行文件、518 应用文件绑定未变。仅新增开发工具和文档，无编译器/Runtime/参数/reference/门限改动，付费/安装/下载/commit/push 0。

r368 三整模型已获用户批准、未启动，12 生成/48 HTTP 原额度未使用；优先级改为补历史源码的缺口，然后原算子/物理布局/bootstrap 计划。新生成用于输入、提示契约和有界修复验收，不阻塞核心迁移。原 12 深多项式、bootstrap 精度、跨密文大模型容量和整网精度保持待完成；x86 原生/空白安装/CUDA 暂缓。r235 新参数试验未批准。

下一命令（已有程序包；无生成或编译）：
~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-historical-agent-replay-r371/linear_square_linear/program --execute
~~~

## 2026-10-01：历史helper与1024元素容量 r377/r386

唯一ARM checkout、lhy-agent-dsl、HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e不变，保留已有修改。r375历史13原源码在Poseidon编译/密态/比较/重载全通过，52组/572值，max8.15395e-8；重载max1.18792e-7。r376来源13审计通过，11helper三公开探针干预匹配原tracing。合r371共27程序案例，不作去重数学模型数或新生成成绩。

备份r378后新增graph-v2/resource_profile和四输入/输出块容量ABI。18新模型+1自动P128+旧Linear/MLP共21/21规则密态通过：84组/36420值max2.72017e-8；6包重载24组/15884值max2.56775e-8，生成/编译0。r385静态94passed/0failed/0skipped，18模型各16双参考探针，27旧请求精确重建。

r381的21tracing启动失败保留（可信layout漏挂载）；补允许列表后r382/r383全部通过。r384 Nix启动失败保留，串行r385通过，底层原因未确认。当前执行579/应用522文件，新绑定，旧575/518结果未重标。源码和文档快照results/poseidon-stage-report-r386。API/安装/下载/参数/编译器/Runtime核心/commit/push变更0。

medium输入/输出各1024已实际验收，范围为逐项、全mean/尾部、标量广播、双输入、对齐concat、reshape。下一步跨块权重/Linear/Conv/Pool、非对齐连接、v2helper/定向及真实Agent。ResNet20物理容量未完成。r235 bootstrap ratio128试验待批准，未启动；r368三模型付费额度未使用，执行前核验当前源码冻结。x86原生/空白安装/CUDA暂缓。

下一条离线命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/agent.py --backend local candidate -- --case scripts/baseline/cases/capacity-v2/capacity_mean_tail_2.json --unified-chunk-period 256 --self-test --compiler-configuration poseidon-cpu-eva-w45-v1
~~~

核心迁移已通过；继续原算子/容量/bootstrap/整模型计划，不重跑全Benchmark，不把迁移当整个项目完成。完整记录[r377](poseidon-historical-helper-acceptance-r377.md)与[r386](poseidon-capacity-acceptance-r386.md)。

## 2026-10-01：r235批准试验已执行 r388

r235的单次有界ratio128试验已获批准并执行为results/poseidon-bootstrap-ratio128-r236，授权已消耗，不能重复使用。参数上下文通过，编译通过；真实原生bootstrap有零组两次刷新通过，有符号第一次刷新与平方通过，第二次刷新15/16384值超冻结门限，max2.15891e-5。立即停，随机/边界未运行。共6阶段5pass/1fail，计划其余6not_run。库、源、原四组输入和门限保持冻结。45.77秒，约13.95MiB，6GiB AS隔离上限，0下载/API；当前Agent参数与Runtime未改，bootstrap仍未开放。

数学诊断显示理想sine近似偏差不足以解释尾部，已有平方误差及归一化/再次刷新新增误差共同累积；carry、噪声、多项式各占比未隔离。继续只读trace分析，新增试验或参数变化另批准。[bootstrap结果](poseidon-bootstrap-ratio128-result-r388.md)。

r387的1024元素mean CLI自检也通过四组真实编译/密态比较；重复现有容量模型，不增加模型计数。原始目录results/candidate-replay-zhovt2h2。核心Poseidon CPU迁移结论保持，跨块权重/算子、ResNet20和bootstrap独立推进。源码579/应用522未改，所有批次已终态，无后台付费任务。

## 2026-10-01：公开张量与跨块 Linear r400

ARM checkout、lhy-agent-dsl、HEAD保持。备份：results/poseidon-capacity-linear-boundary-r389和poseidon-capacity-routing-boundary-r395。新增可信公开向量绑定、跨输入块Linear、batched输出的旋转加权映射；capacity-public-tensors-v1/v2分别绑定与重建，当前规则路径capacity-chunks-v3。旧schema和默认v1保持。应用/执行接口复用原组件，资源、安全和数值门限未改。

r391静态106通过；r397扩展后112passed/0failed/0skipped，双参考及受限算术解释检查16探针/新模型，27历史、21先前容量、15公开向量v1请求精确重建。r390早期测试层次错误、r396五项错误及compact递归版本传递修复均保留历史日志。

r392/r393：9新增模型+6回归，15/15真实编译、密态、解密及逐项比较通过，11,424值max2.18998e-8，3包重载通过。r398：3新batched正例+5回归，8通过/2原矩形batch容量阻塞，9,516值max5.00867e-8；1024元素包重载通过。r399原公开向量v1请求和原候选通过应用组件再编译/密态，32值max3.15176e-10。合24执行/18不同模型，96组/20,972值；4重载/16组/5,184值max9.32353e-9。无skip/not_run，两个阻塞未进入编译，也未记pass。

公开张量、跨块Linear、Linear-square-Linear各三个上下文。跨块batched正例[128,4]、[1,192,3]、[1,2,128,4]，最终1024元素输出包可直接复用。原capacity_batched_linear_1和_2分别需要279/280独立向量，估计1677/1679工作，超过保留的256注册/1024工作预算；数学合法且双参考一致，原案例保留，下一步压缩矩形batch映射。跨块Conv/Pool、部分轴归约、非对齐连接、v2helper/定向和ResNet20物理容量继续推进。

r392/r393执行582/应用525，r398/r399当前584/527文件；源码分别绑定，旧报告未重标。修改仅Agent Python、补充模型、测试和文档，Dacapo/Runtime核心/库/参数/reference/门限未变。本批全部规则候选，新真实Agent/API、安装、下载、commit/push/PR均0。r368三整模型已批准额度仍未使用，旧冻结源码已变化，启动前须建立当前绑定。

r235独立bootstrap批准已消耗，r388有符号第二刷新15值超门限；真实bootstrap与原12深多项式失败仍未解决。x86原生、空白安装、CUDA暂缓。核心迁移保持已验收，全项目目标未完成。

下一条无API命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-routing-numerical-r398/capacity_batched_linear_4/program --execute
~~~

[完整报告](poseidon-capacity-linear-acceptance-r400.md) · [机器可读报告](poseidon-capacity-linear-acceptance-r400.json)。

## 2026-10-01：最终组件复核 r415

公开绑定改为纯Python，均匀权重复用广播时保留标量；旧v1/v2请求身份保持。114passed/0failed/0skipped。共享权重原模型与原DSL在CLI及当前组件均通过真实编译/密态比较；新绑定包CLI重载4组/64值通过，生成/编译0。旧包按验证源码变化拒绝，原证据与拒绝日志保留。当前执行584/应用527源码绑定，二进制不变。

本阶段合26次密态通过/19不同模型/104组/21,100值，5包重载/20组/5,248值。均为规则候选，未增加真实Agent生成。两个原矩形batch容量阻塞、12深多项式、bootstrap重复刷新精度及后续Conv/Pool等缺口保持；API/安装/下载/commit/push0。详情与全部绑定见[r415](poseidon-capacity-final-review-r415.md)，r400保留前一快照。

下一步为矩形batch映射压缩与跨块张量算子。当前包下一命令：
~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-shared-current-r412/program --execute
~~~


## 2026-10-01：矩形 batch Linear r430

唯一ARM checkout、lhy-agent-dsl与HEAD保持。r416保存修改前执行源码；新增公共行宽重排与局部Linear映射，capacity-public-tensors-v3，v1/v2请求按原版本重建。原两个矩形案例不变，连同新第三上下文均在最终587执行/530应用源码绑定下编译和数值通过：12组/7,492值，max6.99028e-7。三个包重载通过；含额外公开CLI复核共16组/10,564值，max6.28365e-7，生成/编译0。最终123passed/0failed/errors/skipped。旧r398两阻塞与本轮早期失败日志保持。

r422八个不同数学模型全部数值通过，32组/12,640值，max8.36250e-7；两个包导出失败定位为缩进JSON撑大文件。统一证据与导出端规范JSON写入后，128KiB、256注册、1024golden工作及沙箱限制全部保留。r424旧导出失败也保留，r426/r428为当前数值绑定，不改写旧结果。[报告r430](poseidon-capacity-tiled-linear-r430.md)及JSON列出全部绑定。

本批全部规则基线，API/安装/下载/commit/push0，Dacapo/Runtime/Poseidon库/参数/reference/门限未改，所有执行进程已结束。核心CPU迁移已验收；继续部分轴归约、非对齐连接、Conv/Pool及容量Agent评测。12深多项式与bootstrap重复刷新精度仍未解决；r235单次试验已消耗，r368三整模型额度未启动，执行前核对新绑定。x86原生/空白安装/CUDA暂缓。

下一条无API命令：
~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tiles-reload-r426/capacity_batched_linear_2/program --execute
~~~


## 2026-10-02：跨块连接与指定轴归约 r441

ARM唯一checkout、lhy-agent-dsl和HEAD保持。r431保存修改前源码，新增capacity-public-tensors-v4及capacity-chunks-v5：静态concat/stack/sum/mean索引映射、连续旋转窗口、suffix行归约与末轴stack的typed helper复用。内部节点名避开用户名字；v1–v3请求保留，8个v3和21个无descriptor历史请求精确重建。

最终132 passed / 0 failed / 0 errors / 0 skipped。本阶段15新数学模型、4兼容模型，共19不同模型，24次编译密态执行全部通过：96组输入、24,296值，按模型去重76组/21,960值，max1.36997235245e-6。八次包重载32组/11,876值，生成/编译0。r435/r437绑定命名修复前源码；r439/r440按最终590执行/533应用文件新绑定，四项受影响回归与当前包CLI重载通过，不改写旧结果。全部为规则基线，新Agent/API为0。

冻结1200集、Dacapo/Runtime核心、库、参数、reference、误差门限和资源上限保持。未安装/下载/提交/推送。跨块Conv/Pool、其他形状/切片组合、容量Agent评测、ResNet20物理布局继续开发；12深多项式编译容量和bootstrap重复刷新精度仍未解决。r235授权已执行消耗，后续参数试验另行批准；r368已批准三整模型额度未使用，旧源码冻结不能直接沿用。x86原生、空白安装、CUDA暂缓。

当前可重放包下一命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-capacity-tensor-cli-r440/program --execute
~~~

[完整报告](poseidon-capacity-tensor-acceptance-r441.md) · [机器可读报告](poseidon-capacity-tensor-acceptance-r441.json)。


## 2026-10-02：跨密文卷积与平均池化 r449

当前594执行源码、537应用文件匹配r448冻结绑定。143项静态测试通过，20/20真实编译、Poseidon CPU密态和数值通过；15新模型加5兼容例，80组/17736值，最大误差1.42573e-6。七个包重载28组/5480值通过，最大误差6.20441e-7。全部规则候选，新Agent/API为0。

新增分组/深度Conv1d/2d、AvgPool1d/2d及卷积残差、空间→Linear组合；v5公开向量无损文件压缩保持旧v1–v4格式与原请求身份。源码、AST、工作量、常量注册与沙箱限制保持。Dacapo/Runtime/库/参数/reference/门限未变，未安装/下载/提交/推送；旧失败及报告保持。

容量真实Agent、其他形状/切片组合和ResNet20物理布局仍待完成。原12深多项式与bootstrap重复刷新精度未解决。此前批准三整模型额度尚未启动，执行前重新核验当前源码、相同公开请求与后端身份；不扩大原12生成/48HTTP额度。x86原生、空白安装与CUDA暂缓。

[验收报告](poseidon-capacity-spatial-acceptance-r449.md) · [机器报告](poseidon-capacity-spatial-acceptance-r449.json)。

## 2026-10-02：真实Agent整模型与中型PyTorch r453/r473

已批准三模型批次已执行：MLP首次通过、RMSNorm修复一次通过、Attention四次编译失败并耗尽三轮修复。共7生成／7 HTTP／0传输重试；两个成功包数值重载通过。原额度已按单例结束，后续付费批次另行批准。[r453](poseidon-whole-models-agent-acceptance-r453.md)

Agent编译诊断现已识别固定Poseidon配置的720类型预算。原Attention候选离线重放仍被拒绝，错误进入受控反馈；未改编译器。当前两份成功Agent原源码重新编译、密态和包重载2/2通过，没有新增生成。

FX新增sum／mean、cat及其别名、stack。中型来源包按声明resource_profile审计；v5元数据列式传输无损恢复原请求；完全相同的Linear helper函数体复用。最终179静态测试通过，0失败／跳过，598执行源码绑定。

三个较大PyTorch模块均通过各16组独立参考。516输入／137输出的归约融合模型完成4组／548值密态验收和包重载；578输入的空间残差在256公开向量预算处阻塞；260输入的Linear融合已编译，执行出现bad_alloc并超时。原案例、r459/r463失败和源码快照保留，未替换模型。

当前问题：奇数宽度步幅池化路由容量、Linear融合中间密文内存、Attention新反馈的真实Agent效果、容量／Python入口真实Agent验收、12深多项式、重复bootstrap精度和ResNet20整网。x86原生、空白安装、CUDA继续暂缓。

本批未安装、下载、调用付费API、修改编译器／Runtime核心、参数、reference或门限，未commit/push。唯一checkout仍为 /home/lhohy/Code/Poseidon，lhy-agent-dsl，HEAD 7588e48093631eaf8385506e3db058e7ca3cfe7e。所有批次已终态，下一步先分析原空间路由与Linear寄存器占用；保持安全上限。

下一条无生成／编译的当前包重载命令：

~~~sh
POSEIDON_PLATFORM=aarch64-linux POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work python3 -B scripts/dsl_bundle.py replay --bundle /home/lhohy/poseidon-work/platforms/aarch64-linux/results/poseidon-pytorch-medium-r470/reduction_fusion/program --execute
~~~

[本轮报告](poseidon-stage3-pytorch-acceptance-r473.md) · [机器报告](poseidon-stage3-pytorch-acceptance-r473.json)。

## 2026-10-02：矩阵FX入口、分块广播与当前程序包（r508）

263项静态测试通过，失败／错误／跳过为0。新增可信FX dot/mm/bmm和Tensor矩阵方法；分块基线使用现有mask0与旋转实现非标量广播。没有修改分解数学、Dacapo、Runtime核心或安全参数。

当前601执行源码绑定下，9矩阵、9广播、2原中型PyTorch组合、2池化回归、2原真实Agent源码共24项真实编译／Poseidon密态／数值通过，24包重载通过。96组／2972值；规则基线22项，历史Agent2项，新增生成/API为0。原6矩阵预检失败已按原模型修复；r495/r497等失败保留。

原516元素归约融合和260元素Linear融合通过；578元素空间残差源码保持，在历史r487仍因中间密文保留失败。r490最后使用释放补丁待批准。MLP/RMSNorm原Agent源码通过；Attention新生成修复、容量／Python真实Agent验收、12深多项式、bootstrap重复刷新精度及ResNet20继续推进。x86原生、空白安装和CUDA暂缓。

[本批报告](poseidon-pytorch-broadcast-acceptance-r508.md)与JSON记录逐例请求、依赖及原始报告绑定。


## 4096元素物理IO r518

resnet20-v1增加capacity V3，保留旧四块ABI。9新模型、一个公开构造变体和两例规则回归12/12通过Poseidon CPU数值验收，10包重载；276静态通过、23旧skip。覆盖逐项／全归约／尾块／四输入四输出。新增Agent生成0。65536中间张量、整网权重、真实ResNet块和4096元素PyTorch继续开发。[报告](poseidon-extended-capacity-acceptance-r518.md)。

## 4096元素PyTorch与真实ResNet20源模型 r535

602执行源码新绑定，300静态通过／23旧skip；开发清单8项通过。五个可信PyTorch案例全部真实编译、Poseidon CPU密态、逐项比较和包重载通过，20组／32800值，生成/API0。标量stack保持真实PyTorch形状，辅助名字避开FX节点。公开CLI导入和请求准备通过。

已核对ResNet20完整拓扑和271098公开值；标准ReLU的PyTorch／独立NumPy对照44组／440值max1.95399e-14。现有1000张子集Top-1 91.4%，不替代完整测试集或训练校准。完整尺寸stem中间16384元素／64块，路由442368超过262144预算，仍阻塞；没有整网编译或密态。后续优先大特征图路由、重复bootstrap及已列出的Agent验收。详见[r535](poseidon-pytorch-extended-acceptance-r535.md)。

## 通道布局与 BatchNorm r547

v8显式布局的六个规则模型及两份历史Agent原源码8/8通过Poseidon编译、密态、数值和重载；163静态通过。原尺寸stem公开计划168绑定/55784字节，数学检查通过，P4096执行ABI待接入。新增生成0。[报告](poseidon-channel-layout-acceptance-r547.md)。

## 大周期与原尺寸首层诊断 r574

当前ARM checkout、lhy-agent-dsl与HEAD保持；本批全部终态。121静态通过，0失败/错误/跳过。V4周期512–4096的5规则案例与2历史Agent已通过r562；原stem大小失败在新配置下解决。r573两个规则案例均完成真实tracing/编译/Poseidon密态/逐项比较/包重载，8组16448值max6.02909e-9；首层64值max3.44480e-9。MLP/RMSNorm原Agent源码在r571重放通过，分别保留源码绑定，新生成/API为0。

新增显式poseidon-cpu-eva-w45-compact-v1（schema4）。原首层IR11138317→6812765字节，168常量688128个f64经固定MLIR位级往返验证。原16MiB前端和8MiB入口上限保持；旧schema2/3默认与哈希保持。静态测试所需的原公开模型/请求/候选已放入repo fixture，去除历史results目录依赖。备份results/poseidon-trace-boundary-r564；失败r565/r568/r569/r571与旧证据保留。

未改Dacapo/Runtime核心/Poseidon库/参数/模型/reference/门限；无安装/下载/commit/push。原尺寸首层输出为ReLU前16通道均值，尚未验收残差块或完整ResNet20。578输入空间残差仍有中间密文保留问题；r490释放草案未应用。bootstrap重复刷新、Attention与新容量/Python真实Agent、整网近似与精度继续推进。x86原生/空白安装/CUDA暂缓。

下一条命令与程序包位置见[本批报告](poseidon-wide-trace-acceptance-r574.md)。开始新批次仍核实进程与冻结绑定。
