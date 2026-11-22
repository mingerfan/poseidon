# Llama 3.1-8B API 版本测试结果及与 Cerium 的差异

> **独立手工模型主线已实现：** 新目标 `llama31_poseidon_manual`，入口 `manual_main.cpp`，直接用 Poseidon API 串联 decoder，不依赖 DSL runtime。显式 plan 管理15个边界的Q/scale/Boot与实际seed消费计数，持有两套不可变profile，末端complex门控、无中间解密。基础与配对SiLU候选分开；旧 `llama31_cerium_source_api` 不被替换。本地std测试11checks/328negatives、实际consumer源码mock23checks/2negatives、6项promotion源码不变量与最终O3主机对象编译通过，**没有新目标链接/GPU精度或延迟验收**。详见 [手工模型入口与优化边界](MANUAL_LLAMA.md)。

> **新任务进行中：**按真实Native Cerium迁移packing/phase/降模/延迟relin，已核实并固定 **N=65536、32768complexslots**，不降N。满N raw代数和实际+i Q编码CPU控制通过；新a50二进制/1e58后端随后完成**一次公共双通道GPU控制：10/10组、40实际Boot，全槽identity最大complex误差1.1722332e−4≤1e−3，新旧real逐metadata/RNS exact**。这不是完整SiLU/Block/32层或性能验收。后续配对all60 SiLU、optional模型pair hook和32Boot公共控制器已写好、独立本地frontend/metric/schema测试通过，**尚未目标链接/启用模型/GPU验证**；新的app-only构建工具已完成最终审阅和37项本地策略测试；定版后的单次上传因现有SSH通道关闭（255）中止，远端新root状态未确认，未启动新构建/GPU且未重连重试。本地继续physical-scale可行性核验，无新GPU排队。下列b3fec…Block0和a50公共primitive结果各自仅绑定其冻结代，不迁移到新改动。见[同N手写迁移进度](../../../experiments/20261115-cerium-manual-align/README.md)。

> **性能实验分支说明（2026-11-14）：** 当前工作树已加入独立、默认关闭的算子优化，
> 不再是冻结的source15源码。下文8020秒结果、`RELEASE_MANIFEST.json`、
> `VALIDATED_BINARIES.json`等保留为**原始基线证据**，不代表新代码的32层性能或精度。
> 新代码的控制变量、真实权重切片、失败记录与复现命令见
> [本轮实验报告](../../../experiments/20261114/RESULTS.md)；
> [算子覆盖与未完成项](../../../experiments/20261114/OPERATOR_COVERAGE.md)。
> 新模型目标已编译，但未完成新的32层验收，也未宣称已达到Cerium整模型单卡延迟。
>
> **A800整Block后续验收：** 非graph组合251单次Block0逐RNS exact，含权重load为304.82→194.91s；
> 加`LLAMA_LINEAR_GROUP_ACCUM=1`的255组合却出现4.73偏差而失败，**不能因微基准/切片通过而用于完整管线**。
> 原参考精度仍失败，非graph配置也没有32层验收。详见[A800续测报告](../../../experiments/20261114-a800/RESULTS.md)。
>
> **后续实验（仍未完成模型目标）：** R2 mode763 Block0约1.94×、逐RNS exact，但参考精度仍失败；见[重复Block控制](../../../experiments/20261114-round2/a800-block-monitor/RESULTS.md)。
> 新增可选 `LLAMA_REAL_CACHE_MB=2048`，默认仍1024 MiB，Api构造时读取，范围0–8192（0不保留）。这是real-mask LRU驻留预算，**不是总显存上限**。
> 同钥代表性AV三对控制中1→2 GiB消除了1.125 GiB工作集的持续thrash，墙时中位1.459→.534s，见[AV报告](../../../experiments/20261114-round3/a800-av-cache/RESULTS.md)。
> 随后的独立Block0两对控制中，2 GiB仅较1 GiB节省4.15/5.14秒（墙时2.7%/3.3%），全部输出逐RNS exact、input/mask immutable、308 Boots，**参考实部误差仍17.244而失败**；不能把AV的2.73×外推成Block或32层倍率。见[Block缓存报告](../../../experiments/20261114-round3/a800-cache-block-monitor/RESULTS.md)。
> 后续同期原始0→1787/pipeline/cache2GiB/Montgomery的完整Block两对墙时为307.04→163.50、304.17→163.49秒（1.878×/1.861×），全输出RNS exact、input/mask immutable、308 Boots，但reference误差16.216仍失败；这既不是各微基准倍率乘积，也未证明比前述约155秒缓存配置更优。见[组合控制](../../../experiments/20261114-round3/a800-composed-block-monitor/RESULTS.md)。
> 当前源码还含默认关闭的FHE精度实验（quotient floor、整数exp单位seed、SiLU rational VR）；它们不属于上述冻结性能二进制。直接EvalMod合成误差约9.8e-4→9e-6，早期完整Boot/RMS/Softmax精选仍0/10通过。后续阶段与raw/encoded权重对照支持C2S编码/解码路径是所试公开样本的主导项（非唯一根因证明）；实体C2S重建后仍0/7双算子通过，再加入更高精度和整数seed后**4/8配置通过限定active38 RMS+Softmax 1e-3门**。这不是完整SiLU、Block或模型接受：完整SiLU首轮0/7、rational/range细化0/11通过；guard0阶段误差约0.002076。后续all60+三次分支刷新（共4 Boot）有**一个候选通过公开Gate±25/Up±8、32768槽源公式门：0.000601661**，off/on RNS exact但非独立重复。这是精度诊断而非延迟优化，也不覆盖Block/模型或数学SiLU；**模型精度仍未修复**。原生Cerium完整SiLU已完成一次非性能GPU诊断，也未过ideal-source/零路门；原getter的独立CPU规划偏差约153.8与GPU误差分列，不能据其0.145s诊断wall宣称同精度性能领先。见[持续状态与独立证据](../../../experiments/20261114-round3/STATUS.md)。


日期：2026-11-13。测试版本：**source15**；与 source12/source13 的 C++ 源文件逐个 SHA256 一致。

> 先前双不可变profile Block0精度对照已完整执行：每次476 Boot、off/on输出RNS exact、输入不可变，但有效38 token最大复数误差 **0.00630305 >1e-3**，未通过。**审计纠正：其冻结Softmax为standard exp ONE PT-scale；日志虽打印integer=true，旧header没有该consumer。**原生Cerium实验性descending-Fnew计划也仍未通过双路门（0.05369／零路0.001546）。均不是性能或模型验收。
>
> **最新限定结果：** **实际启用integer-unit exp PT-scale的新FHE精度计划**完成单次Block0，最终complex误差**0.000289547606≤1e-3**，476 Boot、7个已测边界均≤门限，observer/input/mask不可变；见[七锚点报告](../../../experiments/20261114-round3/a800-mixed-block0-component-anchors-monitor-v1/RESULTS.md)。两代**不只是observer或随机Runtime差别**，PT编码scale由E=T/S改为M=round(E)、保留实际S*M；旧standard计划失败不构成新计划随机不稳定的证据。两run仍未配对，不能归因唯一/主要原因；无跨plan RNS、稳定性、32/Torch/model/性能接受。见[审计纠正](../../../experiments/20261114-round3/SOFTMAX_PLAN_AUDIT_CORRECTION.md)。

## 1. 当前结论

**真实权重、完整32个Decoder Block已在单卡A800上连续完成密文计算，正常退出并产生有限输出；精度没有通过。**

这版实现了本轮“先完整执行并得到结果”的目标，但不能作为输出可靠的模型用于实际推理，也不能宣称已经完全复制了 Cerium 的全部执行实现。

测试容量为128 token，实际输入38个有效token；hidden=4096、FFN=14336、32个Q头、8个KV头。边界与选用的 Cerium benchmark 一致：加密embedding输入，经过32个Block，得到最后一个Block的密文hidden。验收程序仅在全部Block完成后解密，没有中间解密反馈或明文替代计算。本次不包含tokenizer、最终RMSNorm、LM head和增量decode。

## 2. 实测结果

| 指标 | 结果 |
|---|---:|
| 完成时间 | 2026-11-13 16:20，UTC+8 |
| 程序退出码 | 0 |
| 32层计算耗时 | 7558.47秒，即2小时5分58秒 |
| 模型进程耗时，含准备等 | 约8022.75秒，即2小时13分43秒 |
| 构建、组件预检、模型整个后台流程 | 8527.22秒，即2小时22分7秒 |
| 自举调用 | 9856次；每Block 308次 |
| RMM活跃显存分配峰值 | 21.50 GiB |
| RMM显存池保留量 | 30.03 GiB |
| nvidia-smi设备显存采样最大值 | 30.45 GiB |
| CPU进程RSS峰值 | 24.55 GiB |
| CPU资源 | 7核容器，OpenMP=7 |

内存数据对应正常释放不再使用的中间结果、逐层读取权重的执行方式，不是“保留全部中间结果”的最大占用。RMM活跃分配、池保留量和设备采样读数的统计范围不同，不能混作同一指标。

每个Block的自举分布为：Attention RMSNorm 22次、Softmax 208次、FFN RMSNorm 22次、SwiGLU 56次。9856条实际自举日志均为输入Q6、输出Q30、log2(scale)=40。Q表示当前剩余模数个数；每个Block末尾记录的第一个输出密文为Q6。

## 3. 精度结果与对照口径

| 指标 | 结果 |
|---|---:|
| 有效token最大实部绝对误差 | 9,554,791.1420 |
| 有效token RMSE | 1,103,488.9493 |
| 全槽最大虚部绝对值 | 2.9850 |
| 最终实部输出最大绝对值 | 9,554,780.5615 |
| 原1e-3精度门槛 | 未通过 |

对照文件是既有的 **Cerium original_callpath_v2：真实权重、float64近似packed运算、固定jitter**。它不是标准Hugging Face精确算子的输出，也不是当前C++每条指令的浮点重放。

已找回完整生成driver/原shell/组件writer源码；driver使用自写float64 RoPE，不能误套模型helper的`.float()`路径。未改driver单层CPU replay与冻结Block01最大差`1.3877787808e-17`，但SHA不等，严格byte-alignment仍失败。旧v2组件fixtures已完整hash/metadata校验，可提供7个冻结阶段锚点；不使用SHA失败的新reconstruction替代原真值。新七锚点只读诊断模式已完成静态审查、CPU合成/真实hash、主机构建及Main真实CPU绑定检查；单case GPU诊断已完成，最终active38 complex误差`0.000289547606≤1e-3`、七个已观察锚点均≤门限；这是**一次Block0 reference gate通过**。旧mixed-v2的`.00630305244`失败保留，但审计已证实其Softmax无integer-unit实现、新通过代有实现，**完整FHE PT-scale计划不同**；旧SETUP的true仅是请求，不是consumer证据。新单次pass不证明稳定性，旧不同计划的fail也不证明新计划不稳定。不同keys/加密/观察模式仍阻止单因果或跨plan RNS结论，无32/Torch/model/性能接受声明。见 `../../../experiments/20261114-round3/cpu-block0-reference-recon/PROVENANCE_RECOVERY.md`。

因此，这次最终误差同时可能包含参考路径差别及密文数值误差，尚未通过逐阶段隔离试验证明各项原因的贡献。**不能把955万误差直接归因于自举，也不能认为只补上缓存或CUDA Graph就能消除误差。** 本轮按要求不以精度阻断执行，但误差很大这一事实必须保留。

## 4. GPU利用率记录

每约60秒记录一次nvidia-smi设备GPU利用率和显存用量，共135个有效样本。

| 统计范围 | 样本数 | 样本均值 | 中位数 | 范围 |
|---|---:|---:|---:|---:|
| 整个模型进程，含初始化及结束 | 135 | 66.13% | 76% | 0%–100% |
| 第8分钟起，排除结束样本 | 126 | **70.81%** | **76%** | 0%–100% |

后一范围内，101/126个样本达到或超过70%，9个达到或超过90%，8个低于20%。该区间显存样本均为31179MiB，约30.45GiB。

这里的均值是**采样点算术平均**，不是连续性能分析器测得的精确全程忙碌比例，也不是CUDA核心或SM的占用率。60秒间隔可能漏掉短暂空闲或峰值。

![GPU利用率及设备显存记录](GPU_UTILIZATION.png)

线性投影调用累计4900.26秒，占32层计算计时约64.83%。其中Gate、Up、Down合计3975.91秒。这个计时包含公开权重准备、数据传输及GPU计算，不能全部称为GPU kernel耗时；剩余时间也不能全部归为自举。

## 5. 与 Cerium 哪些相同，哪些仍不同

已沿用原始模型结构、主要非线性公式和系数、packed布局、BSGS分块思路、显式自举位置、融合RoPE掩码，以及“全部QK→全部Softmax→全部AV”的阶段顺序。

**未照搬不等于原则上做不到。** 下列差异分为接口契约不能直接替换，以及本版尚未完成的适配工作。特别是线性层和缓存，不能笼统归咎于“库不支持”。

### 5.1 自举算法、模数链和scale：按要求替换了后端，不能把原参数原封不动传入

- **Cerium：** 使用其自举函数、原模数和明文编码scale，例如`bootstrap_32K_35lvl_t1_vec_function`入口使用自己的level/scale契约。
- **本版：** 调用ResNet20使用的原生`GpuEvaluator::bootstrap`，使用当前Q49/P25参数、degree-22 profile及2^40接口scale；每次实际Q6→Q30。
- **为什么不能直接照搬：** 原始level数字、素数、密钥及变换矩阵与当前后端并非同一套参数。库会检查输入Q、scale和profile匹配；只复制数字或改密文元数据，不会得到等价计算。这里的底层自举替换属于用户明确允许的接口适配范围。
- **还可做什么：** 继续对齐输入幅值、实际scale、编码系数和输出语义；但必须重新验证，不能承诺换成相同公式就自动得到相同精度。

还有两处相关的接口差异需要明确：原 runner 的部分系数通过编码scale与声明scale共同表达数值，本版按当前后端实际追踪的物理scale换算系数；这属于表示转换，不能只逐字复制系数文件就假定等价。原SwiGLU函数末尾保留高scale乘积，本版为接入后续scale=2^40接口增加了真实的双Rescale，额外消耗模数并引入舍入。这些均不是新的多项式拟合，但仍需要单独检查误差；不能将“沿用公式”写成物理执行逐指令一致。

依据：Cerium [`bootstrap.py:656`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/benchmarks/llama/bootstrap.py#L656)；API [`gpu_evaluator.cpp:5462`](https://github.com/SL640/FHE_GPU_Agent/blob/55e51804fb6ade2c420c75d3d2f3c727eb52e6de/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L5462)；本版`source_runtime.hpp`。

### 5.2 双lane配对自举：当前profile不能直接替代原双输出契约，配对适配尚未实现

- **Cerium：** 向量自举接受两路，包装函数将两路按其特定表示合并，底层函数返回`z0`、`z1`两个输出。它不是简单把两次独立自举放在一个列表中。
- **本版：** 每个应用输入密文独立刷新，未复制这种配对和拆分；因此应用层自举调用数量不同。
- **为什么没有直接照搬：** 当前S2C-first路径的末尾会将结果与其共轭相加后返回单个输出。直接把两路数据当作任意实部/虚部合并后调用，并不能保证两路都被保留并正确恢复。必须重新适配配对布局、变换和拆分契约。
- **是否完全做不到：** 不能这样断言。库内部本身有实部和虚部分支，其他路径也有复数相关接口；缺少的是本版与选定ResNet20 profile兼容、且已验证的双lane应用适配。不能说整个库“不支持复数”或“不支持自举”。

依据：Cerium [`bootstrap.py:760`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/benchmarks/llama/bootstrap.py#L760)；当前S2C-first输出合并见API [`gpu_evaluator.cpp:5656`](https://github.com/SL640/FHE_GPU_Agent/blob/55e51804fb6ade2c420c75d3d2f3c727eb52e6de/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L5656)。

### 5.3 线性层累加、旋转及Rescale顺序：主要是当前适配未完成，不能说库绝对做不到

- **Cerium：** 在输入tile外准备可复用的baby旋转；跨输入tile累计分组结果，再统一做giant旋转和后续处理，保留较大的共享计算范围。
- **本版：** 对每个输入/输出tile及分支调用一次`dft_double_hoist`，每次调用完成本矩阵的旋转、乘加和Rescale后，才在外部累加。所选固定模式至少做一次Rescale；本版明确设置step=1。这也没有完整复制跨输出tile的旋转复用范围。
- **为什么这一版这样做：** 为先得到可执行的API版本，选用了ResNet20已使用的封装入口，并保持小范围工作区；它一次处理一个输入密文和矩阵组，不直接暴露原来跨多个输入tile的累计状态。这是适配选择及尚未完成的实现，不是证明原顺序不可实现。
- **核查后的更正：** 库底层`multiply_by_diag_matrix_bsgs_double_hoist`支持`rescale_count=0`；相关ResNet20验证代码也有`multiply_by_diag_matrix_bsgs(...,0,...)`再单独Rescale的示例。因此，早先若说“接口完全无法延后Rescale”，表述过强。
- **还需要什么：** 接入并验证不立即Rescale的路径，匹配累加时的Q/scale，恢复跨输入分组累计和可复用旋转的生命周期；只把Rescale参数改成0，仍不足以证明整个BSGS顺序已经与原版一致。这些工作尚未在source15实现和测试。

依据：Cerium [`llama_cerium.py:85`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/benchmarks/llama/llama_cerium.py#L85)；固定高层路径 [`gpu_evaluator.cpp:5165`](https://github.com/SL640/FHE_GPU_Agent/blob/55e51804fb6ade2c420c75d3d2f3c727eb52e6de/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L5165)；底层不Rescale路径 [`gpu_evaluator.cpp:5036`](https://github.com/SL640/FHE_GPU_Agent/blob/55e51804fb6ade2c420c75d3d2f3c727eb52e6de/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L5036)；相关验证示例 [`bootstrap_precision.cpp:65`](https://github.com/SL640/FHE_GPU_Agent/blob/55e51804fb6ade2c420c75d3d2f3c727eb52e6de/gpu_api/Doc/analysis/resnet20_joint_chain_20260911/bootstrap_precision.cpp#L65)。

### 5.4 预编码、缓存重绑定与预取：有条件继续移植，本版采用了有限缓存和流式准备

- **Cerium：** 缓存与其编译后的输入名、partition、remapable内存偏移和执行计划绑定，支持设备内存、UVM或显式复制，并安排预取。
- **本版：** 沿用周期256的紧凑明文准备思路；公开常量按参数ID、精确scale和数值缓存，使用1GiB实数、128MiB复数的保留预算；矩阵流式准备，并预备下一块。没有复制原缓存管理器。
- **为什么不能直接复制原类就使用：** 两边的密文/明文对象、内存指针、编译元数据和分区描述不同，原偏移不能直接指向本后端对象。需要一层输入绑定和对象生命周期适配。
- **是否完全做不到：** 不是。缓存、预编码和预取策略可以继续移植。当前有限缓存是为先完成整模型执行的工程取舍，不能把它说成唯一实现，也没有证据证明A800硬件无法支持更接近原版的缓存策略。

依据：Cerium [`function_io.cpp:675`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/cerium/runtime/execution/function_io.cpp#L675)、[`function_io.cpp:1080`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/cerium/runtime/execution/function_io.cpp#L1080)及[`function_io.cpp:1140`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/cerium/runtime/execution/function_io.cpp#L1140)；本版`source_ops.hpp`、`source_linear.hpp`。

### 5.5 全图调度与CUDA Graph：原实现依赖Cerium编译产物，当前尚未建立等价适配

- **Cerium：** 运行时读取编译后的函数、调用图、内存布局和分区描述，按配置构建CUDA Graph或多线程执行计划；不是仅对几行模型代码套一个Graph开关。
- **本版：** 普通C++按序调用ResNet20 API，以有限的CPU预备重叠GPU执行，保留必要的同步；未接入原调度器或CUDA Graph。
- **为什么不能直接链接原调度器就等价运行：** 原调度器管理的是Cerium生成的函数和缓冲区，不认识当前库对象及其调用生命周期。直接改用Cerium整套生成代码和后端，会偏离“用指定ResNet20 API搭建”的要求。
- **是否完全做不到：** 不是硬件不支持，也不是已经证明这些API不能捕获CUDA Graph。需要为本后端建立图节点/缓冲区适配，检查动态分配、主机同步和捕获兼容性。此项本版未实现、未验证；不能据此承诺移植后能提速多少。

依据：Cerium [`program.cpp:422`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/cerium/runtime/execution/program.cpp#L422)及[`program.cpp:443`](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/cerium/runtime/execution/program.cpp#L443)。这也是可配置功能，不能仅凭源码存在就声称所有原版benchmark都默认启用了Graph。

## 6. 最终评价与证据

当前版本适合交付为**完整执行及性能分析的API benchmark基线**；尚不满足“全部执行细节完全复刻Cerium”，也没有满足精度要求。后续应优先补齐可实现但尚未完成的线性累加/旋转复用与配对接口；缓存和图调度需单独移植验证。任何一项都不能在测试前承诺消除精度问题。

- [完整逐层、level、自举及内存数据](FULL_MODEL_EXECUTION.md)
- [GPU利用率统计说明](GPU_UTILIZATION.md)
- [GPU利用率和显存曲线](GPU_UTILIZATION.png)
- 原始记录：`evidence/model.log`、`model-result.json`、`model-time.txt`、`model-resources.jsonl`。
- 输出SHA256：`2249a7591fd44eefd7334ee326bb99d70bedc4cc3777bb1cfc1701b0ab5dc3a6`。
- Cerium固定版本：`6b487041f69bcd04fb2138b8be87635ddeb128d6`；FHE_GPU_Agent基础版本：`55e51804fb6ade2c420c75d3d2f3c727eb52e6de`。

本报告仅整理现有服务器结果和源码核查；没有因编写报告再运行模型或启动新的GPU测试。


## 7. 本分支交付范围与复现

本目录是source15已完成测试版本的独立交付，C++文件与冻结测试包逐字节一致。
分支基于实测后端提交`55e51804fb6ade2c420c75d3d2f3c727eb52e6de`，
没有将远端main后来的后端改动混入本次测试结论。源码、公开参数、输入、
参考输出、权重版本锁及选取的原始日志随目录保存；约16GB模型权重、后端
二进制和秘密密钥不在本目录中。旧版benchmark分支保持原样。

本次Git交付仅新增可选CMake入口和README，未重跑模型。下方仓库集成入口
未另做构建验证；已完成的服务器测试使用相同C++及记录在证据中的构建wrapper。
请沿用gpu_api/README.md中的依赖配置，并在GPU服务器执行编译与测试。

```bash
cmake -S gpu_api -B build/llama31-source15 \
  -DPOSEIDON_BUILD_LLAMA31_CERIUM_SOURCE15=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
cmake --build build/llama31-source15 --target llama31_cerium_source_api -j 7
```

上述命令需与服务器既有的依赖路径/CMake选项合用。运行参数与输入格式如下。

## Build in the existing API repository

Use SL640/FHE_GPU_Agent with its ResNet20 GPU API (tested base55e51804).
Enable POSEIDON_BUILD_LLAMA31_CERIUM_SOURCE15=ON in the existing gpu_api
CMake configuration, then build target llama31_cerium_source_api. CUDA, RMM and Poseidon come from that
repository. This package does not use a DSL compiler or an external FHE backend.
The portable CMake target links the same API; validation used the recorded
standalone-server CMake wrapper and binary from source15, built from identical
source13 C++ files. The original queued jobs were held before execution.

## Run

```text
llama31_cerium_source_api FIXTURE WEIGHTS INDEX EMBEDDINGS REFERENCE OUTPUT BLOCKS VALID_TOKENS REPORT_JSON
```

Use BLOCKS=32 and VALID_TOKENS=38 for the full tested boundary, with the matching
reference-block32.bin. BLOCKS=1 requires reference-block01.bin. Inputs use the
checked binary/index formats from the existing benchmark. The tested embeddings,
checked index and both reference files are included in inputs/ with verified
SHA256 hashes and their original preparation reports. These preparation reports
describe input generation; MODEL_RESULT.json describes the GPU execution.
Existing output files are rejected. Checkpoint weights, backend binaries and
secret keys are not distributed in this source package. Weights are loaded one
block at a time. The index is tied to the exact checkpoint shard bytes and must
not be reused for a different checkpoint revision without regeneration.
WEIGHT_DOWNLOAD_LOCK.json identifies the tested ModelScope checkpoint files,
per-file revisions, sizes and SHA256 hashes; download the weights on the test
server and verify those hashes before using the supplied checked index.
Set FIXTURE to inputs/r20_fixture.txt, the exact public ResNet20 fixture verified
against the runtime preflight; source_runtime.hpp applies the documented Q49/P25
interface adaptation. The fixture contains public parameters, not secret keys.

Tensor files contain two UTF-8 header lines, followed by exactly 128*4096
little-endian float64 values in token-major order. Line1 is
LLAMA_CERIUM_TENSOR_V1; line2 is "embeddings 128 4096 0" for the input,
"hidden 128 4096 1" or "hidden 128 4096 32" for the matching reference/output.
The runtime validates these headers before executing. The checked TSV index
starts with LLAMA31_HF_SAFETENSORS_V1 and contains tensor name, shard filename,
dtype, absolute byte offset, byte length, rows and columns in that order.

## Provenance and remaining differences

Cerium source is pinned to 6b487041f69bcd04fb2138b8be87635ddeb128d6.
The original nonlinear DAGs, coefficients, layouts and explicit boot positions
are ported; ResNet20 native bootstrap uses different physical primes/scales.
Native DFT rescales per input tile; bounded constant caching and streaming are
backend adaptations. Original dual-lane bootstrap packing, full graph scheduler
and CUDA Graph have not been copied. See SOURCE_MAPPING.md. Reference outputs
are Cerium original_callpath_v2 float64 approximate packed calculations with
fixed jitter, not the exact Hugging Face model or a per-instruction float mock
of this C++ DAG.

The original source15 release had unchanged Source13 C++ bytes; its retained
RELEASE_MANIFEST.json records that historical baseline, not this modified
performance branch. Current source hashes and experiment results are recorded
under experiments/20261114; see the banner above. All GPU measurements ran on
the server; the optimized whole-model target is compiled but not newly validated
for 32 blocks.
