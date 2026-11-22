# 按 Cerium 源码顺序移植：组件与完整模型

需求：2026-11-13 10:59，全部照抄 Cerium，Poseidon 接口层按 ResNet20 替换。此目录是独立的新移植版；原 `cerium-v2` 不能作为已经满足此要求的完整版本。

**2026-11-13最终状态：** source15已于16:20完成真实权重、38有效token/128容量、连续32个Block密文执行，退出0；计算7558.47秒，9856次自举，RMM活跃峰值21.50GiB。精度未通过，最大绝对误差约955万，不能作为可靠推理结果。C++与source12/source13逐文件SHA相同。原集群source13/14任务在执行前hold，不能算作已执行。详细结果及各项差异原因见 `TEST_RESULTS_AND_CERIUM_GAPS.md`（开发目录中位于上一级；交付包内与本文件同级）。下文“第一组”“11:50”等段落记录实现过程，source11仅完成一个Block，不能代替这次完整测试。

## 固定来源和实际参照

- Cerium：`CMU-CAOS/Cerium@6b487041f69bcd04fb2138b8be87635ddeb128d6`。原始文件在 `../../cerium_upstream_20261113/benchmarks/llama/`，不是浮点 Mock 的实现替代。
- 可移交的固定源码链接：[Cerium Llama密文实现](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/benchmarks/llama/llama_cerium.py)、[输入与系数准备](https://github.com/CMU-CAOS/Cerium/blob/6b487041f69bcd04fb2138b8be87635ddeb128d6/benchmarks/llama/llama_runner.py)。上面的本地相对路径只适用于开发工作区。
- 后端：`SL640/FHE_GPU_Agent@55e51804`；本次沿用已校验哈希的同一 GPU 库，应用代码重新构建。
- 实际 ResNet20 入口：`gpu_api/bench/resnet20_s2c_first/main.cpp`，包含 `Doc/analysis/resnet20_joint_chain_20260911/bootstrap_network_precision.cpp`。
- 自举参照：上述文件 `ContinuousBootstrap::prepare_library()`、`run()`，直接调用 `GpuBootstrapProfileBuilder` / `GpuEvaluator::bootstrap`。乘法、重线性化、Rescale 参照 `relu_precision.cpp` 的 `multiply_q_prefix`、`relinearize`、`rescale_many`；不调用旧网络 scheduler。

## 本组已写出的移植

| Cerium 源位置 | 本地函数 | 保留内容 |
|---|---|---|
| `llama_cerium.py:353` / `:398` | `g3_f3` | x²/x⁴共享幂次；x¹/x³/x⁵/x⁷各分支；相同相加次序；共轭；单/双 Rescale 和各分支 ModDrop |
| `llama_cerium.py:1539` | `sign_new2` | g3、g3、f3、f3_new，前三段与末段的不同 Rescale 顺序 |
| `llama_cerium.py:1581` | `silu_internal` | 原自举位置，三条 Sign 分支，原 P/Q 系数，v²/v⁴/v⁶，先 wP/wQ/wv 再门控，最终三项乘积相加后一次重线性化 |
| `llama_runner.py:get_*_inputs` | `source_ops.hpp` / 各公式的 scale 求解 | 明文数值与物理编码 scale 分开，按实际模数乘积求解，不篡改密文 scale 元数据 |

`source_probe.cpp` 仅在服务器执行实际 GPU CKKS；同一组运算全部结束才解密。误差对原系数的独立明文多项式计算记录，精度门槛仅记录。Sign 和 SwiGLU 组件通过不等于完整模型通过。

## 必要接口适配及尚未照搬的部分

1. **自举实现替换**：调用现有 ResNet20 S2C-first degree-22 原生实现，没有应用层正弦逆修正、经验除数、重新拟合、输入中心化或隐式自动自举。
2. **物理参数**：沿用此前实测的 N=65536、Q49/P25、接口 scale=2^40。Q49 是 ResNet20 fixture 去掉索引8的既有适配链，不是原始 Q50，也不是 Cerium 的原始素数。每项 CP 编码 scale 根据真实素数和实际输出 scale 计算；不得将这个差异说成原参数照搬。
3. **共轭后的倍数**：原 runner 将系数编码 scale 减半后以声明输出 scale 解释结果；本后端直接追踪物理 scale，因此系数数值减半并保持正确的公共 scale。编码系数及最后代表的原多项式不变。不是新的拟合或数值修正。
4. **接口输出转换**：SwiGLU 原函数返回高 scale 的原始乘积；当前调用契约为 scale40，末尾增加真实双 Rescale，单独记录消耗。
5. **未移植的向量执行**：本组仅实现单 lane。原 `MakeVector/BreakVector` 的两路执行尚未映射，不声称复制其合批、自举次数或性能。完整32层入口已由source15执行完成；配对缺口仍然存在。
6. **其余模块和调度**：原多GPU路径不是单GPU可直接执行的路径，本次首先对齐原 `llama_blocks_1gpu`。新增模块状态和接口差异见下节；之前候选不能直接冒充已完成。

## 当前测试边界

初次探针：1个 Sign 输入密文，1个 SwiGLU gate/up 密文对，32768槽；仅SwiGLU原始入口处1次真实自举。输出真实误差、物理Q/scale、时间及RMM峰值。任何失败保留原日志后另开修复轮次，不覆盖冻结运行。

## 11:50 后续完整模型接入（source11已完成单Block验证）

- `source_rms.hpp`：原平方求和、旋转归约、selector、四次Rescale、原指数种子、10步Newton和每两步自举、gamma与输入相乘。源逻辑边界10→6→2适配为本后端14→10→6。仍单lane刷新，输入16次而非原配对8次。
- `source_softmax.hpp`：原7轮Max+自举；指数前5次平方→自举→后2次平方；加密因果mask；归约、自举、6次Newton、平方得到倒数。保持输出2倍概率，原1/(2*1.0001)留在AV掩码里。当前每lane13次自举。
- `source_attention.hpp`：两个Q复用K的baby旋转、四个概率矩阵复用V的对角准备；保留QK/AV原乘法、求和、重线性化和Rescale顺序。尚未复制原实/虚双lane打包。
- `source_linear.hpp`：原双分支权重对角、周期256、BSGS尺寸、9个复数RoPE融合mask、旋转后共轭。调用ResNet20 `bootstrap_network_precision.cpp:552`实际使用的 `dft_double_hoist`。**本版所选固定step=1的高层调用在每个输入tile内完成Rescale，尚未保持原跨输入tile先累加再统一处理的顺序与旋转复用范围**；适配版明确打印 `per_input_rescale=true`。底层矩阵接口已有`rescale_count=0`路径，不能说整个库无法延后Rescale；尚需适配累计状态、Q/scale和旋转工作区。保留一个GPU矩阵和一个CPU预备矩阵，6线程只准备公开权重。这是本版流式适配选择，不是复制了Cerium全图缓存、重绑定、数据搬运或CUDA Graph。
- `source_model.hpp` / `source_benchmark.cpp`：按原 `llama_blocks_1gpu` 串联全部子层，入口接受1..32 blocks、128容量/38有效token，真实权重逐层读取，最终block完成后才解密；无最终Norm/LMhead。新入口不引用旧GpuOps精度策略或自动自举调度器。

`source10`在服务器编译阶段发现 `CKKS` 枚举命名空间错误，未进行计算，原日志保留；工作副本已修正为库真实定义。source11随后在服务器重新编译并完成组件和真实权重完整单Block，退出0。精度只记录，不作为推进门槛。

source11冻结后，继续核对 `attention()` 的阶段顺序发现：该包按每组KV执行QK→Softmax→AV，而原源码先做全部QK、全部Softmax、再全部AV。工作副本 `source_model.hpp` 已恢复原三阶段顺序，供下一冻结版本使用；不能把这一改动算作source11已经测试的代码。各算子数学和自举总数不变。

融合RoPE掩码同时参考 `llama_models/models/llama3/model_params.py`，该文件署名Meta并使用Llama 3许可，随工作副本保留 `LLAMA3_REFERENCE_LICENSE.txt`；不能将这一参照文件误标为根目录Apache许可。

工作副本还接入原 `PeriodicPlaintextInput` 的周期常量准备与重复常量复用：实数向量严格逐值检查周期256，使用probe07已验证的紧凑FFT/NTT映射，再重构现有CP接口要求的全Q明文；系数超出int64范围时回到原全尺寸编码，不做近似匹配。缓存按完整参数ID、精确scale和全部公开值匹配。原版按输入名/partition缓存的绑定在此变为值身份查找；1GiB实数、128MiB复数的保留预算及LRU是后端内存适配，不是原缓存管理器的逐行复制。AV同Q/scale掩码跨V组复用，复数RoPE掩码跨输出tile复用；不缓存密文内容、不改变乘法/Rescale/自举计划。这些变化冻结于source12/source13/source15同一份C++，已由source15完成完整模型执行，精度未通过；source13/14被hold而未执行，source11结果只对应其原冻结版本。
