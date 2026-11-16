# Poseidon GPU 密钥准备/加载慢：诊断与解决方案

> 本文保留初次调查的基线数据；后续已实现首次QP准备优化，见 `docs/gpu-qp-first-start-fix-20261115.md`。

日期：2026-11-15。基线：Poseidon `ddcd96cd597d6b6522728914ee8c27eabf32de0b`；独立分支 `investigation/gpu-keyload-20261115`。Cerium 源码基线 `6b487041`。

## 1. 结论摘要

本报告将日常所说的“GPU 密钥加载”拆成 CPU 密钥生成、GPU 参数构建、密钥格式转换/H2D、native Boot profile 的常量准备四类，不能用一个 setup 数字替代它们。

**核心判断：不能把数百秒启动时间直接归咎于磁盘或 PCIe。**

- Poseidon 的 Galois key generation 在 OpenMP 外层并行区域内，用一个 `critical` 包住完整的单 key 生成。主要工作被串行化；等待线程还可能消耗容器 CPU 配额。
- GPU API 默认逐层构建七类完整 NTT root arrays。Q50/P25、N65536 时，仅这七类数组就是 **4,771,020,800 B，4.443 GiB/卡**；这是真正的逐 level 二次展开，但它是**参数表**，不是逐层重复上传整套密钥。
- 当前 ordinary keys 已按每个 rotation 的最高所需 Q 实物裁剪并跨层共享。不能将已经修复的旧 key-copy 问题当成本次根因。
- **本次真实 native Boot profile 准备 262.118s 中，C2S/S2C 的 QP 明文常量转换/压缩/上传合计202.228s，占77.15%，才是该组合阶段的头号瓶颈。** native Galois CPU keygen 32.014s；relin+Galois 上传含预旋转仅1.677s。不能把常量准备称为密钥 H2D。
- Cerium 的密钥生成输出及磁盘格式原生就是 `uint32` NTT limbs；直接读进 pinned pool 后异步上传，免去 Poseidon uploader 的 `uint64→uint32` packing。Cerium 内置 `Average Execution Time` 完全不计上述准备过程。

**优先解决路径：先修计时边界；对本次主要慢项，优先离线预生成/缓存 GPU-ready QP 压缩常量，或让编码器直接保留精确整数并生成 QP，消除启动时重复 CRT 往返。其次优化密钥生成和线程配置，再处理逐层参数 root 表复制。单纯加快 PCIe/key copy 无法解决当前数百秒问题。**

## 2. 实验环境与范围

- 本次独立目录：服务器 `/root/private_data/poseidon-keyload-20261115`；本地 `poseidon-gpu-keyload` worktree。原 `poseidon` 未被修改。
- GPU：NVIDIA A800 80GB PCIe LC，UUID `GPU-e66fc2a8-85c9-46af-52a3-042a0a74e6e9`；驱动 580.95.05；Poseidon 使用 CUDA 11.8，SM80。
- CPU：Hygon C86 7390，64 个可见 CPU；**cgroup `cpu.max=700000 100000`，实际仅 7 核配额**；memory.max 约 245 GiB。正常对照绑 CPU0–6，单列不绑核默认线程诊断。
- 每个样本是独立新进程；启动前记录 GPU PID/显存、主机、CPU quota、实际 affinity 与线程环境。未 reset GPU，未 drop page cache，不能称为硬件/物理磁盘冷启动。
- 所有密钥为本次合成案例新生成；不读取其他实验的私钥或 key/PT/CT 数据。历史实验仅引用其公开参数和报告统计。
- 真实 native 探针使用已存 ResNet20 的公开 Q50/P25 模数链（Q 为30/31/32-bit混合，P 为32-bit）；普通 Poseidon/Cerium 对照均为28-bit、Q32/P16/K8，但素数具体值与随机采样实现不同，只是等形状成本对照，不是同一密码实例或等安全性证明。
- 密钥上传以 CPU 源和 GPU 回读直接比较；该验证在 upload 计时之外。参数全字 fingerprints 是一致性回归，不是密码算法或完整 Boot 数值正确性的证明。
- 本次 native profile 探针只完成真实 profile 准备，不执行 Boot；微基准的 `sec_level=none` 不能作为生产安全参数推荐。

## 3. 本次实测

### 3.1 真实 ResNet20 native Boot profile：262秒花在哪里

同一真实 builder、Q50/P25、N65536、OMP1、Global special-P；不包含 GPU parameter API 构造，未执行 Boot。单次 profile 准备262.118s，完整进程269.276s，其中 CPU context5.299s。

| 阶段 | 秒 | 占 profile |
|---|---:|---:|
| C2S QP diagonals 精确转换/压缩/上传/验证 | 186.252 | 71.06% |
| S2C QP diagonals 同上 | 15.976 | 6.10% |
| native Galois CPU keygen，39个键（含所需共轭键） | 32.014 | 12.21% |
| CPU DFT matrices 生成/编码 | 16.212 | 6.18% |
| EvalMod 常量编码与上传 | 8.873 | 3.39% |
| native relin CPU keygen | 0.892 | 0.34% |
| Galois 上传及 inverse-pre-rotation，2.856GiB payload | 1.635 | 0.62% |
| relin 上传，75MiB payload | 0.042 | 0.02% |
| 其余多项式/metadata/常量 | 约0.218 | 约0.08% |

key uploader 自身 trace：Galois packing1.280s、copy+wait0.299s；relin packing0.0326s、copy+wait0.0077s。copy+wait 包含 pageable staging/CUDA等待，仍不能称作纯 DMA event 时间。

第二个独立新进程（加入更细的 QP trace）复测 profile **262.355s**，与第一轮仅差0.09%。两组各158个 diagonal，细分之和如下：

| QP常量子阶段，C2S+S2C | 秒 |
|---|---:|
| **CPU exact Q→QP：INTT+CRT compose+centered modP+NTT** | **195.135** |
| 周期搜索 | 3.379 |
| compact packing | 2.342 |
| device allocation+H2D+wait | **0.166** |
| compact D2H及CPU全展开精确验证 | 1.097 |

实际只上传927,766,528B的压缩常量；对应未压缩uint32 QP payload4,307,550,208B。195.135s可进一步拆为C2S181.023s、S2C14.112s。**至此直接证实主要是CPU精确基扩展计算，而非H2D**；尚未对 INTT/CRT/modP/NTT 四个内部算法逐项计时，不能把195s全部指定给某一个循环。

注意下载的是 *compact allocation*，随后CPU按完整N展开验证；不是把未压缩4.3GB全部从GPU下载。

### 3.2 等形状 Poseidon/Cerium evaluation keys

N65536、28-bit、Q32/P16、d=2、8个普通左旋键+1个relin，均为**18个digit-key pairs，GPU payload432MiB**；固定同一GPU与7核affinity。各跑3个新进程，均通过全字回读比对；Cerium临时文件均确认删除。

| 阶段 | Poseidon中位数，秒（范围） | Cerium中位数，秒（范围） |
|---|---:|---:|
| CPU Context | 2.346（2.343–2.351） | 0.330（0.330–0.334） |
| SK/PK或secret-RNS准备，不同对象边界，不直接算倍率 | 0.374（SK+PK） | 0.281（secret/ephemeral RNS） |
| **evaluation-key CPU生成：relin+8 rotations** | **4.788（4.778–4.799）** | **0.888（0.861–0.895）** |
| 内存源key→GPU：Poseidon含pack/alloc/wait；Cerium只列pinned H2D | 0.260（0.257–0.263） | 0.031（0.031–0.058） |
| 额外序列化写盘，无fsync | 未执行 | 0.387（0.385–0.404） |
| 额外warm-cache读取到pinned，含pin alloc | 未执行 | 0.292（0.292–0.293） |
| pinned分配，上行读取阶段的子项，勿再相加 | 未执行 | 0.200（0.200–0.201） |

Poseidon一轮代表trace的packing约0.201s、copy+wait0.048s，与总upload约0.260s闭合；Cerium避免这项64→32转换，其CPU key payload也仅432MiB，而Poseidon为864MiB。

解释必须分口径：

- 同形状CPU evalkey生成，Cerium在这里约**5.39倍快**；其按digit-key并行、uint32原生表示和共享secret预计算，与Poseidon的整key critical路径形成实质差异。不能把倍率全部归因于单一因素，也不能推广为整网倍率。
- Cerium若已经有pinned keys，H2D中位31ms；但若从本次写出的文件重新load，**读取+pin分配+device分配+H2D中位0.325s**，并没有比Poseidon“已有CPU keys→GPU”的0.260s更快。因此不是“Cerium任何密钥加载口径都更快”。
- 不含context和验证，Poseidon现场evalkey生成+upload约**5.047s**；Cerium现场生成+序列化+warm reload+upload约**1.615s**。此处Cerium还多做了文件往返；这是等形状合成probe，不是完整Boot程序/等安全性测评。
- 本次Cerium链接已存在的公开runtime，使用CUDA12.2 runtime；Poseidon是本次从源码构建的CUDA11.8/SM80。两者同驱动/同卡，但runtime版本未严格统一，故不将31ms vs48ms等小差异解释为纯实现因果。

### 3.3 参数表：已实测可用的 sparse 构造

相同 N65536/Q50/P25、K8、OMP1，两个独立进程；只改变 `GpuParameterData` 的层集合。此处是单样本 proof-of-concept，不是完整 ResNet 等价加速。

| 参数构造 | 全链 | sparse Q={1,9,34,50} |
|---|---:|---:|
| level 数 | 51，含 key context | 4，Q-only |
| 七类 roots 实际 allocation | 4,771,020,800 B（4.443GiB） | 355,991,552 B（0.332GiB） |
| 全部已统计 parameter buffers | 4,771,986,824 B | 356,063,008 B |
| CPU构建+device allocation+upload | 8.956s | 0.788s |
| parameter阶段 cudaMemGetInfo 差值 | 5,110,759,424 B | 398,458,880 B |

roots payload 减少92.54%，该构造阶段约11.37倍快；显存 driver差值还含 allocation粒度等，不能拿它替代精确 payload。4个共同Q层、28类 arrays、355,991,552字节的全字 fingerprints 一致。此验证不是完整Boot运行回归，也不意味着生产模型只需要这4层。

### 3.4 线程假说：本次没有证实是数百秒的原因

同一 OPENMP=OFF 构建（仍有全局-fopenmp和未保护的Galois pragma）：OMP1的 Galois generation 中位4.217s；OMP7/PASSIVE4.174s，OMP7/ACTIVE4.179s；不绑核且默认64线程4.293s；显式64/ACTIVE且绑7核4.207s。除OMP1为3样本外，其余各1样本。

因此应记录 quota、避免超配，但**本次数据没有显示线程配置导致数量级慢化，也没有证明只改 OMP_NUM_THREADS 就能解决启动问题**。OpenMP ON 的 guarded 内层是另一构建，不在这组结果中；其并行调度不能由此推断。

## 4. 为什么慢：按实际代码路径归因

### 4.1 CPU 密钥生成被混称为“加载”

`src/poseidon/key/keyswitch_base.cpp` 的 `create_galois_keys`：外层 `#pragma omp parallel for`；每个 rotation 对 secret 做 automorphism 后，整个 `generate_one_kswitch_key` 在同一个 `#pragma omp critical` 内执行。HYBRID 每个 key 有 `d=ceil(Q/P)` 个 digits，每 digit 生成两个 QP 分量并做 CPU NTT/模乘加。

设实际非空旋转键数为 K，N 为 degree，L/P 为 Q/P prime 个数：

- CPU key payload：`16 × (K+1) × ceil(L/P) × N × (L+P)` bytes，`+1` 是 relin。
- 完整 GPU key payload：CPU payload 的一半，因为 GPU residues 是 uint32。
- 固定 P 时，digit×limb 本身随 L 近似二次增长；这不是按 level 复制 key。
- `galois_keys.data().size()==N` 是 sparse slots 容器大小，不是 N 个实际密钥；统计应只计非空项。

顶层 CMake 即使 `POSEIDON_USE_OPENMP=OFF`，仍全局加 `-fopenmp`，上述没有宏保护的 Galois pragma 仍生效。仅看 CMake 选项 OFF 无法证明“密钥生成完全单线程”。服务器的64-thread可见性与7-core quota 需要显式控制；但如3.4所示，本次对照未观测到数量级慢化，不将其当作实测主因。

### 4.2 ordinary GPU keys 当前已经按需裁剪，不能错判为逐层重复

`poseidon_gpu_api.cpp` 的 preflight 按每设备、每实际 Galois element 聚合最高 required Q；relin 同样取最高 Q。materialization 按 Q 分桶，每个 element 只属于一个最高层 bucket。`gpu_uploader.cpp` 保留 `Q[0,q) | 原 P`，只上传 `ceil(q/P)` digits。低 Q operation 使用零拷贝 view；P 的存储偏移仍由物理 storage Q 决定。

对 N65536/Q50/P25，单个 full key 75 MiB；q9 的 cropped key 17 MiB；q34 为59 MiB。**这些是当前基线行为，不需要重新实现一次“按 level key 共享”。**

但有两个剩余问题：

1. CPU 端通常仍先生成完整 context 的 keys；GPU crop 不会节约已经花掉的 CPU keygen 时间。
2. 后续 preflight 若抬高某些 key 的 Q，目前会清空并重建全部 combined buckets。多 plan 逐步扩充需求时应测试增量更新，避免重传未改变的 keys。

### 4.3 uploader 的“upload”包含 CPU packing、分配和逐 component 等待

`gpu_uploader.cpp::upload_ciphertext_component_shards` 每个 component/shard 新建 pageable `std::vector<uint32_t>`，逐元素检查并转换 CPU uint64，再 `cudaMemcpyAsync`，随后立刻 `cudaStreamSynchronize`，以保证临时 host buffer 的生命周期。

标准单 shard、D 个 digits 对应 D 次 device owner allocations、2D 次 host packing/H2D/wait；不能把这个 API 的总时间称为纯 PCIe copy。native inverse-pre-rotated keys 另有 GPU automorphism、D2D 写回和四张 pointer tables；这也不是磁盘读取。

初次调查新增 opt-in `POSEIDON_GPU_KEYLOAD_TRACE=1` 诊断：记录 key/digit/shard/bytes、packing wall、copy+wait wall，不输出 key 内容，不改变 key packing 或同步算法。后续QP常量生成已另外实现优化，见独立修复报告。

### 4.4 真正的大规模重复：逐层 GPU 参数表

`GpuParameterData(context,device)` 默认从 key context 遍历完整 Q-only chain。每层分别生成、分配、上传七类 `(q+P)×N` uint32 tables：NTT、INTT、fourstep INTT、normalized INTT 及三个 Montgomery 变体。Q prefix 和同一 P 在多层反复出现，即使最终运行只选择一种 NTT kernel。

七类 root arrays 的全链静态字节数：

```
28 × N × [(L+P) + L(L+1)/2 + LP]
```

Q50/P25/N65536：4.443 GiB；若只存一套最大 QP roots，则同七类理论仅131.25 MiB。两者相差约34.7倍，后者是**设计上限分析，不是已经实现的生产优化**。

此外每层 `q×q` rescale inverse table 累计 O(L³) 小表工作；HYBRID base conversion 小矩阵也逐层重建。可选 tensor/fp64 fused matrices 还能额外放大 setup，比较时必须固定 `POSEIDON_NTT_*` 等环境。

已有 sparse `active_q_counts` 构造可跳过无用层，但普通 API 默认未使用它。仅当编译计划保证所需层集合完整时才能使用 sparse；不能把实验选的少量 Q 层直接套到所有 ResNet/Boot。

### 4.5 native Boot profile 不是普通 key uploader

`gpu_bootstrap_profile.cpp::build_resnet20_manual` 包括：生成多项式、CPU DFT 矩阵/编码、EvalMod metadata/常量编码上传、native relin/Galois CPU keygen、inverse-pre-rotation、Q→QP DFT diagonals 打包上传，以及其他常量。native keys 与 ordinary API keys 是两条准备路径；布局不同，不应不加区分地 alias GPU owner。

本分支在这条真实生产 builder 中增加分段标记 `BOOT_SETUP_PHASE`，而不是用一个小上传测试推断整段 profile 的原因。非默认 level-special-P 模式还有先建 global keys 再建 grouped overrides/替换 owner 的额外准备，应单独优化；本次默认 Global。

## 5. Cerium 为什么相对快，以及它没有做什么

源码：`reference/cerium/cerium/runtime/io/generator.cpp`、`io/ckks-encryptor.cpp`、`execution/function_io.cpp`。

| 环节 | Poseidon 当前路径 | Cerium 路径 |
|---|---|---|
| CPU keygen | Galois 重工作在 critical 内 | 64 个 std::thread，thread-local encryptors、共享预计算 secret RNS/NTT |
| CPU/磁盘表示 | CPU key 为 uint64；upload 再 pack | 原生 uint32 NTT limb 数据，直接序列化 |
| host load | 模型 runner 常现场 keygen，非简单读盘 | ifstream 顺序读到一个 pinned host pool，limb alias 该 pool |
| H2D | 各 component 打包后立即 wait | 每卡一个输入大 pool，逐 limb async copy，末尾统一 wait |
| 内置执行计时 | 需看 runner 字段边界 | run Average 完全排除 keygen/serialize/load/H2D/graph创建 |

关键限制：

- Cerium 是 **CPU keygen**，不是“GPU 生成密钥所以快”。其硬编码 64 worker 在 7-core quota 上也不理想，不能盲目照抄线程数。
- 它不是 mmap key 文件、不是 GPUDirect Storage、不是 GPU 零拷贝读 host key；文件也不是 seed-only 压缩密钥。
- 不同 level/ext/digit partition 进入 key identity，不能声称 Cerium 所有层共用一套最高层 keys；当前 Poseidon ordinary 的 max-Q sharing 反而是已实现优势。
- 标准单卡 bootstrap 编译清单可静态清点为780条引用、195个 digit-key pairs、84个逻辑组、49种 rotation，payload **5,482,479,616 B（5.106 GiB）**。这只是静态 inventory，不是本次整 Boot 运行测量。
- Cerium 固定 N65536；“33 levels”是其 bootstrap 消耗深度标签，不等于只使用 Q33。不能只按这个名字与 Poseidon Q50/P25 比成本。
- 刚生成即读取通常命中 page cache；ofstream close 不等于 fsync。报告不将 warm-cache read 描述成物理磁盘冷读。

## 6. 解决方案与实施顺序

### 首要方案：把 QP 常量准备从每次启动移出

当前压缩路径先对完整 Q 做 INTT，CRT compose 出每个 coefficient 的多字整数，按 Q/2 判符号后逐个 P 做精确整数约简，再对 Q 和 P 做 NTT；随后从1开始搜索周期、构建 compact buffer，上传后还要 compact D2H 并在CPU全展开逐元素验证。压缩虽节省显存，但**在压缩前已经完整计算了大多数字**。

每个diagonal的host复杂度为 `O(N[(2q+p)logN + q² + qp])`。`basics/util/rns.cpp::compose_array` 每个系数要执行q次q-word乘/模加，形成q²项；`modulo_uint` 对每个P一律扫描q个word，即使真实整数magnitude只有1–2个word，也仍做q−1次128-bit约简。两组均158个diagonal，但C2S在q=50/49/48，S2C在q=6/5/4，源码结构解释了C2S为什么远慢于S2C；这不是diagonal数差一个数量级。

低风险首批改造可具体落在：保留原canonical Q-NTT，删除Q forward NTT重做（通用helper须检查/归一化非canonical输入，不能原样复制lazy残基）；每系数先求真实magnitude有效字长w，modP从O(pq)改为O(pw)，保留center判定；完整host周期证明后，用device_compact与compact逐字比较替代CPU全N展开比较；成功周期不重复扫描，复用bit-reversal表。前三项均需以当前exact实现为oracle，不因“看起来等价”免除逐字回归。

最稳妥的第一版方案是离线 prepare 后保存**已经验证的 compressed QP uint32 常量**，索引绑定完整输入内容/模数链/scale/level/N/周期布局/生成版本，启动时做长度和 checksum 校验并直接加载上传。它与 secret key 无关，可独立于密钥缓存部署；仍需使用可信可写目录和严格工件校验。缓存首次构建成本仍在，不能在报告中隐藏。

第二版改编码链：在 `ckks_encoder.h` 的vector encode、**同一次FFT/rounding/大整数分解**之后，保留实际BSGS pre-rotated diagonal的有符号整数系数，直接精确生成Q/P limbs；保留原Q容量/centered范围检查和实际plaintext scale（manual常量为2^45，不是应用2^40），覆盖64/128/multiword分支。不要重跑浮点FFT或decode/re-encode，更不要先编码Q→NTT，再为了补P做INTT→CRT→NTT。保留现有exact实现作为reference，逐字比较Q与P。有限host并行可以按diagonal分工，但必须各自持有qp/composed/magnitude/CRT scratch；不能给共享temp_mpi的compose内部循环直接加pragma。GPU上传仍由同一producer执行，避免worker未继承thread-local stream/resource而逃离调用者生命周期。

**不能用普通 approximate/fast ModUp 替代精确 centered Q→P。** 它可能得到 x+kQ，破坏 double-hoist 延迟 ModDown 的明文乘法语义。也不能仅因 GPU 上压缩周期短，就在未经证明的情况下只对短周期做 CRT；周期是在 NTT/bit-reversed 布局中的性质。

收益边界：如果只消除这202.228s、其它阶段不变，262.118s 的 profile 理想下限仍约59.89s，理想加速约4.38倍；这是 **Amdahl 上限分析，不是缓存已实现或已测的加速**。

### P0：立即可做、低风险

1. **规范计时**：固定输出 `cpu_context / sk_pk / relin_keygen / galois_keygen / gpu_parameters / key_pack / key_h2d_wait / native_constants / graph_compile`。字节数和非空 key/digit 数一起记录；所有 GPU 分段完成同步后结束计时。
2. **显式线程预算**：以 `cpu.max`、affinity 和任务数决定线程，不用 lscpu 数。对当前 Galois critical 路径优先用单线程作为可靠基线；服务正常并行路径的线程数不超过7。`OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`，关掉不必要 BLAS 嵌套线程。具体收益以本报告实测为准，不假定 OMP7 必然快于 OMP1。
3. **复用已准备 context/profile**：一个服务进程、同 context/key identity 的多个请求不要重复创建 GPU API 或 native profile；将启动成本和在线请求成本分报。不能用长驻模式掩盖必须优化的首次准备时间。
4. **利用现有 sparse parameters**：从 RuntimePlan 的完整需求集合提取 active Q，包含所有 boot 内部步骤及 key-context 特殊需求，再在 preflight 时构建/补齐。不能缺层后静默回退到错误参数。

### P1：真正降低 cold-start 的工程改造

**A. 并行且可复用的 CPU evaluation-key generation。**

- 去掉整 key 临界区的前提是解决共享内存池、随机生成器和可变工作区的线程安全；每 worker 独立 scratch、独立安全 RNG/stream，共享只读 secret NTT，预先去重 rotation，将结果写入互不重叠 slot。
- 不建议直接删除 `critical`：现有结构没有提供足够的并发安全保证，还存在 duplicate element 检查/写入竞态风险。
- ordinary 和 native 共享同一 context/SK 的 CPU key cache，按 canonical key type/rotation/digit partition 去重；GPU standard 与 inverse-pre-rotated 仍分 layout owner。
- 构建系统应给所有 OpenMP pragmas 统一宏保护，让 OFF 的含义真实可测，并避免 outer parallel+inner serial/nested 的意外组合。

**B. GPU-ready evaluation-key 工件。**

- 将 CPU 生成和 GPU packing 从服务启动中移到显式离线 prepare，保存 uint32、NTT、已知 digit partition/layout 的公共 evaluation-key payload；不保存私钥。
- 工件索引至少含 version、N、完整 Q/P primes、key type/rotation、digit partition、physical storage-Q、layout、绑定的 opaque key identity、端序/word size、长度和校验。不能只按 rotation 或参数目录名复用，否则可能用错 secret key 的 keys。
- 加载后用有界 pinned staging pool（例如64–256 MiB、双缓冲）批量 async H2D；通过 event 管理生命周期，最后统一等待。不要简单删除现有 synchronize 后让 pageable vector 先析构，也不要为了模仿 Cerium 一次 pin 数 GiB。
- 明确区分首次 keygen+写盘、warm-cache reload 和真实物理冷读；要求冷读时需另行设计可控实验，不能全局 drop caches 干扰他人任务。

**C. 共享 NTT roots，按 kernel strategy 懒建变体。**

- 建 immutable root-table owner，按 `(N, modulus, root-family)` 复用；每 level 只保留 view/小规模 RNS constants。
- 当前 kernel 将 P 放在 active-Q 后面；不能直接让短 Q view 指向最大 QP contiguous array，否则 P offset 错。必须给 descriptor 增加 storage-Q 或分离 Q/P 指针，并同步审计全部 kernel。
- fourstep/Montgomery/tensor 变体按实际 kernel 需求生成，不默认同时建立全部 families。若环境可动态切换 kernel，需在 plan freeze 或首次使用时确保相应 tables 完整。
- sparse 是现有接口可验证的短期节省；跨层 owner sharing 才能在 Boot 需要很多 Q 层时继续获得收益。

### P2：进一步缩短 native profile

- 缓存可公开重建的DFT raw values、编码明文、Q→QP diagonals、EvalMod coefficients/plan；索引必须包含scale的实际位值、有序Q/P、输出层、BSGS pre-rotation、root/layout/encoder版本和profile options。仅parms_id不足以绑定外部P选择；也不能无证明地跨Q-prefix复用P残基，centered representative可能改变。
- inverse-pre-rotation 可以在 GPU-ready prepare 阶段完成并保存；或上传后批量执行，避免重复构建 pointer tables。
- level-special-P 模式先确定最终需求，直接生成 grouped keys，不先创建注定丢弃的 global 全套。
- 增量 preflight 只更新新增/升 Q 的 keys，保持未变 owner 和已验证 view 生命周期。

## 7. 验收标准

1. 完全相同 N、Q/P、word bits、实际 key/digit inventory、CPU affinity/quota 和缓存条件；小探针至少3个独立进程样本，报告中位数/范围。
2. 上载 keys 的 Q 与 P 全字回读对比通过；裁剪 key 的 P offset、inverse layout、copy/move owner 生命周期、跨设备引用都有回归。
3. 参数共享在每个实际 Q level 的七类/所需变体与原实现一致，执行 NTT/INTT、rotate/relin 与 native Boot 数值回归；不能只验证最大层。
4. 同时报告准备 wall、CPU time、peak RSS、pinned/VRAM、H2D bytes/calls/waits，防止仅将开销移至首次请求或增加常驻内存。
5. 生产安全评估独立执行：本报告微基准 `sec_level=none` 和历史非等参数数字不证明安全性或端到端加速比。

## 8. 复现实验与证据

本地完整证据：`test-results/gpu-keyload-20261115/`。

- `summary.json`：可由 `experiments/gpu_keyload/summarize.py` 重建的结构化汇总。
- `results/native-off-omp1/{probe.json,stderr.log,launch.json}`：第一轮真实native阶段、环境和进程wall。
- `results/native-qp-trace-omp1/`：第二轮及 `QP_UPLOAD_PROFILE` 内部细分。
- `results/poseidon-q32-omp1-r{1,2,3}/`、`results/cerium-q32-r{1,2,3}/`：等形状3次独立进程原始样本，含全字验证结果。
- `results/poseidon-q50-{full,sparse}/`、`results/parameter-comparison.json`：参数allocation/层集合/全字fingerprints。
- 各case的 `launch.json`：精确argv、环境、affinity/quota、前后GPU状态、cgroup CPU stats、exit code/timeout。其外部进程wall包含验证和析构，不应替代纯阶段时间。
- `probe-build.json`、构建日志、`cerium-ldd.txt`、`binary-sha256.txt`、`provenance.json`：构建与依赖证据。binary清单是会话结束时的最终工件哈希，不冒称每个历史case都有单独的二进制attestation。

本次增加的可复用工具：

- `bench/gpu_keyload/`：无Runtime依赖的Poseidon standalone probe；sample/full key回读与full/sparse参数比较。
- `bench/cerium_keyload/`：调用原Cerium公开生成/序列化/反序列化API，链接现成runtime，不重编译Cerium。
- `experiments/gpu_keyload/native_profile.cpp`：真实manual native builder的最小准备入口。
- `experiments/gpu_keyload/run_matrix.py`：新进程、GPU忙闲检查、独立case目录、环境与超时记录；锁只协调同一results锁文件，不能替代跨用户GPU协调。
- `src/poseidon/gpu/{gpu_uploader.cpp,gpu_bootstrap_profile.cpp}`：初次调查时仅新增 `POSEIDON_GPU_KEYLOAD_TRACE=1` 可选计时。**后续已实现QP精确重建和并行准备优化，数据与验证见独立修复报告；磁盘缓存和直接QP编码仍是建议，未实现。**

服务器保留同名独立目录。用新的case名称复跑，避免覆盖旧证据，例如：

```bash
D=/root/private_data/poseidon-keyload-20261115
C=replay-q32-$(date +%Y%m%d-%H%M%S)
POSEIDON_GPU_KEYLOAD_TRACE=1 python3 "$D/src/experiments/gpu_keyload/run_matrix.py" \
  --binary "$D/poseidon_gpu_keyload" --results "$D/results" --case "$C" \
  --threads 1 --wait PASSIVE --affinity 0-6 -- \
  16 32 16 8 "$D/results/$C/probe.json" --check full --parameter-mode none

# 不使用GPU的证据重算/参数对照：在worktree根目录运行。
python3 experiments/gpu_keyload/summarize.py test-results/gpu-keyload-20261115/results
python3 bench/gpu_keyload/compare_parameters.py \
  test-results/gpu-keyload-20261115/results/poseidon-q50-full/probe.json \
  test-results/gpu-keyload-20261115/results/poseidon-q50-sparse/probe.json
```

### 主要源码锚点（本worktree行号）

| 路径/行号 | 作用 |
|---|---|
| `src/poseidon/gpu/gpu_uploader.cpp:1101–1225` | exact QP：Q INTT、CRT、centered modP、Q/P NTT |
| 同文件`:1315–1429` | 周期搜索、compact upload、device回读及CPU展开检查 |
| 同文件`:286–361,475–556` | ordinary key packing/逐shard同步与key/digit上传 |
| `src/poseidon/basics/util/rns.cpp:369–422` | O(Nq²) CRT compose |
| `src/poseidon/ckks_encoder.h:433–564` | 同次FFT/rounding保留整数的结构性改造点 |
| `src/poseidon/gpu/gpu_bootstrap_profile.cpp:963–1144` | manual builder真实分段 |
| `src/poseidon/key/keyswitch_base.cpp:205–236` | Galois OpenMP+critical |
| `src/poseidon/gpu/gpu_parameter.cpp:1986–2103,2256–2334` | 全链七套root arrays |
| `src/poseidon/runtime_api/poseidon_gpu_api.cpp:3329–3385,3686–3758` | ordinary key max-Q聚合/分桶共享 |
| `../reference/cerium/cerium/runtime/io/generator.cpp:377–570,975–1071` | 原生32bit并行keygen、序列化、pool反序列化 |
| `../reference/cerium/cerium/runtime/execution/function_io.cpp:214–283,727–835` | pinned backing alias与末尾同步H2D |

SSH凭据仅在访问期间置于本机0700临时目录中的0600临时文件，未进入worktree、报告或实验工件；收尾时清理临时凭据及SSH控制连接，服务器不安装持久登录密钥。

### 历史结果的正确使用

本地原有 `poseidon/test-results/key-prefix-implementation-20261110/direct-resnet20/report.json` 记录：CPU context 10.740s、CPU keys 223.354s、GPU API+native setup 294.076s（其中 native profile 285.039s），ordinary runtime setup 5.449s。这是**11/10历史实验，不是本次 A800 同参数复测**；ordinary key preflight 包含于最后一项，足以否定“294s全是 ordinary key H2D”的解释。原 report 的 transport telemetry 明确排除 direct key uploader，也不能用其带宽推导 key loading。

此前会话提到的 Cerium 51.737s/Poseidon 289.384s 属于另一台 V100 上、非等参数的完整进程启动；本次不以其比值作为实测结论，也不将组合阶段推断为纯密钥时间。
