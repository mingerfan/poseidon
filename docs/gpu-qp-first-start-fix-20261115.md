# 首次启动 QP 常量生成：修复与延迟回归

## 1. 本轮范围

在 `investigation/gpu-keyload-20261115` / `poseidon-gpu-keyload` 中实现，基线为 Poseidon `ddcd96cd` 加前一轮计时插桩。原 `poseidon` 不改动。

本轮测试的是 **N65536、公开 ResNet20 Q50/P25 混合30/31/32-bit链的真实 native manual profile 首次生成**。每个样本是新进程、新生成 secret/evaluation keys、重新编码所有常量；**不加载预生成常量、不复用磁盘key或常驻profile缓存**。

没有修改密钥随机生成算法、Q/P参数、scale、DFT/EvalMod配置或在线CUDA计算kernel。先解决实测195秒的QP准备主瓶颈；约33秒的普通native keygen仍保留。

## 2. 已实现的优化

### 2.1 经全Q核验的精确小整数重建

旧路径对每个diagonal执行Q INTT→完整CRT compose→centered modP→Q/P NTT；CRT compose为O(Nq²)，modP还按q个word扫描。

新路径仍先对独立scratch执行Q INTT，但用前一个/两个32-bit Q primes重建一个有符号候选，再**逐项检查所有Q residues**：

1. 设两prime乘积M=q0·q1，用Garner得到x∈[0,M)，按ceil(M/2)变为centered signed candidate。
2. 两prime均≤UINT32_MAX，M和x的运算使用uint64，乘法/加法上界均不超过M−1<2^64。
3. candidate处于完整Q的centered区间；只有它匹配**全部Q模数**时，才由CRT唯一性认定它就是精确整数，直接对P约简。
4. 任一系数/模数不匹配，整个plaintext退回完整CRT；fallback按实际magnitude有效字长约简P，零值单独处理。

因此不是近似ModUp，不依赖未经核验的“小系数假设”，也不改变centered整数。对较大系数保留正确回退。支持的GPU residues仍为uint32；不得由host fallback推断GPU支持更宽的模数。

### 2.2 保留canonical Q-NTT

直接保存规范化后的原Q NTT数据，在独立scratch做INTT，不再对Q做重复正向NTT；只为新P limbs做正向NTT。对lazy输入先mod qi归一化。回归覆盖等价的[0,q)、[q,2q)、[2q,4q)输入。

### 2.3 压缩消冗与严格验证

- 成功周期只完整证明一次；period=N是identity布局。
- bit-reversal表按matrix复用，不按diagonal重复构建。
- 已知period后直接gather compact代表元，避免重新扫描完整N。
- 保留严格device round-trip：完整host周期证明之后，device compact逐word回读相等，即可推出全部展开Q/P字相等；不再在CPU重复展开整幅数据比较。valid标志只在回读通过后设置。

### 2.4 有界CPU并行，GPU仍在调用线程

压缩diagonal采用显式async CPU任务；在途任务/host结果数≤worker limit。CPU任务只产生普通vector，使用私有scratch/fallback内存池；GPU allocation、copy、验证、map写入和pointer-table组装仍在调用线程，保留其thread-local stream/resource语义。

future在异常展开时等待未完成任务，避免context/plaintext引用提前析构。默认最多4个后台worker；进程内不同调用共享后台worker计数，线程创建失败或无可用slot时在调用线程串行处理。按CPU affinity和cgroup-v2根目录quota尽力裁剪，本机可见64核但只有7核quota；这不是跨进程、嵌套cgroup、并发调用总内存或调用线程数的硬限制。GPU上传仍在调用线程。

## 3. 正确性验证

### 3.1 小规模精确oracle回归：已通过

`bench/qp_prepare`，远端实际GPU执行：24配置、264组整数多项式输入；N1024/4096，active Q=1/2/4，P=2/3，28/30/32-bit及混合模数。覆盖：

- 0、正负数、Q/P整除、完整Q中心边界；
- 接近2^64的两prime乘积、q0>q1混合位宽；
- 超过前两prime候选范围、±q0q1别名、只在最后coefficient且只有最后Q limb不匹配的别名，确保完整Q核验和CRT fallback；
- period=1/8/N，全部压缩数据展开对比，plan的Q/P pointer offset及元数据；
- reference、optimized请求1/4线程直接逐word比较；
- 更宽lazy输入只送入optimized，oracle来自canonical reference/GMP，避免违反旧INTT输入范围；
- 非NTT、未知parms_id、错误shape、非法worker数、部分任务在途时发生异常，以及之后的正常上传恢复；
- 最终边界回归使用调用线程的nondefault CUDA stream，264组输入再次全部通过。合并前复测还覆盖宽lazy输入开启VERIFY_REFERENCE、模拟线程创建失败后的串行回退、两个并发调用各自使用nondefault stream；并发样本不证明线程峰值或内存峰值上界。

GMP独立center/reduce→CPU NTT产生oracle；比较实际GPU回读的完整Q/P与展开compressed数据，不以近似decode替代。

### 3.2 真实profile回归

**已通过全部10个正式新进程样本的公开常量/metadata回归，并另有一次逐字oracle轮。**

- 参考模式、优化1/4/6线程以及oracle轮的公开常量和plan metadata指纹均为 `436dedc230ac395b`，完整inventory一致。
- 指纹覆盖389个公开明文owners、434,972,160个uint32 words（1,739,888,640B），其中两组QP合计316个diagonal，compressed payload927,766,528B；包括其parms_id、scale位值、q/p/N、period、plan和alias引用语义。随机key payload不进入跨进程指纹。
- **另一次 `QP_VERIFY_REFERENCE=1` 的真实profile构建，对316/316个diagonal的全部展开Q/P host words及元数据直接与旧full-CRT结果比较，全部通过**；两组均报告158个 `reference_verified_plaintexts`。展开数据为4,307,550,208B的uint32等价大小；不只检查压缩样本或近似decode。
- 所有性能轮均为316个精确快路径命中、0次full-CRT fallback；fallback仍由小规模GMP回归覆盖。
- Oracle轮准备130.261s，**仅作正确性检查，不进入延迟中位数**。FNV-1a-64只是辅助诊断，不声称无碰撞证明；直接full-CRT逐字oracle补充了新数学路径的验证。
- 合并前另在RTX4060实际执行 **N4096小参数、Q50/P8、2048复数槽**的加密→GPU Boot→解密数值回归。reference/optimized均通过：对输入最大复数绝对误差分别1.499e-4 / 1.491e-4（阈值1e-3），两进程解密结果跨模式最大差1.081e-5。它是 `sec_level=none` 的小参数smoke，不是R20 Q50/P25在线Boot，也不是安全参数推荐；两模式分别生成随机密文。该smoke使用未压缩QP常量路径，压缩并行路径另由上述逐字回归覆盖。
- 正式矩阵、两轮边界回归及两轮开发exploratory共14个GPU测试进程全部exit0、无超时。正式性能表只使用指定A/B样本，排除exploratory和oracle。

## 4. A800首次生成延迟

环境与前轮相同：A800 80GB PCIe，UUID `GPU-e66fc2a8-85c9-46af-52a3-042a0a74e6e9`；Hygon C86；CPU0–6 affinity，cgroup `700000/100000`；GCC11.4，Release，CUDA11.8/SM80，POSEIDON_USE_OPENMP=OFF（顶层仍全局-fopenmp），OMP_NUM_THREADS=1。独立std::async worker数单独设置。

每次启动前检查无GPU PID且显存≤512MiB。reference与optimized使用**同一个编译后二进制**、同一参数文件，按环境开关切换；计时到最终cudaDeviceSynchronize。额外公开常量指纹校验发生在native_profile_s之后，正确性oracle轮不纳入性能统计。

本轮采用新鲜reference样本作同binary对照；不是将之前约262秒历史数据直接当作本轮基线。

| 模式 | 样本数 | native首次准备中位数（范围），秒 | QP两组墙钟中位数，秒 | 测试进程wall中位数，秒 |
|---|---:|---:|---:|---:|
| 旧完整CRT/串行 reference | 2 | **265.702**（265.531–265.873） | **204.601** | 276.597 |
| 优化算法，1个host worker | 1 | 93.055（单样本） | 32.541 | 103.196 |
| **优化算法，默认4 worker** | 3 | **71.044**（70.071–72.352） | **9.352** | 81.000 |
| **优化算法，6 worker** | 3 | **68.516**（68.119–68.586） | **6.828** | 78.335 |

**一次首次生成的实际收益：**

- 默认4线程：native准备节省 **194.658秒（3分14.7秒）**，减少 **73.26%**，约 **3.74倍**。
- 当前7核配额下显式6线程：节省 **197.186秒（3分17.2秒）**，减少 **74.21%**，约 **3.88倍**。
- QP子阶段从204.601s降到6.828s，约 **30倍**；这不是整体profile或在线Boot的30倍。
- 单worker算法优化已经从265.702s降到93.055s，说明收益不只是增加线程；再加4/6个host worker才降到71/69秒附近。

进程wall包括CPU context、CUDA初始化、secret初始化、**额外计时后指纹校验**和析构，不能将其与native阶段相加。相同测试流程下，6线程进程wall从276.597s降至78.335s。

6线程剩余主要阶段中位数：Galois keygen31.990s、DFT生成/编码16.280s、EvalMod编码上传9.670s；这些未在本轮优化。阶段中位数不应强行相加还原某一个样本。

**合并前最终源码复测（2026-11-16）：**加入进程内后台worker slot共享、线程创建失败串行回退、宽lazy oracle规范化之后，同一A800上新进程单样本reference为266.153s、最终源码4 worker为**70.966s**，节省195.187s、约3.75倍；与上表的原三样本结果相近。两次最终源码测试分别生成新密钥及常量，不用缓存。其独立oracle轮再次通过316/316个diagonal完整CRT逐字检查，跨进程公开常量fingerprint为 `436dedc230ac395b`。reference样本使用只差最终trace计数器字段的前一编译版本（reference算法未变），不是把两者称为同一个binary。

注意：这些是首次准备/生成时间，不是Boot在线执行延迟或整网总推理时间。native计时不包含外部CPU context构造、CUDA初始化和最终析构；进程wall还包括额外指纹校验，必须分开看。

## 5. 使用与回退

```bash
# 复现本次GPU准备测试的CPU预算；其他应用的总线程预算需另行协调。
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0

# 默认精确优化路径，最多4个CPU worker（尽力按根quota/affinity裁剪）
unset POSEIDON_GPU_QP_PREPARE
unset POSEIDON_GPU_QP_PREPARE_THREADS

# 显式配置；无需旧key/预生成文件
export POSEIDON_GPU_QP_PREPARE=optimized
export POSEIDON_GPU_QP_PREPARE_THREADS=4
# 当前独占7核配额时可用6；多profile并发应另行控制总CPU预算。
# export POSEIDON_GPU_QP_PREPARE_THREADS=6

# 一键回退旧完整CRT+串行压缩路径
export POSEIDON_GPU_QP_PREPARE=reference

# 只用于正确性验证，不能作为性能样本
export POSEIDON_GPU_QP_PREPARE=optimized
export POSEIDON_GPU_QP_VERIFY_REFERENCE=1
```

`POSEIDON_GPU_QP_VERIFY_REFERENCE=1` 会对每个优化host QP结果再运行旧完整CRT并逐字比较，包括parms_id/scale/shape；延迟测量必须设0/取消。`QP_UPLOAD_PROFILE` 的host时间在并行模式下是任务耗时累加，**不是阶段墙钟**；阶段墙钟看 `BOOT_SETUP_PHASE`。

## 6. 文件与复现

主要修改：
- `src/poseidon/gpu/gpu_uploader.cpp`：精确快路径、完整CRT回退、压缩与bounded async。
- `src/poseidon/gpu/gpu_uploader.h`：行为、开关和线程/stream契约文档。
- `bench/qp_prepare/`：GMP/CPU NTT精确oracle、GPU展开比较与异常回归。
- `experiments/gpu_keyload/native_profile.cpp`：仅为基准增加计时后公开常量/plan指纹，不哈希随机key payload。
- `experiments/gpu_keyload/run_qpfix_matrix.py`、`summarize_qpfix.py`：新进程A/B矩阵及验证汇总。
- `experiments/gpu_keyload/run_qp_merge_validation.py`、`summarize_qp_merge.py`：合并前安全修正后的参考、最终源码及oracle复验与汇总。

原始证据：`test-results/gpu-qp-first-start-fix-20261115/`；合并前完整复验保存在主worktree的 `test-results/gpu-qp-merge-validation-20261116/` 及服务器同名独立目录。Git追踪的compact汇总和oracle输出在前一目录的 `merge-review/`。没有保存私钥或evaluation-key payload。早期exploratory样本仅用于开发定位，不进入正式中位数。

复现实验需要已有GPU构建和空闲、获准的GPU；脚本拒绝覆盖既有case。改变suite名创建新结果目录：

```bash
D=/root/private_data/poseidon-keyload-20261115
GMP_ROOT="$D/tools/gmp" CUDA_PATH=/usr/local/cuda \
  bash "$D/src/bench/qp_prepare/build.sh" "$D/build" "$D/qp_prepare"
bash "$D/src/experiments/gpu_keyload/build_native.sh" "$D"
python3 "$D/src/experiments/gpu_keyload/run_matrix.py" \
  --binary "$D/qp_prepare" --results "$D/qpfix-replay/results" \
  --case exact-regression --threads 1 -- --compare
python3 "$D/src/experiments/gpu_keyload/run_qpfix_matrix.py" "$D" --suite qpfix-replay
python3 "$D/src/experiments/gpu_keyload/summarize_qpfix.py" \
  "$D/qpfix-replay/results" --require-complete
```

## 7. 限制与下一步

- 尚未修改Galois keygen的critical串行化、DFT/EvalMod CPU编码或GPU参数表全链展开。这些仍占剩余首次准备时间。
- 本轮没有缓存，所以不存在将第一次prepare成本藏到计时外的情况；只复用了编译好的程序。
- 合并前补充了N4096小参数GPU Boot解密数值smoke；没有运行R20完整N65536 Boot或整网数值测试，不能将准备加速比当作在线Boot/整网加速比。
- 已测错误输入/异步任务异常；合并前额外模拟OS线程创建失败分支，未做真实线程耗尽、OOM、deque分配失败或多进程内存压力测试。
- 参考路径保留，便于不同参数链或平台上发现问题时回退。其他scale/模数链的快路径命中率可能不同，不能将当前316/316命中或延迟倍率直接外推。
- 默认4线程保留一定CPU余量；当前独占7核配额下可以显式设6。进程内并发调用共享后台worker计数，但已完成结果缓冲和每个调用线程不受该计数限制；多个进程/设备时仍需应用层控制CPU与内存预算。
- 测试期间对协作BERT会话做了CPU型号/boot_id核对，确认型号、boot_id和GPU UUID均不同，未识别到同宿主冲突；不声称物理服务器无任何其他租户负载。
