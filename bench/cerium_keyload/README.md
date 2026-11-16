# Cerium IOGenerator 独立 key-load microbench

只编译这个 C++17 executable，**链接已有 `libCeriumRuntime.so`，不编译/修改 Cerium、不用 Python wrapper、不需要 nvcc**。默认运行会使用 GPU；`--plan-only` / `--cpu-only` 不调用 CUDA，供离线检查。

## 构建

依赖：`g++`、Bash、awk、binutils (`readelf`)，以及外部 Cerium/SEAL/CUDA headers 和现有共享库。所有产物写入本目录 `build/`。

```bash
# 已准备的远端公开源码/二进制目录；本任务未在远端执行这些命令。
base=/root/private_data/poseidon-keyload-20261115/cerium
export CERIUM_SRC="$base"                         # 内有 cerium/runtime/ 和 benchmarks/
export SEAL_INCLUDE="$base/.deps/install/include/SEAL-4.1" # 内有 seal/seal.h + seal/util/config.h
export CERIUM_RUNTIME_SO="$base/lib/libCeriumRuntime.so"
export CUDATOOLKIT_ROOT=/usr/local/cuda            # 只取 include/，不调用 nvcc
export CUDART_SO="$base/lib/libcudart.so.12"       # 明确用 Cerium 所需 CUDA runtime
export LD_LIBRARY_PATH="$base/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash bench/cerium_keyload/build.sh
```

也可将四个必填变量指向原始 `cerium_bundle/{cerium_src,build/cerium/runtime/libCeriumRuntime.so}` 等位置，并显式指定 bundle 中的 `CUDART_SO`。`CXX` 可覆盖 host compiler；C++ ABI、SEAL headers/配置必须匹配现有 runtime。

**服务器 CUDA headers/nvcc 是 11.8，但现有 Cerium runtime 需要 CUDA 12。** 本程序只用稳定 host runtime APIs；不能误链接 toolkit 的 CUDA 11 `libcudart`。脚本检查 Cerium 的 `DT_NEEDED` 与所选 `CUDART_SO` 的 SONAME，拒绝混用 major。未设 `CUDART_SO` 时优先在 runtime 同目录找匹配库，再找 toolkit。`LD_LIBRARY_PATH` 仍应包括 Cerium bundle 的 `lib/`，以解析 `libnccl.so.2` 等传递依赖；必要时以 `ldd bench/cerium_keyload/build/cerium_keyload` 检查，无需运行 GPU。

`extract_primes.sh` 只用 awk 从公开 `benchmarks/bootstrap_32K/primes.py` 提取 64 个常数到 `build/cerium_bench_primes.h`，**不执行 Python**。`PRIMES_SOURCE=/absolute/path/primes.py` 可覆盖来源；也可单独执行：

```bash
bash bench/cerium_keyload/extract_primes.sh "$CERIUM_SRC/benchmarks/bootstrap_32K/primes.py" \
  bench/cerium_keyload/build/cerium_bench_primes.h
```

## 使用

```bash
bin=bench/cerium_keyload/build/cerium_keyload
"$bin" --plan-only                  # 只生成/删除最小 metadata，不生成 keys、不调用 CUDA
"$bin" --cpu-only --q 2 --p 1 --k 1 # CPU serialize/deserialize 精确比对；不是 GPU benchmark
bash bench/cerium_keyload/test_cpu.sh        # 离线 smoke、非法参数、清理检查
bash bench/cerium_keyload/test_cpu.sh --full # 另验证默认 432 MiB payload 的 CPU round-trip

# 默认目标；仅在授权的 GPU 服务器上执行。stdout 为单条 JSON；stderr 立即 flush 阶段进度。
"$bin" --q 32 --p 16 --k 8 --hwt 192 --ephemeral-hwt 192 \
  --device 0 --tmp-root /tmp > result.json 2> stages.log
```

参数：固定 `N=65536`、28-bit primes；`Q>=1,P>=1,Q+P<=64`；`0<=K<32768`。digits **按 Poseidon hybrid 规则取 `ceil(Q/P)`**：连续 P 个 Q indices 为一组，末组可较短；默认正好 digits=2。没有独立的 digits 参数，避免和 Poseidon 的 P-sized decomposition 不一致。`--hwt`、`--ephemeral-hwt` 默认均 192，可改为 `[1,65536]`。

默认库存：

- ordinary `Rot` (type 1)，正向左旋 steps `[1,2,3,4,5,6,7,8]`，不是 powers of two / `RotInv`；另有一个 `Mul` relin (type 0)。
- Q partitions `[0..15]` / `[16..31]`。每个 digit-key 是两个完整 QP polynomial。
- **刻意只构造 48-prime Context**，使用公开 list 的前 48 项；Q IDs `0..31`，P IDs `32..47`。Cerium 从 Context 尾部倒序选择 P，但 limb map 按 ID 排列。不是 64-prime Context 的尾部 `48..63`；此选择使 Context 构造成本也匹配 Q32/P16 的实际 prime 数。
- 最小 `program_inputs` 只有 `Evalkey Stream:`、每 pair 的 c0/c1 两行、独立 `;`。36 个 term references 按公开 ID 去重为 **18 pairs**，不是 36 keys。
- 18 × 2 × 48 × 65536 × uint32 = **452,984,832 payload bytes (432 MiB)**，1728 limbs，每 limb 262,144 bytes。64-bit native format 的完整文件为 **453,000,368 bytes**。
- 数量/bytes 不是只从参数打印：程序审计实际生成和反序列化 map、每个 ID/limb/NTT 标记、生成器返回 element count、精确 file size，并验证实际 host pool slices 无重叠/空洞。

## 计时边界

JSON `timing_ms` 分报：

| 字段 | 边界 |
|---|---|
| `Context` | 仅公开 Context 构造（CPU NTT/Galois tables） |
| `CKKSEncryptorContext` | SK、SK²、ephemeral 的 RNS/NTT 准备；不含合成 secret 采样 |
| `generate_evalkeys` | 原公开 API 调用；返回 map 保留到序列化和 host 校验后，析构不计入生成 |
| `serialize_evalkeys_no_fsync` | 原 API，含 ofstream 析构 close/flush；**没有 fsync / 持久化保证** |
| `deserialize_evalkeys_including_host_alloc` | 原 API 普通 ifstream read，含一次 `cudaMallocHost` pool 建立及 map/limb 构造 |
| `host_pool_alloc_subset_of_deserialize` | 上一阶段内 pool callback 的子计时，不应再次相加 |
| `cuda_initialize` | CUDA context/device/stream 初始化，单列，不属于 H2D |
| `device_pool_alloc` | 一个精确 payload 大小的 `cudaMalloc`，单列 |
| `h2d` | 预建 descriptor 后，每 limb 一次 `cudaMemcpyAsync`，最后一次 `cudaStreamSynchronize` 的 **wall time**；不含分配、D2H、验证、文件清理 |
| `verify_host_roundtrip` / `verify_gpu_readback` | 计时外 correctness 工作的独立耗时；逐字节 `memcmp`，不打印值 |

反序列化通过 `MemoryManager` 公开 subclass/callback 使用一个 `cudaMallocHost` backing arena，与生产 pinned-pool 路径一致。H2D 用一个 device pool、一个 nonblocking stream，按 host pool slice 顺序传完整所有 limbs；GPU 回读用额外**一 limb** pinned scratch，逐 limb D2H+sync，与原 pinned 数据精确比较。原 pinned pool 至少活到所有 H2D/验证结束，整个 key 内容不输出。

**这不是冷磁盘测试**：写入后立即重读，page cache 很可能热，没有 drop_caches/fadvise。无 H2D warmup/重复迭代，每进程只执行一次；CUDA 初始化单列。host round-trip 校验在 H2D 之前，源数据已被 CPU 读过。`--cpu-only` 明确改用 malloc arena，所有 GPU 计时为 null、`verified_gpu_exact=false`，其读盘数值不能冒充 pinned/GPU 结果。

生成器源码固定启动 **64 个 std::thread**（不是 GPU keygen / OpenMP）；`OMP_NUM_THREADS` 不会改变它。JSON 记录线程数、CPU affinity、可读的 `/sys/fs/cgroup/cpu.max`、OMP 环境值。目标服务器可见 64 核而 CPU quota 为 7 核，需在解读 generation 数值时另行核实这一 oversubscription 条件。

## 临时文件 / 隐私

- `--tmp-root` 下用 `mkdtemp` 建立明确的 `cerium-keyload.XXXXXX/` (0700)，路径写入 stderr/JSON；生成 `program_inputs`、`evalkeys.bin` (umask 077)。选本机空间充足的临时盘；默认需约 432 MiB 文件和额外 RAM/pinned/device 内存。
- 正常返回、C++ exception、SIGINT/SIGTERM/SIGHUP/SIGABRT 均 unlink 两个文件并删除目录；**成功 JSON 输出前已验证删除成功**。生成密钥前检查设置 `PR_SET_DUMPABLE=0` 和 `RLIMIT_CORE=0`（后者单独不足以阻止 pipe-based core handler）。SIGKILL、进程/机器崩溃不能保证清理，可按 stderr 的精确路径手动清理。
- 只合成非生产 sparse ternary secret/ephemeral；MT19937 种子来自 random_device，且不记录种子、SK、ephemeral 或任何 coefficient。不要把此采样器当作生产密钥安全实现。
- SK 从不序列化。调用方 secret vectors 显式清零，encryptor/SK 私有副本在序列化前销毁；未改 runtime，**不声称其内部 allocator 会安全擦除 SK 或 evalkey 内存**。unlink 也不是磁盘安全擦除；若必须避免物理盘痕迹，用容量足够的 tmpfs，另注意 swap。输出只含 shape、bytes、计时、验证状态和环境元数据。

## 本地验证范围

已完成 host-only 链接、默认 `--plan-only`，以及默认 Q32/P16/K8 和小规模 `--cpu-only` 的完整公开 API round-trip（另含非整除 Q/P、K=0）。默认实际 map 确认为 18 pairs / 1728 limbs / 452,984,832 payload bytes / 453,000,368 file bytes，所有 host 数据精确相等，临时文件已删除。没有在本任务中运行 GPU、操作远端或重新构建 Cerium；远端 CUDA12 链接及 GPU H2D 计时/回读需由主会话实际执行后确认。
