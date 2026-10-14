# GPU Encode 本地测量（2026-10-11）

RTX 4060 Laptop，SM 8.9，CUDA 13.4。188Server 的代理和直连均未成功，
所以这里是本地 GPU 结果。源码：`39583c0e`，硬件/编译器见 `environment.json`。

测量完整 GPU Encode：实数槽位排列、FP64 cuFFT、twist/scale/round、RNS
展开、负循环 forward NTT。Tensor 版本只在 NTT 使用 INT8 Tensor Core；
FFT/RNS 与 CUDA 版本相同。N=65536、scale=2^40、30-bit Q primes、单 GPU、
Q-only，使用 64 MiB 初始 RMM pool。参数用于算术实验，不代表应用安全参数。

`pooled/` 是正式结果：6 个配置，每配置 3 次独立进程，各模式预热 5 次、
测量 30 次；下表是进程内中位数的中位数，单位 **ms/份明文**。
所有输入已在 GPU 上。individual 逐份提交，batched 合并 FFT 和 RNS/NTT grid。

| Q limbs | Batch | CUDA individual | CUDA batched | Tensor individual | Tensor batched | CUDA batch 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.1735 | 0.1744 | 1.6182 | 1.1795 | 0.99× |
| 8 | 8 | 0.1701 | 0.1265 | 1.1811 | 0.5729 | 1.35× |
| 8 | 32 | 0.1722 | 0.0978 | 1.1829 | 0.5328 | 1.76× |
| 32 | 1 | 0.3457 | 0.3435 | 6.1001 | 4.4569 | 1.01× |
| 32 | 8 | 0.3379 | 0.2450 | 4.4846 | 2.1682 | 1.38× |
| 32 | 32 | 0.2575 | 0.2448 | 4.5053 | 1.9587 | 1.05× |

同一 benchmark 包含 pinned 原始 double H2D + Encode：CUDA batch=32 时，
8/32 limbs 分别为 0.1168/0.2644 ms/份；现有 Tensor batch 分别为
0.5526/1.9804 ms/份。CPU 单线程 Encode-only 分别约 6.415/30.386 ms/份。
CPU 不包含上传，不能据此直接推断 Qwen 的端到端加速。

CUDA events 覆盖提交期间的 GPU 时间线，包含 GPU 等待 CPU 提交的间隙，
不是各 kernel 执行时间之和。固定模式顺序、未锁 GPU 时钟，时延应结合原始
样本及 `case_order` 的温度/时钟看待。batch=1 的不同模式即使算法相同也会
出现时延差异；某些 device-only 结果甚至高于包含 H2D 的另一次测量。
初始化、参数表、cuFFT plans、持久 buffer 分配不计时；现有 Tensor NTT
内部的 scratch 分配/释放计时。wall、p95、分阶段时间和完整样本均保留。

正确性：正式 18 次执行全部通过；每份明文的完整 RNS/NTT 输出比较 CPU
Encode，所有 GPU 模式之间要求逐元素完全一致。CPU/GPU FFT 的舍入次序
存在少量差异，因此 CPU residues 不保证相同；原始报告记录差异数量。
对首尾明文和全部存在 CPU residue 差异的明文进行 CPU decode，正式结果
最大输入误差 3.04e-10，最大 CPU decode 差异 9.12e-13。
`scale25.json`、`scale60.json` 补充低/高 scale；最大输入误差分别
9.57e-6/3.55e-15。NaN 和系数幅度越界拒绝检查通过。

新 Encode smoke 与原 GPU runtime API tests 共 2/2 通过，pool 调整后的
Encode smoke 再次通过，见 `ctest.log`、`pool-ctest.log`。
`profile_cuda_gpu_kern_sum.csv` 是独立 Nsight Systems 单次诊断，不作为正式
时延结果。它确认实际执行 `u8_wmma_mod_batched_gemm_kernel`，并显示旧
Tensor NTT 的 prepare/GEMM/unpack 多次调用。`tensor-instructions.txt` 保存
IMMA 指令证据。硬件性能计数器访问被拒绝（ERR_NVGPUCTRPERM），没有测得
Tensor Core 利用率，不能用这些结果声称利用率提高或降低。

`direct/` 只保留早期直接 cudaMalloc/free 的 **1 次完整进程轮次**，不是
正式重复结果；`run.log` 记录其后中断的第二轮启动，第二轮报告未保留。
根目录 smoke/batch-smoke 是早期检查，其他日志及原始 JSON 用于复查。

复现：

```sh
python3 scripts/benchmark_gpu_encode.py \
  --binary build-runtime-gpu-api-release/bin/poseidon_gpu_encode_bench \
  --output test-results/gpu-encode-new --runs 3 --warmup 5 --repeat 30 --pool-mb 64
```

结论限于当前实现和设备：GPU batch Encode 有收益；当前 Poseidon Tensor
NTT 明显慢于 CUDA four-step。后续 TileLang NTT 应在相同多模数、batch 和
完整 Encode 上重新验证，不能直接套用固定单模数实验结果。
