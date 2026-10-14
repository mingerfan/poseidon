# TileLang GPU Encode 对照（2026-10-11）

原始实验找到于 `/mnt/zhitai/KnowledgeLib/Research/FHE-GPU/experiments/tile_lang_int32/`；CUDA NTT 在 `kernels.py`，Tensor NTT 在 `tensorcore_ntt.py`。`/home/xs/Data/windows-sh/tilelang-int32-source/` 是离线源码包，其中 Tensor NTT 文件与原实验 SHA-256 相同。元数据和哈希见 `environment.json`。

新增实现：[TileLang kernels](../../src/poseidon/tests/gpu_encode/tilelang_ntt.py)、[生成器](../../src/poseidon/tests/gpu_encode/generate_tilelang_ntt.py)。扩展了多模数、batch、运行时 Barrett reduction，直接使用 Poseidon 负循环 NTT roots/TAM matrices，输出保持 Poseidon 的 bit-reversed 顺序。CUDA 版 8 个两阶段融合 kernel；Tensor 版 3 个四阶段 TAM + 2 个 CUDA 尾部 kernel。TileLang 只在构建时使用。

RTX 4060 Laptop、N=65536、Q-only、30-bit primes、scale=2^40、64 MiB 初始 RMM pool。源码 `6b40776d`。Poseidon 主体保留原构建目标 sm_75 + PTX，TileLang 按 sm_89 编译；这是现有实现之间的比较，不能将差异全部归因于语言或 Tensor Core 算术吞吐。188Server 连接不可用，结果来自本机。

每配置 3 个独立进程，每模式/路径至少预热 5 次且累计 GPU event 时间 >=100 ms，再测 30 次。随机打乱模式顺序，seed 保存在每份 JSON；CPU 正确性验证在全部计时之前完成。下表为进程内中位数的中位数，单位 **ms/份明文**。

完整 Encode（输入已在 GPU；batched 提交）：

| Q limbs | Batch | CUDA four-step | 旧 Poseidon Tensor | TileLang CUDA | TileLang Tensor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.1290 | 1.1767 | 0.1359 | 0.1778 |
| 8 | 8 | 0.0929 | 0.5754 | 0.0967 | 0.1868 |
| 8 | 32 | 0.0987 | 0.5323 | 0.2094 | 0.1649 |
| 32 | 1 | 0.2481 | 4.4705 | 0.2593 | 0.4619 |
| 32 | 8 | 0.2519 | 2.1692 | 0.6883 | 0.5482 |
| 32 | 32 | 0.2554 | 1.9558 | 0.6851 | 0.5357 |

仅 NTT 阶段（同一完整 Encode 时间线内，batched）：

| Q limbs | Batch | CUDA four-step | 旧 Poseidon Tensor | TileLang CUDA | TileLang Tensor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.0348 | 1.0833 | 0.0420 | 0.0840 |
| 8 | 8 | 0.0298 | 0.5116 | 0.0335 | 0.1013 |
| 8 | 32 | 0.0356 | 0.4706 | 0.1468 | 0.1022 |
| 32 | 1 | 0.1198 | 4.3387 | 0.1260 | 0.3287 |
| 32 | 8 | 0.1468 | 2.0744 | 0.5916 | 0.4425 |
| 32 | 32 | 0.1480 | 1.8621 | 0.5911 | 0.4340 |

Pinned 原始数据 H2D + 完整 Encode（batched）：

| Q limbs | Batch | CUDA four-step | 旧 Poseidon Tensor | TileLang CUDA | TileLang Tensor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 0.1550 | 1.1988 | 0.1622 | 0.2050 |
| 8 | 8 | 0.1132 | 0.5953 | 0.1170 | 0.1590 |
| 8 | 32 | 0.1169 | 0.5529 | 0.2283 | 0.1836 |
| 32 | 1 | 0.2808 | 4.4927 | 0.2822 | 0.4804 |
| 32 | 8 | 0.2683 | 2.1910 | 0.7079 | 0.5642 |
| 32 | 32 | 0.2651 | 1.9772 | 0.7040 | 0.5503 |

全部逐份提交结果、CPU Encode-only、分阶段时间、wall、p95、按测量顺序的样本和测量前的 GPU 温度/时钟见 `repeated/`。未锁定时钟，CUDA events 包含 CPU 提交间隙；初始化、持久缓冲和表不计时，旧 Tensor 内部 scratch 仍计时。所有路径共享 FP64 cuFFT/RNS；没有测试 Tensor FFT。

正式 18 次执行全部通过边界值/随机 residues 的 NTT 检查，TileLang CUDA/Tensor 均与 four-step 逐元素一致。完整 Encode 的 8 个模式也均逐元素匹配 canonical CUDA Encode。CPU FFT 可能产生不同舍入；每份都与 CPU residues 比较，对首尾及存在差异的明文进行 CPU decode。原始 JSON 包含误差和差异计数。非有限输入及系数越界拒绝检查通过。

Encode CTest 通过；`compute-sanitizer --tool memcheck` 的 Q=3、batch=2 检查得到 0 errors（`memcheck.log`）。该检查和 `smoke.json` 基于实现提交 `33bc1232`；随机顺序改动后的 smoke 见 `shuffled-ctest.log`。`tensor-instructions.txt` 包含 TileLang kernel 的 IMMA.16832.U8.U8 证据，未获得硬件利用率计数器。

`pilot/` 保留 5 个已完成的短预热、固定顺序初测，后续执行被主动中断（`pilot-run.log`）；这些数据不纳入正式汇总。

结论：TileLang 显著改善旧 Tensor NTT 路径，完整 Encode 仍慢于 four-step。是否可以利用空闲 Tensor Core 隐藏 Encode，取决于主算子的明文消费速度及两条 stream 的实际资源竞争，需要另外测量并发时主算子的 slowdown，不能仅用独立 Encode 时延推断。

复现构建方法见 [benchmark README](../../src/poseidon/tests/gpu_encode/README.md)，重复运行：

```sh
python3 scripts/benchmark_gpu_encode.py \
  --binary build-runtime-gpu-api-release/bin/poseidon_gpu_encode_bench \
  --output test-results/gpu-encode-tilelang-new --runs 3 --warmup 5 --repeat 30
```
