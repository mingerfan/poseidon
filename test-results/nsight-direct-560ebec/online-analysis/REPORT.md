# Direct 1GPU/4GPU runtime-online 真实统计

数据源：

- `direct-1gpu.nsys-rep` / `direct-1gpu.sqlite`
- `direct-4gpu.nsys-rep` / `direct-4gpu.sqlite`
- 与报告 SHA256 完全匹配的 1GPU/4GPU runtime plan

只统计代码定义的 runtime online 区间：

| Profile | Online start | Online end | Duration |
|---|---:|---:|---:|
| 1GPU | 23.553067959s | 23.627879758s | 74.811799ms |
| 4GPU | 24.579446692s | 24.644462381s | 65.015689ms |

初始化、输出下载、解密、decode 和正确性验证均不包含在内。

## 总览

| 指标 | 1GPU | 4GPU |
|---|---:|---:|
| Runtime online | 74.812ms | 65.016ms |
| Kernel 数 | 5099 | 5099 |
| Kernel 累计 GPU service time | 26.697ms | 30.010ms |
| `cudaLaunchKernel` P50 | 4.369us | 8.735us |
| `cudaLaunchKernel` P95 | 7.658us | 22.384us |
| `cudaLaunchKernel` mean | 5.882us | 18.680us |

4GPU 相对 1GPU speedup 只有 **1.151x**，相对四卡的并行效率为 28.77%。

## 工作一致性

- 逻辑 compute op：375 vs 375，所有 `op + input level + output level` 数量完全一致。
- CUDA kernel：5099 vs 5099，所有 `kernel name + grid + block + shared memory` 数量完全一致。
- 4GPU 只是在 execution 中额外加入了 218 个 P2P transfer task。
- Rotate：L8 为 102 次，L4 为 23 次，两边完全一致。
- Keyswitch：125 次 rotate 加 1 次 relinearize，对应 126 个 `keyswitch.hybrid` range。

4GPU 分配并不完全均衡：

| GPU | Kernel 数 | 累计 GPU service time |
|---|---:|---:|
| GPU0 | 1151 | 6.695ms |
| GPU1 | 1141 | 6.683ms |
| GPU2 | 1511 | 8.896ms |
| GPU3 | 1296 | 7.736ms |

GPU2 的累计计算量比四卡平均值高约 18.6%。

## CPU worker

图中线程名称已展开：`GPU0 计算线程`负责向 GPU0 发射计算 kernel，`GPU0 通信线程`负责 GPU0 的 P2P transfer task；GPU1–GPU3 同理。`1GPU/4GPU 主线程`表示 runtime 主控线程。

4GPU compute worker 的平均时间构成为：

- Kernel launch：36.63%
- `pthread_cond_wait`：25.98%
- 其他 Nsight 可见活动：6.32%
- 无可见事件：31.07%

逐卡范围：

- Launch：32.9%–38.8%
- `cond_wait`：20.6%–30.8%
- Blank：27.6%–34.3%

4GPU launch P50 是 1GPU 的 2.00x，P95 是 2.92x。Launch 的多线程累计时间为 95.250ms，不能直接作为 wall time，但每个 compute worker 自身约三分之一以上时间处于 launch API 内。

Communication worker 的 `cond_wait` 为 55.8%–62.1%，主要表示通信队列没有 task，不能直接视作通信瓶颈。4G-main 的 98.0%“其他可见活动”主要是等待八个 worker 的 `pthread_join`，同样不是额外的 63.7ms wall overhead。

Blank 是 Nsight 可见区间的补集，可以准确测量，但当前 trace 无法继续区分普通 C++、线程被调度出去和未采集短调用。

`05-online-cpu-timeline.png` 将上述四类时间展开到真实 online 时间轴。它直观显示：

- 4GPU compute worker 的 launch 主要集中在约 3–35ms，并在约 35–53ms 出现大段 `pthread_cond_wait`，之后又恢复一段 launch；这不是单纯的平均 launch 变慢，而是 task 供应/依赖等待呈阶段性变化。
- Communication worker 存在更长的连续 `pthread_cond_wait`，中间穿插短促的 CUDA/OSRT 活动，符合通信 task 按依赖间歇到达的行为。
- 4GPU 各 worker 在约 60ms 后基本不再出现 CUDA/OSRT 事件，而 main thread 的 `pthread_join` 仍延伸到 online 末段。浅灰 blank 本身不能判断 worker 是在执行普通 CPU 代码、被调度出去还是已经进入未采集路径。

## P2P 与计算覆盖

- Execution P2P：218 次
- 总数据量：120.193MB（114.625MiB）
- P2P service time 累计：15.002ms
- 目标端 Rx 加权物理覆盖率：21.27%
- 源端 Tx 加权物理覆盖率：28.26%

各卡 Rx 覆盖率为 15.6%–27.6%，Tx 覆盖率为 26.1%–29.7%。大多数 P2P activity 没有和对应 GPU 上实际执行的 kernel 物理重叠。

通信量最大的方向为 GPU3→GPU1（15.008MB，29 次）和 GPU3→GPU0（14.221MB，25 次）。各方向 trace 中的有效带宽约为 6.77–9.58GB/s。

## 同 level kernel 一致性

通过 runtime plan 中的 rotate/relinearize 顺序与 126 个 `keyswitch.hybrid` NVTX range 一一对应，准确赋予了 4432/5099 个 kernel 的 level。

`04-online-level-consistency.png` 的统计口径为：

- `GPU执行时间 Mean / 全部样本`：4GPU 全部样本平均值除以 1GPU 全部样本平均值。
- `GPU执行时间 P50 / 未与P2P重叠`：4GPU no-P2P 子集中位数除以 1GPU 全部样本中位数。
- `GPU执行时间 P50 / 与P2P重叠`：4GPU with-P2P 子集中位数除以 1GPU 全部样本中位数；格内同时标出子集样本数。
- `CPU Launch API P50/P95`：CPU host thread 上 `cudaLaunchKernel` API 时长的对应分位数之比。

关键结果：

- 不与 P2P 重叠时，主要 keyswitch kernel 的 4GPU/1GPU P50 通常为 0.99x–1.09x，计算本身基本一致。
- 与 P2P 重叠时，`hybrid_forward_ntt_q_two_components_fused_stage` 在 L8/L4 分别达到约 2.27x/2.08x，部分其他 kernel 也达到约 2x。
- 同 level keyswitch launch P50 通常为 1.66x–2.09x，P95 最高达到约 3.95x。

所有 5099 个 kernel 的 GPU service time 比 1GPU 多 3.312ms。按相同 kernel launch signature 的 1GPU 平均时间作为基线：

- 840 个与 P2P 重叠的 kernel 贡献 2.809ms 额外时间。
- 4259 个不与 P2P 重叠的 kernel 贡献 0.503ms 额外时间。
- P2P 重叠 kernel 占全部额外 GPU 计算时间的 84.8%。

## 第一轮判断

目前最明显的三个问题是：

1. 多线程 launch 明显退化，典型时间约翻倍，P95 约为单卡的三倍。
2. Compute worker 有 20.6%–30.8% online 时间处于 `pthread_cond_wait`，说明 ready task 供应不足或依赖未满足。
3. 通信覆盖率低，同时少数与 P2P 重叠的 memory-heavy kernel 会被显著拖慢。不能简单地通过增加重叠解决问题。

现有 profile 足以得到以上第一层结论，不需要在 188Server 重跑。若下一步需要拆开 Blank 或准确回答每次 `cond_wait` 在等哪个 ValueId/远端 producer，才需要增加 CPU context-switch/runtime trace 或轻量 NVTX 后重新采集。

## 图和明细

- `01-online-gpu-timeline.png`
- `02-online-p2p-coverage.png`
- `03-online-cpu-breakdown.png`
- `04-online-level-consistency.png`
- `05-online-cpu-timeline.png`
- `summary.json`
- `op_level_counts.csv`
- `kernel_signature_counts.csv`
- `kernel_level_comparison.csv`
- `p2p_coverage.csv`
- `p2p_matrix.csv`
- `cpu_thread_breakdown.csv`
