# 旧 GPU-only 运行的 packed-read Host 计费上界

本文给出旧运行在修正 Host 公式后的回溯上界，不代表修正后的二进制完成了完整模型，也不替代独立数值验证。本文未填入当前完整模型的最终结果；必须取得匹配的成功退出状态和最终账本后再运行检查器。

## 公式与额度生命周期

审计对象是数学修正 `3d4cff66` 与 Host 修正 `579012ad` 之间的这一项资源查询差异，见 [GPU API](../../src/poseidon/runtime_api/poseidon_gpu_api.cpp)：缓存开启时，每个 GPU Prepare 的临时 `peak.host` 从 `4N` 修正为 `8N`；`peak.raw=8N` 不变。每设备缓存的固定 Host/pinned 额度各为 `4N`，保持不变且已经计入旧账本。缓存关闭时本次修正不增加该 Prepare 的 Host 额度。

设 `A(t)` 为时刻 `t` 已准入、其临时额度尚未回收的缓存 GPU Prepare 集合。它包括尚在排队、尚未开始读取或提交 kernel 的 Prepare，不只是正在执行者。每项额度在完整组准入时预留，在相同的 Ready 完成检查后通过一次 `release(state.temporary)` 一起回收，见 [执行器](../../third_party/ckks-runtime/runtime/runtime.hpp)。所以在旧运行的同一准入及完成轨迹上：

```text
ΔHost(t) = Σ[p∈A(t)] 4Np = GPU_cached_raw(t) / 2
         ≤ old_raw(t) / 2
corrected_host(t) = old_host(t) + ΔHost(t)
                 ≤ old_host_peak + ceil(old_raw_peak / 2) = U
```

同一 ledger 中其他 raw 使用者只会使上界更保守。不同设备及不同 Prepare 的额度都累加到 shared raw；共享输出按已准入的 Prepare 去重，不按消费者重复收费。旧 `host_peak` 包括固定缓存、常驻 bundle、Host 值、workspace 及临时 Prepare 等已管理额度，不能再把固定缓存加一次。Host 和 raw 峰值可以发生在不同时刻；两峰值相加仍是有效上界。

若 `U ≤ HostBudget`，修正后的 Host gate 在旧运行的每个实际准入状态都可行。若 raw budget 是 `67108864` B（64 MiB），额外 Host 至多 `33554432` B（32 MiB）；检查器仍使用观测到的 shared raw 峰值计算更紧的上界。若 `U > HostBudget`，只能说此充分条件未建立，不能据此反推实际越界。

## 未输出峰值的单组 baseline

baseline 使用独立 ledger，其峰值不在最终实际执行摘要里。这里不把两者混为一谈，而使用当前单 rank、固定 GPU 后端、全局拓扑前缀准入方式的支配关系。

对第 `j` 组，比较 baseline 预留本组后的状态与实际执行预留同一组后的状态。baseline 已完成所有前序组，实际只完成其中一个子集，而且未准入后续组。baseline 尚需保留的 storage 在实际账本中也必须保留；实际即使尚未产生输出，也已在准入时预留其 storage。旧 Prepare 在 baseline 中最多保留 resident，实际可能另保留临时额度；本组首次需要的 Prepare 双方预留相同成本。baseline 只保留当前组 workspace，实际可能还保留前序未完成组的 workspace。

`Release` 和 in-place `reuse_input` 仅在完成时标记可回收，且还必须满足剩余引用计数为零并非 external input，才回收 storage。实际不能先于 baseline 回收仍被当前或后续前缀引用的对象。phase/Fence 保留同一 ledger 的固定和常驻额度，边界先等待前段完成，不破坏上述关系。因此 baseline 的修正 Host 用量受对应实际准入点的修正 Host 用量支配。旧 baseline 已通过，修正只增加 Host 维度，所以 `U ≤ HostBudget` 也足以覆盖该 baseline 的 Host 项。

此推导限于上述执行器、相同计划/输入/配置与固定 GPU 后端。Auto 可能改变后端选择，不在本证明范围。不同 rank、独立 runtime/API 实例或其他绕过该 ledger 的并发工作不能合并套用这一结论。

## 可复现检查

最终摘要由 `runtime.hpp` 在 initialization、execution、finalization 三阶段结束后写到 **stderr**，格式严格为：

```text
waterline shared host_peak=<整数> pinned_peak=<整数> raw_peak=<整数>
```

使用该次运行合并 stdout/stderr 的 `model.log`；不能使用 report 的 `runtime_timing.raw_peak_bytes`，后者只是单次 bundle 读取的最大值，不是并发 shared raw 额度。调用者须核对日志、exit-status、构建身份和预算均来自同一旧运行；日志本身不含 Host budget，检查器不会猜测或默认预算。

```sh
python3 test-results/waterline-20261013/packed-host-retrospective-bound.py \
  --model-log /path/to/completed/model.log \
  --exit-status /path/to/completed/model.exit-status \
  --host-budget-bytes 68719476736 > /path/to/packed-host-bound.json
```

示例预算是 64 GiB，使用前须与该次运行实际生效的 `host_budget_bytes` 核对。检查器只读输入、输出 JSON，记录日志、退出状态文件和自身 SHA-256。它要求唯一的 GPU-only QWEN header、唯一的 `FORWARD begin/complete`、顺序正确的唯一 shared 摘要及匹配的 exit0 文件，并拒绝读取期间变动的输入。进行中、失败、缺文件、缺预算、重复/畸形摘要及非固定 GPU 后端均报错（exit2）；有效输入但上界超预算输出 `bound_established=false`（exit1）；建立此上界时 exit0。

输出始终将 `new_binary_full_execution_verified` 和 `independent_numerical_validation_verified` 设为 false。检查器验证的是账本不等式及必要日志格式，不会从 exit0 推导独立解密验证，也无法仅靠这些文件证明调用者提供的预算或构建身份正确。

该结论不约束整个进程 RSS、CUDA/context/key 和图元数据等未托管分配、OS 分配器开销或运行后输出解密/验证阶段；也不声称新二进制的实际调度、耗时和完整运行结果相同。
