# Waterline 清理审计

本文维护已替换与保留的源码边界，逐项引用依据见 [cleanup.tsv](cleanup.tsv)。当前运行状态、提交与测量以 [验证入口](README.md) 为准；反例覆盖和实际限制见 [覆盖审计](waterline-coverage-audit.md)。完整模型尚未数值验收，不以清理完成代表整阶段完成。

## 已替换和保留的路径

| 项目 | 当前行为与可查证入口 |
| --- | --- |
| 旧静态分批 | DaCapo 删除 `PlanPlaintextStreaming.cpp` / `check-streaming.py`，以 `PlanPlaintextWaterline.cpp` / `check-waterline.py` 替换；不按批克隆、重编码或插自动 Fence |
| 旧生产选项 | 删除 `=stream`、`runtime-plan-prefetch-bytes`、`runtime.streaming`；optimizer 只接受 `eager` / `waterline`，运行时使用 `lookahead_groups` |
| HEFT | `AssignPlacement.cpp` / `LatencyEstimator.cpp` 在 `39c1fb2..6e75be2` 无差异；Prepare 改写在 placement/communication 后运行。实际 HEFT 后的四GPU小图保持65条密文计算、37条密文通信及Host Boot的设备与相对顺序；另保留200消费者回归 |
| 单一执行器 | `run_parallel_tasks` 复用原计算/通信执行函数，按完整组和一个拓扑准入前缀推进；没有另一套静态批次执行器 |
| 有用能力 | 保留普通 Encode/Transfer、串行 CPU worker、按需 bundle、binary plan/manifest、Ready、Release/reuse、显式 Fence 和收尾 drain；普通 Sequential 仍可执行，Waterline 要求 PerDeviceWorkers |
| GPU 编码 | 正式 API 使用上下文 N/Q/scale 的完整 FP64 FFT、宽整数 RNS reduction 和 CUDA NTT；独立实验数学回归保留，但不当作模型或加速证明 |

生产源码搜索未发现被删 pass/options 的调用方。runtime/Poseidon 普通 Encode/Fence 回归及 DaCapo 样例名称已按当前语义清理，相关能力和断言保留。MPI runner 的 `streaming_timing` 是既有报告字段，继续保持报告兼容；它不启用已删除的静态分批机制。

## 资源与语义

统一执行器按完整消费者组、拓扑前缀与真实 Ready 推进；同时限制每卡 plain、水位、front/工作区/通信、allocator容量及共享Host/pinned/raw。固定GPU表、key backing和retained workspace先核对reserve。CPU plaintext与pinned上传源同时计Host；GPU raw pinned按每卡固定缓存计费，Ready前禁止覆写。非resident packed读取的字节存储与decoded数组同时存活，动态Host保持N*8，不能因固定缓存另计而减去其中一份，专门正负控见 [Host读取边界](gpu-packed-host-budget-result.json)。失败返回前drain并清理runtime所有权，原始错误直接传播，不自动重试或换后端。

计费Host不包括调用者context、Host keys、fixture/reference和图元数据，另记录进程RSS。RMM live、reserved和driver占用分别报告；外部占用不归给模型。GPU event与Host API时长不相加为端到端墙钟。Native Boot的Waterline工作区界尚不支持，普通显式Native Boot保留。模型Host refresh使用秘密钥，security level none。

用户预先存在的文档整理、AGENTS修改和删除未混入本阶段提交，未整体回退混合提交。后续只持续维护当前协议、使用入口与证据；完整执行和数值结果以实际生成的报告为准。
