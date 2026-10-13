# Waterline 清理与资源覆盖审计

更新日期：2026-10-13，状态核对至20:36（北京时间）。依据适用 AGENTS、最新两份阶段文档、当前源码和本目录日志。完整预训练 Qwen3-0.6B 的最终 BSGS Auto 实际前向正在运行，完整输出和数值验收仍未完成；小图、完整编译和 reader 修复分别记录。

当前已推送引用为 Poseidon `69217e3e`、ckks-runtime `f41b930`、DaCapo `ad35c87`，父仓库引用已更新。包括生产 binary Prepare scanner、CPU Host/pinned 同时存活峰值修复及统一执行器100µs轮询。本文不把工作区其他用户文档整理、删除或 AGENTS 修改归入这些提交。

## 已替换和保留的路径

| 项目 | 当前行为与可查证入口 |
| --- | --- |
| 旧静态分批 | DaCapo 删除 `PlanPlaintextStreaming.cpp` / `check-streaming.py`，以 `PlanPlaintextWaterline.cpp` / `check-waterline.py` 替换；不按批克隆、重编码或插自动 Fence |
| 旧生产选项 | 删除 `=stream`、`runtime-plan-prefetch-bytes`、`runtime.streaming`；optimizer 只接受 `eager` / `waterline`，运行时使用 `lookahead_groups` |
| HEFT | `AssignPlacement.cpp` / `LatencyEstimator.cpp` 在 `39c1fb2..ad35c87` 无差异；Prepare 改写在 placement/communication 后运行。编译器回归检查 200 个消费者的设备和顺序不变 |
| 单一执行器 | `run_parallel_tasks` 复用原计算/通信执行函数，按完整组和一个拓扑准入前缀推进；没有另一套静态批次执行器 |
| 有用能力 | 保留普通 Encode/Transfer、串行 CPU worker、按需 bundle、binary plan/manifest、Ready、Release/reuse、显式 Fence 和收尾 drain；普通 Sequential 仍可执行，Waterline 要求 PerDeviceWorkers |
| GPU 编码 | 正式 API 使用上下文 N/Q/scale 的完整 FP64 FFT、宽整数 RNS reduction 和 CUDA NTT；独立实验数学回归保留，但不当作模型或加速证明 |

生产源码搜索未发现被删 pass/options 的调用方。runtime/Poseidon 普通 Encode/Fence 回归及 DaCapo 样例名称已按当前语义清理，相关能力和断言保留。MPI runner 的 `streaming_timing` 是既有报告字段，继续保持报告兼容；它不启用已删除的静态分批机制。

## 资源和事件修复

| 问题 | 当前修复 | 有效证据与范围 |
| --- | --- | --- |
| 按 active Q 重复上传 keys | 每卡单份 owning Relin/Galois backing；preflight 合并 Q 上界和 rotations，evaluator 使用 active-Q view | `key-backing.log`；`key-backing-n16384-q40-p8-test.log` 在真实 CUDA 的 Q8/24/40 全 8192 complex slots 与 CPU 差值为零，只证明 key/view 路径 |
| 参数表、Encoder持久表及缓存漏计 | `GpuParameterData::allocated_bytes()` 查询真实表；初始化提交前核对持久资源、retained Relin/Rescale 上界及余量 | `core-tests-validation-hook.log`、`local-waterline-metrics.log`；支持最多三组件，Native Boot Waterline bound 明确拒绝 |
| RMM 无硬 cap，外部占用误计模型 | managed pool 指定硬上限并计 live/peak/reserved；scalar 或按 CUDA ordinal 的 cap 列表；记录建池前 driver 占用，后续 pool 外增长保守计入 M | `pool-cap-test.log`、`188-peer-completion-all.log`；existing GPU0 作业保留。快照不是全程 driver peak |
| allocator 小于 M 仍过度承诺 | runtime `f26b72c` 在同一全图/在线账本追加 `front+plain<=allocator_capacity`；Poseidon扣持久分配、retained workspace与64 MiB余量 | `core-tests-allocator-capacity.log` 7/7；实际 managed-pool 容量见188小图，不把 Mock 字节数当V100实测 |
| 后期组不能执行、Host waiter 堵 CPU、跨 Fence Prepare | 初始化前检查全图基线；完整多资源一起准入；只提交就绪 Host work；Prepare 跨阶段注册、在首次消费者所在片段启动，未用 Prepare 不提交 | `core-tests-pressure.log` / `core-tests-failure-multigpu.log`；CPU/GPU 第2次 Prepare 失败直接传播，显式解除注入后同实例再次运行成功 |
| 同线程不同 stream 被错误跳过 | Ready 同时核对真实生产 stream；全部146个 kernel launches 显式使用实际 stream，D2D/zero helpers也保持顺序 | `stream-ready-validation.md`、`stream-reclaim-consumer-wait.log`；AddCP c1 PTDS 正/负对照确定性捕获旧缺陷 |
| 全卡 collect 在调用者 stream 释放别卡 buffer | `ScopedCompletedReclamation` 用各卡合法 PTDS 回收，并恢复 caller CUDA device；keyswitch null-stream copy 修复 | `stream-reclaim-api-repeat-status.json` 10×19/19；188全四卡真实三模式通过，无 invalid resource handle |
| 大型 V4 binary scanner 不识别 tag5 | runtime `9e379fc` 补 Prepare record 扫描，错误含 tag/record/offset | `v4-binary-core-tests.log` 8/8；`v4-binary-compiler-direct-roundtrip.log` 直接编译并由正式 reader 往返；保留首个模型 load 失败日志 |
| 普通/GPU Encode 计时竞态及 Prepare 查表成本 | 共用 timing mutex；descriptor 查询一次排序、逐项 lower_bound，避免完整模型 O(Prepare×Value) | 已纳入 `f26b72c` / `b551ee2f`，不再是工作树待修项 |
| CPU Prepare Host 漏计 pinned 上传源 | CPU uint64 plaintext 与 uint32 pinned staging 同时计入 Host，pinned 另受子预算约束；GPU raw/padded Host 不重复增加 | compiler `17be5ff` degree8 CPU/Auto Host480拒绝、512通过，GPU Host64通过；最新188三份压力日志均包含真实紧预算拒绝/足量全槽通过 |
| 无进展轮询延迟 | `f41b930` 将原执行器 sleep 从1ms缩至100µs，仍按同一Ready、完整组和拓扑前缀推进 | `core-short-poll-tests.log` 8/8和真实GPU回归；本机sleep探针不证明188或完整模型加速 |

预算同时要求每卡 `plain<=alpha*M`、`front+plain+reserve<=M`、后端 allocator bound，以及共享 Host/pinned/raw 约束。未完成使用仍计费。RMM live 是池内活跃量，reserved 包含空闲块，二者不能相加；API 存活时 key/table live 不应归零。账本 Host 不包含整个进程 context、Host keys、fixture/reference 或 graph metadata，需另测 RSS。硬 cap 与保守界不能保证所有碎片或动态外部占用下的分配必然成功，实际失败直接报错。

CPU Encode 统计同步完整 Encode 主机时间；GPU Encode、raw H2D、transport H2D 和消费者 wait 使用实际 CUDA events。transport 不含 direct key upload 或 GPU encoder raw H2D，不能重复相加。API/Encoder counters 为实例累计，transport 为进程累计；连续 CPU/GPU/Auto 应取相邻差值。CSV API 提交时长及各卡 event 时间总和都不能当端到端墙钟时间或并发收益。

## 188Server 与模型证据

本次记录版本 `69217e3e` / `f41b930` / `ad35c87` 的真实188小图使用 N65536/Q32×30-bit/scale40，五轮各卡独立 AddCP 和真实 ring，全部32768 complex slots比较：

| 配置 | CPU/GPU/Auto 最大绝对误差 | CPU/GPU/Auto forward ms | 范围 |
| --- | --- | --- | --- |
| 物理GPU1→逻辑0 | 均8.96891e-7 | 125.398 / 35.5416 / 115.524 | 单卡真实准备、Ready、reserve/pool/allocator与计时 |
| 全部物理0/1/2/3，默认通信 | 均9.3044e-7 | 564.622 / 217.861 / 356.733 | 每模式4个输出；3张peer真实完成均早于延迟GPU0；Auto新增1 CPU+3 GPU Prepare |
| 全部物理0/1/2/3，强制HostStaged | 均9.71521e-7 | 1026.95 / 759.282 / 796.408 | 每模式20次真实post、20次post时观察到pinned；3张peer先于GPU0完成；Auto新增1 CPU+3 GPU Prepare |

日志为 `188-final-waterline-single.log` / `188-final-waterline-all.log` / `188-final-waterline-host-staged.log`，均exit0、pinned live=0、每卡remaining front/plain=0。默认四卡Host/pinned/raw账本峰值268435456 / 67108864 / 2097152 B；强制HostStaged配置Host318767104、pinned50331648、raw1048576 B，三模式峰值分别268435456 / 50331648 / 1048576 B，其中pinned/raw确实达到配置边界。HostStaged实际pinned allocation进程累计峰值33554432 B，区别于逻辑承诺峰值。四卡RMM allocated peak最高1208922248 B、各卡pool reserved/cap为3690987520 B；live保留了API持久资源，不能要求为零。

三份最新日志均包含CPU Prepare紧Host预算157810688 B在消费者提交前拒绝（新增CPU Prepare calls=0），足量162004992 B完整32768实虚槽通过；单卡/默认四卡/HostStaged最大误差依次9.18525e-7 / 9.2084e-7 / 9.17894e-7。较早 `188-peer-completion-*` 对应 `b551ee2f`，其中GPU0外部作业快照20280 MiB与当时测试进程分开；较早 `single.log` / `all.log` 对应 `157999fc` / `04819dd`，均不混作最新版本证据。

公开 `build-188.sh` 已在188实际执行，`188-configure-reproduce.log` 完成configure/generate，`188-build-reproduce.log` 两个正式target完成且time-v exit0。该次为增量复现（0.61s、RSS15744 KiB），不能当作全量冷构建耗时。二进制/源码hash见 `188-latest-build.status`。`run-188-acceptance.sh` 按单卡物理1、全部四卡默认通信、全部四卡强制HostStaged、最终完整BSGS Auto顺序执行；远程期间独立长SSH始终保留，tmux不是保活连接的替代。

真实 checkpoint 为 Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`，SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；311张量、28层、hidden1024、FFN3072、Q/KV heads16/8、head_dim128、vocab151936，tied embedding/LM 相等。客户端查 embedding 后加密 tokens `[9707,11]`、positions `[0,1]`。输出为完整151936 logits和全部层两位置KV，共2输入、261输出。`qwen3-reference*.log` 是独立 dense NumPy FP64 精确/近似方程参考，不是FHE输出。

旧 N65536 单/全四卡容量失败保留在 `qwen3-{single,all}-capacity.log` 与 grouped 对应日志。N16384/8192 slots旧矩形对角线四卡编译通过见 `qwen3-n16384-all-compile.log`（19分01.49秒、RSS43571480 KiB）；单卡旧证据见 `qwen3-n16384-single-capacity.log`。旧GPU-only PID22555实际前向被主执行代理核对own cmdline后SIGTERM，以优先最终BSGS/Host修复/100µs版验收；`qwen3-full-gpu-n16384-v3-superseded.json` 明确未完成数值验收，不是timeout/deadlock诊断。旧time-v虽打印exit0，同时记录signal15，不能算完整模型通过。

BSGS B64 linear保留全部矩阵尺寸、权重对角线和输出。`dsl-linear-bsgs-production-report.json` 的独立dense FP64 oracle通过七形状×dense/short、脏物理尾部和七项bias，max1.1102230246251565e-14，尾部污染为零；它只是计算图布局检查。最终四卡BSGS图 `qwen3-bsgs-final-all-compile.log` 已编译/导出exit0（29分55.87秒、RSS137225352 KiB）；binary1182542616 B、bundle61298262016 B、values9692017、instructions19364300。eager对象估计报告不是Waterline实际峰值。

最新同配置单卡检查 `qwen3-bsgs-final-single-capacity.log` 在instruction #5388456拒绝：front26212827136、plain/workspace3145728、capacity26843545600、waterline16642998272 B，exit1（7分41.38秒、RSS129037504 KiB）。这是此配置/基线下容量限制，不证明全部单卡配置不可行。

完整模型覆盖截至20:36：CPU-only完整模型未执行；GPU-only旧矩形图前向被superseded，最终BSGS GPU-only完整模型未执行；最终BSGS Auto在全部物理0/1/2/3运行，PID30187，尚无完整decoded、runner report、exit-status或独立数值报告。实际目录为服务器 `logs/qwen3-bsgs-auto-final`，监测快照见 `188-read-only-monitor.jsonl`。CSV记录API提交，增长字节数不是完成百分比，小图三模式通过也不补足完整模型覆盖。

## 文档清理及尚未交付项

本次对用户工作树文档只修改旧 stream/static batch 的失效推荐及直接相关 V4 说明，备份位于不入 Git 的 `build-waterline-sync/doc-cleanup-before/`。涉及 runtime README、docs入口、overview、runtime-plan README、DaCapo integration README/communication profile及嵌套DaCapo README。当前工作树只推荐 `eager|waterline`，链接 `docs/runtime-plan/v4.md`，保留普通V3 Encode/Fence兼容语义。基于HEAD的聚焦补丁已单独提交为runtime `140d5da`和DaCapo `b31072b`；未暂存用户混合整理、AGENTS修改或用户删除的旧docs。

尚未交付：最终完整模型前向和261输出按事先固定 `atol=0.001/rtol=0.0001` 核对，完整模型实际CPU/GPU/Auto覆盖范围及冷启动/等待/峰值/收尾汇总，全图HEFT改写前后指纹交付。真实四卡共享有限预算和强制HostStaged已通过上述最新压力日志，不再列为未运行。Native Boot资源界不支持；当前模型Host decrypt/re-encrypt refresh持有秘密钥，security level none，不能称为原生安全FHE推理。固定两token非线性校准不证书化任意输入。

旧全零错误和无法取得stack的hang日志保留；确定性对照证明了stream缺陷，不能据新回归通过反推旧hang的唯一根因。VPN恢复及本机重启不是模型完成证据；远程工作仍须持续保活SSH。最终状态以实际模型输出、当前版本日志与明确限制为准。
