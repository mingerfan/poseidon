# Waterline 实现与 188Server 验证

状态核对于 **2026-10-14（北京时间）**。生产源已推送至Poseidon `c17effa2`，ckks-runtime `369bdc5`、DaCapo `6e75be2`。本地及188最终回归通过，**完整Qwen3-0.6B前向与数值验收尚未完成**。最新四卡GPU-only尝试在资源预检退出1：physical2被另一账号作业占用，MulCP完整组所需额度高于当前实际reserved graph capacity；没有提交图计算。详见 [当前资源阻塞、全部证据与恢复命令](188-resource-blocker.md)。先前旧前向2:29:19后的RMM失败、两次Rescale过计预检及各次修正均保留独立绑定，不混作完整成功结果。

Waterline 已接通后期 Prepare 改写、完整消费者组准入、每设备水位、共享 Host/pinned/raw 预算和事件回收，复用统一执行器的前台计算与通信。自动静态分批和批末 Fence 已替换；显式 Encode/Transfer/Fence、drain、按需读取、Release/reuse 保留，见 [清理清单](cleanup.tsv)。

## Rescale 修复与当前 188 证据

最初仅在按实际 prime 除模后修改 scale 元数据，隔离 q_proj 出现约 1.090078 倍幅值。随后的系数后乘版通过 scale120 小图，却在完整 lazy 图的 `160→40/drop5` 预检被拒绝：先除五个约 30-bit prime 会把原生 scale 降至约 10，事后乘整数不能恢复舍入精度。

当前修复先以正 uint64 整数乘全部活跃 RNS 系数，并同步实际 scale，再执行原生 Rescale；剩余计划层用普通 ModSwitch 消耗，最后发布目标 metadata。`160→40/drop5` 约为先升至 190，再降至 40。目标至少 40-bit、整数对齐相对误差不超过 `1e-9`，预乘 scale 须有限且目标低于剩余模数空间；不满足时拒绝。原 lazy 逻辑及 scale40 计划保留，未以重编译 scale120 图替代修复。详见 [数值说明](rescale-normalization-validation.md)；这些条件不是通用噪声、幅值或安全参数证明。

188 新预乘版检查监督 [exit0](188-prescale-exit-status)，实际源码/二进制绑定见 [hash](188-prescale-build.status) 和 [构建核对](188-prescale-build-verification.log)。下表覆盖真实 V100；CPU/GPU/Auto 指完整 Prepare 后端。

| 检查 | 结果与证据 |
| --- | --- |
| CPU API | 9/9：[日志](188-prescale-cpu-api.log) |
| GPU API，物理 GPU1 | 20/20：[日志](188-prescale-gpu-api-single.log) |
| GPU scalar，物理 GPU1 | 16 组：[日志](188-prescale-gpu-scalar-single.log)；uint64、原地 alias 与独立模乘说明见 [scalar 验证](gpu-wide-scalar-validation.md) |
| 真实 GPU Prepare OOM | 第二次 RMM 分配注入 `bad_alloc`，在途工作自动 drain、动态分配及 pinned live 归零，无消费者提交/隐式重跑/CPU fallback，显式下一次调用成功：[日志](188-prescale-gpu-prepare-oom-single.log) |
| 单卡压力，物理 GPU1 | CPU/GPU/Auto 全通过，maxabs `9.63267e-7`，进程墙钟 35.57s：[日志](188-prescale-waterline-single.log) |
| 全部物理 0/1/2/3 压力 | CPU/GPU/Auto 全通过，maxabs `8.78074e-7`，墙钟 54.83s：[日志](188-prescale-waterline-all.log) |
| 全部四卡强制 HostStaged | CPU/GPU/Auto 全通过，maxabs `1.01899e-6`，墙钟 54.88s：[日志](188-prescale-waterline-host-staged.log) |

压力参数为 N65536/Q32×30-bit/scale40，每模式检查全部 32768 复数槽；测试 pool cap 为 **2560MiB**，见 [配置](188-prescale-pressure-configuration.txt)。首次全部卡检查因外部进程显存增长触发 pool+outside 超过 M 的保护，保留 [失败日志](188-prescale-outside-growth-failure-waterline-all.log) 和 [exit1](188-prescale-outside-growth-failure-exit-status)；后续仅降低测试 pool cap 留出余量，未改 M 或保护条件，未终止其他账号任务。收尾动态 front/plain、pinned live 归零；RMM 活跃分配、保留池和持久资源分别报告，不能要求有意保留的资源归零。

两份全部卡日志的每个模式均记录三张 peer 卡先于延迟 GPU0 实际完成；HostStaged 每模式 20 次 post 全部观察到 pinned。设备快照只说明采样时状态，不证明整个运行独占 GPU。

同版在 188 物理 GPU1 完成逻辑 40/120 × CPU/GPU/Auto 六路组合；每路 **49152 槽**，包含生产 BSGS、polynomial RMS、5 次 Host secret-key refresh，全槽、虚部及尾部均以固定 `atol=0.001, rtol=0.0001` 通过，结束 pinned live=0，[监督 exit0](188-prescale-composite-exit-status)、[构建绑定](188-prescale-composite-build.status)。这是合成组合图，不是完整预训练模型。

| 组合逻辑 scale | CPU maxabs / 报告 | GPU maxabs / 报告 | Auto maxabs / 报告 |
| --- | --- | --- | --- |
| 40 | [8.38985e-5](188-prescale-composite-scale40-cpu-report.json) | [5.13833e-5](188-prescale-composite-scale40-gpu-report.json) | [1.25653e-4](188-prescale-composite-scale40-auto-report.json) |
| 120 | [2.14252e-10](188-prescale-composite-scale120-cpu-report.json) | [2.14253e-10](188-prescale-composite-scale120-gpu-report.json) | [2.14253e-10](188-prescale-composite-scale120-auto-report.json) |

## 本地、core 与编译器回归

| 范围 | 结果与边界 |
| --- | --- |
| CPU 新预乘路径 | 9/9；scale40 的 54 个 dense 组合各查全部 2048 复数槽，含 CP/CC、混合多层乘法、独立 AddCC/AddCP、0/±1 mask、drop5/6，maxabs `5.0261e-8`，独立阈值 `1e-4`；原高 scale、target20 拒绝及 50-bit prime 回归保留：[日志](cpu-rescale-prescale-tests.log) |
| GPU 与组合 | GPU API 20/20：[日志](gpu-rescale-prescale-api-final.log)；本地 lazy40/120 六路各 49152 槽通过：[结果](lazy-rescale-composite-regression.json)；不混用本地与 188 的误差或墙钟 |
| GPU scalar CTest | 16 组、974848 个 RNS word 与独立 128-bit 模乘完全一致，2048 复数槽及非默认 stream 通过：[日志](gpu-wide-scalar-ctest.log) |
| 统一执行器 | 最新 core 8/8：[日志](bundle-read-core-tests.log)；新增 [真实 lazy bundle 缺失 blob](bundle-read-worker-tests.log) 在 CPU/GPU Prepare worker 报错，失败消费者不提交，drain 时源保活、返回后释放；修复源后同 runtime/紧预算完成 20 组。此用例不声称错误瞬间另有 pending Prepare |
| 编译器/HEFT | 10/10：[日志](compiler-heft-preservation-tests.log)；实际四卡小图在同一 placed IR 上比较 eager/Waterline，65 条密文计算、37 条密文通信及 Host Boot 的设备、参数、ValueId、相对顺序一致，8 个 Prepare 输出共享：[逐项证据](compiler-waterline-heft-preservation.log) |

真实 CUDA OOM 清理还保留 [修复前负控](gpu-prepare-oom-local-v5-beforefix-negative-control.log) 与 [修复后正控](gpu-prepare-oom-local-v5-after-cleanup-fix.log)：旧 header 返回时 drain=0、dynamic=229376B、pinned=16384B，新版自动 drain 一次后归零。这是分配路径故障注入，不是耗尽整卡显存。完整 Qwen 同配置 HEFT 前后保持现已通过独立前态审计，见下述证据。

额外修复了GPU缓存路径的Host读取额度：非resident packed读取会同时持有原始字节和decoded double数组，动态Host须保持`N*8`，已存活的pinned缓存另计固定`N*4`。旧公式每次读取少计`N*4`（N16384时64KiB）。本地 [新版API 21/21](gpu-packed-host-budget-tests.log) 通过；真实packed暖缓存、公开Ledger/API边界的紧预算拒绝、足额预算编码全部2048复数槽通过，旧公式隔离负控确实错误接受紧预算并失败，见 [绑定与范围](gpu-packed-host-budget-result.json)。这条专门边界未执行完整Runtime结构，不能扩大为该结构的端到端反例。历史PID153204使用修复前的Host计费公式和已验证的预乘数学路径，现已失败退出；其资源记录不能当作新公式的验证。随后449cf933预检及当前actual-plan workspace版本均包含Host计费修正；数学、编码及HEFT未因此改变。

同一修复在188物理GPU1的 [隔离构建21/21](188-gpu-packed-host-budget-tests.log) 通过，UUID/ECC、精确编译链接命令、二进制及源hash、前后未改共享文件均见 [结果](188-gpu-packed-host-budget-result.json)。[复现脚本](check-packed-host-budget-188.py)读取既有Makefiles的flags，在新目录编译两份更新cpp并优先链接新object；link.map确认旧API cpp成员没有被抽取。该测试与完整模型共存，不用于性能比较；原server源码、build、runner及`logs/latest-build.status`均保持原样。 [压力脚本](check-packed-host-pressure-188.sh)原安排在模型成功结束后复查新API；模型失败后该监督未启动压力。物理池修正后的复测须另用新的构建和日志。失败旧轨迹缺少成功最终账本，不使用回溯上界替代新版完整运行；原临时回溯工具已移除。

物理池与Ready额度修正的行为、真实正负控和边界见 [验证说明](pool-reservation-and-ready-validation.md)。本地API23/23、独立RMM机制3/3、真实GPU Prepare故障收敛通过；188完整复查也已通过，[构建绑定](188-reserved-pool-build.status)、[监督exit0](188-reserved-pool-stage-exit-status)。新完整前向使用相同HEFT逻辑计划、全部四卡的物理顺序`1,2,3,0`；逻辑GPU3的固定key及单组需求较小，将它映射共享的物理GPU0，UUID由runner报告。新binary `449cf933…` 包含Host计费、物理池及Ready修复，仍待完整数值验收。

## 完整模型 HEFT 保持审计

188 使用原 Q50/逻辑40配置及同一编译器，在同一次 `plan-plaintext-waterline` 调用两侧直接保存 IR；只更换输出目录并增加前后打印选项。前后逐项比较 **4,870,497 条密文计算、3,420,801 条密文传输**及密文描述、稳定ID、参数、设备、phase相对顺序、输入/返回签名，均一致，且无遗漏记录；13,275 条 Host Boot 保持。前台规范记录 SHA256 为 `3a587b926bb6406e776bf0b16d38f7a61e5c204a38069098aaafb8bbf893ef24`，见 [完整比较摘要](188-full-qwen-heft-audit-canonical-summary.json)。Prepare/lifetime 与新增 `runtime.value_id` 元数据不属于前台记录。

[监督状态](188-full-qwen-heft-audit-status.json)为 compile=0、canonical=0、complete/passed=true；[新旧计划绑定](188-full-qwen-heft-audit-runtime-plan-binding.json)两份 binary 均为1,185,203,847B，SHA256均为 `9e906c0d32983dccbea866a7d506e677d039b57d537caa4d9bd0c568f61de813`，严格绑定在跑原计划。CPU编译墙钟1:11:17、MaxRSS137280764KiB，流式比较1231.81s；6.41GB诊断IR留服务器，未为本审计全量读取/散列61GB bundle。这是编译期保持证据，**不是模型数值或性能结果**。

精确 [命令](188-full-qwen-heft-audit-command.json)、[输入绑定](188-full-qwen-heft-audit-input-bindings.json)、[审计范围](full-qwen-heft-audit-scope.json)、[可复现监督脚本](188-full-qwen-heft-audit-runner.py)及 [流式比较器](full-qwen-heft-stream-audit.py)一并保留。比较器另经真实小HEFT图的SSA改名正控、设备/参数/类型/顺序及缺前态负控，见 [方法验证](full-qwen-heft-method-validation.json)。

## 完整 Qwen 目标与执行边界

正式模型为 `Qwen/Qwen3-0.6B`，revision `c1899de289a04d12100db370d81485cdf75e47ca`，权重 SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。已核验 311 个 BF16 张量、751632384 个存储参数（含重复 tied LM head；去重 596049920）。模型完整 28 层、hidden1024、FFN3072、16 Q heads、8 KV heads、head dim128、vocab151936；Q/K norm 在 RoPE 前，无 attention bias，共享 embedding/LM head。历史 24 层合成模型不能替代此目标。

客户端为 tokens `[9707,11]`、positions `[0,1]` 查 embedding 后加密，产生 **2 个输入、261 个密文输出**：最后位置 logits 分 149 份（末份 384 个有效值），加 28 层、两个位置的 112 份 K/V（114688 个有效值）。这不是密文 token 查表。独立 NumPy dense FP64 exact/approximate 参考的 logit maxabs `3.787683944267428e-9`、RMS `7.210505850742908e-10`，top1 均为 358。

runner 与 [独立检查器](../../scripts/qwen3_dsl/validate_decoded.py)须核对全部有效值、虚部、零尾部及 signature/fixture/reference/source 绑定，固定 `atol=0.001, rtol=0.0001`。`checker-test.log` 的 20 项使用明确标注的模拟 decoded，仅验证检查器拒绝与映射，不能作为模型数值结果。

生产计划使用 N16384、50 个 30-bit Q、8 个 30-bit P，default/input/逻辑归一化目标 scale40；逻辑 level0..9 对应物理 level4..49，每级跨 5 个物理 level，编译器每op的workspace容量估算上界 1024MiB（runtime另按实际算子分配给出上界）。latency 表 40..49 行沿用旧表末行估算，不是新硬件测量，HEFT 算法不变。提高 waterline 至120会改变 lazy/Boot 规划；临时 scale120 编译已停止且产物不完整。参数入口见 [DSL 说明](../../scripts/qwen3_dsl/README.md)。

| 完整模型阶段 | 当前可支持的结论 |
| --- | --- |
| 旧 Q40 BSGS 四卡编译 | exit0，29:55.87、RSS137225352KiB；binary1182542616B、bundle61298262016B：[日志](qwen3-bsgs-final-all-compile.log) |
| 旧 Q40 单卡容量与前向 | 此预算基线拒绝：[摘要](188-final-single-capacity-summary.json)。GPU-only 已 [superseded](qwen3-full-gpu-n16384-v3-superseded.json)，Auto PID30187 按用户要求 [停止、exit143](qwen3-bsgs-q40-user-stopped.json)，均未完成数值验收；停止记录中 signal15 与 time-v exit0 并存不能算通过 |
| 原 Q50/逻辑40 四卡编译 | exit0，30:00.36、RSS137245980KiB；19401319 指令、9710483 值、binary1185203847B、bundle61298262016B：[命令](q50-all-command.json)、[日志](q50-all-compile.log)、[状态](q50-all-status.json)；新 API 复用此计划 |
| Q50 单卡容量 | instruction #5350733 基线拒绝，required26843807744B 超过扣 reserve 后26843545600B **262144B**，exit1：[命令](q50-single-command.json)、[日志](q50-single-compile.log)、[状态](q50-single-status.json)。仅证明此配置/基线不可行 |
| 后乘 API 的 Q50 GPU-only | PID147482，Rescale 预检失败、exit1、70.05s；trace 仅54B header，无计算提交/decoded，report passed=false：[报告](188-q50-scale40-preflight-model-report.json)、[日志](188-q50-scale40-preflight-model.log)、[退出码](188-q50-scale40-preflight-model.exit-status) |
| 新预乘 API 的原 Q50 GPU-only | PID153204 实际前向2:29:19后RMM分配失败、exit1，未有完整数值结果；CPU-only/Auto 完整验收也未完成，不以小图或编译成功补足 |

[Q50 编译容量摘要](q50-compile-and-capacity-summary.json)仅汇总同一逻辑40计划的编译/容量；其 `historical_hold` 是当时外部占卡快照，**不是当前阻塞状态**。[Waterline 报告](q50-all-qwen3._hecate_qwen3_0_6b.waterline.json)的四卡单组对象/workspace 峰值为 8401649664 / 8391819264 / 8415412224 / 8138522624B，这是编译基线，不是运行峰值。旧 scanner 漏 Prepare tag5、旧 spec schema 失败均已修复，原失败日志保留。

## 局部调度诊断

同一已编译lazy40合成组合图的窗口1/8 × CPU/GPU/Auto，每配置3次轮转，共18次无trace运行及884736个复数槽独立检查均通过，最大误差`1.56393e-4`，见 [比较脚本](prepare-backend-local-comparison.py)、[完整指标与绑定](prepare-backend-local-comparison-result.json)、[日志](prepare-backend-local-comparison.log)。前向中位数（ms）如下；窗口1仍是Waterline单组窗口，并非旧eager或绝对无预取。

| 窗口 | CPU | GPU | Auto |
| --- | --- | --- | --- |
| 1 | 1106.31 | 705.56 | 1115.14 |
| 8 | 963.23 | 663.66 | 744.68 |

窗口8的Auto每次实际选择21份CPU、26份GPU准备，窗口1为47份CPU。窗口8等待中位数CPU/GPU/Auto为3.96/8.97/2.35ms，等待更少未等于端到端最快。重复存在波动，不能推及四卡/完整模型；该比较使用已有预乘数学runner `81f8cf10…`、足量2GiB Host预算，未使用新Host边界公式，不作为该修复的验证。输入、全部小bundle、runner及可用build绑定前后均一致，详细重复报告留build目录。

本地 RTX4060 上，在同一图中将窗口从8改到32/64，中位数约改善2–3%，重复范围重叠，未建立可靠收益，见 [窗口扫描](lookahead-local-sweep-result.json)。隔离 q_proj 的私有 runner 仅修改无进展时的等待语句，100µs/10µs/yield 前向中位数为2733.7/2171.7/1830.6ms，8192槽均通过；Nsight 显示 kernel种类、grid及次数不变、GPU空隙减少，同时 yield 的事件查询和CPU成本增加，见 [诊断](dispatcher-poll-local-diagnostic.json) 与 [原始日志](dispatcher-poll-local-diagnostic.log)。这些是合成单卡图的结果，不能换算完整模型加速率；生产保持100µs，188当前运行未替换。

复现前须完成现有API/runner和DaCapo构建。私有runner脚本读取 **Unix Makefiles** 生成的 `flags.make/link.txt`，不支持直接套用Ninja目录。组合图可由 [公开生成脚本](lazy-rescale-composite-regression.py) 创建；q_proj 使用188工作目录的 `q50-small-regression.tar`，SHA256为 `a57d207fe11dc50d088be0a7427b660d928e535e462c9189ac3c58a80290c192`，69,130,240B，解包后包含 `q_proj/` 与 `profiles/`。它是scale120合成cyclic诊断图，不是生产lazy40模型；大产物留服务器，需显式提供已有输入，不能从空checkout直接运行历史build路径。

```sh
python test-results/waterline-20261013/lazy-rescale-composite-regression.py \
  --output build-lazy-diag-fresh --backends gpu
python test-results/waterline-20261013/lookahead-local-sweep.py \
  --source-directory build-lazy-diag-fresh/scale40 --output build-lookahead-fresh
mkdir build-q50-small-fresh
tar -xf /path/to/q50-small-regression.tar -C build-q50-small-fresh
python test-results/waterline-20261013/lookahead-local-sweep.py \
  --source-directory build-q50-small-fresh/q_proj --output build-q50-lookahead-fresh
python test-results/waterline-20261013/poll-private-runner-sweep.py \
  --source-command build-q50-lookahead-fresh/repeat-0-w8/command.json \
  --output build-poll-private-fresh
python test-results/waterline-20261013/poll-private-runner-cpu-cost.py \
  --sweep-output build-poll-private-fresh
```

私有runner、生成计划、decoded和Nsight大文件留在build目录；仓库仅保留脚本及小型诊断摘要。[资源绘图脚本](plot-model-resources.py)从完整PID采样生成独立PNG及带输入hash的摘要，缺失Host指标不会补成0；当前运行中的预览不作为最终测量。

## 保留的组件与历史证据

- **pinned 缓存与 Auto 开销**：旧 trace 的 Prepare/Host Boot 提交分布见 [快照](188-bsgs-auto-pre-supersede-trace-stats.json)。同二进制 N16384/Q40/P8/scale120 组件对照中，10 次 GPU Prepare 的 pinned 分配由 10 次降至 1 次，提交均值 1.025306→0.219797ms、回收 0.417778→0.012027ms，Encode event 0.272096→0.270313ms，drain 后 pinned live 均为0：[对照](hotpath-pinned-cache-comparison.md)、[绑定](hotpath-cache-188-current-provenance.txt)、[Nsight 范围](hotpath-components-profile-validation.md)。这是组件改善，不是完整模型加速率；固定缓存计入整段 Host/pinned 预算，真实 Ready 前禁止覆写。
- **隔离 Q50 q_proj**：较早后乘 API 的 scale120 合成 cyclic 小图，无 Boot，1025 Prepare/Rescale；188 全8192复数槽 CPU/GPU/Auto maxabs 约 `9.91e-10`：[CPU](188-rescale-q-proj-cpu-report.json)、[GPU](188-rescale-q-proj-gpu-report.json)、[Auto](188-rescale-q-proj-auto-report.json)。它未覆盖生产 lazy160→40/drop5，也未替换正式 BSGS DSL。
- **Q50/P8 key backing**：188 GPU1 在 Q=5（小于P=8）、25、50 下，Relinearize/Rotate/composite/reuse 各查全部8192复数槽，独立 oracle maxabs `4.95184e-8`，与 CPU 差值0；单份 backing159645696B，未按 active-Q 重复保存：[日志](188-key-backing-q50-p8-physical1.log)、[命令](188-key-backing-q50-p8-physical1.commands.json)。不含 Rescale，不是完整模型验收。
- **旧压力与反例**：[后乘版 exit0/hash](188-rescale-build.status)、[失败清理版四卡](188-failure-cleanup-waterline-all.log)/[HostStaged](188-failure-cleanup-waterline-host-staged.log)、[pinned 版 hash](188-pinned-reuse-source-and-binary.status)、[Host 预算边界](cpu-prepare-host-budget-validation.md)、[stream Ready 正负控](stream-ready-validation.md)均保留。旧 pinned 压力与模型有重叠，其墙钟不可作为独占性能对照，旧结果也不代替本轮预乘版验证。

## 复现与剩余交付

大型权重、trace、plan、bundle、decoded、profiler 文件留在 `/home/xuming/poseidon-waterline-20261013`，不提交 Git。远程编译、测试及取回保持独立长 SSH；tmux 仅负责任务持久化。单卡用物理 GPU1，多卡用全部 0/1/2/3，记录 UUID/ECC；按本进程身份采样，保留外部作业。

公开 [build-188.sh](build-188.sh)使用188离线依赖、CUDA arch70，构建 CPU/GPU API、scalar、GPU Prepare OOM、Qwen runner 五个 target，保存日志/hash；[check-rescale-188.sh](check-rescale-188.sh)核对实际构建绑定并执行 API、OOM、三配置压力和 q_proj 三后端。小图 tar 与新日志目录须按脚本准备，禁止覆盖旧证据；旧 `188-build-reproduce.log` 仅证明旧版增量复现。

```sh
base=/home/xuming/poseidon-waterline-20261013
POSEIDON_188_WORKSPACE="$base" bash build-188.sh
# 先按脚本准备 "$base/q50-small-regression.tar" 和新的检查目录。
POSEIDON_188_WORKSPACE="$base" POSEIDON_RESCALE_CHECK_LABEL=rescale-prescale-fresh \
  bash check-rescale-188.sh
```

[组合图脚本](run-lazy-rescale-composite-188.sh)在物理GPU1重跑两个已生成小图的六种模式，输入目录与新日志label可显式配置；具体生成与转移见 [Rescale说明](rescale-normalization-validation.md)。

[run-188-acceptance.sh](run-188-acceptance.sh)通过 `POSEIDON_QWEN_BACKEND=cpu|gpu|auto` 选完整 Prepare 后端；复用压力须满足脚本的同构建 hash、既有 exit0 与配置要求。原 Q50 精确编译命令见上表 JSON，`--waterline=40` 保持有效；用新构建及新 label 执行，重新编译只允许 1/4 卡顺序运行，失败最终返回非零。启动脚本本身不构成模型通过。

`monitor_process.py`在 CUDA setup 前按 PID/身份采样 RSS/HWM 和 driver 显存；采样峰值不是连续峰值。runner 分列冷启动、setup/init/forward、首个/全部输出、Host Boot、Prepare、GPU events、H2D、等待和收尾，外部 time-v 覆盖进程冷启动。首输出时间指完整run返回后首个解密结果，并非流式首个密文Ready时刻。共享Host/raw承诺峰值取最终stdout的Waterline账本统计，`runtime_timing.raw_peak_bytes`仅是单次读取峰值。CSV 字节数、提交数、ordinal 均不能当 CUDA 完成率或剩余时间，利用率快照也不足以证明卡死。 [流式trace汇总器](summarize-api-trace.py)只读取已关闭trace，统计Host API调用；跨worker与嵌套调用的时间会重叠，不能作为墙钟占比或CUDA kernel时长。

仍需交付完整261输出和独立数值报告、实际完整后端覆盖、每卡/Host/pinned峰值、收尾清单；完整模型同配置 HEFT 前后保持证据已通过，见 [覆盖审计](waterline-coverage-audit.md)。本参数为 `sec_level_type::none`；计划含13275条 Host decrypt/re-encrypt refresh，使用秘密密钥，**不是原生 FHE Boot**。固定两 token 校准不证明其他输入精度；Native Boot 的 Waterline workspace 查询尚未接通。当前没有完整模型端到端加速证据。

旧PID153204的closed trace已完成一次流式汇总：[统计](188-q50-lazy40-failed-api-trace-summary.json)、[命令及输入绑定](188-q50-lazy40-failed-api-trace-command.json)、[性能口径](188-q50-lazy40-failed-api-performance-interpretation.json)。这是失败的部分前向，不能当完整模型统计。Transfer/Rescale/Host refresh的大量主机API调用有实际耗时；不同worker可重叠、计时可嵌套，累计量不代表墙钟占比或CUDA kernel时间。CSV无绝对时间，不能事后重建队列等待。

`consumer_wait_ms`只记录foreground CUDA stream对所有CT/Plain输入事件的等待区间，不含Prepare发布value前的CPU、队列或准入等待，也不独立归因为明文。启动填充计入cold/运行总时间，未独立单列；`initialization_ms`不等于填充耗时。未据这些指标声称准备完全隐藏或完整模型加速。

Rescale工作区计费现使用完整已验证plan的实际最大降层数，保留Qfull和三分量上界；支持能力50不再当作每次执行50层。本地 [24/24及实际三分量分配](actual-rescale-workspace-api-validation.json)、188 [同版完整检查](188-plan-bound-checks-summary.json)通过，源API为`9a74df35…`，runner为`bdfb7483…`。新图/重复或失败验证会重新发布或清除该最大值；数学与HEFT不变。完整模型仍须重新预检并完成数值对照。

包含以上修正的完整GPU-only复跑于188时间18:31启动，PID185043，源提交`00a65df7`、runner `bdfb7483…`；[启动记录](188-plan-bound-full-launch.json)与 [实际进程绑定](188-plan-bound-full-live-binding.json)核对同一原计划、argv、物理四卡映射和binary。该次随后在#17049920 Rescale预检退出，差7,350,368B，零compute；[报告](188-plan-bound-preflight-model-report.json)、[层级上界诊断](188-plan-bound-preflight-workspace-analysis.json)。正在按已验证target+maxgap的Q前缀上界继续修正，不以启动成功当作验收。

最新分层上界已通过 [本地24/24](level-aware-rescale-api-validation.json)及 [188最终同版检查](188-level-aware-checks-summary.json)：Rescale临时Q按target+已验证最大gap推导，固定scratch仍按完整Q预留，详见 [分配证明与边界](pool-reservation-and-ready-validation.md)。源API `af90f470…`、测试 `962f909e…`、188 runner `c261b9a5…`。此前`00a65df7`及两次预检失败分别保留原始绑定，均没有完整数值结果。

在188保留SSH长连接后，可用最新源码执行 [构建脚本](build-188.sh)和新的`POSEIDON_RESCALE_CHECK_LABEL`运行 [回归脚本](check-rescale-188.sh)。完整运行继续使用 [同一验收入口](run-188-acceptance.sh)：设置`POSEIDON_QWEN_CUDA_VISIBLE_DEVICES=1,2,3,0 POSEIDON_QWEN_BACKEND=gpu`及全新的`POSEIDON_ACCEPTANCE_LABEL`；仅在`latest-build.status`完全一致时，用`POSEIDON_VERIFIED_PRESSURE_DIR`指向成功的`logs/level-aware-rescale-final-checks-20261014`。不设置旧initial/max pool环境变量，保持模型及既定容差；入口会核对261输出并运行独立检查器。

另核对了level map复制这一可疑开销：[局部CPU审计](context-level-map-audit.md)。本地每百万次按值查找约450ms，const引用约3ms；未测其在完整模型中的次数或墙钟贡献，不能解释约149分钟慢跑或声称模型收益。本阶段不改这一公开getter、不引入私有缓存，继续完整模型验收。

最新分层上界的源提交为`c17effa2`，已推送；全模型新进程PID188271于188时间18:56:53启动，[launch](188-level-aware-full-launch.json)、[实际exe/argv/plan绑定](188-level-aware-full-live-binding.json)。该次随后在logical1/physical2的MulCP完整组预检退出1，workspace=0，现有reserved graph额度不足；没有图计算提交。详见 [当前资源阻塞及复现](188-resource-blocker.md)。

**当前资源阻塞，完整模型未验收。** 最新源`c17effa2`及全部回归已推送；全模型GPU-only的最新尝试因physical2他人作业占用，保证式单组预检拒绝，见 [完整证据、限制与恢复命令](188-resource-blocker.md)。没有完整261输出成功数值报告。旧失败模型的 [已关闭资源采样图](188-q50-lazy40-failed-resource-history.png)仅作失败历史测量，不能替代新代码完整验收。
