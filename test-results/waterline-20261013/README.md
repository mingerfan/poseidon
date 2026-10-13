# Waterline 实现与 188Server 验证

当前已通过编译器 10 项回归、运行时 8 项回归与本地 GPU API 20 项回归。Waterline 使用后期 Prepare 改写、完整消费者组准入、每设备水位及共享 Host/pinned/raw 账本，并复用前台计算和通信执行函数。没有自动批末 Fence。HEFT 源码没有改动。

本地真实 CKKS 生产路径已执行 CPU-only、GPU-only、Auto，N=65536、Q=32 个 30-bit 素数、scale=2^40，32768 个实数槽。图有五次 AddCP，使用真实加密输入与完整 dense 明文，并按全部槽位与明文算术参考比较。三种模式 max absolute complex error 均为 9.21178e-7，容差预设 1e-4；结束时计费 Device 明文和 pinned live 均为零。Auto 在这一单卡图中选择 CPU，不把该结果当作混合路径已覆盖。

对应日志：`compiler-regressions.log`、`core-tests-commit.log`、`core-waterline.log`、`local-waterline-dense-current.log`。清理引用依据见 `cleanup.tsv`。临时失败和调试构建日志未作为通过证据。

```sh
cmake -S third_party/ckks-runtime -B build-waterline-core \
  -DCKKS_RUNTIME_BUILD_TESTS=ON -DCKKS_RUNTIME_BUILD_EXPERIMENTS=OFF
cmake --build build-waterline-core -j6
ctest --test-dir build-waterline-core --output-on-failure --timeout 60
ctest --test-dir third_party/ckks-runtime/third_party/dacapo/build/nix --output-on-failure
cmake --build build-runtime-gpu-api-release --target poseidon_runtime_gpu_api_tests -j6
POSEIDON_WATERLINE_ONLY=1 POSEIDON_WATERLINE_LOG_N=16 \
POSEIDON_WATERLINE_Q_COUNT=32 POSEIDON_WATERLINE_SCALE=40 OMP_NUM_THREADS=1 \
build-runtime-gpu-api-release/bin/poseidon_runtime_gpu_api_tests
```

服务器验收和完整 Qwen3-0.6B 尚未完成。本地旧 Qwen25 24 层 DSL 的配置与目标不符，不能复用作完整目标模型。官方目标 revision `c1899de289a04d12100db370d81485cdf75e47ca` 为 28 层、hidden size 1024、head dim 128、16 Q heads、8 KV heads、词表 151936、Q/K norm、共享 embedding/LM head。权重下载使用该固定 revision；权重未提交到 Git。

GPU Encode 接受实数槽及上下文模数范围内的有限 FP64 系数；宽整数按 significand/exponent 作 RNS reduction。常数 scale 40/60/80/120 与 CPU 每个 RNS word 一致。配置 reserve 已核对实际参数表、单份 owning key backing、编码器表和持久 workspace 上界；RMM 有硬上限，并分别报告实际活跃分配和保留池空间。这里不声称完整模型加速。Native Boot 的 Waterline workspace 查询暂未接通，普通显式 Native Boot 能力保留。CPU decrypt/re-encrypt Boot 的安全与数值语义必须与原生 Boot 分开报告。

后续资源阶段的有效日志为 `core-tests-validation-hook.log`（7/7）、`key-backing.log`（Q=3/5/7共享key且全槽与CPU一致）、`gpu-api-h2d-final.log`（19/19）、`h2d-timing.log`、`gpu-encode-timing.log`、`local-waterline-metrics.log`。后者是真实 managed-pool N=65536/Q32/scale40 图，三模式最大误差8.64e-7，实际持久表约1.15GB，RMM活跃峰值约1.19GB，保留池1.75GiB、硬cap3520MiB；结束pinned live和计费Device明文为0。时间指标是本机本次测试，不能据此推断V100或完整模型性能。

`qwen-runner-metrics-smoke.log` 仅为2输入/2输出的小图，证明多输出入口、固定容差与实际事件计量接通，不是完整Qwen。旧N65536单卡与四卡容量预检失败日志保留。当前N16384完整四卡旧对角线图已编译并进入GPU-only实际前向；最终BSGS图也已完成四卡编译，见`qwen3-bsgs-all-compile.log`（exit0，31分13.68秒，RSS137224028KiB）。两份图均保留28层、全权重和完整词表；尚无完整CKKS数值通过证据。

2026-10-13 后续 stream 阶段已通过真实验证：所有计算 kernel 与 D2D/zero helper 使用实际执行 stream，跨设备回收暂用各设备合法 PTDS，并恢复调用者当前设备。`stream-ready-validation.md` 包含 AddCP c1 的确定性正/负对照；当前 helper 在 PTDS 阻塞时仍全部 28672 words 正确，旧 helper 全部错误。最新单卡 CUDA 库 `stream-reclaim-api-repeat-status.json` 连续 10×19 通过，旧数值失败和无法取 stack 的 hang 日志保留；不为旧 hang 虚构独立根因。

188Server 新小图证据为 `188-peer-completion-single.log` 和 `188-peer-completion-all.log`。单卡物理 GPU1，四卡物理0/1/2/3；全部三模式已通过。四卡每模式全四个输出、32768个complex槽最大误差9.37752e-7，全部三张peer卡的真实计算完成都早于延迟GPU0。Auto新增1 CPU+3 GPU Prepare，不用队列名称当执行证据。结束各卡明文计费与pinned live均为零。GPU0已有其他作业仍保留，初始outside快照与执行器RMM预算分开；`188-peer-completion-devices.csv`的GPU0快照为20280MiB，不能把它当本进程或整个run峰值。

完整模型N16384的旧矩形对角线图已完成四卡编译/导出，见 `qwen3-n16384-all-compile.log`；单卡在该预算配置下仍容量失败，见 `qwen3-n16384-single-capacity.log`。四卡Waterline基线峰值约4.68/6.22/9.37/11.55GB，Prepare数1417857、消费者及Release等组9726229；普通串行对象报告按eager Prepare记账，不能替代Waterline基线。完整模型首次runner启动暴露生产并行binary扫描器漏Prepare tag5，已修复并通过直接编译往返，保留 `qwen3-n16384-binary-scan-failure.log`。第二次OperatorSpec说明字段不符合schema；保留原spec/plan后仅重绑定合法spec摘要，metadata外1356009044B body SHA256保持 `9fcdbfcb46bab88f92a3d90c3cfef4e947bf470632e03885e81e3801a8c9f709`。当前旧图已在全部四卡进入实际前向，服务器 `logs/qwen3-full-gpu-n16384-v3.log`，仍未完成全部输出验收。最终BSGS计划直接绑定合法spec，单组峰每卡约6.3GB；FP64独立布局检查不能代替其CKKS验收。

CPU Host/pinned资源修复已通过：CPU Prepare的uint64 Host plaintext和uint32 pinned上传源同时存活，二者都计入Host，pinned另有子预算。degree8真实编译CPU/Auto Host480拒绝、512通过，GPU Host64仍通过；修复前误接收的反例日志保留为`compiler-cpu-pinned-before-fix.log`。本地真实CUDA的N4096/Q7与N65536/Q32在消费者#8提交前拒绝紧预算，精确足够预算全实虚槽通过、动态front/plain/pinned归零。见`cpu-prepare-host-budget-validation.md`、`cpu-prepare-host-bound-full-api.log`及legal/mode0/mode1日志。Host-staged测试入口已有明确实际post计数与pinned分配观测；本机仅单卡，全四卡真实路径仍待188运行。

`scripts/qwen3_dsl/validate_decoded.py`核对fixture、signature、参考和runner报告绑定，分别汇总完整151936有效logits与全部KV的最大误差、RMS、cosine、top1，检查所有虚部和零尾部，固定atol=0.001/rtol=0.0001。`checker-test.log`的20项使用明确标注的模拟decoded，只证明映射/拒绝，不是模型CKKS结果。当前独立NumPy exact/approximate参考logit max3.787683944267428e-9、RMS7.210505850742908e-10、top1均358。N16384/Q40/P8、图scale120及Host秘密钥辅助decrypt-reencrypt refresh使用security level none，不声称标准安全FHE或原生Boot。

`run-188-acceptance.sh`依次运行物理1、全部四卡默认通信、全部四卡强制Host-staged与完整BSGS Auto。全四卡staged每后端要求实际post20次、有真实pinned观测及全四个结果正确。服务器大型artifact保存在`/home/xuming/poseidon-waterline-20261013`；远程期间始终保留独立SSH长连接。`monitor_process.py`按PID采样driver内存及RSS，区间采样不是连续峰值，晚启动披露首个timestamp。runner记录main到首个/全部输出、setup/init/forward及保存；外部time-v计进程冷启动。完整模型最终输出、独立数值结果、最新全部四卡压力和资源测量仍待完成。

执行器无进展轮询由1ms缩至100µs，仍按原Ready/完整组/顺序检查；没有新调度路径或配置。`core-short-poll-tests.log`核心8/8，`core-short-poll-gpu-*`完整20/20及N65536/Q32 mode0/1各一次通过，源码/库hash保持一致。它可能增加CPU查询开销并改变Auto比例；本机host sleep探针与回归耗时不能外推188或完整模型加速。最终计划使用含Host/pinned修复的新编译器，产物目录为服务器`qwen3-plans-bsgs-final/gpu4`。
