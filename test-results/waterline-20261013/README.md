# Waterline 实现与 188Server 验证

当前已通过编译器 10 项回归和运行时 7 项回归。Waterline 使用后期 Prepare 改写、完整消费者组准入、每设备水位及共享 Host/pinned/raw 账本，并复用前台计算和通信执行函数。没有自动批末 Fence。HEFT 源码没有改动。

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

`qwen-runner-metrics-smoke.log` 仅为2输入/2输出的小图，证明多输出入口、固定容差与实际事件计量接通，不是完整Qwen。原始完整Qwen3图已生成无损12GB bytecode；单卡与四卡在31GiB M/4GiB reserve的容量预检失败，四卡报前台对象约27GiB。正在收窄逐头归一化和宽区间SiLU的算术图后重新编译；完整模型未执行、未通过数值验收。

2026-10-13 后续 stream 阶段已通过真实验证：所有计算 kernel 与 D2D/zero helper 使用实际执行 stream，跨设备回收暂用各设备合法 PTDS，并恢复调用者当前设备。`stream-ready-validation.md` 包含 AddCP c1 的确定性正/负对照；当前 helper 在 PTDS 阻塞时仍全部 28672 words 正确，旧 helper 全部错误。最新单卡 CUDA 库 `stream-reclaim-api-repeat-status.json` 连续 10×19 通过，旧数值失败和无法取 stack 的 hang 日志保留；不为旧 hang 虚构独立根因。

188Server 新小图证据为 `188-peer-completion-single.log` 和 `188-peer-completion-all.log`。单卡物理 GPU1，四卡物理0/1/2/3；全部三模式已通过。四卡每模式全四个输出、32768个complex槽最大误差9.37752e-7，全部三张peer卡的真实计算完成都早于延迟GPU0。Auto实际后端比例见累计CPU/GPU计数，不用队列名称当执行证据。结束各卡明文计费与pinned live均为零。GPU0同时存在其他进程约13.6GiB占用，保留原进程，初始outside快照与执行器RMM预算分开。

完整模型N16384的旧矩形对角线图已完成四卡编译/导出，见 `qwen3-n16384-all-compile.log`；单卡在更晚消费者处仍容量失败，见 `qwen3-n16384-single-capacity.log`。四卡Waterline基线峰值约4.68/6.22/9.37/11.55GB，Prepare数1417857、消费者及Release等组9726229；普通串行对象报告按eager Prepare记账，不能替代Waterline基线。完整模型首次runner启动暴露生产并行binary扫描器漏Prepare tag5，见 `qwen3-n16384-binary-scan-failure.log`，正在修复。当前仍没有完整模型执行/数值通过证据。BSGS lowering的FP64独立矩阵检查通过，后续新图仍需单独CKKS验收。
