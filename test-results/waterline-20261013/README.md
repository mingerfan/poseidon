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

GPU Encode 仅接受实数槽和可由 uint64 幅值表示的完整系数，错误明确暴露。当前 workspace 账本使用保守后端上界，配置 reserve 还须覆盖持久缓存、密钥与参数表；此记录不声称已证明总显存上界或完整模型加速。Native Boot 的 Waterline workspace 查询暂未接通，普通显式 Native Boot 能力保留。CPU decrypt/re-encrypt Boot 的安全与数值语义必须与原生 Boot 分开报告。
