# 单卡手写 ResNet20 与 API Trace

手写性能入口来自 `SL640/FHE_GPU_Agent` 的 `gpu_api`（main `55e51804`）。默认参数
N65536、Q49/P25、scale 2^40，完整9个 BasicBlock、18次 degree22/DA3/K12/h32
native Boot；卷积使用 merged masks：输入 Q31 → stem/Boot Q30 → ReLU Q8 →
Conv Q6，最终 logits Q3。主秘密 h32768 与 Boot 临时秘密 h32 是不同的秘密。
Q50 和旧 degree59 路径仍可通过环境变量选择。

**安全边界：** `--unsafe-*` 使用未经安全批准的研究参数；性能或数值测试不能证明
密码学安全性。完整应用前向只使用一张 GPU。准备阶段生成密钥、权重明文、Boot
矩阵和 workspace；在线阶段不进行中间解密或重加密。

## 构建、手写运行

需要 CMake ≥3.26.4、CUDA/CUPTI 及本仓库的离线依赖：

```bash
bash bench/resnet20_s2c_first/build.sh /tmp/poseidon_resnet20_s2c_first
FIXTURE=Doc/analysis/resnet20_joint_chain_20260911/relu_precision_fixture.txt
CUDA_VISIBLE_DEVICES=0 POSEIDON_S2C_CUPTI=0 /tmp/poseidon_resnet20_s2c_first \
  --unsafe-performance-only "$FIXTURE" 0
/tmp/poseidon_resnet20_s2c_first --metadata-only "$FIXTURE" 0
```

`--unsafe-accuracy-only` 使用带 observer 的旧 degree59 诊断，不验证 Q49 低阶默认路径。
上游 `summarize_performance.py` 仍是旧 Q50/dnum2 契约，不能用作新版结果验收。

## 实际 API Trace → MLIR → RuntimePlan

`--unsafe-trace-plan fixture image empty-dir` 先完成常驻准备和 warmup，执行一次原手写
参考，然后对同一份程序的 observer-free replay 做 Trace：实际 GPU API 经轻量 facade
转发，记录 SSA 依赖和真实 Q、浮点 scale、常量/密钥资源，而不是手工重建网络拓扑。
27个输入是已在 GPU 上的外部密文，编码、上传和准备检查不进入 trace。

- `rotate_many_hoisted` 录为一次 `rotation_hoist` 和每个旋转各一条
  `rotate_from_hoist`。前者产生独立只读 Buffer（含 Q/P/dnum/布局和分配字节），
  后者同时消费原密文与 Buffer；原有 Galois、key乘加和 ModDown 内核保留。
- native Boot 按实际执行顺序展开为每个 S2C/C2S 线性层、ModRaise、两路 EvalMod
  和实虚拆分/重构算术等 `ckks.apicall`。线性层保留内部 double-hoist 和融合内核，
  EvalMod 仍是单独的完整 API；每一步使用同一份已准备的 profile，不重新规划
  scale、rescale 或 Boot 位置。手写默认 S2C-first profile 每个 Boot 有 6 个线性层。
- 编译器执行单卡 HEFT placement、通信物化、内存生命周期规划；V8 RuntimePlan
  把 Hoist Buffer 作为一等值，在最后一个消费者之后 Release。ApiCall 不启用输入复用。
- runtime `SequentialRuntime<TracedPlanApi>` 对计划逐条 dispatch 到原 GPU API，
  绝不把准备阶段或 Boot 替换成解密/重加密。单 rank、单卡、单流；不支持将 Hoist
  Buffer 跨设备搬运，也未测试并行 stream。该拆分用于本进程的 API Trace，通用
  `PoseidonGpuApi` 的 native Boot 仍是独立接口。
- 已注册的外部输入是回放时使用的同一份 GPU 密文，必须保持不变；Trace 禁止将
  GPU API 的输出写回注册输入。需要原位操作时，先通过受追踪 API 生成中间值。
- 密钥、常量及 Boot 内部 workspace 仍为进程内常驻资源，**不在编译器图值内存报告中**；
  报告不是完整进程显存上界。计划与该次 trace 的本进程资源绑定，不能作为其他
  密钥/模型进程的通用可移植程序。
- Trace 捕获的 CPU API 调用时长只用于粗略调度代价，不是精确的 GPU profiling 数据。

使用已构建的 LLVM/MLIR 18 `hecate-opt`，准备目录必须为空。应用会保留同一批 keys 与
inputs，写出 `trace.mlir` 和 `trace.mlir.ready` 后等编译器；编译完成将
`plan.json`、`operator-spec.json` 放入目录，最后**原子创建** `ready` 文件；失败时
创建 `error` 文件。最多等待一小时：

```bash
# 本地 compiler 与GPU应用在同一机器时，两终端：
D=/tmp/resnet20-trace-$USER
mkdir "$D"
CUDA_VISIBLE_DEVICES=0 POSEIDON_S2C_CUPTI=1 /tmp/poseidon_resnet20_s2c_first \
  --unsafe-trace-plan "$FIXTURE" 0 "$D"

# 第二终端，在 trace.mlir.ready 出现后：
python3 bench/resnet20_s2c_first/compile_prepared.py "$D/trace.mlir" --format json \
  --output-dir "$D/compiled" --hecate-opt /path/to/new/hecate-opt
cp "$D/compiled/scheduled.gpu_api_trace.runtime-plan.json" "$D/plan.json"
cp "$D/compiled/operator-spec.json" "$D/operator-spec.json"
touch "$D/ready.tmp" && mv "$D/ready.tmp" "$D/ready"
```

若编译器只在本地、GPU 在 `188Server`，可在服务器启动应用后，于本地执行：

```bash
python3 bench/resnet20_s2c_first/compile_trace_remote.py 188Server \
  /absolute/server/empty-dir --output-dir /tmp/compiled-trace \
  --hecate-opt /path/to/new/hecate-opt
```

最终 `API_TRACE_RESULT` 报告相同输入及密钥下手写/trace/计划的精确 RNS 对照、
Boot ModRaise、线性层、EvalMod 和 Hoist 数及 Hoist 实际释放统计。开启 CUPTI 时还检查在线 H2D/D2H；
最终解密在计时之后，仅检查前10个 logits 与参考预测，不代表全槽或数据集精度。
离线编译和准备时间不属于在线执行时间。

## 测试

```bash
ctest --test-dir build-runtime-gpu-api-release -R '^poseidon_gpu_bootstrap_trace_e2e$' --output-on-failure
ctest --test-dir third_party/ckks-runtime/build-api-trace --output-on-failure
ctest --test-dir third_party/ckks-runtime/third_party/dacapo/build/trace --output-on-failure
CUDA_VISIBLE_DEVICES=0 build/bin/test_gpu_split_hoist --extended
```

GPU split 回归对比原始 `rotate_many_hoisted` 和拆分实现的逐 RNS 结果；
另有 key 裁剪、非默认流上传下载以及 Boot 密钥切换回归。`check_trace_run.py`
核对真实计划中的 hoist/旋转依赖、最后使用后 Release、单设备输入和 GPU 结果日志。
