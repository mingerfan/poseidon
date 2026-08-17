# 65536 CPU、4 GPU 和 8 GPU Runtime 负载

这个目录可以随 Poseidon 仓库一起打包。它包含六份 RuntimePlan：

- `plans/cpu/mlp.runtime-plan.json`：单进程 CPU MLP。
- `plans/cpu/probe.runtime-plan.json`：单进程 CPU 高并行 probe 的等价串行计划。
- `plans/gpu/mlp-4gpu.runtime-plan.json`：编译器生成的单进程 4 GPU MLP。
- `plans/gpu/probe-4gpu.runtime-plan.json`：4 GPU 交错计算和环形 P2P 通信 probe。
- `plans/gpu/mlp-8gpu.runtime-plan.json`：编译器生成的单进程 8 GPU MLP。
- `plans/gpu/probe-8gpu.runtime-plan.json`：8 GPU 交错计算和环形 P2P 通信 probe。

四份计划共用 `fixture/`。MLP 使用 `plaintext-bundle/` 中的模型常量；probe
没有明文常量，但运行入口仍传入这个目录，以保持命令格式一致。

## 构建

在 Poseidon 仓库根目录运行：

```bash
cmake -S . -B build-runtime-gpu-api-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON
cmake --build build-runtime-gpu-api-release --parallel \
  --target poseidon_runtime_cpu_mlp_e2e poseidon_runtime_gpu_mlp_e2e
```

如果使用其他构建目录，运行时设置 `POSEIDON_BUILD_DIR=/path/to/build`。

## CPU

```bash
runtime-artifacts/65536/run_cpu.sh mlp
runtime-artifacts/65536/run_cpu.sh probe
```

CPU probe 很长，主要用于比较相同算子数量的 CPU/GPU 执行时间。默认报告写到
`runtime-artifacts/65536/reports/`。也可以把第二个参数指定为其他报告路径。

## 4 GPU 和 8 GPU

```bash
runtime-artifacts/65536/run_4gpu.sh mlp
runtime-artifacts/65536/run_4gpu.sh probe
runtime-artifacts/65536/run_8gpu.sh mlp
runtime-artifacts/65536/run_8gpu.sh probe
```

也可以使用统一入口：

```bash
runtime-artifacts/65536/run_gpu.sh 4 mlp
runtime-artifacts/65536/run_gpu.sh 8 probe
```

脚本默认使用从 0 开始的 4 张或 8 张可见设备。需要选择其他物理 GPU 时，在命令
前设置 `CUDA_VISIBLE_DEVICES`，但设备数量必须与计划一致。运行模式固定为每张卡
一个朴素 FIFO device worker。

probe 的输出本来就不是 MLP 数值结果。入口脚本只在完整报告已经落盘时忽略数值
比较失败；计划加载、内存不足或 GPU 执行错误不会被忽略。
