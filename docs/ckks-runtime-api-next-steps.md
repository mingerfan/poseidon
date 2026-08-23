# CKKS Runtime Api 状态与后续工作

这是一份临时开发文档，记录 Poseidon 与 CKKS Runtime 的当前边界和剩余工作。稳定后，内容应合并到正式架构文档并删除本文件。

## 当前边界

- Runtime 固定在 `third_party/ckks-runtime` submodule；打开 `POSEIDON_BUILD_CKKS_RUNTIME_API` 时默认从该目录构建，本地联调可用 `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` 覆盖。不打开该选项时不检查也不构建 Runtime。
- `PoseidonCpuApi` 支持单进程 Host 执行，也可选通过 MPI 在多进程间执行显式 Transfer。
- `PoseidonGpuApi` 支持单进程单卡和多卡，使用 RuntimePlan 已分配的设备并执行显式 CUDA 拷贝；每个值仍是位于一张卡上的完整对象，不支持多 shard 算子。
- CPU/GPU Api 都支持计划中明确写出的 `decrypt_reencrypt` Host Boot；`PoseidonGpuApi` 还支持预先配置 profile 的原生单卡 GPU Boot。跨进程 GPU 通信尚未实现。
- Runtime 核心不依赖 Poseidon；Api 实现放在 Poseidon 仓库。

## 已完成

- GPU 明密文、参数、密钥上传、Evaluator、handlers 和 CUDA kernels 已收敛到独立的 `poseidon_gpu` target。
- `PoseidonCpuApi` 和 `PoseidonGpuApi` 都直接实现 Runtime Api；旧 mgpu IR/调度/执行系统以及 HEVM/CST frontend 已删除。
- 同进程 CUDA 拷贝、CPU MPI 通信和 Host `decrypt_reencrypt` Boot 已接入 RuntimePlan 执行路径。
- `GpuBootstrapProfileBuilder` 可确定性生成并上传当前 degree-22、baby-4、三次二倍角的 StC-first native Boot profile；RuntimePlan 通过 profile id 和静态 Device place 调用优化后的 `GpuEvaluator::bootstrap`。
- 拷贝由 RuntimePlan 的 Transfer/Replicate 驱动，算子不会在失败后临时搬运数据或切换后端。

原生 Boot 的最小接入方式如下：

```cpp
PoseidonGpuApi api(context_id, context, cuda_device_id);
auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
    context, key_generator, cuda_device_id);
auto operator_profile = poseidon::runtime_api::make_native_boot_profile(profile);
api.configure_native_bootstrap(0, std::move(profile));
```

`operator_profile` 必须加入编译器使用的 `OperatorSpec::boot_profiles`。默认 profile 使用与当前优化测试一致的 `StC(q=6) -> ModRaise -> CoeffToSlot[5,4,3,3] -> EvalMod(degree=22, baby=4, DA=3) -> real+i*imag` 顺序，输入固定为 level 5。EvalMod 后增加一次数值补偿的明文乘与 rescale，把约 `2^45.07` 的物理 scale 精确归一到 RuntimePlan 可声明的 `2^45`，输出为 level 11。builder 的结构性参数均显式传递，不读取 EvalMod 分裂或线性变换模式环境变量；旧专项测试仍保留环境变量入口。

`poseidon_runtime_gpu_bootstrap_e2e` 在 `N=65536` 上同时运行一次直接 GPU evaluator 和一次完整 RuntimePlan（Host transfer -> Device Boot -> Host transfer），并比较两者。V100 实测 `runtime_direct_max_abs_error=0`，源消息误差为 `6.78437`；后者来自当前 degree-59 系数截断到 degree-22 的近似误差，不是 Runtime 调度误差。

Runtime 的 RMM 初始池默认取当前空闲显存的四分之一（上限 24 GiB），避免在 24 GiB GPU 上因固定预分配 24 GiB 而启动失败。需要预热更大的池时可显式设置 `POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB`，后续分配仍可由 RMM 按需增长。

## 剩余工作

1. 接入 GPU-aware MPI 或其他跨进程 GPU 传输。
2. 在多卡机器上补齐真实跨卡 MLP/ResNet 端到端测试和通信性能测试。
3. 固化 native Boot 的性能基线；正确性 E2E 已在 V100 上通过，尚未把 setup、StC、ModRaise、C2S、EvalMod 和输出归一化的分段耗时纳入 Runtime artifact。
4. 为 65536 Runtime artifacts 生成与其模数链匹配的 native Boot profile；现有 artifact 仍明确使用 `decrypt_reencrypt`。
5. 继续校验 lazy rescale 数值误差，尤其是完整 MLP 与 CPU/Python 结果的差异。

未实现的能力在 preflight 直接拒绝，不提供自动降级。
