# CKKS Runtime Api 状态与后续工作

这是一份临时开发文档，记录 Poseidon 与 CKKS Runtime 的当前边界和剩余工作。稳定后，内容应合并到正式架构文档并删除本文件。

## 当前边界

- Runtime 固定在 `third_party/ckks-runtime` submodule；打开 `POSEIDON_BUILD_CKKS_RUNTIME_API` 时默认从该目录构建，本地联调可用 `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` 覆盖。不打开该选项时不检查也不构建 Runtime。
- `PoseidonCpuApi` 支持单进程 Host 执行，也可选通过 MPI 在多进程间执行显式 Transfer。
- `PoseidonGpuApi` 支持单进程单卡和多卡，使用 RuntimePlan 已分配的设备并执行显式 CUDA 拷贝；每个值仍是位于一张卡上的完整对象，不支持多 shard 算子。
- CPU/GPU Api 都支持计划中明确写出的 `decrypt_reencrypt` Host Boot；`PoseidonGpuApi` 还支持预先配置 profile 的原生单卡 GPU Boot。MPI/NCCL GPU 通信通过独立的实验性构建选项接入，首期只传输完整的单设备对象。
- Runtime 核心不依赖 Poseidon；Api 实现放在 Poseidon 仓库。

## 已完成

- GPU 明密文、参数、密钥上传、Evaluator、handlers 和 CUDA kernels 已收敛到独立的 `poseidon_gpu` target。
- `PoseidonCpuApi` 和 `PoseidonGpuApi` 都直接实现 Runtime Api；旧 mgpu IR/调度/执行系统以及 HEVM/CST frontend 已删除。
- 同进程 CUDA 拷贝、CPU MPI 通信、MPI/NCCL bootstrap、跨 rank Host-to-Device staging 和 Host `decrypt_reencrypt` Boot 已接入 RuntimePlan 执行路径。
- `GpuBootstrapProfileBuilder` 可确定性生成并上传当前 degree-22、baby-4、三次二倍角的 native Boot profile；RuntimePlan 通过 profile id 和静态 Device place 调用优化后的 `GpuEvaluator::bootstrap`。
- 拷贝由 RuntimePlan 的 Transfer/Replicate 驱动，算子不会在失败后临时搬运数据或切换后端。
- 多 rank `PerDeviceWorkers` 为每个 MPI rank 使用一个有序通信提交线程；它只提交跨 rank NCCL，不逐条等待完成。device worker 负责本地计算和同 rank CUDA copy；initialization 仍顺序执行。
- CUDA copy 和 NCCL 接收在提交后立即发布带完成 event 的 Device 输出。后续计算或通信在自己的 stream 上等待 event，CPU 只在 Host 物化、错误检查和最终收尾时等待完成。

原生 Boot 的最小接入方式如下：

```cpp
PoseidonGpuApi api(context_id, context, cuda_device_id);
auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
    context, key_generator, cuda_device_id);
auto operator_profile = poseidon::runtime_api::make_native_boot_profile(profile);
api.configure_native_bootstrap(0, std::move(profile));
```

`operator_profile` 必须加入编译器使用的 `OperatorSpec::boot_profiles`。builder 使用生产 `ModRaise -> CoeffToSlot -> EvalMod -> SlotToCoeff` 顺序；实验性的 StC-first probe 尚未作为 Runtime Boot implementation 暴露。builder 的结构性参数均显式传递，不读取 EvalMod 分裂或线性变换模式环境变量；旧专项测试仍保留环境变量入口。

Runtime 的 RMM 初始池默认取当前空闲显存的四分之一（上限 24 GiB），避免在 24 GiB GPU 上因固定预分配 24 GiB 而启动失败。需要预热更大的池时可显式设置 `POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB`，后续分配仍可由 RMM 按需增长。

## 剩余工作

1. 在多卡机器上补齐真实跨卡 MLP/ResNet 端到端测试和通信性能测试。
2. 完善跨 rank Device-to-Host 和 Host-to-Host value 传输；当前只支持 Host-to-Device 通过源 rank 的逻辑 GPU 0 进行 NCCL staging。
3. 在可用 GPU 驱动环境运行 `poseidon_runtime_gpu_bootstrap_e2e`，固化 native Boot 的精度和性能基线；当前会话只完成了编译和无设备 Skip 验证。
4. 为 65536 Runtime artifacts 生成与其模数链匹配的 native Boot profile；现有 artifact 仍明确使用 `decrypt_reencrypt`。
5. 继续校验 lazy rescale 数值误差，尤其是完整 MLP 与 CPU/Python 结果的差异。

未实现的能力在 preflight 直接拒绝，不提供自动降级。
