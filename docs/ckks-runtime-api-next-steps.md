# CKKS Runtime Api 状态与后续工作

这是一份临时开发文档，记录 Poseidon 与 CKKS Runtime 的当前边界和剩余工作。稳定后，内容应合并到正式架构文档并删除本文件。

## 当前边界

- Runtime 固定在 `third_party/ckks-runtime` submodule；打开 `POSEIDON_BUILD_CKKS_RUNTIME_API` 时默认从该目录构建，本地联调可用 `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` 覆盖。不打开该选项时不检查也不构建 Runtime。
- `PoseidonCpuApi` 支持单进程 Host 执行，也可选通过 MPI 在多进程间执行显式 Transfer。
- `PoseidonGpuApi` 支持单进程单卡和多卡，使用 RuntimePlan 已分配的设备并执行显式 CUDA 拷贝；原生 Boot 已支持 real/imag 双卡调度、四卡 C2S 正确性分片和 degree-22 EvalMod 根分区。每个跨阶段值仍是位于一张卡上的完整对象，不支持 RNS multi-shard 算子。
- CPU/GPU Api 都支持计划中明确写出的 `decrypt_reencrypt` Host Boot；`PoseidonGpuApi` 还支持预先配置 profile 的原生单卡 GPU Boot。跨进程 GPU 通信尚未实现。
- Runtime 核心不依赖 Poseidon；Api 实现放在 Poseidon 仓库。

## 已完成

- GPU 明密文、参数、密钥上传、Evaluator、handlers 和 CUDA kernels 已收敛到独立的 `poseidon_gpu` target。
- `PoseidonCpuApi` 和 `PoseidonGpuApi` 都直接实现 Runtime Api；旧 mgpu IR/调度/执行系统以及 HEVM/CST frontend 已删除。
- 同进程 CUDA 拷贝、CPU MPI 通信和 Host `decrypt_reencrypt` Boot 已接入 RuntimePlan 执行路径。
- `GpuBootstrapProfileBuilder` 可确定性生成并上传当前 degree-22、baby-4、三次二倍角的 StC-first native Boot profile；RuntimePlan 通过 profile id 和静态 Device place 调用优化后的单卡或多卡调度。四卡安装默认按实测成本选择 C2S 1 卡、EvalMod 2 卡，强制四卡根分区仍可配置。
- 拷贝由 RuntimePlan 的 Transfer/Replicate 驱动，算子不会在失败后临时搬运数据或切换后端。

原生 Boot 的最小接入方式如下：

```cpp
PoseidonGpuApi api(context_id, context, cuda_device_id);
auto profile = poseidon::gpu::GpuBootstrapProfileBuilder::build(
    context, key_generator, cuda_device_id);
auto operator_profile = poseidon::runtime_api::make_native_boot_profile(profile);
api.configure_native_bootstrap(0, std::move(profile));
```

`operator_profile` 必须加入编译器使用的 `OperatorSpec::boot_profiles`。默认 profile 使用与当前优化测试一致的 `StC(q=6) -> ModRaise -> CoeffToSlot[5,4,3,3] -> EvalMod(degree=22, baby=4, DA=3) -> real+i*imag` 顺序，输入固定为 level 5。双重 hoist 的 baby tile 也由 profile 固定为 15，不再依赖进程环境。EvalMod 后增加一次数值补偿的明文乘与 rescale，把约 `2^45.07` 的物理 scale 精确归一到 RuntimePlan 可声明的 `2^45`，输出为 level 11。builder 的结构性参数均显式传递，不读取 EvalMod 分裂或线性变换模式环境变量；旧专项测试仍保留环境变量入口。

`poseidon_runtime_gpu_bootstrap_e2e` 在 `N=65536` 上默认执行 2 次预热和 5 次计时，分别测量直接 GPU evaluator、仅返回 Device 值的 RuntimePlan，以及包含 Host transfer 的完整 RuntimePlan，并比较最后一次输出。预热和迭代次数可通过 `POSEIDON_RUNTIME_BOOTSTRAP_WARMUP` 与 `POSEIDON_RUNTIME_BOOTSTRAP_ITERATIONS` 调整。V100 实测 `runtime_direct_max_abs_error=0`，源消息误差为 `6.78437`；后者来自当前 degree-59 系数截断到 degree-22 的近似误差，不是 Runtime 调度误差。

同一张 V100 上两轮默认配置的基准结果如下。原专项测试的 StC-first 自举为 `74.7508 ms`；它结束于约 `2^45.07`、Q=13，未包含 Runtime 契约要求的精确输出 scale 归一化。当前直接 evaluator 的完整 profile 为 `80.6910/81.6666 ms`，Runtime Device-final Boot 为 `78.7626/79.0153 ms`，两者的负差值属于顺序执行时的测量波动，没有观察到 Runtime 分发本身造成的额外延迟。Runtime Boot 加 D2H 为 `91.5031--91.9668 ms`；再加 H2D 与 setup 后，完整 RuntimePlan 为 `96.2218--96.2485 ms`。因此，公平预热后设备端计算相对原专项测试约慢 `4.14 ms`（约 `5.5%`），这包含额外输出归一化及实现/测量差异，不能全部归因于 Runtime 调度。

Runtime 的 RMM 初始池默认取当前空闲显存的四分之一（上限 24 GiB），避免在 24 GiB GPU 上因固定预分配 24 GiB 而启动失败。需要预热更大的池时可显式设置 `POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB`，后续分配仍可由 RMM 按需增长。

## 剩余工作

1. 接入 GPU-aware MPI 或其他跨进程 GPU 传输。
2. 在多卡机器上补齐多个独立 bootstrap 的吞吐调度，以及真实跨卡 MLP/ResNet 端到端测试。
3. 将现有 Boot 内部分段计时进一步汇入 Runtime artifact，而不只由端到端回归程序读取。
4. 为 65536 Runtime artifacts 生成与其模数链匹配的 native Boot profile；现有 artifact 仍明确使用 `decrypt_reencrypt`。
5. 继续校验 lazy rescale 数值误差，尤其是完整 MLP 与 CPU/Python 结果的差异。
6. 若要降低单个 degree-22 Boot 的四卡延迟，实现 RNS/keyswitch 粒度的分布式算子；当前对象级 PCIe 子树拆分受重复基生成和通信下限限制。

未实现的能力在 preflight 直接拒绝，不提供自动降级。
