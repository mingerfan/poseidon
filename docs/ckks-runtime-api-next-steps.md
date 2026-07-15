# CKKS Runtime Api 后续工作

这是一份临时开发文档，只记录当前 `PoseidonCpuApi` 之后的工作。稳定后，内容应合并到正式架构文档并删除本文件。

## 当前边界

- Runtime 通过 `POSEIDON_CKKS_RUNTIME_SOURCE_DIR` 指向外部源码目录；不开 `POSEIDON_BUILD_CKKS_RUNTIME_API` 时不检查该目录。
- `PoseidonCpuApi` 第一版只支持一个进程、Host 位置和同步执行。
- 第一版不提供通信后端，不支持 Boot。
- Runtime 核心不依赖 Poseidon；Api 实现放在 Poseidon 仓库。

## 4. 抽出独立 GPU 算子库

从现有 `poseidon_mgpu_gpu_runtime` 中抽出 `poseidon_gpu_core`，包含 GPU 明密文、参数、密钥上传、Evaluator、handlers 和 CUDA kernels。

旧 mgpu 执行后端与新的 Runtime GPU Api 都链接 `poseidon_gpu_core`。新的 Api 不允许依赖旧 mgpu IR、调度器、验证器或解释器。

完成条件：

- 默认 CPU 构建不查找 CUDA/RMM；
- 旧单 GPU 和可选 mgpu 测试行为不变；
- GPU 源文件只在一个底层 target 中编译。

## 5. 实现单进程单卡 PoseidonGpuApi

新增 `PoseidonGpuApi`，直接消费 RuntimePlan 已确定的 Device 位置，不做 placement，不插入拷贝。

第一步只支持单进程单卡：

- Host Encode 继续使用 Poseidon CPU；
- 显式 Host→Device/Device→Host Transfer 负责上传和下载；
- Device compute 调用 `poseidon_gpu_core`；
- Value 使用完整的单设备对象，不支持多 shard；
- 缺 CUDA、RMM、设备或密钥时在 preflight 直接报错。

构建开关应独立于旧 `POSEIDON_BUILD_MGPU`，并且默认关闭。

## 6. 通信和 Host Boot

GPU 单卡计算稳定后再做：

1. 把现有同进程 CUDA peer copy 和对象物化代码迁到 Runtime Api 的通信实现；
2. 扩展 `PoseidonCpuApi` 到 MPI 多进程通信；
3. 接 GPU-aware MPI 或其他跨进程传输；
4. 在 `PoseidonGpuApi` 内组合 CPU 能力，执行计划明确写出的 `decrypt_reencrypt` Host Boot；
5. 保持所有拷贝由 RuntimePlan 的 Transfer/Replicate 驱动，不在算子失败后临时搬运或切换后端。

每一步都使用同一份计划和 Vec/CPU/GPU 差分测试。跨进程和 GPU 能力未实现前直接拒绝对应计划，不提供自动降级。
