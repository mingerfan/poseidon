# ResNet20 resident runtime 实验入口

**仅供研究：此入口使用的实验参数禁用安全级别校验，不应据此宣称满足密码学安全要求，也不适合生产部署。**

`resident_gpu.cpp` 是独立实验入口，复制自历史证据目录
`../../test-results/resnet20-resident-v100-20261108/resident_gpu.cpp`，原文件 SHA-256：
`331072fbdb4c94a9cf0cffa2cae71ac083de320d9c3829bc5351a3aa6aaaf334`。
此副本将普通旋转键生成改为可选策略、增加策略报告字段和输出路径预检查，并调整入口提示文字；不改变历史目录、报告、二进制或构建产物。整个 `test-results/` 已从 Git 排除，上述历史路径只在本地/外部归档中存在，不能假定新 clone 附带它；详见[实验产物约定](../../docs/experiment-artifacts.md)。这里不提供预构建二进制。

## 旋转键策略

本入口使用 `poseidon::runtime_api::rotation_key_steps` 选择普通 Galois 旋转键：环境变量 `POSEIDON_RUNTIME_ROTATION_KEY_POLICY` 未设置、为空或为 `direct` 时默认直接生成所需的归一化旋转键；仅实验对照可设为 `binary`，生成二进制分解基。其他值会报错。运行报告新增 `rotation_key_policy`，原有 `rotation_key_basis` 等字段保留；`rotation_key_basis` 是实际生成的普通旋转键步长集合，direct 时不一定是二进制基。

该环境变量由采用共享策略的 plan 驱动读取，不能改变历史 resident 可执行文件或自动为外部提供的密钥补键。direct 是键生成策略，不是取消运行时分解 fallback：runtime 对直接键优先、缺失时仍可分解。普通旋转键与 native Bootstrap 自身使用的键是不同集合。

## 构建与输入前提

需要兼容此入口的 C++/CUDA、Poseidon GPU 和 ckks-runtime 头文件及库，包括定义上述策略 API 的当前 `rotation_key_basis.h`，以及匹配的旧 native ResNet20 Boot 实现、ABI 和参数。runner 要求 native profile ID `poseidon-resnet20-manual-q50-scale40`，并会核对其与 OperatorSpec 声明的参数。历史构建使用特殊 Q50/P25 native ResNet Bootstrap（N=65536，scale 2^40），**不能假定普通 Qwen runner 或 profile 可替代，也不保证能从普通 Qwen runner 构建**。

还必须提供彼此匹配的 OperatorSpec、编译出的 RuntimePlan、plaintext bundle 和 GPU fixture（含输入及参考输出）。历史计划及输入的出处和摘要可参阅历史目录的 `README.md`、`run-command.json`、`provenance.json`；完整编译 bundle 并未保证随此目录提供。不要用不同参数集的计划、权重或 fixture 混搭。

从仓库根目录配置 `POSEIDON_BUILD_CKKS_RUNTIME_API=ON`、`POSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON` 和 `POSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON` 后，可构建 `poseidon_runtime_gpu_resnet20_resident` CMake 目标。运行仍要求上述匹配的 native ResNet20 Boot profile、计划和 fixture；编译成功不表示旧实验可直接复现。

历史 `build.py` 依赖当时服务器上 native ResNet20 构建的头文件、预编译 runtime/GPU 库以及目标目录的 `flags.make` 和 `link.txt`；`build-commands.json` 和 `provenance.json` 记录了命令及部分摘要，但不能证明归档库完整源码或提交对应关系。移植时需自行准备并验证相容的构建环境和依赖，确保编译器实际找到包含新策略 API 的头文件。不要直接在历史目录执行其构建脚本：它会写入二进制、对象、日志和构建记录。

## 使用方式（需先有兼容构建）

以下仅展示 runner 参数，不保证现有环境可构建或运行。将输出放在**新的实验目录**，不要指向历史证据路径；入口会在启动前拒绝已存在、相同或落入输入目录的输出路径。该检查不是针对并发进程的原子文件预留。选定适当的单 GPU 可见性和资源配置后：

```bash
# 默认 direct；使用实际路径替换大写占位符。
env -u POSEIDON_RUNTIME_ROTATION_KEY_POLICY ./build/bin/poseidon_runtime_gpu_resnet20_resident \
  PLAN OPERATOR_SPEC GPU_FIXTURE NEW_REPORT NEW_DECODED \
  --bundle PLAINTEXT_BUNDLE --trace NEW_TRACE

# 仅用于 binary 策略对照；同样使用新的输出文件。
POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary ./build/bin/poseidon_runtime_gpu_resnet20_resident \
  PLAN OPERATOR_SPEC GPU_FIXTURE NEW_BINARY_REPORT NEW_BINARY_DECODED \
  --bundle PLAINTEXT_BUNDLE --trace NEW_BINARY_TRACE
```

支持的额外参数见入口的 usage；report 的 `passed` 表示严格全槽验证是否通过，完成执行不等同于验证通过。历史运行命令和环境只供参考，不应原样使用其输出路径。

## 驻留与历史测量的边界

入口在初始化期间编码并上传权重明文，移除设备权重明文的 Release，并重新编号和验证计划，使其保留至整个在线前向结束；密文临时量仍按计划释放。在线阶段禁止编码、准备和通信。历史 binary 实验使用 27 个基础键，执行 1,709 次逻辑 Rotate 和 5,637 次普通 key switch（Bootstrap 另计）；历史记录的设备权重 payload 约 6.45 GiB，managed allocator 峰值约 21.95 GiB，NVML 采样峰值 31,030 MiB（主要反映预留池）。历史完整全槽严格数值验证未通过，详见历史 README。

这些都是历史 binary 的结果，**不是当前入口的 direct 策略测试结果**。本次仅对当前入口做编译和 CLI 输出预检查，未重新进行整网 GPU 测量；不能由历史 binary 的显存、速度或精度推断 direct 策略能在 32 GiB V100 上运行。
