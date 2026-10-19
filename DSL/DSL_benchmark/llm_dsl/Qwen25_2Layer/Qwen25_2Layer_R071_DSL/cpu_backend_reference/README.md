# 071 CPU 后端参考（070）

这里保留 070 的五个并行补丁、CMake 配置、短测程序和源码哈希证据，用于说明 071 的 32 线程结果来自哪些后端改动。它们不是 GPU 必需文件，也不是可直接套到任意新版 Poseidon 的补丁集合。

历史应用顺序：`patch_rns_ntt.py` → `patch_rns_poly.py` → `patch_rns_named_ops.py` → `patch_rescale_conversion.py --mode combined` → `patch_rotation_moddown.py --mode combined`，参数均指向**独立的兼容旧核心源码副本**。具体基底逐文件哈希在 `evidence/backend-source-manifest.json`。不要直接修改当前主仓库。

071 还使用了隔离 PoseidonCpuApi native Boot 回调与旧 48 阶 CPU 自举核心。此源代码包没有包含这些外部库、回调头文件、编译产物或密钥，所以这里的 CMake 是历史构建参考，不是开箱即用的 CPU 构建入口。`frozen/tools/native_probe.cpp` 的 include 和链接依赖必须先与对应 Runtime/Poseidon 环境匹配。

历史环境：`OMP_NUM_THREADS=32`、`OMP_DYNAMIC=FALSE`、`OMP_MAX_ACTIVE_LEVELS=1`、`OMP_WAIT_POLICY=PASSIVE`；绑定到上海机器 32–63 物理核。其他机器应按自身拓扑选择，不照搬 CPU 编号。平均实际 CPU 用量约 4.267 核，不代表全程满载 32 核。

GPU 测试使用目标仓库的 GPU 后端、真实自举和匹配的编译配置；CPU 并行补丁不直接用于 GPU。
