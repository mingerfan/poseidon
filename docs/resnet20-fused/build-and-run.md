# D31 重建和验证

## 快照与环境

完整源码归档为 `source.tar.gz`，SHA-256：`1fda81e15f8775be606159d6a36baa0f52ac8ebd4cb7ed9fa5730bbb12ff69d2`。A800 上原件在 `/root/resnet20-d31-audit-20261118.eixwDN/snapshot.9rX2Yd/`，其中 `files.sha256` 可对逐文件核验。解压应在有足够空间的独立目录进行，归档包含符号链接，建议在 Linux 上保留其原样解压。不得覆盖已有源码目录或稳定分支。

实验所用 A800 为 80GB、CUDA 11.8、CMake 3.27.9、GMP 开发头文件及库；归档包含离线第三方源码、Trident 融合参数、模型及 1000 张测试样本，但不包含 CUDA、GMP 或编译器。路径不同的机器需相应调整。源码是实验快照，默认不改变任何 V100 分支。

```bash
sha256sum source.tar.gz
mkdir d31-rebuild
tar -xzf source.tar.gz -C d31-rebuild
cd d31-rebuild
export CUDA_ROOT=/usr/local/cuda
export GMP_ROOT=/path/to/gmp  # 其下包含 include/ 和 lib/libgmp{,xx}.so
export PATH="$CUDA_ROOT/bin:$PATH"
arch=80                    # A800；V100 使用 70，H100 使用 90
cmake -S migration -B "build-sm${arch}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_GPU=ON \
  -DPOSEIDON_BUILD_RESNET20_S2C_FIRST=ON -DPOSEIDON_USE_ZLIB=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF -DSPDLOG_FMT_EXTERNAL_HO=OFF \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/migration/third_party/spdlog" \
  -DGMP_INCLUDE_DIR="$GMP_ROOT/include" -DGMPXX_INCLUDE_DIR="$GMP_ROOT/include" \
  -DGMP_LIBRARY="$GMP_ROOT/lib/libgmp.so" -DGMPXX_LIBRARY="$GMP_ROOT/lib/libgmpxx.so"
cmake --build "build-sm${arch}" --target resnet20_wide_functional \
  functional_wide_ntt_probe functional_wide_bv_probe --parallel 8
```

部分发行版将 GMP 库放在 `lib/x86_64-linux-gnu`，并将 `gmp.h` 放在 `include/x86_64-linux-gnu`；此时分别设置上述头文件和库路径。A800 的现有依赖在 `/root/resnet20-a800-20261117.KgGe5I/deps/usr`，`gmp.h` 与库位于对应的 `x86_64-linux-gnu` 子目录，`gmpxx.h` 位于 `include`。

## A800 D31 组件及整网

以下为本轮 A800 的选项；60 GiB 是内存池上限而不是完整设备需求，显存不足的设备不要直接套用此值。保留这些开关以复现本次优化组合，并通过取消相应开关检查回退路径。输出会包含模型预测、GPU/明文误差、各阶段活动及内存信息；`PASS` 代表 GPU 与 D31 明文一致且误差不超过 0.01，不代表标签预测正确。

```bash
export CUDA_VISIBLE_DEVICES=0
export POSEIDON_WIDE_NTT_FUSED=1 POSEIDON_WIDE_NTT_TILE=1024
export POSEIDON_WIDE_POOL_GIB=60 POSEIDON_WIDE_FULL_KEYS=1
export POSEIDON_WIDE_HOIST_DIGITS=1 POSEIDON_WIDE_HOIST_KEYCACHE=1
export POSEIDON_CTS_LOG_BSGS_RATIO=1 POSEIDON_WIDE_FAST_LIFT=1
coeff=trident-functional/resnet20/functional_params
build-sm80/bin/functional_wide_ntt_probe 31
build-sm80/bin/functional_wide_bv_probe 31
build-sm80/bin/resnet20_wide_functional 31 0 "$coeff" --benchmark
# 图像编号 0 开始共 100 张，复用模型与密钥但逐张重新加密及计算明文参考
build-sm80/bin/resnet20_wide_functional 31 0 "$coeff" --batch=100
```

整网计时排除了输入和公开材料的上传，日志应与设备、源码哈希、环境和样本一起保存。内置 CUPTI 活动追踪不能替代受到权限限制的 Nsight Compute 硬件计数器；不要将 kernel 耗时之和直接当作端到端时间。

H100 的构建接口见 `h100-retest.sh`；`SOURCE_ROOT` 指向解压后的目录、`GMP_ROOT` 指向 GMP，先传 `build`。有真实 H100 后设置 `BUILD_DIR`、`COEFF_DIR`，运行 `h100-benchmark.sh`，默认执行两项组件检查和 5 次 D31 图像 0 预热后在线计时，分别保存日志、设备和显存采样；需要时设置 `POSEIDON_WIDE_POOL_GIB`、`REPEATS`、`IMAGE_ID` 和 `OUTPUT_PARENT`。脚本检查选中设备名是否包含 H100，已在 A800 验证拒绝运行；完整基准、Nsight 硬件计数器可用性和性能结果均需在 H100 真机检查。已有 `sm_90` 交叉编译记录，尚无 H100 实机结果。用户本阶段不要求新增 V100 实测，之前的 V100 记录见 `../resnet20-reproduce.md`；本轮 `sm_70` 在 A800 交叉编译通过，不能引用 A800 数据充作 V100 结果。

```bash
export SOURCE_ROOT=/path/to/d31-rebuild
export GMP_ROOT=/path/to/gmp
export AUDIT_TOOLS=/path/to/d31-audit-20261118
bash "$AUDIT_TOOLS/h100-retest.sh" build
export BUILD_DIR="$SOURCE_ROOT/build-sm90-audit"
export COEFF_DIR="$SOURCE_ROOT/trident-functional/resnet20/functional_params"
export SOURCE_ARCHIVE=/path/to/source.tar.gz
export CUDA_VISIBLE_DEVICES=0 REPEATS=5
# 先确认 GPU 剩余显存；足够时可设置 POSEIDON_WIDE_POOL_GIB=60 复现 A800 配置。
bash "$AUDIT_TOOLS/h100-benchmark.sh"
```

性能、正确性和限制见 `progress.md`；逐项实验、原始日志与汇总脚本位于同目录。当前密码学状态为 `SECURITY_NOT_VALIDATED`。
