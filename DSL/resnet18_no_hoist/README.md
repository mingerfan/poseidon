# ResNet18：模型级 Python DSL / Native no-hoist

入口是 **[ResNet18.py](ResNet18.py)**，交付形式与相邻resnet20_no_hoist对齐：
模型源码、可换路径构建入口、框架补丁、配置、测试数据、验证报告及真实RuntimePlan。

```text
ResNet18.py → Hecate/Earth → 完整 Dacapo depth-dp → RuntimePlan
            → 正常 SequentialRuntime → Poseidon GPU / Native NoHoistBsgs
```

Python不读取native物理trace，不手写bootstrap/rescale/密文level。
模型保留原102个权重、d13多项式ReLU、边界归一化20、3×3 stride2平均池化、
2/2/2/2 BasicBlock和1000类FC。N65536/32768 slots，8个输入密文。
本目录替换混合旧HEVM与服务器实验脚本的交付；旧版本可从Git历史查阅。

## 已验证结果

| 检查 | 结果 |
| --- | --- |
| 前端槽算术 vs 独立dense FP64 | 最大差异8.881784197001252e-15 |
| 正常明文Runtime vs Python | 最大差异0 |
| 整网加密GPU / 最终独立审计 | PASS / PASS |
| 样本 / 预测 / 标签 | image0 / 65 / 65 |
| Compute / Native Boot | 423312 / 4705 |
| logits最大误差 / RMS | 0.008928450882893735 / 0.002623180777834832 |
| 最大虚部 | 9.927378136936084e-5 |
| 在线时间 / GPU进程总耗时 | 11292.59秒 / 3:11:15 |

1000输出满足原门槛abs(error)<=0.1+0.006*abs(reference)，虚部门槛0.01。
仅为单样本，不是数据集准确率或同密文逐位一致结论。
最终配置Q50/P25、H192、context scale51、内部EvalMod scale60、ratio128、
degree58/generation59、NoHoistBsgs=4、实数投影；Boot输出level15/scale60。
原ratio32/scale52整网失败，修正未放宽门槛。V10032GiB/31GiB池，无固定FHE种子、Host刷新。

## 文件

- ResNet18.py、imagenet_layout.py：模型与槽布局；check_semantics.py/check_layout.py独立校验。
- compile_model.py：原artifact CLI的模型注册/按需编码入口，runtime路径由参数指定。
- generate_profiles.py、profiles/：从实际组件日志生成匹配配置，包含ratio128读取。
- patches/：与ResNet20一致的固定框架基础版本和修复。
- build-support/：复用正常框架目标，覆盖实际通过的测试入口和optimizer driver源码。
  支持8输入/1000输出、传输handle回收、描述符索引和按需明文编码，不引入模型专用执行器。
- fixtures/：原102个权重/测试图像archive、d13、运行fixture及参考。
  native-encrypted-reference.json是历史加密参考，精确数值参考仍为独立dense方程。
- evidence/：完整输出、组件/语义/独立审计和压缩的真实RuntimePlan。
- verify_result.py：标准库复核交付SHA、真实计划和1000输出原门槛。

不上传二进制、约25GiB Earth bytecode或约26GBplaintext bundle；从模型重新生成。
压缩plan仅供审计，不能省略重新编译的bundle后单独运行。

## 1. 准备匹配的 framework

以下命令在Linux Bash执行，Python需NumPy、Torch、einops（依赖沿用Dacapo）。
需要匹配的LLVM/MLIR18工具链、CMake≥3.26.4、CUDA及GMP；本次实际LLVM18.1.2、
CUDA12.2、sm70。先进入本目录，然后设置绝对路径：

```bash
set -euo pipefail
export PKG="$PWD"
export WORK="/absolute/path/to/new-resnet18-work"
mkdir -p "$WORK"
git clone --branch framework https://github.com/mingerfan/poseidon.git "$WORK/framework"
export FRAMEWORK="$WORK/framework"
git -C "$FRAMEWORK" checkout --detach 6dc8cbfb65e3311271563000f6f5dce92d0324a8
git -C "$FRAMEWORK" submodule update --init --recursive third_party/ckks-runtime
export RUNTIME="$FRAMEWORK/third_party/ckks-runtime"
export HECATE="$RUNTIME/third_party/dacapo"
test "$(git -C "$RUNTIME" rev-parse HEAD)" = 0d3ba226b0329932ca9aed4a8eaa1797f5893965
test "$(git -C "$HECATE" rev-parse HEAD)" = 6e75be29a52b69bc56404fc07e44465fe8971d29
git -C "$FRAMEWORK" apply --check "$PKG/patches/poseidon-framework.patch"
git -C "$RUNTIME" apply --check "$PKG/patches/runtime-packed-input.patch"
git -C "$HECATE" apply --check "$PKG/patches/dacapo-llvm18.patch"
git -C "$FRAMEWORK" apply "$PKG/patches/poseidon-framework.patch"
git -C "$RUNTIME" apply "$PKG/patches/runtime-packed-input.patch"
git -C "$HECATE" apply "$PKG/patches/dacapo-llvm18.patch"
```

子模块URL沿用上游SSH配置，需要相应Git访问权限；仅有HTTPS权限时可在本次
clone中设置对应HTTPS URL后初始化，不必修改全局Git配置。

补丁保留实际通过的源码：secret sampler边界/初始化/拒绝采样修复、Native Boot
输出系数归一化、测试入口的完整余弦/实数配置和packed输入支持，以及LLVM18
`dyn_cast`作用域修复。没有修改Dacapo规划算法，也没有引入自定义HEVM执行循环。
`patches/revisions.json`列出全部改动文件，SHA与实际服务器构建源码相符。

## 2. 构建当前框架和正常runtime入口

设置已有LLVM/MLIR安装目录；Dacapo自己的Nix环境也可使用，见其README。

```bash
export LLVM_DIR="/absolute/path/to/llvm18/lib/cmake/llvm"
export MLIR_DIR="/absolute/path/to/llvm18/lib/cmake/mlir"
export HECATE_BUILD="$WORK/build-dacapo"
cmake -S "$HECATE" -B "$HECATE_BUILD" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR" -DMLIR_DIR="$MLIR_DIR" -DBUILD_TESTING=ON
cmake --build "$HECATE_BUILD" --target hecate-opt HecateFrontend -j2
ctest --test-dir "$HECATE_BUILD" --output-on-failure
python "$PKG/build-support/build_optimizer.py" --hecate-build "$HECATE_BUILD" --output "$WORK/compiler-driver"
export PYTHONPATH="$HECATE/python/hecate:$HECATE/python/poly${PYTHONPATH:+:$PYTHONPATH}"

export CUDACXX=/usr/local/cuda/bin/nvcc
export BUILD="$WORK/build-runtime"
cmake -S "$PKG/build-support" -B "$BUILD" \
  -DPOSEIDON_SOURCE="$FRAMEWORK" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DPOSEIDON_BUILD_DEPS=OFF \
  -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DPOSEIDON_USE_INTEL_HEXL=OFF -DPOSEIDON_USE_OPENMP=OFF \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON \
  -DCKKS_RUNTIME_BUILD_TESTS=OFF -DCKKS_RUNTIME_BUILD_EXPERIMENTS=ON
cmake --build "$BUILD" --target poseidon_runtime_gpu_bootstrap_e2e \
  poseidon_runtime_gpu_mlp_e2e dacapo_mlp_vec_e2e -j2
export BIN="$BUILD/poseidon/bin"
```

CUDA/CUPTI若没有自动找到，按本机CUDA路径补充
`-DCUDA_cupti_LIBRARY=.../extras/CUPTI/lib64/libcupti.so` 和
`-DCUDAToolkit_CUPTI_INCLUDE_DIR=.../extras/CUPTI/include`。
构建入口只复用RMM已提供的spdlog头文件，没有另一份模型执行器。

## 3. 恢复数据、检查语义并导出Earth

```bash
python "$PKG/prepare_data.py" --output "$WORK/data"
export DATA="$WORK/data/resnet18"
export WEIGHTS="$DATA/pretrained_parameters/resnet18_imagenet"
export D13="$PKG/fixtures/d13.txt"
python "$PKG/check_semantics.py" --weights "$WEIGHTS" --relu-coefficients "$D13" \
  --values "$DATA/testFile/test_values.txt" --labels "$DATA/testFile/test_label.txt" \
  --native-logits "$PKG/fixtures/native-encrypted-reference.json" \
  --report "$WORK/semantics.json" --runtime-fixture "$WORK/fixture.json"
python "$PKG/ResNet18.py" --weights "$WEIGHTS" --relu-coefficients "$D13" \
  --output "$WORK/traced" --bytecode
```

输出目录/报告须尚不存在，必须使用含真实常量的ResNet18.mlirbc。

## 4. 组件、完整Dacapo规划、正常明文Runtime

选择无人占用、空闲至少32256MiB的健康GPU；以下GPU1可按机器调整。

```bash
for name in ${!POSEIDON_@}; do unset "$name"; done
export CUDA_VISIBLE_DEVICES=1
export POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=31744
export POSEIDON_RUNTIME_BOOTSTRAP_DEGREE=65536
export POSEIDON_RUNTIME_BOOTSTRAP_LINEAR_MODE=no_hoist
export POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE=1
export POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE=1
export POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE=1
export POSEIDON_RUNTIME_BOOTSTRAP_DIAGNOSTICS=1
export PRECISION_PROBE_OUTPUT="$WORK/native-component.json"
"$BIN/poseidon_runtime_gpu_bootstrap_e2e" > "$WORK/native-probe.log" 2>&1
echo NATIVE_NOHOIST_PROBE_EXIT=0 >> "$WORK/native-probe.log"
python "$PKG/generate_profiles.py" --runtime-root "$RUNTIME" \
  --native-probe-log "$WORK/native-probe.log" \
  --compiler-profile-template "$PKG/profiles/compiler-profile.json" --output "$WORK/profiles"
python "$PKG/compile_model.py" --runtime-root "$RUNTIME" resnet18 \
  --operator-spec "$WORK/profiles/operator-spec.json" \
  --compiler-profile "$WORK/profiles/compiler-profile.json" \
  --traced-mlir "$WORK/traced/ResNet18.mlirbc" \
  --hecate-opt "$WORK/compiler-driver/hecate-opt" --hecate-build "$HECATE_BUILD" \
  --boot-placement depth-dp --waterline 40 --lazy-rescale-level-factor 3 \
  --device-counts 1 --runtime-plan-memory --runtime-plan-memory-report \
  --inline-payload-max-bytes 0 --plan-format json --output-dir "$WORK/compiled"
export PLAN="$WORK/compiled/resnet18.optimized._hecate_ResNet18Frontend.runtime-plan.json"
export BUNDLE="${PLAN%.runtime-plan.json}.bundle"
"$BIN/dacapo_mlp_vec_e2e" "$PLAN" "$WORK/profiles/operator-spec.json" "$BUNDLE" \
  "$WORK/fixture.json" "$WORK/clear.json" "$WORK/clear.diff.log"
```

每一步成功再继续。按需编码driver仅延迟Encode/上传到首次使用前，交由正常Runtime执行释放，
避免权重一次全部驻留；不修改Dacapo规划算法、不绕过Waterline native资源边界检查。
单rank每256次传输完成handle回收，避免长期累积异步任务。

## 5. 全量加密运行与历史结果复核

确认GPU仍空闲后：

```bash
for name in ${!POSEIDON_@}; do unset "$name"; done
export POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=31744
export POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary
export POSEIDON_RUNTIME_TRACE="$WORK/full.trace.csv"
"$BIN/poseidon_runtime_gpu_mlp_e2e" "$PLAN" "$WORK/profiles/operator-spec.json" \
  "$BUNDLE" "$WORK/fixture.json" "$WORK/clear.json" "$WORK/full.json" \
  "$WORK/profiles/native-config.json" > "$WORK/full.log" 2>&1
```

必须传最后的Native配置。仍是框架通用E2E入口经正常SequentialRuntime执行，完整运行数小时。
不自动重试数值失败，不放宽门槛。

只复核随包历史成功证据，无需GPU或第三方Python库：

```bash
python "$PKG/verify_result.py"
```

审计解压约245MiB真实plan，需要数GiB主机内存，核对4705个Native Boot、文件SHA和1000输出。
此命令不把新运行文件冒充历史结果；原provenance服务器路径只是证据，不是脚本依赖。
本次打包/构建核验见evidence/packaging-validation.json。历史完整GPU证据仍来自原运行，
没有宣称本次新构建又完成了整网加密执行。
