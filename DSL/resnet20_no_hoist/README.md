# ResNet20：模型级 Python DSL / Native no-hoist

入口是 **[ResNet20.py](ResNet20.py)**，参考当前 framework 的 `MLP.py` 写法。
它直接描述模型计算，读取原生 `resnet20_new` 的全部116个文本参数，使用原
ReLU d13 多项式近似、边界归一化40、Option-A残差、平均池化和原分类头。
不是 SiLU，也不是精确的 `max(0,x)`。

实际执行流程：

```text
ResNet20.py → Hecate/Earth → 完整 Dacapo 规划 → RuntimePlan
            → 正常 SequentialRuntime → Poseidon GPU / Native NoHoistBsgs
```

前端不读取 native 物理 trace，不手工指定 bootstrap、rescale 或密文层级。
编译器自行规划本次466次自举，不强制复刻旧原生18次物理计划。
自举是真正的同态 Native no-hoist，不使用 Host 解密再加密刷新。

本目录已替换旧 trace→CKKS MLIR→HEVM 包，删除其执行器、sidecar、native源码
快照和生成产物。旧版本可从 Git 历史查阅；其他模型目录不属于本次更新范围。

## 已验证结果

| 项目 | 本次结果 |
|---|---|
| 前端 FP64 vs 原生独立 C++ CPU oracle | 最大误差 `1.509903313490213e-14` |
| 编译计划正常明文 Runtime vs Python | 最大误差 `0` |
| 完整加密 GPU 运行 | `exit0`，`passed=true` |
| 样本 | image0，label3；CPU/GPU均预测3 |
| Compute / Native no-hoist Boot | 16222 / 466 |
| logits 最大误差 / RMS | `0.06999352187590024` / `0.03890168834121847` |
| 最大虚部 | `3.9892700148298886e-05` |
| 在线运行时间 | `612.189` 秒 |

这是单样本验证，不是测试集准确率，也没有证明本路线与原生同密文逐位一致。
报告中的 E2E 内部数值门限不是用户指定的验收标准。加密随机性会使复测数值波动。

最终 Native 配置：N65536、Q50（混合28..32bit）、P25、H192、context scale51、
EvalMod scale52、完整有效degree58（generation59）、`NoHoistBsgs=4`，同态实数投影。
Boot输出physical level21/scale52；前端输入scale40由计划规定。它保留模型计算语义，
不声称复刻旧native的所有加密参数或物理计划。GPU验证使用V100 32GiB、pool26GiB。

## 文件

- `ResNet20.py`：前端；复用 framework 现有 Hecate/poly 模型计算助手。
- `check_semantics.py`：同一前端的槽计算、独立CHW方程和native CPU参考对照。
- `generate_profiles.py`：从通过的Native组件日志生成编译/运行能力配置。
- `profiles/`：本次实测匹配的三份配置；算子延迟仍是框架估算值。
- `patches/`：本次验证所需的少量框架修复及精确基础版本/源文件SHA。
- `build-support/`：复用正常runtime测试入口的CMake构建入口。
- `fixtures/`：本次image0的输入/packed slots与原生CPU参考；不是模型权重副本。
- `evidence/`：原始数值报告、组件日志、语义报告和压缩后的真实RuntimePlan。
- `verify_result.py`：标准库独立复核交付文件SHA、计划、自举类型及报告数值。

不上传二进制、557MiB的Earth bytecode或大型plaintext bundle；这些应从 `.py`
经框架重新生成。压缩的RuntimePlan仅供审计；运行需要本地重新编译生成的bundle。

## 1. 准备匹配的 framework

以下命令在Linux Bash执行，Python需NumPy、Torch、einops（依赖沿用Dacapo）。
需要匹配的LLVM/MLIR18工具链、CMake≥3.26.4、CUDA及GMP；本次实际LLVM18.1.2、
CUDA12.2、sm70。先进入本目录，然后设置绝对路径：

```bash
set -euo pipefail
export PKG="$PWD"
export WORK="/absolute/path/to/new-resnet20-work"
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
cmake -S "$HECATE" -B "$HECATE_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR" -DMLIR_DIR="$MLIR_DIR" -DBUILD_TESTING=ON
cmake --build "$HECATE_BUILD" --target hecate-opt HecateFrontend -j2
ctest --test-dir "$HECATE_BUILD" --output-on-failure
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

## 3. 检查语义并从Python导出Earth

权重、d13、image0已在上述固定framework提交中，直接使用，不需要另下载checkpoint。

```bash
export DATA="$FRAMEWORK/bench/resnet20_s2c_first/support/data/resnet20"
export WEIGHTS="$DATA/pretrained_parameters/resnet20_new"
export D13="$DATA/relu_param/d13.txt"
python "$PKG/check_semantics.py" --weights "$WEIGHTS" --relu-coefficients "$D13" \
  --images "$DATA/testFile/test_values.txt" \
  --native-oracle "$PKG/fixtures/native-cpu-reference.json" \
  --report "$WORK/semantics.json" --runtime-fixture "$WORK/fixture.json"
python "$PKG/ResNet20.py" --weights "$WEIGHTS" --relu-coefficients "$D13" \
  --output "$WORK/traced" --bytecode
```

输出目录/报告文件需尚不存在。必须使用包含真实常量的 `ResNet20.mlirbc`；
不能把省略常量的inspect MLIR拿去编译。完整bytecode约557MiB。

## 4. Native组件、完整Dacapo规划、正常明文Runtime

先通过 `nvidia-smi` 选择无人占用且空闲≥28GiB的GPU，下面以GPU3为例。
先清除本shell继承的 `POSEIDON_*`，再显式设置本次配置，避免旧实验override。

```bash
for name in ${!POSEIDON_@}; do unset "$name"; done
export CUDA_VISIBLE_DEVICES=3
export POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=26624
export POSEIDON_RUNTIME_BOOTSTRAP_DEGREE=65536
export POSEIDON_RUNTIME_BOOTSTRAP_LINEAR_MODE=no_hoist
export POSEIDON_RUNTIME_BOOTSTRAP_FULL_COSINE=1
export POSEIDON_RUNTIME_BOOTSTRAP_MODEL_PROFILE=1
export POSEIDON_RUNTIME_BOOTSTRAP_REAL_PROFILE=1
export POSEIDON_RUNTIME_BOOTSTRAP_DIAGNOSTICS=1
if "$BIN/poseidon_runtime_gpu_bootstrap_e2e" > "$WORK/native-probe.log" 2>&1; then
  echo NATIVE_NOHOIST_PROBE_EXIT=0 >> "$WORK/native-probe.log"
else
  echo 'Native component failed; stop before compilation/model execution.' >&2
  exit 1
fi
python "$PKG/generate_profiles.py" --runtime-root "$RUNTIME" \
  --native-probe-log "$WORK/native-probe.log" \
  --compiler-profile-template "$HECATE/profiled_HEAAN_GPU.json" --output "$WORK/profiles"

python "$RUNTIME/integrations/dacapo/generate_model_artifacts.py" resnet20 \
  --operator-spec "$WORK/profiles/operator-spec.json" \
  --compiler-profile "$WORK/profiles/compiler-profile.json" \
  --traced-mlir "$WORK/traced/ResNet20.mlirbc" \
  --hecate-opt "$HECATE_BUILD/bin/hecate-opt" --hecate-build "$HECATE_BUILD" \
  --boot-placement depth-dp --waterline 40 --lazy-rescale-level-factor 3 \
  --device-counts 1 --runtime-plan-memory --runtime-plan-memory-report \
  --inline-payload-max-bytes 0 --plan-format json --output-dir "$WORK/compiled"
export PLAN="$WORK/compiled/resnet20.optimized._hecate_ResNet20.runtime-plan.json"
export BUNDLE="${PLAN%.runtime-plan.json}.bundle"
"$BIN/dacapo_mlp_vec_e2e" "$PLAN" "$WORK/profiles/operator-spec.json" "$BUNDLE" \
  "$WORK/fixture.json" "$WORK/clear.json" "$WORK/clear.diff.log"
```

每一步成功后再执行下一步。明文Runtime中的Boot是理想恒等，不能代替加密验证。

## 5. 全量加密运行与历史结果复核

确认GPU仍空闲后，清除组件阶段环境，只留下模型运行所需配置：

```bash
for name in ${!POSEIDON_@}; do unset "$name"; done
export CUDA_VISIBLE_DEVICES=3
export POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=26624
export POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary
export POSEIDON_RUNTIME_TRACE="$WORK/full.trace.csv"
"$BIN/poseidon_runtime_gpu_mlp_e2e" "$PLAN" "$WORK/profiles/operator-spec.json" \
  "$BUNDLE" "$WORK/fixture.json" "$WORK/clear.json" "$WORK/full.json" \
  "$WORK/profiles/native-config.json" > "$WORK/full.log" 2>&1
```

该入口仍是framework现有的MLP通用E2E runner，经正常SequentialRuntime执行
编译后的ResNet20；不能省略最后的Native配置参数，否则不能作为本次no-hoist验证。

仅复核本次已保存的历史结果不需要GPU或第三方Python库：

```bash
python "$PKG/verify_result.py"
```

原始日志/provenance保留当时服务器路径，路径是历史记录，不是运行脚本依赖。
审计会解压真实plan、核验SHA/全部466个Native Boot，独立重算logits误差与分类。
