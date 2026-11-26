# gpu-api 的四条原生 GPU 自举路径

编译器通过一个原生 `Boot` 算子调用以下四个 `operator_profile`。四者均使用分支锁定的 `ckks-runtime`（`369bdc55`）、OperatorSpec V1 和 RuntimePlan V1。输入固定为 N=65536、32768 槽、两个 NTT 分量、Q6、scale `2^40`。Q 的数字表示活动 Q 素数枚数，Runtime 的 level 从 0 起算。[公开接口](../src/poseidon/runtime_api/s2c_bootstrap.h)

## 整体性能、输出与精度

2026-11-26，Tesla V100-SXM2-32GB，Release / sm70，从项目根目录用锁定的 Runtime 干净构建。表中每行的误差和 RMS 取该路径常规输入的首次全槽解密，延迟取同一进程 3 次热身、18 次同步单次调用的墙钟中位数。[正式回归与独立密钥记录](../test-results/s2c-first-v100-20261126/integer-runtime.txt)

| `operator_profile` | 输出语义 | 输出 Q / scale | 在线中位数 | 最大绝对误差 | RMS |
| --- | --- | --- | ---: | ---: | ---: |
| `s2c-first-degree59-da2-q50p25` | 末端投影到实数 | Q31 / `2^40` | 104.423 ms | 8.71975e-04 | 2.77627e-04 |
| `s2c-first-degree30-da3-q34p9` | 保留实部与虚部 | Q14 / `2^50` | 66.628 ms | 6.15344e-05 | 1.77117e-05 |
| `s2c-first-degree22-da3-q43-dnum4` | 实虚两支 EvalMod 后投影到实数 | Q24 / `2^40` | 60.087 ms | 8.61674e-04 | 2.77603e-04 |
| `s2c-first-degree22-da3-q43-dnum4-real-only` | 仅实支 EvalMod，输出实数 | Q24 / `2^40` | 49.014 ms | 8.60839e-04 | 2.77602e-04 |

计时从驻留输入调用 `api.compute(Boot)` 开始，到 `api.synchronize(output)` 返回为止，包括发射、GPU 计算、输入输出的设备内复制和最终等待。密钥、矩阵与输入上传在窗口之前，取回主机与解密在窗口之外。本轮一次性准备分别为 59 阶 290.4 s、30 阶 148.6 s、两条 22 阶共用 316.2 s。两条 22 阶在同卡、同密钥、同一份输入下配对测量。59 阶与 30 阶分别使用独立卡和各自参数。单独的 22 阶 benchmark 另有 CUPTI 活动时间。[计时代码](../src/poseidon/tests/runtime_api/poseidon_gpu_bootstrap_e2e.cpp) · [独立 benchmark](../bench/bootstrap_agent22/README.md)

各路径均按所有 32768 个复数槽计算 `abs(解密值 − 明文参考)`，实数输出使用实值输入并计入虚部残差。正式回归的统一退出门槛是最大绝对误差 ≤ `1e-3`。30 阶另外逐槽记录超过 `1e-4` 的数量，并检查解密前的 Q14 和精确输出 scale。

## 30 阶如何兼容整数 scale

30 阶保留完整 30 阶系数、三次倍角和实虚两路 EvalMod。S2C 将 Q6 降至 Q2，先在 Q2/P2 上切至重量 32 的临时密钥，模升到 Q34，再在 Q34/P9 切回重量 192 的主密钥。C2S 从 Q34 降到 Q30，EvalMod 从 Q30 降到 Q14。最终返回的复数密文 scale 精确等于 `2^50`，Runtime 计划写整数 `50`。[参数与密钥配置](../src/poseidon/runtime_api/s2c/native_boot.cpp) · [系数与倍角定标](../src/poseidon/runtime_api/s2c/native.patch)

原 30 阶排程自然产生 `log₂(scale) ≈ 50.11407545`。仅将标签改成 50 会让解码幅度放大约 8.23%，幅度 0.5 的消息仅此一项就偏约 0.041。现实现把比例 `2^50 / 2^50.11407545 ≈ 0.923974` 折入已有的 EvalMod 多项式系数和三个倍角常数。最终合并处先检查折算出的 scale 和目标相符，再写入精确的 `2^50`。这一步复用已有乘法，Q14、倍角次数和在线 kernel 排程保持一致。应用应从注册返回值生成计划并准备匹配的评估材料。

30 阶还修正了两处独立的近似误差。C2S 按模升后的实际 scale 对齐正弦周期，稀疏临时密钥收窄了模升后整数项的分布。S2C 的明文对角线把消息幅度缩为四分之一，复数合并处乘 128 恢复幅度，降低复数角点近似误差。旧版直接在重量 192 的主密钥下模升时实测整数项绝对值达 22，超出 ±15 的插值中心范围。修复前、消融和小数 scale 中间版本的历史数据分别存档。[误差诊断](../test-results/s2c-first-v100-20261126/precision.txt) · [消融与小数 scale 记录](../test-results/s2c-first-v100-20261126/degree30-repair.txt) · [初次迁移记录](../test-results/s2c-first-v100-20261125/measurements.txt)

正式回归使用一套新生成的、经检查的密钥，四类明文各独立加密 3 次。另在三张 V100 上用三套独立密钥做过同样的整数 scale 兼容实验。下表分别给出两个实验中每类输入的全槽最坏误差，RMS 为正式回归每类 3 份密文中单份 RMS 的最大值。三卡兼容实测的 18 次调用中位数分别为 66.771、66.615、63.900 ms。[逐次记录](../test-results/s2c-first-v100-20261126/integer-runtime.txt)

| 30 阶输入 | 正式回归最大误差 | 正式回归最大 RMS | 三套独立密钥兼容实验最大误差 |
| --- | ---: | ---: | ---: |
| 叠加正弦，实部幅度 ≤0.5、虚部 ≤0.3 | 6.30673e-05 | 1.77555e-05 | 6.65606e-05 |
| 复数网格，最大实部 17/32、最大虚部 11/64 | 7.67978e-05 | 2.66536e-05 | 7.67191e-05 |
| 复数角点，实部和虚部各取 ±0.5 | 9.08755e-05 | 7.38539e-05 | 9.45908e-05 |
| 零消息，非零密钥加密 | 1.78725e-05 | 5.35253e-06 | 2.20718e-05 |

## 应用如何选用

使用同一上下文生成输入、密钥并配置设备。在准备阶段从注册返回的 `BootProfile` 读取输出 level、整数 log₂(scale) 和分量数，将其加入 `OperatorSpec`。设备上 `Boot` 的属性应使用同一份描述。[GPU API 声明](../src/poseidon/runtime_api/poseidon_gpu_api.h) · [完整运行示例](../src/poseidon/tests/runtime_api/poseidon_gpu_bootstrap_e2e.cpp)

```cpp
using namespace poseidon;
using namespace poseidon::runtime_api;

const auto name = s2c_bootstrap_profiles[3];
auto context = make_s2c_bootstrap_context(name);
auto secret = make_s2c_bootstrap_secret(context, name);
KeyGenerator keys(context, secret);
PoseidonGpuApi api("my-context", context, 0);
const auto profile = api.configure_s2c_bootstrap(0, keys, {name}).at(0);

fhegpu::BootAttrs attrs{profile.output_level, profile.output_scale_log2,
                        profile.output_components, profile.profile_id,
                        fhegpu::BootImplementation::Native};
```

22 阶双支和单实部可一次注册 `{s2c_bootstrap_profiles[2], s2c_bootstrap_profiles[3]}`，共享同设备的矩阵、评估密钥和工作区。前者执行两路 EvalMod 后投影到实数，后者只执行实支 EvalMod。59 阶也在末端投影到实数。需要保留虚部时选 30 阶。不同 Q/P 链各自在匹配的上下文注册。旧调用方仍可用原接口注册自定义名称。[设备执行与工作区复用](../src/poseidon/runtime_api/s2c/native_boot.cpp)

## 如何复测

从项目根目录构建。需要 CUDA、CMake ≥ 3.26.4、Python 3、tar、patch、GMP 和支持 PIC 的 fmt/spdlog。先初始化项目固定的 Runtime 依赖。下例使用空闲 V100，已有 fmt/spdlog 前缀可直接用于 `CMAKE_PREFIX_PATH`。[项目依赖说明](../README.md)

```bash
git submodule update --init --recursive third_party/ckks-runtime
work=build_bootstrap_migration
prefix="$PWD/$work/prefix"
cmake -S third_party/fmt -B "$work/fmt" -DFMT_TEST=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build "$work/fmt" --target install --parallel 8
cmake -S third_party/spdlog -B "$work/spdlog" -DSPDLOG_FMT_EXTERNAL_HO=ON \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_PREFIX_PATH="$prefix" -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --build "$work/spdlog" --target install --parallel 8

cmake -S . -B "$work/native" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_PREFIX_PATH="$prefix" -DPOSEIDON_USE_OPENMP=ON -DPOSEIDON_BUILD_DEPS=OFF \
  -DPOSEIDON_USE_ZLIB=OFF -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DPOSEIDON_BUILD_BENCH=OFF -DPOSEIDON_BUILD_EXAMPLES=OFF \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_API=ON -DPOSEIDON_BUILD_CKKS_RUNTIME_TESTS=ON \
  -DPOSEIDON_BUILD_CKKS_RUNTIME_GPU_TESTS=ON
cmake --build "$work/native" --parallel 8 --target poseidon_runtime_gpu_bootstrap_e2e
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8
ctest --test-dir "$work/native" -R '^poseidon_runtime_gpu_bootstrap_e2e_' --output-on-failure
```

CTest 的三项测试覆盖四条路径。也可向 `"$work/native/bin/poseidon_runtime_gpu_bootstrap_e2e"` 传入表中的完整名称，逐条运行。无参数会用相同密钥和输入跑两条 22 阶路径。全槽精度见 `PRECISION`，驻留在线中位数见 `ONLINE`。有 CUDA 设备并实际出现这两类输出，才表示执行了 GPU 回归。

## 密钥、安全性与已知限制

- 正式精度验证和延迟测量均使用操作系统随机源。固定重量采样要求 `1 ≤ h ≤ N`，检查实际秘密的非零性、三元性、重量和 RNS 一致性。主密钥重量为 59/30 阶 192、22 阶 32768，30/22 阶临时密钥重量为 32。CPU 负向测试含零重量、超范围重量、导入全零及不一致的 RNS。旧 GPU 加法演示仍把 CKKS 固定重量设为零，运行到密钥生成时会报错。[密钥检查](../src/poseidon/keygenerator.cpp) · [负向测试](../src/poseidon/tests/runtime_api/poseidon_cpu_api_test.cpp) · [旧演示](../src/poseidon/tests/gpu_data_struct/demo_gpu_ciphertext_add_handler.cpp)
- 两条 22 阶在 Q43 用 P11、dnum=4。实际生成的评估密钥中，最大 log₂(PQ) 在 Q37/P19，为 **1775.35 < 1782 bit**。这验证了当前模数预算。128 位安全性还需对实际主秘密、临时秘密和噪声联合估算。59 阶 Q50/P25 为研究配置。30 阶此次验证覆盖正确性、输出格式和性能，尚无独立安全评估。[分层密钥核验](../bench/bootstrap_agent22/README.md)
- 单次原生自举已在单卡 `SequentialRuntime` 上验证。双卡 GPU API 回归覆盖通用算子。一次自举跨卡拆分、V4 Waterline 原生 Boot 显存上界均在此次验证范围之外。
- `--legacy-diagnostic` 显式进入原 N=16384 的自举诊断，其历史默认参数在非零密钥下超过原精度门槛。旧的独立目标 `test_gpu_bootstrap_modraise` 还引用当前分支缺失的 CPU 方法 `eval_mod_high_precision`。四条公开路径的构建和回归入口见上节。[历史原始记录](../test-results/s2c-first-v100-20261125/measurements.txt)
