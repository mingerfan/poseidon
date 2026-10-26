# Poseidon GPU ResNet20 — 默认22阶自举

这是原生 C++ 单GPU加密 CIFAR-10 ResNet20，不是CPU版或DSL版。密文卷积、ReLU、自举、池化和FC使用 `poseidon::gpu::GpuEvaluator`；CPU用于参数/密钥/明文编码、数据加载和可选的解密参考验证。

网络保持 Trident ResNet20 语义：输入3×32×32并除以boundary40；3/3/3个BasicBlock，通道16/32/64；跨阶段使用stride2卷积与Option-A shortcut；ReLU多项式保持 `[15,15,27]`；最后全局平均池化和64×10 FC。完整图片始终执行 **18次自举**，降低自举阶数不改变网络、权重或ReLU阶数。

## 当前默认与修复

默认采用 **22阶、k=25、5次倍角**。30阶使用4次倍角；59组保持原生离散拟合及2次倍角，偶多项式实际最高58次。22/30直接生成固定阶余弦Chebyshev展开（Bessel系数），不是截断59阶系数，也不把k缩成12/16。

H192密钥下旧的小区间出现越界：k16/k12时归一化输入实测达到1.0625/1.4167。现在统一保留k25。GPU EvalMod工作scale改为2^52，应用scale保持2^40，并校正S2C增益，避免工作精度变化带来的幅度错误。

**自举末尾不再裸ModRaise。** 工作链预留足够预算，真正刷新后的结果仍高于应用Q36，随后仅向下降模；预算不足时直接报错。30阶叶节点规划不能恢复生成基函数时已消耗的Q素数，该修复用 `POSEIDON_GPU_RESNET20_FIXED_DEGREE_BOOTSTRAP` 限定在本目标，不改变ResNet18/50的编译路径。复用分支中已修好的非零密钥采样器。

共同参数：

| 参数 | 值 |
|---|---|
| N / slots | 65536 / 32768 |
| Hamming weight | 192（非零三元密钥） |
| Q / P / dnum | 70 / 35 / 2 |
| message ratio / k | 32 / 25 |
| EvalMod工作scale | 2^52 |
| 应用输出 | Q36、scale 2^40 |
| 应用逻辑层 | 17，加两物理素数基底 |

GPU物理素数不超过32位。`sec_level_type::none`；扩链改变了Q/P总位数，本次未认证安全位数，也不宣称安全等级不变。

## 59 / 30 / 22阶实测对比

**2026-10-22实际结果**，基于冻结提交 `943515e4e9a3e6010b295892ad7fcb4edb5db274` 的独立修复副本。在同一V100-SXM2-32GB、同一进程/密钥、同一组图片0—9和同一模型上依次测试。三组使用相同Q70/P35及组合旋转。每张18次自举，各完成10张，共540次验证自举（不含组件与计时重放）。

| 自举阶数 | 倍角次数 | 图片通过 | CPU/GPU预测一致 | 整网中位数/秒 | 比59减少耗时 | 最大logits误差 | 合并logits RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 59（实际58） | 2 | 10/10 | 10/10 | 78.190152 | — | 0.0164434920 | 0.0063681966 |
| 30 | 4 | 10/10 | 10/10 | 71.408415 | 8.67% | 0.0028502634 | 0.0011037836 |
| **22（默认）** | **5** | **10/10** | **10/10** | **68.510075** | **12.38%** | **0.0057163887** | **0.0020503007** |

30阶在本组中误差最小；22阶最快，比30阶再减少约4.06%整网耗时。完整验证保留原中间层0.02和最终logits 0.1门槛，并要求CPU/GPU预测一致，没有放宽阈值。合并RMSE基于10张图的100个最终logits，独立从CPU/GPU输出重算。10张均命中标签，但这不是完整CIFAR-10测试集准确率；本组只使用一把密钥，不能外推多密钥成功率。

整网时间为输入/模型缓存准备完成后，图片0三次重复的中位数，包含密文stem至FC，不含密钥生成、缓存准备和验证解密。三次重放logits误差均为0。

| 阶数 | 三次整网耗时/秒 |
|---|---|
| 59 | 78.190152231, 78.217020511, 77.218263257 |
| 30 | 71.408414793, 71.303292874, 71.604092635 |
| 22 | 68.510075152, 68.408482798, 68.737412880 |

### 自举组件与层数

组件输入为全部32768槽的 `0.5*sin(0.7*i+0.3)`，同一密钥及同一输入密文Q4；一次预热后同步计时三次。

| 阶数 | 单次自举中位数/秒 | 最大误差 | RMSE | 升模Q→原始输出Q→归一化Q→应用Q | 算术消耗物理Q数 |
|---|---:|---:|---:|---|---:|
| 59 | 0.862920583 | 2.90925959e-5 | 6.69957981e-6 | 70→43→42→36 | 28 |
| 30 | 0.798270581 | 9.26339036e-6 | 9.74150049e-7 | 70→40→39→36 | 31 |
| 22 | 0.794779013 | 2.18035511e-5 | 1.53411505e-6 | 70→38→37→36 | 33 |

Q数是**物理素数个数**，不等于CPU逻辑Level。消耗含输出归一化，但不含最后丢弃到Q36的素数。低阶组为保留覆盖范围增加倍角，**没有节省层数**；速度收益属于整个候选配置，不能解释为固定倍角数下单独降阶的收益。

逐图CPU/GPU logits、18次自举计数和独立重算指标见 [验证记录](validation/20261022-bootstrap-22-30-59.json)。这些是10月22日实测；10月26日的发布整合不会冒充新一轮GPU测试。

### 与旧的约9秒数据区别

此前约7.9—9.6秒来自零密钥H0、短链Q36/P18和直接旋转预加载，不能与本表比较。修复后的非零密钥配置需要Q70/P35预算；109组全链直接旋转密钥与全缓存同时驻留会使32GB显卡内存不足。因此本目标的 `--gpu-only` 和对比模式都复用39组bootstrap DFT密钥的组合旋转，不再默认预载109组直接密钥。直接旋转低显存上传优化尚未实现，不能承诺修复后仍有旧配置约9秒的速度。

## 构建和运行

需要CMake >=3.26.4、C++20、CUDA工具链及仓库内的third_party依赖；默认CUDA目标SM70。确保 `nvcc` 在PATH中，或设置 `CUDACXX=/usr/local/cuda/bin/nvcc`。`CMAKE_BIN` 可指定合适版本的cmake，`BUILD_JOBS` 默认2。自行选择空闲GPU；运行脚本不会轮询GPU。

```bash
cd benchmark/resnet20_gpu
./run.sh --topology-check
./run.sh --weights-check

# 默认22阶，完整9个BasicBlock，逐层验证
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1 ./run.sh --infer 0

# 缓存预热后，对同一图片重复三次完整GPU推理
CUDA_VISIBLE_DEVICES=0 ./run.sh --gpu-only 0

# 同一进程/密钥/输入比较三阶：10张逐层验证，然后各三次计时
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_VALIDATE_BLOCKS=1 ./run.sh --bootstrap-suite 10

# 单独比较自举组件（同一密钥/输入）
CUDA_VISIBLE_DEVICES=0 ./run.sh --bootstrap-cmp

# 单独切换30或59；未设置时默认为22
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE=30 ./run.sh --infer 0
CUDA_VISIBLE_DEVICES=0 POSEIDON_GPU_RESNET20_BOOTSTRAP_DEGREE=59 ./run.sh --infer 0
```

`RESNET20_CMP_DEGREE` 作为旧实验命令的兼容别名保留；新变量优先。`--repair-suite` 是 `--bootstrap-suite` 的兼容别名。`--infer IMAGE MAX_BLOCKS` 可只跑前缀，`MAX_BLOCKS=0` 为stem。原 `--smoke`、`--shortcut-check`、`--hoist-check`、`--head-check IMAGE` 入口保留；它们不等于本表的完整网络对比。

输入、权重和原始ReLU系数在 `data/resnet20`，可用 `POSEIDON_TRIDENT_RESNET20_ROOT` 指向另一数据目录。每次run.sh生成带时间戳的output日志；默认构建目录可用 `POSEIDON_GPU_RESNET20_BUILD_DIR` 覆盖。普通运行的elapsed包含初始化/CPU验证，不能当作GPU-only耗时。

不需要GPU的配置回归检查（从本目录运行，不要添加-DNDEBUG）：

```bash
g++ -std=c++20 -O2 -I. gpu_config.cpp tests/test_bootstrap_profile.cpp -o /tmp/resnet20_bootstrap_profile_check
/tmp/resnet20_bootstrap_profile_check
```
