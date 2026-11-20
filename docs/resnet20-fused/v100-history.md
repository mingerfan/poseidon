# ResNet20 V100 基线复现

V100 工作区是 `~/resnet20-fused-v100-20261117.P6pXY0`。根目录下的
`baseline/` 来自 Poseidon `integration/resnet20-manual-opt`，固定在
`a0ca0a894746956550afce68609a1694fc41b236`，标签为 `v100-baseline`。
`trident-functional/` 固定在 Trident `gs/feat-qwen` 的
`1ce3d5c447e92d81d624b6ff44e3c77a3cc4b518`。
`migration/` 使用独立分支 `work/resnet20-fused-v100-migration`。

## V100

在 V100 工作区根目录运行。构建使用 CUDA 12.2、CMake 3.27 和 GCC 11，
采用仓库内的依赖源码，关闭不参与此应用的压缩和 GSL 选项。

```bash
CUDA_ARCHITECTURES=70 bash build-baseline.sh
"$HOME/cmake-new/bin/ctest" --test-dir build-sm70 --output-on-failure \
  -R 'resnet20_s2c|homomorphic_dft_repack'
export CUDA_VISIBLE_DEVICES=GPU-5aaf51d1-a1db-e8f5-19bd-db86599054c5
bash run-baseline.sh performance
bash run-baseline.sh accuracy
```

性能模式使用上游默认的库内 22 阶自举，开启 CUPTI，并在计时后解密校验最终 logits。
逐阶段正确性模式由上游实现固定使用 59 阶自举。两种模式都运行输入 0，
每次运行在 `logs/` 生成新目录，保留原始日志、退出码、校验和、GPU 状态及上游验收输出。

## H100 复测命令

将两份脚本及同一基线 checkout 放到 H100 主机。脚本与 `baseline/` 同级，
`baseline/` 保留 `v100-baseline` 标签。先用 `nvidia-smi -L` 核对空闲卡，
以下命令以设备 0 为例。该配置尚未在 H100 上运行。

```bash
export CUDA_ARCHITECTURES=90 CUDA_VISIBLE_DEVICES=0
bash build-baseline.sh
ctest --test-dir build-sm90 --output-on-failure -R 'resnet20_s2c|homomorphic_dft_repack'
bash run-baseline.sh performance
bash run-baseline.sh accuracy
```

## 融合迁移核对项

Trident 每次融合激活从 L2 进入，StC 后到 L0。提升模数后将 scale 解释为 q0。
CtS 后共轭相加，接着求值 D31 复数载波、连续平方三次、求值 D15 或 D31 复数
谐波多项式，再共轭相加回到 L4、scale 2^40。这里 L 表示从 0 起算的层级。
两档均为 N=32768、P1（51 bit）、H=32，Q 分别为 19 和 20 个素数。
D15/D31 的冻结系数出自同一个 Trident 提交。

Poseidon GPU 代码中的 hybrid digit count 为 ceil(Q/P)，Trident 的单素数 P1
对应 BV 逐 Q 分解。迁移分支的组件测试覆盖两套参数下的旋转、共轭和重线性化，
在最高层和 L2 检查解密结果与 CPU BV 参考的差异。


## V100 实测

单图 0，CUDA 12.2，Tesla V100-SXM2-32GB（GPU 1），一次预热加一次带
CUPTI 的在线测量。在线段覆盖 9 个块、18 次自举，记录 83,308 项 GPU 活动。
在线 H2D/D2H、逐阶段同步和中间重加密均为 0。预热后剩余 1470.375 MiB。

| 指标 | 时间（ms） |
|---|---:|
| 在线 wall | 4394.637 |
| CUDA event | 4396.162 |
| GPU 活动区间并集 | 4318.358 |
| Host enqueue | 4378.896 |
| Conv+BN | 596.403 |
| ReLU | 818.024 |
| Bootstrap | 2866.617 |
| Shortcut | 25.044 |
| Pool+FC | 11.763 |
| Residual/copy | 0.508 |

明文及密文均预测类别 3，真实类别 3。最大 logit 绝对误差为
0.009823980640262242，最大虚部为 0.0001726562155781673。
上游性能验收通过。在线时间基于单次测量。CTest 3/3 通过，Python 39 项中
31 项通过、8 项跳过。原始性能日志及组件对照日志见 `v100-baseline-evidence/`。

## 组件测试观察

迁移分支提交 `7b413eae` 增加 `functional_keyswitch_probe`。
按原参数构建 D15/Q19 与 D31/Q20，上下文都选择 BV。两档均在创建
`GpuParameterData` 时因 45 位 Q 超过 32 位 `GpuWord` 失败。
使用 29 位 Q、30 位 P 的 P1 对照后，两档上传成功，首次旋转均返回
`HYBRID parameter shape mismatch`。该对照用于定位密钥切换接口，
保留原始 D15/D31 参数作为主验证配置。复现日志中包含每次退出码。

精确参数需要支持 45 位 Q 和 51 位 P 的 GPU RNS/NTT 运算，以及
P1/BV 的密钥切换。当前还没有 D15/D31 的 GPU 融合耗时或正确性结果。

两份提交的 116 个权重与 BN 参数文本，以及 CIFAR-10 的输入和标签文本，
逐文件 SHA-256 一致。后续融合精度验收应对照 Trident 同阶冻结近似的
明文输出，同时报告相对真实标签的分类结果。

完成宽模数 RNS/NTT 和 BV 后，先通过组件探针，再逐阶段核对
StC、ModRaise、CtS、载波和输出。两档组件均通过后接入整网的 19 个融合激活点，
并用原先的图像与计时边界对比。重复测试应保存每次原始时间、峰值显存和 logits
误差，再计算多次测量的中位数。

探针的构建和运行命令如下。以下 V100 示例用 GPU 2 跑组件，避开 GPU 1 的基线进程。

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_GPU=ON \
  -DPOSEIDON_BUILD_RESNET20_S2C_FIRST=ON -DPOSEIDON_USE_ZLIB=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/migration/third_party/spdlog"
cmake --build build-migration-sm70 --target functional_keyswitch_probe --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
build-migration-sm70/bin/functional_keyswitch_probe 15
build-migration-sm70/bin/functional_keyswitch_probe 31
build-migration-sm70/bin/functional_keyswitch_probe 15 --low-word-control
build-migration-sm70/bin/functional_keyswitch_probe 31 --low-word-control
```

对照验证补充：D15 的 29 位 Q / 30 位 P 双素数 HYBRID 配置在最高层和 L2 的
旋转、共轭、平方与重线性化均通过，比较输出的最大误差为 0。
对应临时测试源码和日志在 `v100-baseline-evidence/`。
`sm_90` 已通过 CMake 配置，H100 编译及实机测试待执行。

## V100 59 阶逐阶段正确性回归

该回归已完成，退出码为 0，上游整网验收为 `PASS`。9 个残差块、
18 次自举、18 次块内卷积、9 次残差相加均完成，中间重加密为 0。
真实标签、明文参考和 GPU 预测均为 3。最大 logit 绝对误差为
0.01043864925906401，最大网络边界误差为 0.0003319959973810063。
18 次链路诊断中有 5 次超过独立的 1e-4 端到端诊断阈值，
整网验收使用各自的边界阈值。

18 次同步计时的 GPU 自举阶段平均 211.489 ms，中位数 211.623 ms，
合计 3806.810 ms。计时范围为自举 GPU 阶段，排除离线准备、参考校验和
卷积等整网阶段。4.395 秒仍是上方 22 阶配置的单图在线 benchmark。

验收摘要、原始日志与 SHA-256 保存在
`v100-baseline-evidence/accuracy-sm70.jx24X2/`。

## 2026-11-17：Trident 优化核对与 GPU BV 组件

Trident `1ce3d5c4` 的 CPU 实现包含 DFT 共享分解、PS 多项式幂次并行、
求值计划与系数明文缓存。其精度修复版将 exp carrier 从 C31 截到 C23，
谐波激活保持 D31。该分支报告同密钥组件均值 8.158744→7.329639 秒，
十图纯推理历史均值 170.2733→159.5777 秒。这两组均为 Trident CPU 数据，
使用 dnum=4 的 881-bit 参数。P1 版的实际分解数是 ceil(Q/P)，Q19/Q20
分别为 19/20。`resnet20_p1` 的验证使用 Q20、最高 39 位 Q 和 51 位 P，
同样超过当前 GPU 字长。该版单图最大 logits 误差为 0.055064567，
超过 0.01 门槛。

迁移提交 `39d34096` 在 GPU 参数上传时用 P1 建立逐 Q 分解表，并在 ModDown
把 P 残基居中，与 CPU BV 的舍入一致。原来的低位宽 P1 对照在上传后报
HYBRID 参数形状错误。补表后普通旋转可执行但有误差，补齐居中后
D15/Q19 和 D31/Q20 在最高层与 L2 的普通旋转、共轭、平方与重线性化
均与 CPU 最大差异为 0。维持之前的 29 位 Q、30 位 P 控制参数。
P2/HYBRID 对照通过，CTest 3/3 通过，Python 39 项中 31 通过、8 跳过。

P1 共享分解旋转试验在两档最高层的最大偏差分别为 0.0727972 和 0.12261，
正式提交中的探针验证该路径会明确拒绝执行。该提交的两档原参数探针仍因
45 位 Q 超过 32 位 GPU 字长而在参数上传退出。实现宽模数 RNS/NTT 后，
可先用普通 BV 跑原参数对照，再分别验收共享分解和 C23 截断。当前无融合整网
GPU 结果。日志见 `v100-baseline-evidence/bv-final-sm70.N8ecqo/`
和 `v100-baseline-evidence/bv-hoist-sm70.x2Ythi/`。

## V100 宽模数 NTT 和 ModRaise

迁移提交 `58dcb8cb` 新增独立的 `GpuWideNtt`，用 64 位残基及预计算的 Shoup 乘法常量
执行批量负循环 NTT。CPU 与 GPU 使用相同的根表和残基排列。单 q0 ModRaise
的两个密文分量各自在 GPU 上完成逆 NTT、居中扩基、全 Q 正 NTT，
再下载结果与 CPU `Bootstrapper::mod_raise` 逐字比较。

| 原始参数 | Q/P 正逆 NTT 残基数 | q0 → Q 的两分量残基数 | 结果 |
|---|---:|---:|---|
| D15，N=32768、Q19、P=1×51 bit | 655360 | 1245184 | 精确一致 |
| D31，N=32768、Q20、P=1×51 bit | 688128 | 1310720 | 精确一致 |

在 V100 工作区根目录，用已经按上文配置的 `build-migration-sm70` 复现：

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DSPDLOG_FMT_EXTERNAL_HO=OFF
cmake --build build-migration-sm70 --target functional_wide_ntt_probe --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
build-migration-sm70/bin/functional_wide_ntt_probe 15
build-migration-sm70/bin/functional_wide_ntt_probe 31
```

现有 32 位 `GpuEvaluator` 仍使用原有路径。此次重新构建后，3 项 CTest 和两档低位宽 BV 对照均通过。接下来需要在宽模数下实现
BV 的逐 Q 扩基、密钥乘加和居中 ModDown，再串起 CtS、载波及谐波求值。
本阶段的逐字比对尚未产生融合整网 GPU 性能数据。日志目录为
`v100-baseline-evidence/wide-modraise-sm70.g8J7VS/`。


## V100 原参数宽模数 BV 密钥切换

宽模数 BV 在 GPU 上完成逐 Q 扩基、正 NTT、评估密钥乘加、P 逆 NTT、
居中 ModDown 及密文累加。旋转和共轭的 NTT 置换，以及平方的三分量乘法
也在 GPU 执行。探针在主机侧生成评估密钥、打包当前 Q 前缀和 P，并计算
CPU 正确性参考。每项比较两个输出密文分量的全部 RNS 残基。

| 原始参数 | 最高层与 L2 | L0 |
|---|---|---|
| D15，N=32768、Q19、P=1×51 bit | 旋转 +1、-3，共轭，平方及重线性化，全部逐字一致 | 两次旋转和共轭逐字一致 |
| D31，N=32768、Q20、P=1×51 bit | 旋转 +1、-3，共轭，平方及重线性化，全部逐字一致 | 两次旋转和共轭逐字一致 |

L0 的单个 45 位 q0 无法容纳应用 scale 的平方，保留该层用于旋转和共轭验证。
运行同一构建目录中的探针：

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DSPDLOG_FMT_EXTERNAL_HO=OFF
cmake --build build-migration-sm70 --target functional_wide_bv_probe --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
build-migration-sm70/bin/functional_wide_bv_probe 15
build-migration-sm70/bin/functional_wide_bv_probe 31
```

日志目录为 `v100-baseline-evidence/wide-bv-final-sm70.jOU76G/`。
该路径是独立宽模数组件，后续仍需接入 StC/CtS 线性变换和 ResNet20 融合链。
迁移提交为 dd93f6ea。宽模数 NTT、ModRaise、低位宽 BV 回归及 3 项 CTest 全部通过。

## V100 宽模数 StC 与 CtS

探针使用 Trident 固定提交 `1ce3d5c4` 的 D15/D31 相位系数、两层 DFT、
载波域 K=12 以及 CtS BSGS 比例 2。在 StC 的最后一层乘入 Trident 的
增益，ModRaise 后将 scale 的解释从 2^40 改为 q0。输入是合成实数槽位，
使用相同密文、旋转键及明文对角矩阵对比 CPU 和 GPU。

GPU 用宽模数 BV 执行 BSGS 的大小步旋转、明文乘加、逐层 rescale、ModRaise
和 CtS 共轭相加。CPU 负责初始加密、矩阵编码、生成及打包评估密钥和
正确性参考。各阶段逐字比较两个密文分量，阶段间延续 GPU 输出。
组件探针逐次上传所需密钥和对角明文，当前实验口径为正确性。

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DSPDLOG_FMT_EXTERNAL_HO=OFF
cmake --build build-migration-sm70 --target functional_wide_dft_probe --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
for degree in 15 31; do
  build-migration-sm70/bin/functional_wide_dft_probe "$degree" \
    "trident-functional/resnet20/functional_params/relu_d${degree}.txt"
done
```

日志与二进制和系数 SHA-256 在 `v100-baseline-evidence/wide-dft-sm70.FFXbJL/`。

迁移提交为 ecd1d2ee。V100 两档各 6 个检查点全部逐字一致。
D15 的层级为 L2→L1→L0→L18→L17→L16，D31 为
L2→L1→L0→L19→L18→L17，最后的共轭相加维持 CtS 输出层级。
宽模数 NTT/ModRaise、两档 BV 正确性、低位宽 BV 与 3 项 CTest 回归通过。

## V100 完整融合自举组件

在 StC、ModRaise、CtS 和实部投影后，GPU 执行 31 阶复数 Chebyshev 载波、
三次平方、D15 或 D31 的复数谐波多项式，以及最终的共轭相加。
多项式采用按二次幂拆分的求值顺序，复用密文幂次，使用普通宽模数 BV
重线性化。峰值临时量、键和常量留给整网接入后的 benchmark 测量。

在同一个 GPU 缓冲区链上延续各阶段输出。主机负责密钥生成、系数编码、
调度和对照。检查点解密的是下载的副本，后续计算继续使用 GPU 密文。
线性阶段与 CPU 密文逐字比较，复数载波和每次平方与独立明文运算比较，
最终全部 16384 槽与 Trident 冻结拟合比较。场景采用打乱的 [-1,1] 网格，
包含 0 和 ±1e-8。两套配置在两张 V100 上并行进行正确性验证。

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DSPDLOG_FMT_EXTERNAL_HO=OFF
cmake --build build-migration-sm70 --target functional_wide_dft_probe --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
coeff=trident-functional/resnet20/functional_params
for degree in 15 31; do
  build-migration-sm70/bin/functional_wide_dft_probe "$degree" \
    "$coeff/relu_d${degree}.txt" "$coeff/carrier_k12_d31.txt"
done
```

日志及输入 SHA-256 在 `v100-baseline-evidence/wide-fused-sm70.C4ekCj/`。

迁移提交为 d0e0cc57。两档原参数 N=32768、P=1×51 bit 都通过。
线性阶段各 6 个检查点逐字一致，最终均为 L4、scale=2^40。
D15/Q19 相对冻结拟合的最大槽误差为 3.48168e-7（门槛 3.29403e-4），
D31/Q20 为 4.11494e-7（门槛 1.52081e-4）。
冻结拟合相对精确 ReLU 的最大偏差分别是 0.00329403 和 0.00152081。
宽模数 NTT/ModRaise、BV、低位宽 BV 及 3 项 CTest 回归通过。
本次完成一次融合激活验证。19 个激活点的整网调度、最终 logits
及同口径 V100/H100 性能对照仍待验证。

## V100 ResNet20 全网融合验证

独立应用 `resnet20_wide_functional` 复用 64 位 GPU BV、StC/CtS、
ModRaise 及载波和谐波求值。打包以 16 个固定的 32×32 平面存储特征，
降采样后用每个平面的列相位容纳更多通道。卷积和 BN 折叠后的缩放通过
稀疏对角矩阵执行。两个跨阶段残差用 Option-A，下采样并在通道维居中填零。
分类头在 GPU 完成全局平均和 FC，前向只有初始一次输入加密和最后一次
logits 解密。

明文参考独立使用 CHW 卷积和 Trident 冻结拟合。验收要求正好 19 个
融合激活，最终 logits 最大差异不超过 0.01，预测与明文参考一致。
两档图像 0 在两张 V100 上并行进行正确性验证。该程序仍按需编码对角明文
并上传旋转键，进程耗时包含这些准备与传输，后续需要按基线计时口径
补充常驻方案和在线 benchmark。

```bash
export PATH="$HOME/cmake-new/bin:/usr/local/cuda/bin:$PATH"
cmake -S migration -B build-migration-sm70 -DSPDLOG_FMT_EXTERNAL_HO=OFF
cmake --build build-migration-sm70 --target resnet20_wide_functional --parallel 4
export CUDA_VISIBLE_DEVICES=GPU-0768eaf5-15f4-6cca-f263-141df9bae9c6
coeff=trident-functional/resnet20/functional_params
build-migration-sm70/bin/resnet20_wide_functional 31 0 "$coeff"
build-migration-sm70/bin/resnet20_wide_functional 15 0 "$coeff"
```

日志与二进制和系数 SHA-256 位于 `v100-baseline-evidence/wide-network-sm70.Gzt5AE/`。

迁移提交 `852d7e39`。V100 图像 0 两档均完成全部 19 次融合，
每次回到 L4、scale 2^40。最终 logits 位于 L2，GPU、冻结拟合明文
与真实标签均为类别 3。

| 配置 | 最大 logit 绝对误差 | 进程总用时 | 进程峰值主机 RSS |
|---|---:|---:|---:|
| D15 / Q19 | 4.2742429e-5 | 260.52 s | 17.49 GiB |
| D31 / Q20 | 8.315962315e-5 | 279.75 s | 18.87 GiB |

总用时和 RSS 来自 GNU time，包含两档并行运行期间的密钥准备、
模型与对角明文编码、上传和最后的校验。该测量未达到此前 22 阶基线的
整网在线计时口径。D31 单次融合回归与 3 项 CTest 也通过。

### H100 宽模数应用复测

以下 `sm_90` 编译已在现有主机验证，二进制包含 `sm_90` CUDA 代码。
H100 实机运行和性能数据仍待测。将迁移与 Trident checkout 放在同一目录下：

```bash
cmake -S migration -B build-migration-sm90 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPOSEIDON_BUILD_EXAMPLES=OFF -DPOSEIDON_BUILD_GPU=ON \
  -DPOSEIDON_BUILD_RESNET20_S2C_FIRST=ON -DPOSEIDON_USE_ZLIB=OFF \
  -DPOSEIDON_USE_ZSTD=OFF -DPOSEIDON_USE_MSGSL=OFF -DSPDLOG_FMT_EXTERNAL_HO=OFF \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="$PWD/migration/third_party/spdlog"
cmake --build build-migration-sm90 --target resnet20_wide_functional --parallel 4
export CUDA_VISIBLE_DEVICES=0
coeff=trident-functional/resnet20/functional_params
build-migration-sm90/bin/resnet20_wide_functional 31 0 "$coeff"
build-migration-sm90/bin/resnet20_wide_functional 15 0 "$coeff"
```

### V100 常驻重放与稳定明文索引

在迁移目录旁执行（根据实际空闲 GPU 选择设备）：

```bash
export CUDA_VISIBLE_DEVICES=2
coeff=trident-functional/resnet20/functional_params
build-migration-sm70/bin/resnet20_wide_functional 15 0 "$coeff" --benchmark
build-migration-sm70/bin/resnet20_wide_functional 31 0 "$coeff" --benchmark
```

每次运行预热一次、再计时一次。CUPTI 严格拒绝在线主机传输；最终才解密校验。
2026-11-17 图像 0 单次实测：D15 墙钟 11377.919 ms、GPU 活动并集
7762.313 ms、最大误差 4.968e-5；D31 墙钟 12652.410 ms、GPU 活动
并集 8799.739 ms、最大误差 4.878e-5。两者分类均为 3，19 个激活，
在线主机传输为 0。日志在 `v100-baseline-evidence/wide-index-real-sm70.O8t6mC/`。
计时窗口仍包含部分 CPU 矩阵准备和调度，不应与冻结基线直接作为完全同口径
比较；单次结果也不构成统计性能结论。H100 运行时可将上述命令中的
`build-migration-sm70` 替换为 `build-migration-sm90`，根据实际设备选择 GPU。
