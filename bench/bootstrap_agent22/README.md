# Agent 22 阶 S2C-first 自举

| 调用名 | 用途 |
| --- | --- |
| `s2c-first-degree22-da3-q43-dnum4` | 双支 EvalMod 后投影到实数，简称 `dnum4` |
| `s2c-first-degree22-da3-q43-dnum4-real-only` | 单卡实数应用专用，采用 Agent ResNet20 的单实部 EvalMod |

两条路径使用完整 22 阶系数、3 次倍角、`K=12` 和 Double-Hoist。模数链为 Q6 → Q2 → Q43 → Q39 → Q24，输入和输出 scale 均为 `2^40`。峰值 Q43 使用 P11、dnum=4。两条入口共用已验证的 C2S 三分解 baby-step 和 giant-step 内核。 [Agent 自举计划](https://github.com/SL640/FHE_GPU_Agent/blob/be4bc9f5f5993dd10aa11fa9c48ec7e8455a0f6c/gpu_api/src/poseidon/gpu/gpu_bootstrap_profile.cpp#L948)

单实部入口使用上游提交 `be4bc9f` 的 `BootRealOnly` 分支。S2C、模升和 C2S 仍完整执行，C2S 末尾的融合拆分可能同时写出实、虚两份中间密文。只有实支进入完整 EvalMod，末端保留乘 16 与共轭求和。该变体适用于自举输出需要实数投影的应用，本地独立入口使用实值槽输入。 [上游分支选择](https://github.com/SL640/FHE_GPU_Agent/blob/be4bc9f5f5993dd10aa11fa9c48ec7e8455a0f6c/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L7418-L7428) · [EvalMod 执行](https://github.com/SL640/FHE_GPU_Agent/blob/be4bc9f5f5993dd10aa11fa9c48ec7e8455a0f6c/gpu_api/src/poseidon/gpu/gpu_evaluator.cpp#L7537-L7555)

## 单实部首版与双分支的同卡对照

2026-11-25，单张 V100-SXM2-32GB。两条路径共用一组经检查的随机主密钥和临时密钥，以及同一份 Q6 输入密文。输入是 32768 槽的实值正弦与余弦叠加，幅度上界为 0.5。先测单实部，再测双支，各热身 3 次并计时 18 次。在线延迟取墙钟中位数，CUPTI 在另外 18 次连续运行中采样。共用准备时间 309.361 s。

| 每次完整自举 | 在线中位数 | CUPTI 活动均值 | 最大复数绝对误差 | RMS |
| --- | ---: | ---: | ---: | ---: |
| 双支 | 59.958 ms | 59.884 ms | 8.64148×10⁻⁴ | 2.77601×10⁻⁴ |
| 单实部 | 48.834 ms | 48.686 ms | 8.63358×10⁻⁴ | 2.77595×10⁻⁴ |

单实部的在线中位数缩短 **11.124 ms（18.55%）**。两者输出均为 Q24、scale `2^40`，各自通过 `1e-3` 门槛，两份解码输出逐槽最大复数差为 **7.652×10⁻⁶**。每个采样窗口的主机与设备间拷贝均为 0 B。主秘密与临时秘密实测重量分别为 32768、32，两者的三元分布与全部 65 个 RNS 分量均通过检查。实际评估密钥最大 `log₂(PQ) = 1775.35322 < 1782`。本次测量覆盖独立单次自举及其重放。 [完整基准输出](results/v100-real-only-20261125.txt)

## 历史 Q43/P22 对照

2026-11-23，Tesla V100-SXM2-32GB，CUDA 12.2，驱动 535.288.01，Release / sm70。每组独立生成秘密并准备材料，热身 3 次后测量 18 次。

| 分层 P 阈值 | GPU | 在线墙钟中位数 | CUDA event 中位数 | CUPTI 活动均值 | 最大绝对误差 | RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 9，上游源码默认 | GPU 1 | 56.643 ms | 56.634 ms | 56.550 ms | 8.606×10⁻⁴ | 2.776×10⁻⁴ |
| 25，Step H 报告 | GPU 2 | 56.883 ms | 56.867 ms | 56.760 ms | 8.629×10⁻⁴ | 2.776×10⁻⁴ |

两组的 CUPTI 窗口均记录 8,928 项 GPU 活动，H2D/D2H 为 0 B。一次性准备耗时分别为 324.11 s、324.57 s，RMM 显存池预留均为 17,664 MiB。阈值 25 在这组素数上只把 Q24 实数投影从 P12 改为 P22，仍为 `dnum=2`。两个 V100 样本间约 0.21 ms 的活动差包含卡间波动。 [阈值 9 原始日志](results/v100-dnum1-9.txt) · [阈值 25 原始日志](results/v100-dnum1-25.txt)

上游 Step H 将 ResNet20 里的 18 次自举单独归类，报告 A800 为 31.85 ms/次、RTX 4090 为 26.19 ms/次。V100 的 56.760 ms 是独立自举，分别为两条记录的约 1.78 倍和 2.17 倍。复测使用了相同规模的 Q/P 链和两支 EvalMod，设备和调用场景有差异，同卡复现仍待验证。 [上游 Step H CUPTI 记录](https://github.com/SL640/FHE_GPU_Agent/blob/be4bc9f5f5993dd10aa11fa9c48ec7e8455a0f6c/gpu_api/Doc/analysis/resnet20_a800_optimization_summary_2.md#L55)

## 峰值 dnum=4 的分层 P 配置

两条 dnum4 路径沿用同一 Q 链、scale 和拟合系数。每个密钥切换层使用满足 `log₂(PQ) < 1782 bit` 的最小 `dnum`，按 `P 枚数 = ⌈Q 枚数 / dnum⌉` 取 P 前缀。Q43 的 C2S 旋转和临时密钥切回主密钥均在 Q43/P11 生成。评估密钥按各组的实际 Q/P 生成并上传，测试逐组核算存储模数。输入在 Q6 对称加密，避免生成全局 Q43/P22 公钥。

| 实际密钥切换层 | `dnum` | 代表性 Q/P |
| --- | ---: | --- |
| Q6、Q5、Q4，前置 S2C | 1 | Q6/P6 |
| Q43 | 4 | Q43/P11 |
| Q42、Q41、Q39 | 3 | Q42/P14、Q39/P13 |
| Q37 到 Q24 的后续密钥切换 | 2 | Q37/P19、Q24/P12 |

Q43/P11 的 `log₂(PQ)` 为 **1708.40 bit**。逐组核查 GPU 实际保存的密钥后，最大值是 Q37/P19 重线性化键的 **1775.35 bit**。主密钥切到临时密钥使用 Q2/P2（127.97 bit），全局评估密钥后备为空。 [分层密钥核验日志](results/v100-layered-dnum4.txt)

2026-11-23 的双分支基线在同一张 V100 GPU 1 上，3 次热身后测量 18 次。在线墙钟中位数为 **61.250 ms**，CUDA event 中位数为 **61.240 ms**，CUPTI 活动均值为 **61.173 ms/次**，比峰值 `dnum=2` 基线增加 **8.2%**。CUPTI 窗口 H2D/D2H 为 0 B，一次性准备耗时 308.35 s。输出为 Q24 / `2^40`，32768 槽最大绝对误差 **8.618×10⁻⁴**，RMS **2.776×10⁻⁴**。 [V100 原始日志](results/v100-layered-dnum4.txt)

1782 bit 是本次采用的模数预算。确认 128 位安全性需要对实际秘密分布、噪声和各评估密钥做安全估算。 [Cerium 参数口径](https://arxiv.org/html/2512.11269v2#S6.SS1)

## 未采用的优化

### 最后一次倍角改用 Q26/P26 单分解

2026-11-23 在单张 V100（GPU 1）上做了配对实验。实验在最后一次倍角把 Q26/P13、`dnum=2` 换成 Q26/P26、`dnum=1`。P26 由原 P22 加已经离开活动 Q 的 q27–q30 组成。两组共用输入密文、拟合系数、其余评估密钥和 Q26 → Q24 的 rescale 计划，每组热身 3 次并测量 36 次完整自举。

| 最后一次倍角 | 在线墙钟中位数 | CUPTI 活动均值 | 32768 槽最大绝对误差 |
| --- | ---: | ---: | ---: |
| Q26/P13，`dnum=2` | 61.261 ms | 61.192 ms | 8.622×10⁻⁴ |
| Q26/P26，`dnum=1` | 61.289 ms | 61.209 ms | 8.622×10⁻⁴ |

单分解将每次自举的 GPU 活动数从 537 减到 534，在线墙钟中位数却高了 0.029 ms。差距很小，这次实验的结论是延迟持平，未观察到加速。已删除实验实现和 `q26-dnum1` 入口，Q26 重线性化继续用 P13、`dnum=2`。

Q26/P26 密钥的实际 `log₂(PQ)` 为 **1651.326 bit**。两组均输出 Q24、scale `2^40`，在线 CUPTI 窗口 H2D/D2H 为 0 B。两组解码结果逐槽最大差为 **4.317×10⁻⁶**，RMS 差为 **6.094×10⁻⁷**。 [配对实验原始日志](results/v100-q26-dnum1-ab.txt)

## 运行

在仓库根目录运行。依赖 CMake 3.26.4 或更高版本、CUDA Toolkit（含 CUPTI）、GMP 和仓库自带的 `third_party`。

```bash
# 本机的 CMake 3.27 和 V100 GPU 1。
export CMAKE=/opt/conda/envs/apollo/bin/cmake
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 CUDA_ARCHITECTURES=70

# 保留的两条 Q43/dnum4 路径。3 次热身后测量 18 次完整自举。
./bench/bootstrap_agent22/run.sh 18 s2c-first-degree22-da3-q43-dnum4
./bench/bootstrap_agent22/run.sh 18 s2c-first-degree22-da3-q43-dnum4-real-only

# 用同一组密钥和同一份输入密文测量两条路径，分别校验并逐槽比较。
./bench/bootstrap_agent22/run.sh 18 s2c-first-degree22-da3-q43-dnum4-real-only --compare

# 历史对照，上游 Q43/P22 的分层 P 默认阈值为 9。
./bench/bootstrap_agent22/run.sh 18 9

# Step H 报告中的自举分层 P 阈值为 25，实际 P 仍受 P22 的上限约束。
./bench/bootstrap_agent22/run.sh 18 25

# 峰值 Q43/P11，低层随 Q 下降选择 dnum。
./bench/bootstrap_agent22/run.sh 18 dnum4
```

A800 用 `CUDA_ARCHITECTURES=80`，RTX 4090 用 `89`。环境变量 `CMAKE`、`NVCC`、`BUILD_DIR` 和 `BUILD_JOBS` 可选。传 `--build-only` 只编译。`dnum4` 和旧全名 `s2c-first-degree22-da3-full-coefficients-q43-layered-dnum4` 都指向新命名的双支配置。省略第二个参数时使用 `9`。`9` 和 `25` 均使用 Q43/P22，按层选 P 的阈值不同。

- `SETUP` 单列参数、评估密钥、DFT 明文和输入的生成与上传。
- `ONLINE_WALL` 从提交一次完整自举到 GPU 完成。`CUDA_EVENT` 在同一 CUDA stream 上包围完整自举。
- `CUPTI` 在独立采样中连续提交全部自举，最后同步一次，统计 GPU kernel、设备内拷贝和 memset 活动区间的并集。该窗口检查无 H2D/D2H。
- `EVALMOD_OUTPUTS` 检查热身后独立 workspace 中实际生成的 EvalMod 输出。单实部应为 `real=1 imag=0`，双支为 `real=1 imag=1`。
- `PRECISION` 在计时窗口外解密，比对全部 32768 槽与原始实数输入，复数绝对误差上限设为 `1e-3`。主秘密和自举临时秘密分别使用 OS 随机源派生的固定重量 32768 和 32，生成时检查实际重量、三元分布和完整 RNS 一致性。Q43/P22 的最高 `log₂(PQ)` 约 2060.23 bit，高于 1782 bit 的 128 位安全预算，供正确性与性能研究。 [参数对照](../../docs/gpu-bootstrap-migration.md)

## 源码来源

独立构建从 `vendor/upstream-be4bc9f5.tar.gz` 提取 Agent 提交的 CPU/GPU 库及选项头文件，复用本仓库的 CUDA 依赖和 CUPTI 计时器。归档包含 228 个上游文件，均核对 Git blob SHA。构建时应用补丁，接入离线构建和秘密检查。该目标单独链接 Agent 的库，和本地 59 阶、30 阶以及 ResNet20 应用各自构建。 [固定上游提交](https://github.com/SL640/FHE_GPU_Agent/tree/be4bc9f5f5993dd10aa11fa9c48ec7e8455a0f6c) · [本地补丁](vendor/local.patch)
