# ResNet20 同一实际混合链的 Q/scale 联合复核（2026-09-11）

> **2026-09-13 完整网络更新：**图像 0 的 9 个 BasicBlock、18 次 S2C-first
> degree-59/DA2 自举、18 次卷积、9 次残差以及最终 average-pool/FC 已在同一
> 连续 GPU 密文链上执行通过，中间重加密为 0。明文与密态预测均为 3，logits
> 最大误差 0.010449，全网边界最大误差 3.096e-4，未 OOM。详见
> [完整网络正确性报告](FULL_NETWORK_STATUS.md)。该结果仍是未获安全批准的
> Q50/P25 准确性诊断，不是生产性能/安全验收。

> **最新完整block进展：**图像0的首个 `layer1_0` 已连续执行完成：3次自举
> （含额外诊断stem前缀）、2次真实卷积、1次真实残差相加，正常加密仅一次。
> 8个应用边界均低于1e-4；最终Q9/scale40，最大误差6.29e-5，未OOM。
> 48项回归/证据检查通过。仍非完整网络、生产逐层P路径或安全性验收。
> 详见[首个BasicBlock报告](BOOTSTRAP_BLOCK_STATUS.md)。

> **最新单层进展：**真实图像0的连续“自举 → scale40 → ReLU → 原应用ConvBN”
> 已在非零随机密钥下执行，准确到达Q6/scale40；卷积局部误差1.79e-6，末端
> 对原应用明文多项式+卷积误差1.76e-5。单卡未OOM，42项回归通过。仅一个
> 真实卷积层，非完整网络或逐层P优化运行时验收。详见
> [连续自举/ReLU/Conv报告](BOOTSTRAP_RELU_CONV_STATUS.md)。

> **后续连续子链测试：**已实现并运行同一密文的“修复周期后的自举 → scale40
> 系数补偿 → 原应用 ReLU”，没有中途重新加密。两轮独立非零密钥的接口和局部算术检查通过，
> 但包含自举近似误差的末端1e-4检查未通过。详见
> [连续自举/ReLU报告](BOOTSTRAP_RELU_STATUS.md)。这不是整网验收。

> **2026-09-12 后续：新增恢复历史定标的独立自举入口。**
> Q34/P9 两组输入、Q50/P25 历史输入在非零随机密钥下均达到旧测试的 0.002
> 容差，但未达到 1e-4；没有出口 scale40 折叠，也没有执行完整网络。详见
> [恢复基线实测报告](BOOTSTRAP_BASELINE_STATUS.md)。
> 这是独立配置，不会让下面保留的旧 Q50 witness 自动变成正确配置。

> **2026-09-12 正确性更新：当前 Q50 witness 的首次自举已实测失败。**
> 两套非零随机密钥的最大恢复误差分别约 6.97、6.92。已定位到 ModRaise 后
> C2S/EvalMod 的整数周期归一化不一致。详见
> [自举精度与失败定位报告](BOOTSTRAP_ACCURACY_STATUS.md)。
> 下文的静态账本仅说明层数和 scale 可衔接，不能再用来断言该自举方案正确。

2026-09-12 新增独立实密文测试：22-Q ReLU 的三轮末端精度检查通过，
中间统一阈值检查未通过；误差传播分析与适用边界见
[ReLU 精度报告](RELU_ACCURACY_STATUS.md)。自举失败定位和后续恢复基线见上方两份报告，整网未运行。

> 后续安全性检查：本文件只证明静态 Q/scale 可行性。Q50/P25（dnum2）和
> Q50/P13（dnum4）已在实际 Q×P、SparseTernary(96,96)、CBD(21) 模型下被
> 128-bit 安全预检拒绝，不能直接部署。详见
> [应用安全验证状态](/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/SECURITY_VALIDATION_STATUS.md)。

## 结论与范围

**Q50 通过联合静态复核**：同一组实际素数上，S2C 前置的 degree-59 / DA2 自举、
优化调度的 ReLU(15,15,27)、ConvBN 和下一次 S2C 可以闭环。
这不是把不同模数链上分别得到的最小消耗相加。

这里 `Qk` 表示还剩 k 个物理 Q 素数，不是从零开始的 level 索引。
N=65536，logSlots=15，应用 scale=2^40；所有 Q/P 素数都小于 2^32，
没有切换到 64-bit GPU 算术，也没有使用 Cheddar 式有理 rescale。

这是**新增的规划/验证工具和配置记录**，不是已经修改并执行的 ResNet20 应用。
本节静态账本生成没有密钥生成、加密、GPU 执行或零秘密密钥试验。元数据上下文使用
`sec_level_type::none`；该记录没有认证安全等级、密钥切换噪声或最终解密精度。

## 同链完整账本

下表的 scale 数值均为 log2(scale)，使用实际素数的 log2 计算，未将其取整为位宽。

| 阶段 | Q 变化 | 删除 Q 数 | 输出 log2(scale) |
|---|---|---:|---:|
| S2C，三次矩阵乘 | 6 → 5 → 4 → 2 | 1+1+2=4 | 53.010295 → 66.021522 → 47.045351 |
| 升模前整数准备 | 2 → 2 | 0 | 58.973499 |
| ModRaise | 2 → 50 | 0（恢复 Q 基） | C2S 采用源码校准后的输入 44.973499 |
| C2S，三次矩阵乘 | 50 → 49 → 48 → 46 | 1+1+2=4 | 60.021304 → 73.025934 → 54.035946 |
| EvalMod 多项式部分 | 46 → 34 | 12 | 45 |
| 两次 double-angle | 34 → 33 → 31 | 1+2=3 | 58.007504 → 53.030946 |
| 与上述系数联合设计的出口定标 | 31 → 31 | 0 | 40 |
| ReLU 第一个 P15 | 31 → 25 | 6 | 40.1 |
| ReLU 第二个 P15 | 25 → 19 | 6 | 44 |
| ReLU P27 | 19 → 12 | 7 | 89.871122 |
| ReLU 最后乘原始 x | 12 → 9 | 3 | 40 |
| ConvBN 权重/支撑/选择掩码 | 9 → 8 → 7 → 6 | 3 | 最终回到 40 |
| 下一次 S2C 和整数准备 | 6 → 2 | 4 | 与本表起点完全相同 |

因此，升模后的链长度分配为：

```
50 = C2S(4) + EvalMod含DA(15) + ReLU(22) + ConvBN(3) + S2C(4) + 保留Q(2)
```

自举输出 Q31，ReLU 后 Q9，卷积后 Q6，S2C 后 Q2。
S2C 的 4 个 Q 位于下一次升模之前，不能再次从 Q50 的自举输出里重复扣除。
源码请求 degree=59；生成的余弦多项式有效最高次数为 58（本来的奇数最高项为零），
没有改用低阶拟合或删去非零项。

## 出口定标如何不额外消耗 Q

未经补偿的自举出口 scale 为 S=2^53.03094629532368，不能直接把标签写成 2^40。
本记录使用原多项式系数和两次 double-angle 常数的联合缩放：

- α=2^40/S；r=α^(1/4)，log2(r)=-3.25773657383092。
- 原多项式的**所有系数**乘 r，不重新拟合、不截断。
- 两次 double-angle 的常数分别乘 r^2、r^4。
- 若原递推是 y1=2*y0^2-c1、y2=2*y1^2-c2，则修改后的结果为 r^4*y2=α*y2。
- 计算结束时将 scale 从 S 改成 α*S=2^40，理想算术下表示的函数值不变。

因此这不是免费更改标签，也不是额外做一次任意实数乘法：常数补偿已融入原有
多项式系数与 DA 常数的编码。重新调用现有 EvalMod 上传器的元数据规划模式后，
Q/scale 轨迹和非零项次数集合保持不变。独立复核器还检查系数比例、DA 常数比例
及递推恒等式。实际编码/舍入误差仍应在后续正常秘密密钥试验中验证。

升模前准备也没有隐藏的额外 rescale：S2C 出口乘整数 **3897**，同时更新 scale，
其真实 log2(scale)=58.97349886989023，并非强行等于目标值
58.97332848776823。之后按当前源码的校准公式传给 C2S。

## 实际模数链

完整十进制素数和所有节点见 `q50.json`，`q_bottom_first` 从最低保留素数向上排列。

- q[0..8]：9 个 32-bit 素数，供保留 Q2、S2C 和 ConvBN 使用。
- q[9..30]：22 个 30-bit 素数，使用此前 22-Q ReLU 的实际素数序列。
- q[31..49]：19 个混合 30/31/32-bit 素数，供 C2S 和 EvalMod 使用。
  从低到高的位宽为 `[31,32,32,30,31,32,31,32,32,31,31,31,32,32,31,32,32,32,30]`。
- P：25 个与 Q 不重复的 32-bit 素数，仅用于建立此元数据规划上下文；
  这不是已经经过密钥切换正确性/安全性验证的最终 P/dnum 选择。

复核器验证 Q/P 两两不重复、素性、每个素数都满足 q≡1 mod 131072，以及每次
rescale 确实删除相应位置的实际素数。两条旧见证中冲突的素数已在构造时替换为
同位宽的未使用素数，替换后的整链重新经过源码规划。

## 验证结果与负例

`verification.json` 给出 `PASS_SCALE_LEDGER`，独立重放 **168 个状态**，检查：

- 密文乘法 scale 相加、明文乘法 scale 相加、按实际模数乘积 rescale。
- EvalMod 和 ReLU 各分支的 Q/scale 对齐，而非只检查最深分支层数。
- 中间 scale 不超过当前 Q 的容量，以及整个循环的接口一致性。
- EvalMod 最低非平凡系数编码 logscale：47.048597。
- 计入上述 r 缩放后的 EvalMod 最低有效编码 scale 指标：43.790861。
- ReLU 最低非平凡明文编码 logscale：40.030797。
- 最大中间 logscale：179.765042；最小 log2(Q)-log2(scale)：4.999830。

最后两种 scale/容量指标**不是明文有效精度或噪声预算证明**；容量还需容纳真实
消息幅度和误差。它们用于严格排除静态 Q/scale 不可行配置，不冒充端到端精度测试。

`test_joint.py` 的 12 个测试全部通过，覆盖正例，以及伪造出口标签、未补偿 DA 常数、
错改多项式系数、接口 scale 错误、缺失/重复素数、分支层数错误、scale 溢出、
漏计卷积 rescale、错误准备整数、错误 DA rescale 等负例。
此前 `resnet20_scale_chain_20260911/test_metadata.py` 的 6 个测试也通过。

`q49_trial.json` 保存同一构造族的 Q49 尝试：自举出 Q30、ReLU 后 Q8、卷积后 Q5，
S2C 后不足 Q2，失败。因此不能直接把这套已复核调度减为 Q49。
这不是穷举所有混合素数排列、scale 和多项式调度后的全局最小性证明。

## 复现

仓库根目录下，只复核现有见证（纯 Python、无 GPU/密钥/上下文构造）：

```bash
python3 -B Doc/analysis/resnet20_joint_chain_20260911/verify_joint.py \
  Doc/analysis/resnet20_joint_chain_20260911/q50.json
python3 -B Doc/analysis/resnet20_joint_chain_20260911/test_joint.py
python3 -B Doc/analysis/resnet20_scale_chain_20260911/test_metadata.py
```

从已有 Poseidon 构建产物重新导出见证：

```bash
audit_dir=$(mktemp -d /tmp/resnet20-joint-check.XXXXXX)
bash Doc/analysis/resnet20_joint_chain_20260911/build_bridge.sh "$audit_dir/bootstrap_bridge"
python3 -B Doc/analysis/resnet20_joint_chain_20260911/joint.py \
  --bridge "$audit_dir/bootstrap_bridge" --output "$audit_dir/q50.json" \
  --full-q 50 --prefer-relu --fold-output
python3 -B Doc/analysis/resnet20_joint_chain_20260911/verify_joint.py "$audit_dir/q50.json"
```

`build_bridge.sh` 只编译一个 C++ 文件并链接本机已有的 Poseidon/GPU 对象，
不重新构建 CUDA 工程；需先有脚本中指定的构建产物。虽然链接了 CUDA 运行库，
程序仅执行 CPU 元数据规划，不初始化 GPU、不编码明文载荷、不上传数据。

## 新增工具与源码依据

- `bootstrap_bridge.cpp`：调用现有 `GpuUploader::upload_eval_mod_high_precision`
  的 `metadata_only=true` 模式，导出完整 EvalMod DAG，并重新规划缩放后的系数。
- `joint.py`：构造同一实际素数链，连接 S2C/准备/C2S/EvalMod/ReLU/ConvBN。
- `verify_joint.py`：独立重放和代数补偿检查。
- `test_joint.py`：正例和损坏账本的拒绝测试。
- `build_bridge.sh`：本机低内存编译入口。
- `q50.json`、`verification.json`、`q49_trial.json`：成功见证、重放结果和失败试验。

EvalMod 源码：`src/poseidon/gpu/gpu_uploader.cpp` 的高精度上传器；动态 rescale
规则：`src/poseidon/gpu/gpu_scale_planner.h`。ReLU 调度器复用相邻目录
`resnet20_scale_chain_20260911/search.py`，再用独立复核器检查实际输入链。
卷积 3 次 rescale 依据应用目录
`/home/liufuyao/Work/poseidon_gpu_other/resnet20-9.3/benchmark/resnet20_gpu/gpu_multiplexed_tensor.cpp`
的权重、支撑掩码和选择掩码路线；没有修改该应用文件。
