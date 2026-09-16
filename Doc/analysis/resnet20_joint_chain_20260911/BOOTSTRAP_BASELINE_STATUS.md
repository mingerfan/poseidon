# 恢复历史 S2C 前置 degree59 自举基线

日期：2026-09-12。这是独立 GPU 精度验证入口，不是完整 ResNet20 推理，也没有批准安全性。

后续：已新增使用同一C2S密文对照原生/补偿出口，并连续接入原ReLU的诊断。
该路径与本页的原生基线分开，结果及误差口径见
[连续自举/ReLU报告](BOOTSTRAP_RELU_STATUS.md)。下文保留原生基线的历史结果。

结论：Q34/P9 两组输入、Q50/P25 历史输入均达到旧版 0.002 容差，但均未达到
额外的 1e-4 检查。Q50 的第一次尝试触发 12 GiB 人工显存池限制；调整诊断
缓冲区生命周期和受限池容量后，第二次尝试完成。没有把未完成的尝试当成精度通过。

## 本阶段实现

`bootstrap_baseline_precision.cpp` 恢复历史成功日志中的定标组合，调用现有
Poseidon 的 double-hoist DFT 和高精度 EvalMod API，没有重写 GPU 算子：

- N=65536、32768 个槽；正常公钥加密，秘密密钥为 OS 随机的非零 h=192
  （96 个 +1、96 个 -1）。没有零秘密密钥试验，也没有中途重新加密。
- S2C 前置，输入 Q6 / scale=2^40，三次矩阵运算删除 1+1+2 个 Q，得到 Q2。
- 恢复历史 `q0_over_message_ratio()` 的取整策略：目标为
  `2^round(log2(Q0/32)) = 2^59`，而不是精确的 `Q0/32`。
  这里 Q0 是底部两个素数的乘积。
- S2C 后乘整数 3969 并同步更新 scale，实际 log2(scale)=58.99991050072101；
  准备过程不删除 Q、保持表示的数值。
- ModRaise 后使用固定的 C2S 逻辑 scale=2^45，保持原 C2S 矩阵系数。
  显式验证每个整数倍 Q0 对应 EvalMod 的一个完整周期。
- 原始 CosDiscrete、请求 degree=59（有效最高次数 58）、DA=2、K=25、
  message ratio=32、arcsine_degree=0；原多项式和 DA 常数不做出口系数折叠。
- 原生出口 log2(scale)=53.03094629532368。没有把它直接改标签为 40，
  也没有额外 rescale 或把该输出送入目前要求 scale=2^40 的 ReLU。

两个显式配置：`historical-q34` 精确复用历史 Q34/P9 素数；`q50-fixed45`
仅换成已保存的 Q50/P25 实际素数，保持上述定标组合。
它们对应的全局 dnum 分别为 4 和 2，不能把二者视为同参数性能比较。

当前独立测试源码存在后加的动态 C2S 校准分支；直接运行其当前默认值不等于
复现历史成功日志。因此本入口将固定 scale 写成显式契约，避免依赖环境默认值。
历史源码的 h=0 入口没有被调用。

## 已完成的实测

下表为独立非零随机密钥、一次完整自举的最大绝对误差。`historical` 输入
逐槽为 `((i%17+1)/32, (i%11+1)/64)`，与历史测试一致；`grid` 为
[-0.5,0.5] 实数网格，显式包含 0 和两端点。

| 配置 / 输入 | 出口 Q | 输出相对输入 | GPU 相对同一明文多项式 | 旧容差 0.002 | 严格目标 1e-4 |
|---|---:|---:|---:|---|---|
| Q34/P9 / historical，第 1 轮 | 15 | 0.001041236638 | 4.742677708e-6 | PASS | FAIL |
| Q34/P9 / grid，第 2 轮 | 15 | 0.000869037519 | 3.227200047e-6 | PASS | FAIL |
| Q50/P25 / historical，受限内存复测 | 31 | 0.001040035492 | 4.433802557e-6 | PASS | FAIL |

三轮完成试验的 EvalMod 输入均没有槽超出 [-1,1] 近似域。Q50 已完成一次历史
复数输入的精度测试，**尚未完成 Q50 grid 或多轮连续自举测试**。

对应日志：

- [Q34 历史输入](bootstrap_baseline_q34_historical_run1.log)
- [Q34 实数网格](bootstrap_baseline_q34_grid_run2.log)
- [Q50 第一次尝试：人工池限制中止](bootstrap_baseline_q50_historical_run1.log)
- [Q50 受限内存复测](bootstrap_baseline_q50_historical_run2.log)

严格目标是额外报告的精度检查，不是修改旧测试的容差。进程退出 0 表示达到
旧容差、GPU 对照误差不超过旧容差、且未超出近似域；不能理解为严格目标、
全网精度或安全性通过。日志明确输出 `full_network_tested=false`。

## 实际 Q/scale 路径

下表 scale 均为 log2(scale)，Q 是物理素数数量，不是 level 索引。

| 阶段 | Q34/P9 | Q50/P25 |
|---|---|---|
| 输入 | Q6 / 40 | Q6 / 40 |
| S2C | Q6→Q5→Q4→Q2 / 47.0453506537 | 同左 |
| 整数准备 | Q2 / 58.9999105007 | 同左 |
| ModRaise | Q2→Q34；随后 C2S 逻辑 scale=45 | Q2→Q50；随后 C2S 逻辑 scale=45 |
| C2S | Q34→Q33→Q32→Q30 / 54.0149941823 | Q50→Q49→Q48→Q46 / 54.0624466989 |
| 原始 EvalMod + DA2 | Q30→Q15 / 53.0309462953 | Q46→Q31 / 53.0309462953 |

即：升模后消耗 C2S(4)+EvalMod 含 DA(15)=19 个 Q。S2C 的 4 个 Q 在升模前
消耗。这里没有新增最终降 scale 的 rescale；因此**尚未闭合应用 scale40 接口**。

## 误差解释与边界

Q34/P9 历史输入这一轮：

| 对照 | 最大绝对误差 |
|---|---:|
| 理想正弦映射相对原始输入 | 0.001041373065 |
| degree59 明文多项式相对理想正弦映射 | 1.400716368e-7 |
| GPU 输出相对 degree59 明文多项式 | 4.742677708e-6 |

这说明此前约 6.97 的周期错误已在这个恢复基线中消除，但剩余约 1e-3 主要
来自原版不带 arcsine 恢复的正弦映射及其消息定标，不是 degree59 多项式没有
正确执行。整数准备取整后的小信号增益为 0.9999379657513102，本阶段没有再
加输出增益补偿。上述误差参照不同，不能直接将各行最大值相加。

历史成功日志 `bootstrap_slim_stc_evalmod_phase4_n65536.log` 的 source error
本来就是 0.00103294，容差是 0.002。因此本次结果与旧基线数量级一致，
但不能将“59 阶高精度”这个名称当作已达到 1e-4 或完整网络精度的证据。

Q50 受限内存复测的理想正弦映射误差为 0.001040366319，degree59 相对正弦
误差为 1.427408971e-7，GPU 相对该多项式误差为 4.433802557e-6，与 Q34 的
误差分解一致。本轮消除了此前定标组合导致的约 6.97 大错误，但没有声称
消除了原方案固有的近似误差。

此前失败的 `q50.json` / `bootstrap_precision.cpp` 及日志保留为回归样本。
新基线是另一显式入口，不能用它的成功结论覆盖旧 witness 的失败结论。

## 资源、验证与下一步

- 仅复用已有 GPU 对象并编译小型诊断入口，不进行全量 CUDA 编译。
- Q34 两轮在 12 GiB RMM 池中完成；grid 运行中采样到本进程 GPU 内存占用
  6452 MiB，非峰值测量。Q50 第一次尝试在 double-hoist C2S 阶段触发
  `Maximum pool size exceeded`，退出码 2，**没有最终精度结果**。
  这是人工池上限的分配失败，不是整张 V100 的可用显存耗尽。
- 之后缩短缓冲区生命周期：S2C 完成即释放其 GPU 矩阵和工作区，C2S 完成即
  释放其 GPU 矩阵、工作区和旋转密钥，再进入只需重线性化密钥的 EvalMod。
  不改变算法、系数、实际素数，也不修改生产运行时的内存策略。
  当前独立入口 Q34 池上限 12 GiB、Q50 上限 18 GiB；分别要求启动时至少
  20/26 GiB 空闲，额外保留 8 GiB 余量。单卡串行运行。
  Q50 复测采样到本进程使用 14134 MiB，卡上空闲 18364 MiB；同样不是峰值证明。
- 本目录 27 项 CPU/元数据回归测试通过，包含历史实际素数、Q50 素数不变、
  周期对齐、原生出口 scale、拒绝未知配置/残缺素数文件/缺失精度实验标志。
  相邻 scale-chain 工具的 6 项测试也通过，共 33 项。
  元数据测试不初始化 GPU 或生成密钥；记录见 `bootstrap_baseline_unit_tests.log`。
- 未修改应用生产入口、安全门禁、CPU Golden Reference 或 security estimator。
- 下一阶段需要单独验证原生自举出口到应用 scale=2^40 的数值接口，再将真实
  密文连续送入 ReLU/Conv；还需考察单次约 1e-3 误差在网络中的传播，不能据此
  宣称 18 次自举或完整分类正确率已达标。

## 复现

在 `/home/liufuyao/Work/poseidon_gpu` 中：

```bash
baseline_dir=Doc/analysis/resnet20_joint_chain_20260911
baseline_tmp=$(mktemp -d /tmp/resnet20-bootstrap-baseline.XXXXXX)
bash "$baseline_dir/build_relu_precision.sh" \
  "$baseline_tmp/bootstrap_baseline_precision" bootstrap_baseline_precision.cpp

# 无 GPU、无密钥的配置测试。
POSEIDON_BASELINE_BINARY="$baseline_tmp/bootstrap_baseline_precision" \
  python3 -B -m unittest discover -s "$baseline_dir" -p 'test*.py' -v

# 以下逐条运行；明确仅进行精度实验，不能据此绕过生产安全门禁。
CUDA_VISIBLE_DEVICES=0 "$baseline_tmp/bootstrap_baseline_precision" \
  --unsafe-accuracy-only historical-q34 historical
CUDA_VISIBLE_DEVICES=0 "$baseline_tmp/bootstrap_baseline_precision" \
  --unsafe-accuracy-only historical-q34 grid
CUDA_VISIBLE_DEVICES=0 "$baseline_tmp/bootstrap_baseline_precision" \
  --unsafe-accuracy-only q50-fixed45 historical "$baseline_dir/relu_precision_fixture.txt"
# 可选附加输入测试；是否实际完成以本报告实测表为准。
CUDA_VISIBLE_DEVICES=0 "$baseline_tmp/bootstrap_baseline_precision" \
  --unsafe-accuracy-only q50-fixed45 grid "$baseline_dir/relu_precision_fixture.txt"
```

每次 GPU 启动都会生成新的非零随机密钥，误差不会逐位相同。程序明确区分
`legacy_tolerance_0_002` 和 `strict_target_1e_4`；不要只看退出码。

改动文件：新增 `bootstrap_baseline_precision.cpp`、`test_bootstrap_baseline.py`
和本报告；构建脚本增加新入口；`bootstrap_precision.cpp` 仅为原 main 加入
条件编译保护，以复用诊断工具，不改变旧失败入口的算法行为。README 和旧失败
报告增加后续结果链接。没有替换原 witness；其 SHA-256 仍为
`eaf476ef2707290d7d5895bad2c072b8dc174c3009aa1795a146fa2cfd160bbf`。
