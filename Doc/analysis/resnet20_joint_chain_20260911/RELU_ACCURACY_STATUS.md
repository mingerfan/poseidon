# Q50 方案第一阶段：22-Q ReLU 实密文精度验证

日期：2026-09-12。结论限于独立 ReLU，不是完整 ResNet20 的验收。

> 后续已有恢复定标后的连续“自举→scale40→ReLU”实测，见
> [连续子链报告](BOOTSTRAP_RELU_STATUS.md)。本页保留独立ReLU的原始测试口径，
> 不用后续不同参照的检查替换本页的中间guard失败记录。

> 后续自举实测已发现当前 Q50 定标的严重周期错误，并停在首次自举边界。
> 见 [自举失败定位报告](BOOTSTRAP_ACCURACY_STATUS.md)。本页的独立 ReLU
> 结果仍成立，但不能推及该自举输出或整网精度。

## 结论

三套独立、真实非零随机秘密密钥下，22-Q ReLU 的最终输出均通过本次
`1e-5` 最大绝对误差检查。每轮 32768 个实数槽，输入为 [-1,1] 网格，
并替换部分槽覆盖 0、±1、接近零的正负 2 的幂。误差按完整复数差的模计算，
**没有丢弃虚部误差**。这是一组确定输入的抽样验证，不是所有输入的误差上界。

| 项目 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|
| ReLU 输出相对原多项式的最大误差 | 5.07597e-6 | 1.15325e-6 | 2.37689e-6 |
| 同上，RMS | 1.17729e-7 | 1.14257e-7 | 1.14241e-7 |
| 输入噪声传播对照后的额外计算误差（最大） | 未加入该检查 | 5.97627e-9 | 7.38121e-9 |
| 输出 Q / log2(scale) | 9 / 39.999999999984 | 同左 | 同左 |
| 末端 1e-5 检查 | PASS | PASS | PASS |
| 中间阶段统一 1e-5 保守检查 | FAIL | FAIL | FAIL |

第 2、3 轮直接编译、调用应用原有 `polynomial_relu_reference()` 交叉验证，
与本测试 long-double 参考的最大差异为 **5.05672e-16**。
没有重拟合或替换多项式，也没有删掉原程序实际使用的项。

## 实际层数及中间误差

以下是三轮中的最差值；“最大误差”均相对从未加密的原始输入出发的明文计算。

| 阶段 | Q 变化 | 消耗 Q | 输出 log2(scale) | 最大误差 |
|---|---|---:|---:|---:|
| 第一段 P15 | 31 → 25 | 6 | 40.1 | 4.88179e-5 |
| 第二段 P15 | 25 → 19 | 6 | 44 | 3.57172e-4 |
| P27 | 19 → 12 | 7 | 89.871122387085 | 2.74131e-4 |
| 加 0.5、乘回原始 x、rescale | 12 → 9 | 3 | 39.999999999984 | 5.07597e-6 |

真实 GPU 计算确实只消耗 **22 个 Q**。所有节点的 Q/scale 都按已保存的
`q50.json` 执行和核对；最大 scale 偏差为 1.58523e-11 bit，是浮点记账差异。
加减前仅允许极小浮点舍入差的 metadata 对齐，没有任意覆盖 scale 来绕过检验。

中间误差不能直接等同于新调度的算术误差。加密输入本身已有约 9.2e-7～9.6e-7
的最大误差，多项式会传播/放大它。第 2、3 轮另从**最初解密出的复数输入**
出发，使用 long-double 复数明文递推；整个过程中没有重置为后续解密值。
相对此对照，三个多项式阶段的额外误差最大分别为：

| 阶段 | 第 2、3 轮最差额外误差 |
|---|---:|
| 第一段 P15 | 2.65407e-8 |
| 第二段 P15 | 1.21860e-6 |
| P27 | 1.63195e-7 |
| 最终 ReLU | 7.38122e-9 |

这支持“本组试验中的中间大误差主要是输入噪声传播，而非新调度实现错误”的判断。
不能由此外推自举输出噪声更大时仍一定达标，也不能把不同位置的最大值相减当作
严格误差分解。

诊断程序仍保留 `precision=FAIL / final_output=PASS / stage_guard=FAIL`，
退出码为 1。统一中间 `1e-5` 是本轮设置的保守诊断线，**不是论文或分类准确率
给定的验收标准**；没有为了使输出全绿而放宽它。

## 多项式逼近误差与 HE 计算误差不是同一件事

在本组输入上，即使完全不加密，原多项式相对 `max(x,0)` 的最大误差也有
**1.1509801e-4**，RMS 为 2.05722e-5。GPU 最终输出相对理想 ReLU 的最大误差
三轮约为 1.15101e-4～1.15142e-4。

因此，若“达标”要求相对理想 ReLU 的最大误差小于 1e-5，原拟合本身就不能
在这组输入上满足；若要求执行该多项式时新增的 HE 误差小于 1e-5，本次末端检查通过。
分类准确率是否满足要求，仍需真实应用数据验证。

## 配置、实现与边界

- Tesla V100-SXM2-32GB，单卡，N=65536；全局 Q50/P25，dnum=2。
- Q/P 使用 `q50.json` 原始实际素数，均小于 2^32，调用现有 32-bit GPU 路径。
- 秘密密钥使用 OS 随机源，恰好 96 个 +1 和 96 个 -1；正常公钥加密及重线性化。
- 只生成一个密文需要的公钥和重线性化密钥，没有生成整网旋转密钥。
- 运行前要求至少 16 GiB 空闲显存，RMM 池上限 12 GiB；三轮均完成，没有 OOM。
- 这是全局 P25 的隔离测试，**尚未验证应用逐层 P 映射、dnum=3/4 或卷积旋转**。
- 尚未测试 S2C 前置 degree59/DA2 自举、53-bit 出口联合折叠到 scale40、
  自举→ReLU→Conv 连接及 18 次自举后的整网精度。
- 尚未验证真实网络所有激活是否落在本次测试域内。
- 正式应用的安全门禁未修改；此入口明确要求 `--unsafe-accuracy-only`，
  不代表 Q50/P25 参数安全性获批。
- 本程序含逐节点下载/解密，不应用于性能计时。

## 复现

在 `/home/liufuyao/Work/poseidon_gpu` 中执行：

```bash
precision_dir=Doc/analysis/resnet20_joint_chain_20260911
precision_tmp=$(mktemp -d /tmp/resnet20-relu-precision.XXXXXX)
python "$precision_dir/export_relu_precision.py" --output "$precision_tmp/fixture.txt"
bash "$precision_dir/build_relu_precision.sh" "$precision_tmp/relu_precision"
CUDA_VISIBLE_DEVICES=0 "$precision_tmp/relu_precision" --unsafe-accuracy-only "$precision_tmp/fixture.txt"
python -m unittest discover -s "$precision_dir" -p 'test*.py' -v
```

GPU 执行需要可见的 CUDA 设备。退出码 1 应结合上述 `RESULT` 解读，不应忽略。
构建复用当前已有的单卡 GPU 对象文件，不重编整个 CUDA 工程；若修改底层 GPU
源码，必须先更新这些对象文件再重建测试。

新增测试源码：`export_relu_precision.py`、`relu_precision.cpp`、
`original_relu_reference.cpp`、`build_relu_precision.sh`、`test_relu_precision.py`、
`summarize_relu_precision.py`。原始证据见 `relu_precision_run1.log`～
`relu_precision_run3.log`，机器可读汇总见 `relu_precision_summary.json`。

下一阶段：单独验证真实密钥下 S2C 前置 degree59/DA2 自举与出口系数折叠。
在它通过前，不宣布 Q50 全链正确性验证完成。
