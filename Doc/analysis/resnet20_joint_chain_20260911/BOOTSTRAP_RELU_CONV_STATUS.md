# 连续自举 → ReLU → 原应用 ConvBN：单层精度诊断

日期：2026-09-12。当前阶段只接一个真实卷积，不是完整 ResNet20 验收。
生产应用、安全门禁、CPU Golden Reference、security estimator 均未修改。

## 验证对象与限制

从原应用加载 CIFAR 图像和真实预训练参数，用原 `plain_conv_bn()` 离线计算
stem 的输出。将 16×32×32 个值按原 multiplexed 布局装入前 16384 个槽，
后 16384 槽填零；这一输入仅加密一次，随后连续执行：

```text
真实 stem 输出（离线明文前缀；没有验证 GPU stem）
  → 一次正常非零随机密钥加密，Q6 / scale40
  → S2C：Q6→Q5→Q4→Q2
  → 整数3969准备 + ModRaise：Q2→Q50
  → 固定逻辑scale45的 C2S：Q50→Q49→Q48→Q46
  → 原degree59/DA2，全系数/常数出口补偿：Q46→Q31 / scale40
  → 原ReLU(15,15,27)：Q31→Q25→Q19→Q12→Q9 / scale40
  → layer1_0.conv1 + BN：Q9→Q8→Q7→Q6 / scale40
```

`scale40` 表示 scale=2^40。沿途解密只读取诊断副本，不把任何解密值重新加密
插回计算链。使用 N=65536、32768 槽、Q50/P25、全局 dnum=2、真实 OS 随机
balanced sparse h=192（96个+1、96个−1）秘密密钥。

卷积直接编译原应用 `gpu_multiplexed_tensor.cpp`，仅替换诊断类型名，未复制或
重写卷积的 mask、旋转列表、求和、BN 和 rescale 调度。
独立 CHW 参考直接编译原 `gpu_resnet20_inference.cpp` 的明文函数。
链接器丢弃未调用的完整应用入口；不创建原应用运行时，不绕过其安全门禁。

**边界：**诊断运行时调用当前主工程的单卡 GPU evaluator，用全局 P25 和直接
旋转密钥。它不等于原应用的逐层 P、hoisted/batched rotation 或 prepared/offline
运行时验收。它包含临时明文编码、上传以及诊断解密，不能用于性能计时。
原生自举与出口补偿的差分对照也不属于一次正常应用推理的操作数。

参与本轮构建的原源码 SHA-256（用于后续原工程发生修改时核对）：

```text
gpu_multiplexed_tensor.cpp b3c2dddf5113992af01ac839ac3ef543d49b97e8044b9a15f39ac24c2aae5477
gpu_resnet20_inference.cpp 24056f5386cea58d36cb0b9953b5a9aa865093f30bcf56345db3aab1274c32aa
resnet20_weights.cpp      f25d0b15c92ce588e8d76f0c3b011895cd0a5c944e7b291434ea7d1f6d2b972b
```

## 原源码的实际 Conv 调度

16输入通道、16输出通道、32×32、3×3、stride1、k=1；单密文包含32页，
16页有效。原路径内部复制输入两份，输出分8组，采用延迟明文累加和输出 BSGS。

| 原码运行时调用 | 次数 |
|---|---:|
| 非零旋转 | 49 |
| 密文加法 | 33 |
| multiply_plain | 10 |
| multiply_plain_accumulate | 72 |
| rescale | 3 |
| add_plain（BN bias） | 1 |
| 同层复制/drop 接口 | 26 |

这些是运行时操作调用数，不是 kernel launch 数。GPU 和明文模拟必须产生相同的
操作计数与按 Q 分组的旋转集合，否则诊断失败。

旋转密钥集合由原 Conv 源码的明文模拟采集，而非照搬旧 Q7/Q8 应用列表：

- Q9：`1,31,32,33,1024,2048,4096,8192,16384,18432,20480,22528,24576,26624,28672,30720,32735,32736,32737,32767`。
- Q7：`15360`。

共21种不同非零旋转。自举旋转 GPU 密钥和工作区释放后，再用同一秘密密钥生成
并上传这些公开求值密钥，避免同时保留两组大工作区。

## 校验口径

1. 无密钥、无 CUDA 的原 SIMD 源码明文模拟，比较原 CHW 卷积参考；覆盖实际
   图像和非图像边界测试值，校验旋转方向、空间越界 mask、通道布局及空槽。
2. 卷积局部算术：实际密文 ReLU 输出的解密副本，经过完整复数 SIMD 参考，与
   实际 GPU Conv 比较。包括全部32768槽，不忽略输入空槽的 HE 噪声。
3. 连续多项式参考：从自举 C2S 的解密副本开始连续计算原 EvalMod、ReLU、Conv，
   中途不重新初始化该参考；与最终密文比较。
4. 原应用参考：对最初未加密 stem 输出执行原 ReLU 和 CHW Conv，包含全部
   自举近似误差、编码误差和 GPU 算术误差。另报末端1e-4是否通过。

比较使用完整复数差的模，并报告虚部及空槽泄漏。卷积局部误差、空槽最大值均
要求≤1e-5；连续多项式误差应≤“已测量的输入误差传播”+1e-5。
后一检查是误差归因的一致性检查，**不等于全链满足固定1e-5/1e-4精度目标**。
固定末端1e-4结果单独报告，不随测试数据放宽。
上游自举/出口补偿/ReLU 的所有阶段 guard 保持上一阶段的值，不因接卷积而放宽。
误差均在应用的 boundary=40 归一化表示中计算，不是最终分类 logits 误差。

## 实测证据

一次真实非零随机密钥、图像0的连续GPU运行完成，进程返回0，所有阶段
integration guard通过。原始证据：[完整日志](bootstrap_relu_conv_image0.log)。

| 边界/指标 | Q / log2(scale) | 最大复数绝对误差 |
|---|---|---:|
| 原生自举对照 vs 最初未加密输入 | Q31 / 53.0309462953 | 1.369156660e-5 |
| scale40补偿自举 vs 最初未加密输入 | Q31 / 40 | 3.143028843e-5 |
| 连续ReLU vs 原应用明文ReLU | Q9 / 39.99999999998415 | 2.324369312e-5 |
| ReLU末端局部算术 | Q9 / 39.99999999998415 | 5.197609056e-9 |
| Conv新增局部算术，包含全部槽 | Q6 / 39.99999999998415 | 1.792481549e-6 |
| Conv末端 vs 连续degree59/ReLU/Conv多项式参考 | Q6 / 39.99999999998415 | 1.344593266e-5 |
| Conv末端 vs 原应用明文ReLU+CHW Conv | Q6 / 39.99999999998415 | 1.759113707e-5 |

最终对原应用参考的RMS误差3.340718262e-6，最大虚部1.759113443e-5，
后16384个空槽最大幅值5.164828094e-9。固定末端1e-4检查通过。
卷积输入算术误差的明文传播量为1.345021503e-5；自举近似误差经过
ReLU/Conv后的明文传播量为1.754620929e-5。它们是不同参考之间的差，
不能将这些最大值直接相减得出局部误差。

该层原应用明文输出最大幅值0.09009965873024799，结合最终误差可得实际
密文解码输出幅值上界约0.09012，小于当前测试域1；尚未执行下一次自举。
图像0起点最大幅值为0.08547231963851412。这比之前[-0.5,0.5]网格范围窄，
本轮原生自举的近似误差相应更小。**不能用这一张图的通过结果覆盖之前网格
测试的失败，也不能宣布18次自举后或完整网络的精度达标。**

另外，当前ReLU末端对理想max(x,0)误差为1.180459581e-4，并未达到1e-4。
表中的通过目标是原应用使用的近似ReLU多项式，二者参照不同，不能混淆。

运行未触发OOM，RMM池上限18GiB；准备阶段采样显存14134MiB（约13.8GiB），
不是全程峰值证明。运行结束后进程退出并释放GPU资源。

纯明文 SIMD/CHW 差最大6.354102357122859e-17；非图像布局检查差最大
1.615732287546523e-16。它们只证明布局/函数等价，不是 HE 精度证明。

新构建的 baseline、ReLU、Conv 三个入口完成本目录36项回归测试，全部通过；
相邻 scale-chain 目录6项测试通过，合计42项。新 Conv 测试含4项，覆盖图像0/1
的原码布局/层数调度及非法图像/运行模式。见 `bootstrap_relu_conv_unit_tests.log`。
这些单元测试不生成密钥、不初始化 CUDA。

## 复现与新增文件

在 `/home/liufuyao/Work/poseidon_gpu` 下：

```bash
chain_dir=Doc/analysis/resnet20_joint_chain_20260911
chain_tmp=$(mktemp -d /tmp/resnet20-bootstrap-conv.XXXXXX)
bash "$chain_dir/build_relu_precision.sh" "$chain_tmp/probe" bootstrap_relu_conv_precision.cpp

# CPU metadata、原源码布局和负例检查，不生成密钥/初始化CUDA。
POSEIDON_BOOTSTRAP_RELU_CONV_BINARY="$chain_tmp/probe" \
  python3 -B -m unittest discover -s "$chain_dir" -p 'test_bootstrap_relu_conv.py' -v

# 串行单卡、精度诊断；要求启动时至少26GiB空闲，RMM池上限18GiB。
CUDA_VISIBLE_DEVICES=0 "$chain_tmp/probe" --unsafe-accuracy-only \
  "$chain_dir/relu_precision_fixture.txt" 0
```

新增：

- `bootstrap_relu_conv_precision.cpp`：真实输入、原应用 Conv 连续入口和分层参考。
- `conv_probe_runtime.h`：原源码共用的复数 SIMD 参考/单卡 GPU 精度适配器。
- `original_conv_reference.h/.cpp`：原 CHW 参考、真实参数、离线 stem 输入。
- `test_bootstrap_relu_conv.py`：两图布局/Q/scale检查、非法输入及模式负例。
- 本报告与实测/回归日志。

修改现有诊断：baseline增加可选应用输入和公开旋转求值密钥工厂；ReLU增加
仅在上游guard通过后调用的可选后续阶段。原独立入口默认行为不变。
构建脚本增加新入口和原参数读取源码；不全量重编 CUDA。
旧 `q50.json` 失败 witness、Q/P实际素数、ReLU fixture 和原系数均保持不变。

下一步是完整 BasicBlock：第二次真实卷积、残差支路层数/scale对齐，以及再次
自举的连续闭环。再之后才是逐层 P 路径验证、18次自举和完整分类头/预测一致性。
