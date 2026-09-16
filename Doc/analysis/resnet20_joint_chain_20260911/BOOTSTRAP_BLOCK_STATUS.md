# 首个 BasicBlock 的连续密态正确性验证

日期：2026-09-12。本阶段扩展独立诊断，不修改生产应用、安全门禁、CPU Golden
Reference、security estimator、Q/P实际素数或ReLU系数。不是性能或安全性验收。

## 实际执行的链

复用前一阶段的真实 CIFAR 图像 stem 输出作为离线明文前缀；仅在此处正常
加密一次，使用新的非零 OS 随机 balanced sparse h=192（96正、96负）密钥。
从此全部主分支、残差以及后续自举都接收已有密文，不解密/重新加密替代计算。

```text
离线明文 stem 输出 → 一次正常加密：Q6 / scale40
  → 诊断前缀自举：Q6→Q2→Q50→Q46→Q31 / scale40
  → stem ReLU：Q31→Q9
       ├─ 保存真实密文 shortcut：Q9 ──────────────────────┐
       └─ Conv1+BN：Q9→Q6                              │
            → act1 自举：Q6→Q2→Q50→Q46→Q31            │
            → act1 ReLU：Q31→Q9                        │
            → Conv2+BN：Q9→Q6                          │
            → residual_add：Q6 + drop_Q(shortcut,6) ◄──┘
            → act2 自举：Q6→Q2→Q50→Q46→Q31
            → act2 ReLU：Q31→Q9 / scale40
```

此处 scale40 指 scale=2^40；原生自举53.030946…出口仅作为差分对照，实际
后续使用已验证的全系数/DA常数补偿出口。每次自举的S2C仍消耗4Q，C2S消耗4Q，
EvalMod+DA消耗15Q；ReLU消耗22Q，每个ConvBN消耗3Q。

首个 `layer1_0` block 的真实顺序为保存输入残差、Conv1、自举/ReLU、Conv2、
残差相加、自举/ReLU，与原应用一致。block自身有2次自举；加上诊断stem前缀
总共3次。原应用stem处不一定有这次前缀自举，不能据此声称执行了原完整网络。
两个卷积分别使用原预训练参数索引1/2和对应BN，不是重复测试同一组权重。
本block为16→16、32×32、stride1、k=1，无Option-A下采样。

残差相加复用原 `residual_add()` 源码。支路仅做Q9→Q6 modulus drop，保持scale，
不做rescale、不擅自升模、不用改scale标签掩盖差异。保存的Q9密文在主支路
计算前后必须保持不变；相加后单独检查局部算术误差≤1e-7。

## 参考与通过条件

- 原GPU Conv/残差源码的复数SIMD明文模拟，对照原CHW函数组合，验证完整block
  的布局与函数语义；独立参考从最初未加密stem出发，不用解密中间结果重置。
- 每次自举内部继续保留原生/补偿对照和degree59局部参考；每次ReLU/Conv
  继续保留已有局部算术检查。所有上一阶段阈值保持不变。
- 主分支和残差的解密值只用于完整复数差诊断，空槽及虚部不忽略。
- `BLOCK_BOUNDARY`检查stem.relu、Conv1、act1原生自举、act1.relu、Conv2、
  残差相加、act2原生自举、act2.relu，共8个应用边界；每个与从原始stem出发
  的应用明文参考比较，固定目标1e-4。记录首次超标位置，不因末端误差缩小而
  抹去之前的超标。非有限值/幅值超过1或局部guard失败则停止。
- 最后再次调用独立原CHW first-block参考，要求连续明文参考与之相差≤1e-10。
  只有真正完成3次自举、2次卷积、1次残差且到达Q9，才打印`BLOCK_RESULT`。

这些误差以应用boundary=40的归一化值为单位，参考是原应用近似ReLU多项式，
不是理想max(x,0)，也不是分类logits。每次自举后的degree59局部参考来自该次
C2S诊断副本；它不能称为贯穿三次自举的独立全链算术参考。全程未重置的是
原应用明文参考，最终全链误差以它为准。

## 资源与实现边界

单卡V100-SXM2-32GB，N=65536，32768槽，Q50/P25，全局dnum=2。
RMM池上限18GiB；启动要求至少26GiB空闲；不全量重编CUDA。

抽取已验证的自举算术到 `bootstrap_ciphertext_stage.h`，使其可以接收已有Q6
密文。只将正常初始加密保留在外层。主机端缓存同一秘密密钥对应的公开求值
密钥，后两次自举不用重新生成39个旋转密钥；上传和工作区仍在每个GPU阶段
结束后释放，避免同时保留三份高Q工作区。下一次自举前释放已完成ReLU的basis。

卷积仍使用精度适配器的直接旋转和全局P25，没有验收生产应用逐层P/hoisting、
prepared/offline计时路径。诊断包含重复原生EvalMod对照、临时明文编码/上传、
下载/解密，不能把本次运行耗时或操作次数当作应用推理性能。
没有修改生产安全门禁，也没有认证该参数链安全性。

## 实测与日志

图像0、一次新非零随机密钥的完整连续执行完成，进程返回0。
原始证据：[完整GPU日志](bootstrap_block_image0.log)。

| 应用边界 | Q | log2(scale) | 对原应用明文参考的最大复数绝对误差 |
|---|---:|---:|---:|
| stem ReLU | 9 | 40 | 2.651270925e-5 |
| Conv1+BN | 6 | 40 | 2.156270698e-5 |
| act1 原生自举对照 | 31 | 53.0309462953 | 2.438537096e-5 |
| act1 补偿自举+ReLU | 9 | 40 | 2.970348530e-5 |
| Conv2+BN | 6 | 40 | 2.728647884e-5 |
| 残差相加 | 6 | 40 | 3.594886533e-5 |
| act2 原生自举对照 | 31 | 53.0309462953 | 6.302536610e-5 |
| act2 补偿自举+ReLU，即block出口 | 9 | 40 | 6.292898796e-5 |

8个边界均低于1e-4，没有首次超标点。表中scale40的实际日志值为
39.99999999998415，与目标相差约1.59e-11 bit，未用额外rescale纠正此浮点差。
最终RMS误差5.793003777e-6，最大虚部2.974395063e-5，输出最大幅值
0.1522129066871257。最终Q9可以接下一block的Conv1，但本轮未继续执行。

局部检查用于区别“算子算错”和“上游误差累积”：

- Conv1、Conv2新增局部算术误差分别为2.031659105e-6、2.036361881e-6。
- 已保存shortcut前后解密差为0，Q仍为9；残差相加局部误差4.164057179e-17，
  输出Q6，rescale次数为0。
- 三次ReLU末端的局部算术误差分别为5.344164972e-9、5.605563976e-9、
  6.070353886e-9。
- 最后一次原生自举相对其实际密文输入解码值的恢复误差3.439882256e-5；
  相对degree59局部多项式参考的算术误差3.977338059e-6。
  该次原多项式参考相对原应用参考的误差6.302508750e-5，与实测
  6.302536610e-5接近。当前主要增长出现在自举恢复/近似误差的累积，
  不是残差相加错误。不能把表中不同位置的最大值直接相减当作局部误差。

最终连续明文参考与独立原CHW first-block参考最大差4.110174437e-18。
末尾ReLU若改用理想max(x,0)作为该处参考，误差为1.252673996e-4，
未达到1e-4；此对照只替换最后一个ReLU，**不是全网络理想ReLU参考**。
因此本次通过仅针对原应用实际采用的近似多项式。

图像0完整block的纯明文SIMD/独立CHW差为5.551115123125783e-17；这不是HE精度证明。
全量元数据/负例回归执行40项并通过，另6项相邻scale-chain测试通过；运行完成后
补做2项GPU日志验收并通过，合计48项通过。新block元数据包含图像0/1，GPU仅图像0。
证据：[回归日志](bootstrap_block_unit_tests.log)、[GPU日志验收](bootstrap_block_gpu_evidence_tests.log)。
回归日志中的2项“skipped”就是后者已单独补做的GPU日志验收，不是遗漏的测试。

运行未发生OOM；准备阶段多次采样本进程GPU占用14134MiB（约13.8GiB），
不是全程峰值证明。第三次自举期间主机RSS采样约28GiB；诊断递归保留的参考和
主机矩阵不适合直接当作低内存生产执行器。测试进程已退出并释放GPU资源。
两次后续自举均命中39个旋转求值密钥的主机缓存，Conv2也复用21个旋转密钥。

本轮源码重构还逐项核对了抽取前后的自举算术段：除求值密钥工厂和等价Q2
parms_id查找外保持一致；没有重新拟合、删项或修改自举/ReLU层数公式。

## 复现

在 `/home/liufuyao/Work/poseidon_gpu`：

```bash
chain_dir=Doc/analysis/resnet20_joint_chain_20260911
chain_tmp=$(mktemp -d /tmp/resnet20-block.XXXXXX)
bash "$chain_dir/build_relu_precision.sh" "$chain_tmp/block" bootstrap_block_precision.cpp

# 纯CPU预检，不生成密钥/初始化CUDA；GPU日志检查在未指定日志时跳过。
POSEIDON_BOOTSTRAP_BLOCK_BINARY="$chain_tmp/block" \
  python3 -B -m unittest discover -s "$chain_dir" -p test_bootstrap_block.py -v

CUDA_VISIBLE_DEVICES=0 "$chain_tmp/block" --unsafe-accuracy-only \
  "$chain_dir/relu_precision_fixture.txt" 0 > "$chain_tmp/image0.log" 2>&1

# 完整执行后，审计3次已有密文自举、同一密钥缓存、8个边界和最终状态。
POSEIDON_BOOTSTRAP_BLOCK_BINARY="$chain_tmp/block" \
POSEIDON_BOOTSTRAP_BLOCK_LOG="$chain_tmp/image0.log" \
  python3 -B -m unittest discover -s "$chain_dir" -p test_bootstrap_block.py -v
```

新增 `bootstrap_block_precision.cpp`、`bootstrap_ciphertext_stage.h`、
`test_bootstrap_block.py` 和本报告/日志。原单层入口增加可选后续回调，默认仍
只运行一层；原CHW包装器增加第二组真实卷积参数和独立完整block参考。
`relu_precision.cpp`仅增加诊断scratch释放接口，不改变ReLU计算和阈值。

未覆盖：GPU stem、更多图像/独立密钥、其余8个block、stride2/Option-A、
全局平均池化、分类头、18次自举后的logits及预测一致性。
