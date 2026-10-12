# 连续自举 → scale40 → ReLU 正确性检查

日期：2026-09-12。范围是单个密文的连续子链，不是完整 ResNet20 推理。
生产应用、安全门禁、CPU Golden Reference 和 security estimator 均未修改。

## 实测结果

已完成两轮同密文连续 GPU 检查，每轮独立生成非零随机秘密密钥。
原生/补偿自举、ReLU 局部算术 guard 均通过。
但相对原始未加密输入的末端1e-4检查未通过，不能据此宣布整网精度达标。

| 测试输入 | scale40 vs 同源原生出口 | ReLU 末端局部算术误差 | 子链合计算术误差 | 末端 vs 原应用 ReLU |
|---|---:|---:|---:|---:|
| 第1轮：网格+近零正负数 | 2.599562476e-5 | 5.250682479e-9 | 2.606730415e-5 | 0.000898058093 |
| 第2轮：完整实数网格 | 2.444476300e-5 | 6.400804661e-9 | 2.468081580e-5 | 0.000905309264 |

第1轮原生自举恢复误差0.000867303468，补偿后0.000867189278。
ReLU末端相对理想max(x,0)误差0.000891259774；相对原应用多项式的误差
应看上表，二者参照不同。原始证据：[第一轮日志](bootstrap_relu_run1.log)。

第2轮原生自举恢复误差0.000870333828，补偿后0.000877028327；ReLU末端
相对理想max(x,0)误差0.000900888116。证据：[第二轮日志](bootstrap_relu_run2.log)。
两轮实际ReLU出口均为Q9 / log2(scale)=39.99999999998415，确实消耗22个Q；
没有为实现接口而额外删除模数。

第1轮三个ReLU阶段的局部算术最大误差分别为
2.349264872e-8、1.062341420e-6、1.537319168e-7，末端为5.250682479e-9。
第2轮对应为1.874720453e-8、1.168970581e-6、1.562354083e-7，末端6.400804661e-9。
这里的复数参考从ReLU最初输入出发，没有在后续阶段重新初始化参考。

第1轮的明文连续参考预测：自举近似误差传播到ReLU末端后，相对原应用多项式
误差为0.000898329338，与实测0.000898058093接近。因此目前主要问题不是
ReLU算子算错，而是原自举近似仍存在约1e-3的最差误差，以及scale40补偿引入
了约2.6e-5的额外误差。第2轮的同一明文预测为0.000902228305，实测0.000905309264。
这些子链测试尚不能证明误差在全网会不会影响分类结果。

日志中相对原始未加密输入的P15中间误差可达0.0517，T16基函数误差可达1.00；
这不是它们的局部GPU算术误差，也不能作为最终ReLU误差。高阶中间函数会放大
输入差异，末端组合和乘回输入后可能缩小；应结合上述独立连续复数对照判断。

## 实现与对照

新增入口 `bootstrap_relu_precision.cpp` 复用已恢复的 Q50/P25 自举：
N=65536、32768 槽、全局 dnum=2、非零 OS 随机 h=192 秘密密钥。
所有 Q/P 实际素数保持不变，仍为现有 32-bit GPU 路径。

一次正常公钥加密后，连续执行：

```text
Q6 / scale40
  → S2C：Q6→Q5→Q4→Q2
  → 整数3969准备：Q2 / scale58.9999105007
  → ModRaise：Q2→Q50
  → 固定逻辑scale45的C2S：Q50→Q49→Q48→Q46
       ├─ 原版EvalMod+DA2：Q31 / scale53.0309462953（对照）
       └─ 全系数和DA常数补偿：Q31 / scale40（真实后续输入）
            → 原ReLU P15：Q31→Q25 / scale40.1
            → 原ReLU P15：Q25→Q19 / scale44
            → 原ReLU P27：Q19→Q12 / scale89.8711223871
            → 加0.5、乘回ReLU输入、rescale：Q12→Q9 / scale40
```

这里的 scale 数字表示 log2(scale)。原生对照与补偿路径使用**同一套密钥和
同一份 C2S 输出密文**，不是两次独立加密的比较。补偿后的密文直接进入 ReLU。
解密只用于读取诊断副本，没有把解密值或明文参考重新加密插回 GPU 链。

## scale40 接口如何实现

复用之前已规划的出口补偿方式，但这次只接在**周期已修复**的自举上：

- 原生物理出口 scale 为 S=2^53.03094629532367。
- alpha=2^40/S=0.0001194797467008398，r=alpha^(1/4)=0.1045498883353879。
- 原多项式的全部系数乘 r，两次 DA 常数分别按 r²、r⁴ 补偿。
- 先检查 GPU 结果在原生物理 scale 下表示的是 alpha 倍原值，再将对应的
  表示 scale 乘 alpha，使其恢复原值且 scale=2^40。
- 不对原生输出裸改 scale 标签，不删多项式项，不重拟合，也不额外 rescale。
- 对比上传器生成的 basis、叶项、合并节点、明文 Q/scale、DA rescale 和
  重线性化 Q 前缀，要求补偿前后调度完全一致。

元数据测试中，使用实际存储的补偿系数计算，明文恒等差最大为
2.214970828279261e-15。ReLU 参考与应用原始 `polynomial_relu_reference()`
的最大差为 1.188014530500991e-16。这些静态数值不是密文精度证明。

`q50.json` 仍保留旧失败定标作为回归样本；本入口只读取其中导出的实际 Q/P
及 ReLU 调度，不采用它的旧准备整数3897或动态 C2S scale。

## 检查口径

程序分别报告以下参照，不能把不同参照的最大误差直接相减：

1. 补偿出口 vs 同一 C2S 输入的原生出口：检测 scale40 接口的额外误差。
2. 补偿出口 vs 原 degree59 明文计算：检测补偿路径的 HE 计算误差。
3. ReLU 各阶段 vs 从**最初 ReLU 输入解密副本**出发的完整复数多项式递推：
   检测 ReLU 额外算术误差，后续阶段不会重新从解密值开始计算参考。
4. 子链末端 vs 从 C2S 解密副本出发、连续经过原 degree59 和原 ReLU 的参考：
   检测出口补偿和 ReLU 的合计算术误差。它不隔离 C2S 以前的误差。
5. 子链末端 vs 对原始未加密输入执行的原应用 ReLU：包含自举近似和全部 HE
   误差，是更接近应用语义的误差；另报相对理想 max(x,0) 的误差。

所有比较使用完整复数差的模，**不丢弃虚部误差**。ReLU 入口检查幅值不超过1。
测试输入 `grid-near-zero` 是 [-0.5,0.5] 实数网格，显式包含0、±0.5，以及
±2^-1 到 ±2^-32，重点覆盖 ReLU 的近零敏感区。

阶段 guard：原生/补偿自举恢复误差不超过旧容差0.002；接口差分及子链算术
误差不超过1e-4；ReLU 局部各阶段和末端算术误差不超过1e-5。
`integration=PASS` 只表示这些局部检查通过。程序另外报告
`end_to_end_target_1e_4`，**不会因整体验证更难而修改或隐藏该项结果**。
旧 ReLU 单测中相对未加密输入的统一中间1e-5 guard仍保持原来的失败记录；
本轮的局部算术对照参照不同，不能混为同一检查。

## 资源与回归

- 单卡串行，RMM 池上限18 GiB，启动要求至少26 GiB空闲。
  两轮均完成，未触发显存池限制；第一轮准备阶段采样到本进程占用14134 MiB，非峰值证明。
- 在已同步的诊断边界释放 S2C/C2S 矩阵、旋转密钥和不再使用的 EvalMod
  工作区；ReLU 不保留变换工作区。没有全量重编 CUDA。
- 本目录32项测试、相邻scale-chain目录6项测试通过，共38项。
  新增5项覆盖补偿元数据、原始函数参照、拒绝复数数据集、错误运行模式、
  错误末端层数、篡改系数（部分测试包含多个检查）。
- 原生基线增加可选诊断 continuation hooks；默认无 hook 的独立入口仍不做
  出口补偿。continuation只能在原生基线guard通过后执行。
- 旧 witness 和 fixture 的 SHA-256 保持不变。

## 复现

在 `/home/liufuyao/Work/poseidon_gpu` 下：

```bash
chain_dir=Doc/analysis/resnet20_joint_chain_20260911
chain_tmp=$(mktemp -d /tmp/resnet20-bootstrap-relu.XXXXXX)
bash "$chain_dir/build_relu_precision.sh" "$chain_tmp/bootstrap_relu" bootstrap_relu_precision.cpp
bash "$chain_dir/build_relu_precision.sh" "$chain_tmp/bootstrap_baseline" bootstrap_baseline_precision.cpp

# 纯CPU元数据/负例，不生成密钥、不初始化CUDA。
POSEIDON_BASELINE_BINARY="$chain_tmp/bootstrap_baseline" \
POSEIDON_BOOTSTRAP_RELU_BINARY="$chain_tmp/bootstrap_relu" \
  python3 -B -m unittest discover -s "$chain_dir" -p 'test*.py' -v

# 同一密文连续计算，仅精度实验；不是绕过生产安全门禁。
CUDA_VISIBLE_DEVICES=0 "$chain_tmp/bootstrap_relu" --unsafe-accuracy-only \
  "$chain_dir/relu_precision_fixture.txt" grid-near-zero
CUDA_VISIBLE_DEVICES=0 "$chain_tmp/bootstrap_relu" --unsafe-accuracy-only \
  "$chain_dir/relu_precision_fixture.txt" grid
```

该入口含逐节点下载/解密，**不能作为性能计时入口**。密钥生成耗时不是自举
GPU延迟。尚未接入真实ConvBN、残差分支或分类头，也未验证整网所有激活的取值域。

## 下一阶段的验收对象

1. 保留当前子链作为回归基线，接入真实ConvBN权重、BN系数和应用打包/旋转布局，
   核对Q9→Q6及scale40，不用恒等算子冒充卷积。
2. 对照应用现有`plain_conv_bn()`，分别测卷积的局部算术误差和此前误差的传播；
   检查卷积输出的实际取值域是否仍适合下一次自举。
3. 再接一个完整BasicBlock：两次真实卷积、残差支路对齐和后续自举/ReLU，
   沿途检查同一密文的连续状态。通过后才扩展到18次自举及完整分类头。
4. 最终用真实图像的logits误差、相对明文实现的预测一致性及分类准确率验收，
   不用这两轮合成输入的局部PASS替代整网正确性证据。

新增实现/测试：`bootstrap_relu_precision.cpp`、`test_bootstrap_relu.py`。
`bootstrap_baseline_precision.cpp`增加可选continuation接口和近零数据集，
构建脚本增加新入口；没有修改已保存的失败witness或原ReLU系数文件。
