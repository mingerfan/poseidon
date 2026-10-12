# ResNet20 连续密态全网正确性状态

日期：2026-09-13。图像 0 的完整 ResNet20 已在单张 V100 上执行通过。该结果是
Q50/P25、随机非零 `h=192` 密钥下的**准确性诊断**，不构成参数安全批准或性能结论。

## 已完成的真实密态路径

```text
加密 CIFAR-10 输入 patch -> GPU stem ConvBN -> ReLU
  -> 9 个 BasicBlock
     每个 block: ConvBN -> S2C-first bootstrap -> ReLU
                 -> ConvBN -> residual -> S2C-first bootstrap -> ReLU
  -> global average pool -> FC -> 解密 10 个 logits
```

- 完成 9/9 BasicBlock、18/18 次自举、18 次 block 卷积和 9 次残差相加。
- 两个 stride-2 block 的 Option-A shortcut 均执行并通过独立 CHW 参考检查。
- 自举使用满槽 `N=65536 / slots=32768`、degree-59/DA2、实部投影路径。
- 所有中间层沿同一密文链继续计算；中间重加密次数为 0。
- stem 按原应用实现加密 27 个原始图像 im2col patch。这些都是输入编码，
  不是用解密后的中间值重新加密。
- 边界解密只用于观察和与独立明文参考比较，不参与下一层输入。

## 本次结果

| 项目 | 结果 |
|---|---:|
| 真实标签 | 3 |
| 明文参考预测 | 3 |
| GPU 密态预测 | 3 |
| 10 个 logits 最大绝对误差 | 0.0104490249856819 |
| 全网中间边界最大绝对误差 | 0.0003095991874335351 |
| 最终 block 状态 | Q9 / log2(scale)=39.99999999998415 |
| 每个 block 后 GPU 可用显存 | 17.9339599609375 GiB（稳定） |

最终 logits：

```text
-6.6128274709  2.3368999641  1.8481904677  22.4489919035  -5.8440306738
 5.6112920186 -3.4778886940 -7.5987820517  -2.0109499928  -6.6967947192
```

独立原应用 CHW 参考 logits：

```text
-6.6069642899  2.3473489890  1.8511820891  22.4578666832  -5.8519356767
 5.6138431538 -3.4807083062 -7.6067586726  -2.0131181348  -6.7066615256
```

最终验收要求 `max_logit_error <= 0.1` 且明密文 argmax 一致，本次为 `PASS`。

## 内存与离线准备

原递归诊断会在嵌套回调中保留前一次自举工作区，不适合 18 次连续执行。当前入口
在每次自举/ReLU 完成后只复制 Q9 输出，随后退出回调并释放该轮临时对象。9 个
block 后可用显存保持相同，未观察到随 block 数增长的显存下降。

S2C/C2S 的 CPU 编码矩阵在第一轮创建后缓存：完整运行记录 1 次 `matrices=miss`
和 17 次 `matrices=hit`。GPU 矩阵与公开旋转键仍按轮上传并释放，因此该程序是
正确性入口，不是已经完成常驻资源优化的性能入口。

## 复现

```bash
chain=Doc/analysis/resnet20_joint_chain_20260911
bash "$chain/build_relu_precision.sh" \
  /tmp/resnet20_network_lcdnn bootstrap_network_precision.cpp

# 无 GPU：检查完整拓扑与参数元数据
/tmp/resnet20_network_lcdnn --metadata-only \
  "$chain/relu_precision_fixture.txt" 0

# 先跑 3 个 block 的显存/精度门槛
CUDA_VISIBLE_DEVICES=0 /tmp/resnet20_network_lcdnn \
  --unsafe-accuracy-only "$chain/relu_precision_fixture.txt" 0 \
  --max-blocks 3

# 完整 9-block/18-bootstrap 推理
CUDA_VISIBLE_DEVICES=0 /tmp/resnet20_network_lcdnn \
  --unsafe-accuracy-only "$chain/relu_precision_fixture.txt" 0
```

`--unsafe-accuracy-only` 明确表示 Q50/P25 尚未获安全批准；它不会启用零密钥、
不会绕过非零秘密密钥生成，也不会把解密后的中间结果送回密态计算。

## 尚未完成

1. 将本诊断中已经验证的 S2C-first 调度接入应用的生产 `GpuCkksRuntime`。
2. 将 DFT 数据和公开旋转键做显存安全的常驻/分阶段复用，消除分钟级上传准备。
3. 在新的安全参数下复跑完整精度验证；当前 Q50/P25 已知不是安全批准配置。
4. 实现真正 reduced-slot BOOT14/13/12；本结果使用 32768 满槽，不冒充 reduced-slot。
