# LCDNN S2C 前置实部投影迁移状态

日期：2026-09-13。本页记录单卡 V100 上的正确性优先诊断结果；不是安全批准或
生产性能结论。后续完整 ResNet20 分类已通过，见
[完整网络正确性报告](FULL_NETWORK_STATUS.md)。

## 已验证的算法边界

当前路径为满槽 `N=65536 / slots=32768` 的 S2C 前置自举：

```text
输入 Q6 / scale 2^40
  -> S2C，Q6 -> Q2
  -> 整数准备 + ModRaise，Q2 -> Q50
  -> C2S real/imag，Q50 -> Q46
  -> 两次 degree-59 / DA2 EvalMod，Q46 -> Q31
  -> 16 * v + 16 * conjugate(v)
  -> 输出 Q31 / scale 2^53.030946...
```

最后一步等价于在历史输出乘数 32 中折入 LCDNN 的
`Re(v)=v/2+conjugate(v)/2`，只增加一次共轭 KeySwitch，不增加 rescale，
因此不消耗 Q。

LCDNN 原文的 imaginary-removing bootstrapping **没有删除 imaginary C2S 或
第二次 EvalMod**。此前只运行 real EvalMod 的试验虽然 GPU 对其明文多项式误差
仅约 `4.19e-6`，但该多项式相对原输入误差约 `5.08`，已经删除，不作为可用路径。

## 单次自举实测

输入为 CIFAR-10 图像 0 经原始 stem 明文参考形成的真实应用槽，使用 OS 随机、
非零、平衡 `h=192` 秘密密钥；只加密一次。

| 路径 | GPU 阶段总计 | 最终重组 | 最大虚部 | 相对原输入最大误差 |
|---|---:|---:|---:|---:|
| 普通复数 S2C 前置 | 314.515 ms | 0.249 ms | 1.111e-5 | 1.355e-5 |
| LCDNN 实部投影 | 317.655 ms | 3.022 ms | 2.683e-10 | 1.446e-5 |

LCDNN 路径的一次分解：

| GPU 阶段 | 时间 |
|---|---:|
| S2C | 27.074 ms |
| prepare | 0.438 ms |
| ModRaise | 0.682 ms |
| C2S | 120.693 ms |
| EvalMod real | 82.996 ms |
| EvalMod imag | 82.749 ms |
| 重组 + 实部投影 | 3.022 ms |
| 合计 | 317.655 ms |

计时在每个实际 GPU 操作前后执行设备同步；不包含矩阵编码、密钥生成/上传、
解密参考检查。相对普通复数路径增加约 `3.14 ms`（约 `1.0%`），本次随机密钥
之间的小幅阶段波动不解释为算法收益。

## 应用 scale 与 ReLU 连续验证

同一 C2S 密文上的 EvalMod 系数折叠将输出实际接到 `Q31 / scale=2^40`，没有
额外 Q：

- scale40 自举输出相对原输入最大误差：`2.790897421990975e-5`；
- 连续执行原 `{15,15,27}` ReLU 后为 `Q9 / log2(scale)=39.99999999998415`；
- ReLU 输出相对原应用明文 ReLU 最大误差：`1.948664432684577e-5`；
- GPU 相对连续明文多项式最大误差：`1.644108423987556e-5`；
- ReLU 末端最大虚部：`4.541023532455358e-7`。

两处 `1e-4` 正确性边界均通过。后续已在本路径上完成真实卷积、9 个 BasicBlock
和完整网络；本节保留的是最初的单次自举/ReLU 证据。

## 稀疏 DFT 修复与限制

Poseidon 的 `log_slots < log_n-1` DFT 移植原来存在三个矩阵层问题：repack
矩阵为空、蝶形向量错误按 `n` 而非 `2n` 分配、C2S 后半清零按值遍历而无效。
本轮补齐后，完整稀疏 S2C/C2S 矩阵生成单测通过。

这只证明矩阵生成契约，不证明 LCDNN 的 `BOOT14/BOOT13/BOOT12` 密文语义已经
实现。Poseidon 当前 CKKS encoder 固定使用 `N/2` 槽，没有 Lattigo 式的逐密文
`LogDimensions`；把 16384 个有效值简单放在前半、后半补零后直接套稀疏 DFT，
实测目标函数误差约 `5.08`。因此生产路径仍使用 32768 槽满槽 DFT，真正 reduced-
slot bootstrapping 必须另做编码/打包契约，不能把当前矩阵单测冒充完整 BOOT14。

## 复现

```bash
chain=Doc/analysis/resnet20_joint_chain_20260911
bash "$chain/build_relu_precision.sh" \
  /tmp/resnet20_block_lcdnn_s2c_real bootstrap_block_precision.cpp

# 无 GPU 的元数据与负向测试
POSEIDON_BOOTSTRAP_BLOCK_BINARY=/tmp/resnet20_block_lcdnn_s2c_real \
  python3 "$chain/test_bootstrap_block.py"

# 单次自举及分解计时
CUDA_VISIBLE_DEVICES=0 /tmp/resnet20_block_lcdnn_s2c_real \
  --unsafe-accuracy-only "$chain/relu_precision_fixture.txt" 0 \
  --lcdnn-s2c-first-real --bootstrap-only

# 单次自举折叠到 scale40，并连续执行原 ReLU
CUDA_VISIBLE_DEVICES=0 /tmp/resnet20_block_lcdnn_s2c_real \
  --unsafe-accuracy-only "$chain/relu_precision_fixture.txt" 0 \
  --lcdnn-s2c-first-real --relu-only
```

`--unsafe-accuracy-only` 只是显式标记当前参数未通过安全审批，不绕过零密钥禁令；
测试仍使用随机非零密钥。

## 未完成项

1. 将已验证顺序封装为独立生产 GPU bootstrap scheduler；当前生产
   `GpuEvaluator::bootstrap` 仍固定为 ModRaise/C2S/EvalMod/S2C。
2. 在应用生产运行时切换到该 scheduler；诊断入口的完整 BasicBlock/全网验证已完成。
3. 单独实现和验证 BOOT14/13/12 的 reduced-slot 密文编码与转换语义。
4. 对最终 Q/P 参数重新做安全估计；当前 Q50/P25 仅用于准确性验证。
