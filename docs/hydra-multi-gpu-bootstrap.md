# Hydra 思路的多 GPU Bootstrap

本文记录将 Hydra 的任务映射方法迁移到 Poseidon GPU Bootstrap 的实现状态。
Hydra 的原型硬件是多块 FPGA；这里复用的是任务分解、通信与归约思路，
底层通信改为同进程 CUDA P2P，并在 P2P 不可用时由现有传输层直接报错或按
RuntimePlan 指定的方式执行。

## 两卡版本

当前两卡版本保持外层 RuntimePlan 的 Boot 契约不变：输入和最终输出位于协调卡，
Bootstrap profile 在安装时被编译成一个内部两卡计划。执行顺序如下：

```text
GPU0: StC -> ModRaise -> raw C2S DFT
                         | P2P
             +-----------+-----------+
GPU0: extract real -> EvalMod(real)   |
GPU1:               extract imag -> EvalMod(imag)
             +-----------+-----------+
                         | P2P GPU1 -> GPU0
GPU0: real + i*imag -> output ratio -> exact scale normalization
```

原始 C2S DFT 作为阶段边界，只复制一次。real/imag 提取分别在两张卡上执行，
避免 Runtime IR 需要表达一个计算节点的两个输出。两张卡使用同一套 CPU
Relinearization/Galois key，分别上传 GPU；如果每张卡独立随机生成评估密钥，
虽然语义仍正确，但无法再对多卡与单卡密文结果做严格的逐槽零差异回归。

外层 Runtime artifact 完成后会同步所有已配置设备并释放已经完成的
in-flight 临时资源。此前资源只追加不释放，连续 warmup/iteration 会导致后半段
H2D/D2H 和设备分配时间异常增长。

在两张 V100-SXM2-32GB 上，`N=65536`、degree-22、baby-4、DA=3、
2 次预热和 5 次计时的结果为：

- 单卡 Runtime Device-final Boot：约 `78.89 ms`（此前固定基线）。
- 两卡 Runtime Device-final Boot：`65.8548 ms`。
- 两卡 Boot + D2H：`84.7308 ms`。
- 两卡完整 H2D + Boot + D2H：`88.8044 ms`。
- 两卡相对单卡设备端延迟降低约 `16.5%`。
- 两卡与单卡输出最大差异：`0`。
- 源消息最大误差：`6.78437`，与单卡一致。

测试入口是 `poseidon_runtime_gpu_bootstrap_e2e`。通过
`POSEIDON_RUNTIME_BOOTSTRAP_GPU_COUNT=1|2` 选择单卡或两卡；默认执行 2 次预热
和 5 次计时。

## 四卡版本的下一步

两卡版本只利用了两个独立 EvalMod 分支。四卡版本将进一步拆分 double-hoist
DFT：每张卡重复 baby-step，按 giant group 分配局部矩阵乘，随后使用显式二叉树
加法归约。是否拆分每一层由实测计算与通信代价决定；giant group 太少的层继续
保留单卡执行。
