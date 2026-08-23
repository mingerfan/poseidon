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
`POSEIDON_RUNTIME_BOOTSTRAP_GPU_COUNT=1|2|4` 选择单卡、两卡或四卡；默认执行 2 次预热
和 5 次计时。

## 四卡正确性版本

四卡版本进一步拆分 post-ModRaise C2S double-hoist DFT。每张卡重复完整 baby-step，
按连续 giant-group 区间计算局部矩阵乘；每层局部结果按 `4 -> 2 -> 1` 二叉树归约，
GPU0 的归约结果作为下一层输入。C2S 完成后仍由 GPU0/GPU1 并行执行 real/imag
EvalMod，GPU2/GPU3 在首个正确性版本中不参与 EvalMod。

不能让每张卡独立完成 outer ModDown/rescale 后再归约 Q 密文。该顺序在首次测试中
产生了 `1.1525631740552091e-7` 的单卡/四卡逐槽差异，略高于严格的 `1e-7`
回归阈值。当前实现让每张卡停在 outer QP accumulator，跨卡同时归约 Q、P，随后
只在 GPU0 执行一次共享 ModDown 和该层 rescale。这既恢复了单卡运算顺序，也把
最终差异降为严格的 `0`。

在四张 V100、`N=65536`、无预热、单次执行的正确性运行中：

- C2S 四层 giant-group 数为 `4,4,1,1`；前两层使用四卡，后两层自然退化为单卡。
- 四卡与单卡逐槽最大差异为 `0`，staged/monolithic 差异也为 `0`。
- 源消息最大误差为 `6.78437`，输出 level 为 `11`，scale 为精确 `2^45`。
- 冷态 Device-final Boot 为 `203.361 ms`；同一进程后续一次 Boot+D2H 为
  `99.1505 ms`。这组数据只验证冷热差异，不能作为最终性能结论。

## 四卡性能版本的下一步

正确性基线会保留 QP 统一归并。性能阶段首先加入 StC、ModRaise、每层 C2S 的
broadcast/partial/reduce/shared-ModDown、两个 EvalMod 分支和 finalize 的独立计时，
然后依据计算/通信比选择需要分卡的 C2S 层。需要重点消除每层临时 `std::async`
线程、硬 stream synchronize、重复设备分配，并复用跨卡传输缓冲区。由于本机
四张 V100 的可见拓扑为 PCIe `PIX` 而不是 NVLink，QP 归并增加的 P-side 流量也
必须纳入分片收益模型。
