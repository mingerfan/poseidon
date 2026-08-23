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

## 四卡性能版本

性能版本加入了 StC/ModRaise、每层 C2S fanout/partial、QP reduction、共享
ModDown/rescale、EvalMod 分支、结果回传和 finalize 的分段计时。实测表明本机
四张 V100 之间是 PCIe `PIX`，对单个 degree-22 密文，物理卡数不等于有效并行度：

- 单卡 C2S 四层约为 `10.5,10.4,4.9,4.8 ms`。
- 前两层使用两卡后，重复 baby-step、QP 传输和归约使每层增长到约 `13.3 ms`。
- 前两层强制四卡更慢，因此最低延迟计划让 C2S 只使用 GPU0。
- 单个 EvalMod 约 `19.3--20.0 ms`，其中基生成约 `9.1--9.5 ms`、BSGS 合并约
  `6.1--6.3 ms`、三次二倍角约 `3.3--3.4 ms`；叶计算只有约 `0.72 ms`。

为验证 Hydra 的多项式树映射，当前实现提供 degree-22 根分区：GPU0/GPU2 协作
real，GPU1/GPU3 协作 imag。coordinator 计算叶 0--1、根商子树和 `Q*T16` 原始
size-3 乘积；helper 计算叶 2--5 和根余数子树，并跳过不需要的 T16 基。helper
只回传一个低层级 size-3 余数，两部分相加后统一执行一次 HYBRID keyswitch，保持
与单卡 lazy-relinearization 相同的运算顺序。分支输入通过 CUDA source-ready event
连接到 P2P stream，避免读取未完成的 real/imag 提取结果。

四卡根分区的正确性结果为：Runtime/direct、partitioned/direct 最大差异均严格为
`0`，源消息误差仍为 `6.78437`。但在 2 次预热、5 次计时下，强制四卡根分区为
`65.97 ms`，没有超过两卡 real/imag 分支。这是因为 degree-22 的 T2/T3/T4/T8
基链仍需在两个分区重复，而两次 PCIe 传输无法被较小的子树完全掩盖。

因此编译后的默认低延迟计划是“四卡安装、按成本选择活跃子集”：C2S 使用 1 卡，
real/imag EvalMod 使用 2 卡。稳定结果如下：

- 直接单卡完整 profile：`85.1568 ms`。
- 默认 Runtime Device-final Boot：`65.4409 ms`，相对同轮直接执行为 `1.301x`。
- 两个 EvalMod 分支：`23.4150 ms`。
- Runtime/direct、partitioned/direct 最大差异：`0`。
- Boot+D2H：`79.1030 ms`；完整 Host-final RuntimePlan：`83.7438 ms`。

`POSEIDON_RUNTIME_BOOTSTRAP_C2S_DEVICE_LIMIT=1|2|4` 控制 C2S 活跃卡数；
`POSEIDON_RUNTIME_BOOTSTRAP_EVALMOD_DEVICE_LIMIT=2|4` 控制 EvalMod 选择普通
real/imag 双卡路径还是四卡根分区路径。四卡根分区保留为可复现实验路径，也可在
更快互连或更高阶多项式上重新评估。下一阶段若追求四卡收益，应优先调度多个独立
bootstrap 获得吞吐并行，或者实现 RNS/keyswitch 粒度的数据并行，而不是继续拆分
degree-22 单密文的窄依赖链。
