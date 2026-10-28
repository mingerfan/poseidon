# 无手动约束的自动 scale + 25Q

2026-10-28，在188Server使用当前实验分支二进制，重新编译完整原始 ResNet20。明确移除 `--native-target-scale-goals` 与 `--native-target-bootstrap-schedule`，scale 与 Boot 位置都由当前 target + greedy 决定。使用同一手动 Q50/P25 链、waterline=40，Boot 入口 Q6/s40、出口 Q31/s40；每次 Boot 后最多有25Q可用于算术及入口对齐。

|初始密文|自动 Boot 次数|Rescale 操作数|编译时间|C++ RuntimePlan 校验|
|---|---:|---:|---:|---|
|Q31/s40|**38**|1686|18.81秒|PASS，16892个值|
|Q34/s40|**37**|1676|18.81秒|PASS，16962个值|
|Q34/s40，提供手动阶段 scale 目标，先前对照|18|1218|约9～14秒|PASS|

本次两份完整编译并行执行，各自编译子进程峰值RSS约1.16GiB。模数最大32bit，没有修改模数链。两份计划的每个 Boot 都满足实际入口和出口 level/scale。逐个复核 Rescale 方程 `s_out=s_in−log2(q_current)`，最大差值为0（按计划所存浮点数计算）。Q31结果的 RuntimePlan SHA256 与上轮无约束38次计划相同。

这里的“自动版”指新 target 策略不使用手动阶段目标；Boot 插入采用局部 greedy。target 尚未接入 DaCapo optimized，本次结果不是 optimized 的 Boot 次数或最优下限。当时另有多状态搜索使用另一条 threshold 路径，不计入这两份结果；该任务随后已按用户要求停止，并由单状态实现替换。

## 差距在哪里

单独复编译同一 ReLU、不给手动目标，结果是 **23Q**：Q31/s40→Q8/s≈40，66个 Rescale 操作，没有 Boot。手动约束版为22Q，到Q9。后续卷积即使按3Q计算，23+3=26也超过25，无法按手动22+3=25的方式安排同一片段。增加预算不会自动让当前局部 scale 选择达到手动的阶段出口。

此外，Q34完整计划的37个 Boot，按所刷新值的原始源码位置分为：

- 18个来自 ReLU 尾乘，`ResNet20.py:111`。
- 18个来自多项式共享基底计算，`ResNet20.py:79`。
- 1个来自末端平均池化中的加法，`MPCB.py:241`。

因此额外的 Boot 并非单纯因为初始少了几个 Q；中间基底的 scale/level 选择、局部刷新以及末端处理仍有改进空间。上述是所刷新值的来源位置，不表示加法自身消耗了乘法深度，也不说明37或38已经最优。

## 产物

[Q31完整命令和结果](auto25/q31/result.json)、[Q34完整命令和结果](auto25/q34/result.json)、[单次ReLU](auto25/one-relu/result.json)、[37个Boot的来源](auto25/q34/boot-source-locations.json)、[逐Boot入口出口](auto25/q34/boots.json)、[复现脚本](auto25/run.py)。

当前编译器 SHA256：`6482cc341572337edc7c382e50b58f3dbc8eb37f226b9c3d4075942ce9ae3c9d`。本次只增加实验和记录，没有修改规划策略，也未 commit/push。完整模型仅做编译与计划校验，没有新增整网密态推理测试。上轮手动约束及GPU验证见[25Q实验报告](budget25.md)。

远端完整计划、bundle、bytecode及Q34的带源码位置IR：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/auto25-unhinted-20261028T011542Z/
```
