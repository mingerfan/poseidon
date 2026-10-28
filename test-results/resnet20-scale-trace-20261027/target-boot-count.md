# 新scale管理下的ResNet20 Boot次数

> 本文记录23Q预算时的历史实验。25Q、阶段scale约束及18次自动Boot的后续结果见[更新报告](budget25.md)。

本次统计完整实际ResNet20编译计划的Boot插入次数。输入图、Q50/P25手动链、Boot profile和waterline=40均相同，工作Q预算仍为23，没有取消原来的2Q对齐预留。所有规划与验证在188Server执行，不进行整网密态推理。

## 结果

|scale管理与Boot规划|Boot次数|rescale次数|说明|
|---|---:|---:|---|
|原threshold + greedy|182|4811|本次重跑|
|新target + 局部greedy|139|1633|本次自动插入|
|原threshold + optimized|27|4397|此前已验证的原计划|
|新target + 原optimized的27个切分位置|27|1599|固定位置重新规划scale并验证|

新局部greedy比原greedy少43次Boot，约23.6%。但目前没有得到少于原optimized的27次Boot的可行计划。固定原27个位置后，rescale从4397降到1599，约少63.6%，Boot数量仍为27。

rescale统计是RuntimePlan中的操作数量；native模式下每个操作除去一个Q。这不是关键路径深度，也不是GPU性能测试。

两种greedy都使用“当前计算放不下时刷新输入”的局部思路，但管理单元不同：原策略按单次乘法判断，新策略按共享值之间的算术树规划，失败时按消耗Q从多到少刷新输入。因此这不是固定所有Boot决策、只替换scale公式的严格隔离实验；不能据此将139与原optimized的27直接比较算法优劣。

## 尝试从原27次中删除Boot

使用原optimized输出的`btp_schedule`，在新策略中复用同一组原始opid位置。该验证模式禁止自动添加额外Boot。

原27个位置在新策略下通过完整编译、物理level/scale转换和C++ RuntimePlan验证。随后依次尝试删除每一个位置，共27份26-Boot候选；当前规划器均未找到满足预算和scale约束的计划，因此最终保留27个。

这只说明在这些固定位置、当前有限scale搜索及预算下，未能直接删去其中一次Boot。没有同时移动切分位置，也没有用target策略重新运行DaCapo optimized搜索，不能把27解释成新策略的最优下限。

## 验证

三个新生成的完整计划均通过C++校验，每个Boot都满足输入runtime level=5、s≈40，输出level=30、s=40；s=log₂(scale)，对应Q6输入和Q31输出。新target自动计划有17421个值，固定27位置计划有17312个值。

11项target检查通过，包含连续明文乘法、自动插入、固定schedule、空schedule不补Boot、输入输出条件及明确拒绝行为。原25项实际模数规划和9项手动Boot对齐检查通过。此前GPU测试的`ReLU → Boot`、`ReLU → Boot → ReLU`计划重新生成后字节相同；本次不重新生成GPU Boot资源。

测试过程中曾加入一个会提前固定系数和scale的连续明文乘法实现，造成回归，计数为238。该版本已修复并废弃，没有纳入本报告比较。最终实现对未知的嵌套明文乘法输入设定waterline目标并继续倒推，保留原ReLU系数和的可调整性。完整GPU计划一致性检查防止这一回归再次混入结果。

## 本次实现和限制

- target增加局部greedy自动插入：按当前算术树的共享输入状态决定刷新，刷新后继续倒推；全部刷新仍不能规划时直接报错。
- 增加`--native-target-bootstrap-schedule=PATH`，JSON条目为`[cut,target1,...]`，复用现有BootstrapPlacement。该模式只验证指定位置，不自动增加Boot。
- 补齐完整网络最后线性层出现的连续明文乘法。默认threshold路径及其模数预算保持原样。
- target尚未接入DaCapo的候选段覆盖分析、段成本评估及optimized搜索。减少片段内rescale还不等于已经降低整网最优Boot次数。

## 产物

[汇总](target-boot-count/summary.json)、[原greedy完整结果](target-boot-count/threshold.json)、[新局部greedy完整结果](target-boot-count/target.json)、[新策略固定27位置结果](target-boot-count/target-fixed.json)、[27次删除尝试](target-boot-count/deletion-results.json)、[原始27位置](target-boot-count/original-schedule.json)、[既有GPU计划一致性](target-boot-count/gpu-plan-unchanged.json)。

复现脚本：[自动计数](target-boot-count/count.py)、[固定位置计数](target-boot-count/count-fixed.py)、[逐个删除](target-boot-count/delete-fixed.py)。脚本中的路径对应188Server已有离线工具链和数据。

远端完整RuntimePlan、明文bundle和日志保留在：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/automatic-boot-count/
```
