# DaCapo 单状态搜索核对与回归

2026-10-28。本轮开始时核对确认：主代码仍是 `map<cut, SmallVector<Plan>>`，按不同出口level/scale保留多个前缀，再对等价状态只留最低成本者。之前合入的是等价状态合并和结构模板缓存，并未把主实现切换为每个切分点一份方案。早先约193秒的单状态测试使用独立实验二进制。

这次已实际改写并测试严格单状态版本：`map<cut, Plan>`，按累计估算延迟替换切分点的唯一方案，携带所有存活值（含bypass）的实际类型和scale。不再遍历每个切分点的多份状态；同时移除单状态下没有重复使用机会的结构模板缓存。源码从234行减到191行。

## 完整网络结果

188Server、同一原始ResNet20、同一实际Q50链、初始Q31/s40、Boot Q6/s40→Q31/s40，预算25Q。使用 **threshold + DaCapo optimized**，不是新target scale策略。

|项目|本次严格单状态|
|---|---:|
|完整编译时间|196.14秒|
|编译子进程峰值RSS|1288248 KiB，约1.23 GiB|
|Boot|27|
|Rescale|4397|
|MulCC|513|
|Rotate|1709|
|C++校验|PASS，21919个值|

单状态二进制SHA256：`fa0914c3e769ab6473a7bed5a3dad6dcf12573c32b89a5f71d27138889745214`。完整计划SHA256：`ef589ad15b5bab5be8038fa325321a529fe9a3403f02a10d58eaa6fff29ca5c5`。

[完整命令和结果](single-state/full-q31/result.json)、[编译计时](single-state/full-q31/compile.log)。这份结果来自本次重编译，不能直接套用先前独立原型的28次统计。

原先运行的多状态进程仍使用旧二进制，截至本次收尾已运行约21小时、单核满载、RSS约8.4GiB，停止前尚未给出完整结果。用户选择严格单状态后，该旧进程已终止。不能仅因修改了源码，就把旧进程看作在运行新算法。

## 单状态实际会漏掉可行方案

旧threshold图测试包含180个optimized用例。参考多状态二进制得到165份有效计划，另15例预算确实不可行，完整测试通过。严格单状态得到159份有效计划、15例正确拒绝，并将额外6例有解输入拒绝为无解。因此本次 **旧图回归没有全部通过**，没有放宽项目测试的断言。

这6例为：

```text
40-q1-optimized-cc-double-rescale
40-q1-optimized-cc-scale60
40-q2-optimized-cc-scale60
45-q1-optimized-cc-double-rescale
45-q1-optimized-cc-scale60
45-q2-optimized-cc-scale60
```

一个具体反例是在工作预算1Q的测试profile中计算：

```text
a = x * 1.01
b = y * 1.01
out = a * b
```

参考版本找到2次Boot的有效方案；严格单状态报无解。这个profile用于短链回归，不是手动ResNet20的25Q profile。

原因是到同一切分点的不同前缀，可能给绕行值留下不同的level/scale。便宜前缀可能保留一个已经耗尽Q的bypass，后续在当前候选刷新集合下无法继续；较贵前缀提前刷新过该值，反而可以继续。记录被选中方案的状态，不能恢复已经被剪掉的另一份前缀。因此“一点一份最低成本”是可能漏解的剪枝，不是保证等价的性能优化。

[完整审计结果](single-state/pruning-audit/summary.json)、[多状态对照](single-state/multi-state-control/summary.json)、[官方测试的原始失败](single-state/threshold.log)、[反例输入](single-state/counterexample/pruning-audit/input.mlir)、[反例的有效2-Boot计划](single-state/counterexample/multi-state-control/result.graph.runtime-plan.json)。外部审计脚本只为继续收集全部180例的结果，仍检查成功计划的数学语义与scale方程，额外拒绝计作失败；没有修改仓库测试。

独立的25个实际模数规划、9个手动Boot对齐、18个native产物、15个预算/初始前缀检查及Boot边界/形状契约检查均通过。未新增GPU测试。

## 当前代码状态

用户已明确选择严格单状态并接受上述剪枝限制，现已直接替换[主实现](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/DaCapoPlanner.cpp)。不保留多状态模式、状态枚举或备用搜索；原来长期运行的多状态任务已停止。没有保留另一份候选源码或切换补丁。

正式图测试将这6个有解但被剪掉的案例单独标为 `expected_single_prefix_rejection`，不与物理无解混淆，其余输入仍严格检查。找不到续接方案时，编译器明确报告 `no feasible DaCapo plan after single-prefix pruning`。

复现文件：[整网脚本](single-state/run-full.py)、[剪枝审计脚本](single-state/audit-pruning.py)。远端保留独立的单状态二进制及wrapper，完整产物位于：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/single-state-main-20261028/
```

`run-full.py --output-dir /new/remote/path` 使用当前默认单状态编译器，输出目录必须是新目录。上表保留首次试验的数据；最终提交版本[复测](single-state/final-full-q31/result.json)为195.69秒、27次Boot、4397次Rescale，计划SHA256与首次试验完全相同。最终编译器SHA256为 `f393ef293185743b782b57764642bf447fc43e3ab2e2c0a3267ed349ffd95c3a`。

最终版本的618项检查通过，其中540个图检查为489份有效计划、45个预算无解及6个明确记录的单状态剪枝拒绝；另有Boot边界和算术语义检查通过。见[最终图检查汇总](single-state/final-tests/threshold-summary.json)及同目录日志。
