# target scale策略的真实Boot测试

2026-10-28，在188Server的物理GPU2（V100 32GB）完成。分支仍为`experiment/native-scale-planning`。

两条路径都执行了真实native Boot并通过解密检查：一次完整15/15/27 ReLU接Boot，以及ReLU接Boot后再执行一次完整ReLU。两条路径各有1次显式Boot。使用原手动Q50/P25素数，全部Q/P均≤32 bit，没有更换模数链或Boot算法。

## 输入输出与scale

这里s表示log₂(scale)，Q表示当前有效素数数量，runtime level=Q−1。

|完成位置|Q|runtime level|s|
|---|---:|---:|---:|
|第一个ReLU的输入|31|30|40|
|第一个ReLU完成|8|7|40.000000000993|
|Boot输入对齐完成|6|5|40.000000000993|
|Boot完成|31|30|40|
|第二个ReLU完成|8|7|40.000000000993|

Boot前只增加一次ModSwitch，从Q8降到Q6，没有增加×1、upscale或rescale。输入scale与40相差约9.93e−10，在既有1e−8容差内，没有强行改写密文scale元数据。Boot出口遵守手动profile的Q31/s40约定；后续算术重新从该状态规划，每次ReLU消耗23Q。

## 密态结果

输入样本−0.75、−0.5、−0.25、−0.125、0、0.125、0.25、0.5、0.75、0.875重复填满32768个槽位，最大误差统计覆盖全部槽位。

参考为相同ReLU多项式的实数计算，以及现有手动Boot的正弦映射。该手动profile使用arcsine_degree=0，参考不是严格恒等Boot，也不是理想ReLU。

|路径|Boot次数|最大绝对误差|最大虚部|门槛|
|---|---:|---:|---:|---:|
|ReLU → Boot|1|2.4191228245e−5|4.5124311902e−6|1e−4，通过|
|ReLU → Boot → ReLU|1|2.3499755013e−5|2.3066556582e−6|1e−4，通过|

若将Boot当作严格恒等映射，两个测试的最大偏差分别约为0.00452991、0.00452775。这包含手动Boot正弦映射自身的近似偏差，不能把上表误差解读成相对恒等映射的误差。

测试进程总耗时340.30秒，包含上下文、密钥、Boot变换资源准备和两次完整计划执行。前5分钟日志仍处于资源准备阶段。两组测试在同一进程复用context和Boot资源；该时间不是一次Boot的在线执行时间，也不能用于宣称加速比。

## 本次代码变化

- `RemoveBootstrap.cpp`：target模式保留输入图中的显式Boot边界。默认threshold模式仍执行原有删除与重新规划流程。
- `TargetScalePlanning.cpp`：将Boot的输入和输出设为算术片段边界；复用现有`applyThresholdScale`核对Boot出口约定，并用刷新后的level/scale继续规划。没有扩展Boot DP搜索状态。
- `GreedyBootstrapPlacement.cpp`和`Common.h`：向target规划传递所选profile的Boot输出scale，输出实际Boot数量。
- `dacapo_target_scale_tests.py`：加入显式Boot前后算术、实际Boot数量及输入输出约定检查，以及出口scale冲突拒绝检查。集成说明同步更新。

## 验证和范围

8项target检查、25项实际素数规划检查、9项手动Boot对齐检查均通过。最终编译器重新生成的两份Boot计划与真实GPU执行的计划字节相同。未插Boot的完整ReLU在target与threshold模式下分别与修改前的计划逐字段相同。Python检查通过ruff和语法检查，Git diff空白检查通过。

本次证明显式Boot与新scale策略可以衔接。这组密态测试没有运行完整ResNet20的卷积和分类流程。后续自动插入的完整计划计数见[target Boot次数实验](target-boot-count.md)，没有整网分类精度结论。

## 产物与复现

[GPU摘要与前10槽样本](target-boot/gpu-summary.json)、[校验记录](target-boot/checks.json)、[实际GPU命令](target-boot/gpu-command.json)、[GPU日志](target-boot/gpu-run.log)、[测试输入生成与计划验证](target-boot/test.py)、[GPU探针构建脚本](target-boot/build-gpu.py)、[探针源码](target-boot/gpu/target_boot_gpu.cpp)。

两份原始输入图为[ReLU → Boot](target-boot/relu-boot.mlir)和[ReLU → Boot → ReLU](target-boot/relu-boot-relu.mlir)。它们复制此前实际trace的ReLU算术，只增加显式Boot及第二段同样的ReLU。完整RuntimePlan和全部32768槽位的解密结果保留在：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/boot-tests/
```

生成仍使用`--scale-strategy=target --boot-placement=greedy`。两份[编译命令之一](target-boot/relu-boot-command.json)和[另一份](target-boot/relu-boot-relu-command.json)记录了实际参数；profile与之前无Boot实验相同，见[OperatorSpec](target-strategy/production-operator-spec.json)。GPU测试保留native Boot资源构建与配置，不能用此前移除了Boot配置的单ReLU探针替代。

计划SHA256：

```text
ReLU → Boot:        65035ca9da87c0bb82c17dcc3670209eb28721989d4652e3ec174936baa1d3ae
ReLU → Boot → ReLU: 6bf7dc8f4c8777eed00d546634ae9e7b4ea9594972554b571e203285f779e41a
```
