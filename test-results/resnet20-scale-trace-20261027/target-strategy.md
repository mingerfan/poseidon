# 实验分支：目标scale规划

分支：`experiment/native-scale-planning`。Poseidon、CKKS Runtime和Dacapo三层仓库均已建立该分支，之前未提交的latency表引用和Boot规划缓存改动保留。本次scale策略没有扩展原有Boot DP搜索状态。

已实现并通过真实GPU解密测试的原型入口是`--scale-strategy=target --boot-placement=greedy`。默认仍是`threshold`，在最终编译器中，默认策略导出的整份ReLU RuntimePlan与旧计划逐字段一致。

## 策略与代码

- 以共享密文为边界，按原IR依赖对单使用的算术树倒推目标level和实际scale，为系数重新选择编码scale。没有读取手动见证表或写入15/15/27专用调度。
- 加法树优先在合并后rescale；局部已知基底可延后到下一次乘法前处理。对可调整的共享出口，有限尝试waterline到waterline+20、步长0.1的目标，并只保留当前选定方案，没有进行全局最优搜索。
- 实际浮点scale与整数类型分开传递。系数、非零×1对齐的编码精度至少为waterline；s=0的精确常数1仅用于兼容整数类型。固定结构不得被后续canonicalization或EarlyModswitch改写。

延后rescale、跟踪真实scale的思路参考[OpenFHE官方FLEXIBLEAUTO示例](https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/advanced-real-numbers.cpp)，编译期管理也参考[EVA](https://github.com/microsoft/EVA)。这是本仓库的有限搜索原型。

主要实现：[TargetScalePlanning.cpp](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/TargetScalePlanning.cpp)，运行与限制见[集成说明](../../third_party/ckks-runtime/integrations/dacapo/README.md)，检查脚本为[dacapo_target_scale_tests.py](../../third_party/ckks-runtime/tests/dacapo_target_scale_tests.py)。配套修改还使Earth常量和Upscale的实际编码scale进入CKKS物理化，而不是被整数类型覆盖；默认路径没有这些新属性。

## 同一ReLU对照

测试是此前实际trace中的一次完整15/15/27 ReLU，输入Q31、s=log₂(scale)=40，使用原手动Q50/P25素数，全部Q/P均≤32 bit。计算图、主要乘法的依赖和系数均不变，未执行Boot、卷积或整网。编译、规划和GPU运行均在188Server完成。

|项目|旧编译器策略|新target策略|手动执行参考|
|---|---:|---:|---:|
|主要CC乘法|27|27|27|
|系数CP乘法|30|30|30|
|全部rescale除素数次数|94|66|55|
|×1乘法|42|21|5|
|关键路径消耗Q|25|23|22|
|输出Q|6|8|9|
|输出s|44.845572|40.000000001|40|

rescale统计覆盖所有并行分支，不是关键路径深度，也不能直接当作CUDA kernel数量。手动多素数rescale按实际除去的素数展开。新策略的21个×1中，6个确实提高scale，编码s均≥40；其余15个使用s=0的精确常数1，保留为整数类型兼容操作。它们也计入成本，没有从统计里隐藏。

新策略各阶段的完成状态包含最终加法之后的rescale：

|阶段|新增Q消耗|完成Q|完成s|
|---|---:|---:|---:|
|P1(d15)|6|25|40.100000001|
|P2(d15)|6|19|41.900000001|
|P3(d27)|7|12|121.861846847|
|尾部|4|8|40.000000001|

目前仍比手动多1Q：手动阶段消耗为6/6/7/3，本原型为6/6/7/4。中间scale高不等于计划更差，比较应包含后续处理和最终接口；阶段目标的联合选择仍有改进空间。

## 密态精度与Boot输入条件

GPU测试使用物理GPU2、V100 32GB，原Q50/P25实际上下文。输入为−0.75、−0.5、−0.25、−0.125、0、0.125、0.25、0.5、0.75、0.875，重复填满32768个slot；参考是同一多项式的实数计算。没有把多项式近似ReLU本身的误差混入执行误差。

旧编译器方案最大绝对误差为 **0.033630883**；新策略为 **8.80108064e-07**，最大虚部为 **7.76650117e-07**，通过1e−4门槛。最终构建重新导出的计划与GPU测试的计划字节相同，摘要为`sha256:23970e2a4604ce311f0d32038258e37725176649b05a11e457a9de6d8cfd18fa`。测试未测量可复用的GPU加速比。


在正式手动Boot profile保守预留2Q后的23Q预算内，使用用户CLI参数也能完成编译并通过C++ RuntimePlan验证。新出口Q8/s≈40只需一次ModSwitch到Q6，不需要额外×1或rescale，已用实际物理化pass核对。本节记录当时的Boot输入对齐验证；后续真实Boot接入与测试见[target Boot实验](target-boot.md)。

## 验证与适用范围

- 57个主要数学乘法逐一对应，系数和主要依赖相同。
- 6个新策略检查：线性系数、±1系数、共享基底、预算拒绝、外部明文拒绝、非支持Boot模式拒绝。
- 默认路径：540个图检查（495份计划、45个预期拒绝）、25个实际模数规划、9个手动Boot对齐、18个native产物检查均通过。
- 最终源码通过diff空白检查；新增Python测试通过ruff和语法检查。

原型支持能完整放入现有Q预算的算术片段，以及用显式Boot连接的多个片段。明文须局部编码，可倒推的CC乘法至少一侧有确定的scale；不支持的图或预算会直接报错。局部greedy自动插入后的完整ResNet20计数见[target Boot次数实验](target-boot-count.md)。DaCapo optimized和depth-dp尚未接入target联合规划，未声称全网密态精度通过。

[比较数据](target-strategy/comparison.json)、[详细计划统计](target-strategy/one-relu-inline-summary.json)、[数学节点核对](target-strategy/differences.json)、[GPU结果](target-strategy/gpu-summary.json)、[检查记录](target-strategy/checks.json)、[Boot前对齐IR](target-strategy/boot-aligned.mlir)。

## 复现

从CKKS Runtime仓库根目录，使用已构建的实验编译器：

```bash
python3 integrations/dacapo/generate_model_artifacts.py mlp \
  --operator-spec /path/to/operator-spec.json \
  --compiler-profile /path/to/compiler.json \
  --hecate-opt /path/to/hecate-opt \
  --traced-mlir /path/to/one-relu.mlir \
  --boot-placement greedy --scale-strategy target \
  --device-counts 1 --plan-format json --output-dir /tmp/target-relu
```

[实际执行命令](target-strategy/target-command.json)。远端完整产物在`188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/`；本地保存了[构建](target-strategy/build.py)、[单ReLU检查](target-strategy/test.py)和[核对](target-strategy/audit.py)脚本。
