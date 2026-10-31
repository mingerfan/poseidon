# 18次与27次Boot的scale开销定位

2026-10-28。结论：当前 DaCapo optimized 在选 Boot 切分点，每个候选内部都使用同一套 threshold 规则；它不搜索手动版那样的系数编码scale、保留高scale的共享中间值、合并后的Rescale位置。换模数链改变了局部减掉多少bit，却没有提供这些决策。当前18位置对照在该规则下确实超预算，不能只归因为Boot搜索漏选。

本轮新增诊断脚本与结果，生产编译器代码沿用上一轮状态；target未恢复到framework。所有重放与编译在188Server进行。

## 公平对照

比较此前保存的带手动阶段scale约束的18次自动计划，与当前threshold单前缀optimized的27次计划。两者都使用完整原始ResNet20、初始Q34/s40、同一手动Q50/P25实际素数、N=65536、Boot Q6/s40→Q31/s40及25Q预算。因此这里threshold的Rescale总数是4339，不能与初始Q31时的4397混用。

18次方案由调用方提供阶段scale约束，内部由实验性target规划器完成；不是DaCapo自己搜索出所有scale，也不是每个内部原语都逐项复制手动实现。它证明同一图与模数链存在18次可行规划。完整密态分类仍未验证。

[compare_plans.py](compare_plans.py)按RuntimePlan构建算术表达式DAG：常数按原始明文数据SHA256匹配，保留旋转和算术依赖，去除Boot/Rescale/ModSwitch/Relinearize/传输及编码1乘法，交换律操作数归一化。已断言完整出口与19个ReLU出口的算术DAG全部一致。两者均为513个MulCC、2775个非单位MulCP、1709个Rotate、3677个AddCC、229个AddCP、76个Negate。该等价不包含明文编码舍入误差。

|操作计数|手动阶段scale，18次|threshold，27次|增量|
|---|---:|---:|---:|
|Rescale总数|1218|4339|3121|
|ReLU内部Rescale|1121|1878|757|
|ReLU外Rescale，主要是卷积|97|2461|2364|
|编码1的MulCP|380|828|448|
|ModSwitch|254|737|483|
|Boot|18|27|9|

约75.7%的额外Rescale操作位于ReLU之外。**这是全分支操作总数，不是关键路径多消耗3121个Q**。原语融合的GPU收益也没有计入本比较。

ReLU范围通过19个`x*(P3+0.5)`的实际依赖回溯确定，截止各自入口的x；包括其内部Boot、保存x分支的管理操作和尾乘后的Rescale。其余计算单独归为ReLU外。各范围互不重叠，计数加总覆盖全部计算步骤。

## 1. 卷积：先逐乘积Rescale，失去求和后共享一次的机会

完整网络ReLU外有2205个相同的非单位MulCP：

|这些MulCP之后的立即Rescale数|手动阶段scale方案|threshold|
|---|---:|---:|
|0|2172|0|
|1|33|1965|
|2|0|240|

threshold对这2205个乘积做了2445个Rescale；手动阶段方案只在其中33个后面立即Rescale，其余多在加法归约后统一处理。对应到整个网络，手动阶段方案有1071个Rescale的最近算术上游是加法，threshold为0。这里连续单Q Rescale分别计数，最近上游的查找会跳过已有Rescale、ModSwitch和Relinearize。

一个入口条件完全相同的首层权重乘法：

|操作|手动阶段scale方案|threshold|
|---|---|---|
|密文输入|Q34/s40|Q34/s40|
|相同权重的明文编码s|54.976558|40|
|乘积s|94.976558|80|
|立即Rescale|0，保留以供累加|1，得到Q33/s48.007504|

在第一段主体卷积上，两份整网计划有相同88个MulCP、112个AddCC、56个Rotate。手动阶段方案只有3个Rescale，threshold有104个；后者入口已经被前面的scale管理改变，分别为Q9/s40与Q21/s58.197757。为避免入口不同的干扰，另外截取同一真实卷积，从Q9/s40单独重放threshold，仍有88个Rescale操作，最长路径消耗2Q，输出Q7/s56.018906。

手动阶段方案的对应卷积是Q9/s40→Q6/s40，消耗3Q并可直接Boot。threshold虽然局部只消耗2Q，却没有得到s40：按现有至少30bit单位明文精度，对齐s56.018906还需要2Q，总共4Q，超过Q9到Q6可用的3Q。这说明“局部少Rescale一个Q”可能让下一个边界更差。

首层也有同样的现象：手动阶段方案Q34/s40→Q31/s40，threshold为Q34/s40→Q32/s56.015141。后者保留了一个额外Q，却把更高的scale送进了ReLU。

## 2. 多项式：系数固定40、过早归一化和合并时上调

当前[MulOp的操作数处理](../../third_party/ckks-runtime/third_party/dacapo/include/hecate/Dialect/Earth/IR/EarthOps.td:465)先把系数明文编码s设为waterline=40。[Common.cpp](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/Common.cpp:49)在乘法后反复删除Q，直到再删一个会低于40；[applyThresholdScale](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/Common.cpp:132)只对Mul/Boot调用归一化，并未搜索加法归约之后的Rescale位置。

在还有可用Q时，这个循环结束后满足：

```text
s_out < 40 + log2(q_next) < 72   （每个q≤32bit）
```

而原手动ReLU会保留H14的s≈100.0315、H4的s≈130.0692等中间结果，等待父节点使用。GPU32并不禁止这样的scale；scale由整个RNS前缀承载。禁止保留它们的是当前局部归一化规则。因此仅改变≤32bit的q，无法让当前规则逐项复现这些手动中间状态。这不是“任何其它低scale方案都不可能18次”的证明。

此前相同输入Q31/s40的逐节点跟踪如下，Q消耗按阶段出口累计：

|检查点|原手动ReLU消耗Q／s|threshold消耗Q／s|
|---|---|---|
|P1出口|6／40.100000|6／60.069203|
|P2出口|12／44.000000|14／41.758472|
|P3出口|19／89.871122|23／68.826002|
|ReLU最终出口|22／40.000000|25／44.845572|

P1最初并没有多耗Q，问题先表现为出口scale不适合后继；进入P2后才多消耗2Q。P3再多消耗2Q，尾乘少消耗1Q，合计多3Q。threshold最后还不是s40，已到Q6而没有入口对齐余量。详细节点和系数见[原逐步对照](../resnet20-scale-trace-20261027/comparison.md)。

合并分支也受到整数类型对齐的影响。例如T3实际s≈60.003349，类型向上取整为61；另一条校正分支被上调到61，物理转换又补一个小数scale的×1。原手动实现则在共同的高scale完成校正后再一起Rescale。原手动单ReLU有5个×1管理乘法，threshold有42个；这既增加操作，也可能带来低精度编码误差。

[AbsorbCoefficientScale](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/OptimizePolynomialArithmetic.cpp:110)只吸收已经产生的上调，且明确保留原有立即Rescale数量。这是局部清理，不会倒推全树scale或把多个分支的Rescale合并到求和之后。

## 3. Boot入口：每段出口未对齐，继续占用Q

18次方案的18个入口均已达到s40，无需额外Rescale对齐。27次方案有21个入口需要2Q、6个需要1Q，共48个入口Rescale，使用编码精度至少30bit的单位明文乘法。该项已包含在上面的总数中，不能重复累加。

这48个操作占总额外Rescale的比例不大，但对贴近25Q上限的单个片段很关键。并非统一预留两Q导致浪费：当前按各值实际scale按需计算，18方案能用满25Q，threshold则经常必须提前留出一到两个Q。

2026-10-30复核：上述“至少30bit”是编译器在`BootstrapInputAlignment.h`中硬编码的保守策略，不是Boot profile本身的输入条件，也不是数学上必须消耗2Q。从Q7/s56.018905618出发，一次对齐可以将scale抬到s71.990059633，再除当前q=4265476097，得到Q6/s40。所需单位明文编码s≈15.971154，因低于30被当前策略排除。仅按标量常数`round(2^s)/2^s`估算，此例乘1的相对偏差约5.41e−6；这不是完整CKKS误差或Boot误差测试。因此这里准确的结论是“现有门槛判定需要2Q”，一次Rescale候选的可用性需要真实精度验证。此前48个入口Rescale及固定位置失败统计均是在原30bit门槛下得到的。

## 4. 固定手动位置，直接验证片段是否可行

[build_audit_driver.py](build_audit_driver.py)只在远端独立目录构建诊断driver，复用当前编译器的pass静态库。增加的代码读取外部schedule、设置与当前native路径相同的参数并标记原始opid；未改变threshold、Boot预算或物理算术规则。[fixed_schedules.py](fixed_schedules.py)跳过搜索，直接重放固定位置。

|固定位置与输入|结果|
|---|---|
|现有27位置，原链、初始Q34|通过；出口s66.9485344932141，与现有27计划一致|
|手动18位置，原链、初始Q34|首个ReLU的P3内原始opid263超Q预算|
|手动18位置，全32bit工作链、初始Q34|首个ReLU尾乘opid291超Q预算|
|手动18位置，混合链`mix-30-20-2`、初始Q34|首个ReLU的P3内opid254超Q预算|
|去掉stem，从Q31/s40开始，手动18位置、原链|首个ReLU后第一个卷积权重乘法超Q预算|

最后一行排除了首层入口scale的影响：相同Q31/s40输入下，threshold已把25Q用在ReLU上，随后卷积的第一个乘法就需要第26个Q；原手动22Q ReLU后还有3Q可以做卷积。

这些结果说明当前算术规则不能容纳相应的手动片段。它们不证明所有18位置方案都不可行，也没有精确分摊“多出的9个Boot中几个归因于算术、几个归因于搜索”。当前严格单前缀搜索仍可能丢失更好的状态，见[此前剪枝审计](../resnet20-scale-trace-20261027/single-state.md)。但增加搜索状态本身不能让上述失败片段变可行。

18方案的Boot均位于阶段入口；27方案包含19个多项式内部Boot及8个外部边界Boot，位置整体发生变化，净增加9。不能把19个内部Boot都算作在原18位置上额外添加。

## 修复方向

优先让scale规划器能为一个算术区域联合决定系数编码scale、合并点和Rescale位置，再让Boot规划器调用该区域规划器判断25Q是否可行。至少应覆盖卷积的权重乘积与归约、多项式叶项与父节点，以及片段出口的Q6/s40合同。实际浮点scale应参与分支对齐，避免先按整数上调再补低scale的×1。

这不要求恢复多状态穷举。可以先保持每个切分点一份前缀，用有限的区域scale候选改善片段本身；候选还需检查编码精度与实际模数容量。先修复能复现的“同图、同25Q片段装不下”问题，再比较相同Boot规划算法的次数。

## 证据与限制

- [整网对比汇总](evidence/comparison.json)，[手动阶段方案详细计数](evidence/manual_scales_18.json.gz)，[threshold详细计数](evidence/threshold_27.json.gz)。
- [固定位置命令与原始诊断](evidence/fixed-schedule-results.json)，[独立卷积状态](evidence/conv-summary.json)。诊断卷积临时允许更低算术下界以测量深度，不是合格的生产Boot计划。
- [诊断driver源码](evidence/audit-optimizer.cpp)，[静态pass库SHA256](evidence/pass-archive-hashes.json)，[诊断二进制SHA256](evidence/audit-binary.sha256)。
- 完整原始图SHA256：`d29b1693f705976e75aa02b146251040811418ded92ed2dfe3ce2e6e21350816`。当前生产编译器SHA256仍是`395e6fb9178bb673a197f51de79c2964da7ddb0fc76a7f0dbea7c2b9d7eb7060`，本轮诊断driver是另一个独立二进制。
- 本轮没有新增GPU执行或修改精度门限。此前手动阶段单ReLU/Boot片段通过、threshold单ReLU未过门限的记录见[预算及精度实验](../resnet20-scale-trace-20261027/budget25.md)和[模数链实验](../resnet20-chain32-fixed25-20261028/README.md)。

远端完整诊断产物位于：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/scale-overhead-20261028/
```

本地仅保存压缩后的计数和小型诊断。构建脚本复用当前远端离线工具链；应先核对静态库SHA256，不要将之后其它版本的库混入本次比较。
