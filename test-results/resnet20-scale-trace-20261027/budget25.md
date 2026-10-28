# 25Q 预算、手动阶段 scale 与 ResNet20 Boot 对照

2026-10-28，在 `experiment/native-scale-planning` 分支继续实验；编译、完整模型规划和 GPU 测试均在 188Server 执行。代码尚未 commit/push。本报告接续 [23Q 时的统计](target-boot-count.md)，下列结果不能与旧预算混用。

已得到完整原始 ResNet20 的 **18 次自动 Boot**：提供阶段 scale 约束，不提供 Boot schedule，输出 RuntimePlan 与指定手动 18 个位置的计划逐字节相同。它仍是带调用方 scale 约束的局部 greedy，尚未接入 target + DaCapo optimized，也不是编译器自行发现了这些 scale。

## 25Q 如何使用

Qn 表示有效前缀有 n 个素数，Runtime level=n−1；s 表示 log₂(scale)。实际模数链仍为同一份手动 Q50/P25 fixture，所有 Q 素数位宽不超过 32 bit。

Boot 仍要求 Q6/s40，返回 Q31/s40。之前对所有片段统一预留两个 Q，因此只给算术使用 23Q。现在由规划和物理转换共用同一计算：

- 已到 s40：入口对齐不做乘法或 Rescale，Q31→Q6 的 **25Q 可全部用于算术**。
- 未到 s40：寻找满足至少 30 bit 明文编码精度的最少 Rescale 数。所需编码 scale、实际 Q 和乘法前容量均检查；可能需要一个、两个或更多 Q。
- Q 或容量不足即拒绝。没有直接修改密文 scale 元数据，也没有绕过实际除 Q。

因此不是把预算常数从 23 改成 25 就结束，而是让候选切分点检查该值能否完成真实入口对齐。

## 完整模型结果

|输入和规划|Boot|Rescale 操作数|验证|
|---|---:|---:|---|
|旧 threshold + optimized，23Q|27|4397|此前已验证，历史基线|
|threshold，25Q，每个切分点只留一个最低延迟状态|28|4255|C++ PASS，约194秒|
|同上，先最少 Boot 再最低延迟|28|4255|C++ PASS，约193秒|
|target，无手动 scale 约束，初始 Q31|38|1686|C++ PASS|
|target，手动阶段约束，初始 Q31，指定19个位置|19|1219|C++ PASS|
|target，手动阶段约束，初始 Q34，指定18个位置|18|1218|C++ PASS|
|target，同一约束，初始 Q34，**自动选位置**|**18**|**1218**|C++ PASS，与上一行计划字节相同|
|target，P2 出口改为 s50，初始 Q34，自动选位置|18|1142|C++ PASS，额外候选|

25Q 的两个单状态 DaCapo 对照使用 threshold，并未使用新 target 策略；它们是远端独立二进制，未替换当前主规划器。预算、可行性判定和候选剪枝改变后，得到28而非旧27，不意味着原问题的最优值升高。只增加 Q 不会自动修复不合适的 scale 和候选选择。

手动约束包括 19 次 ReLU 的 P1/P2/尾乘出口，以及卷积共享中间结果的高 scale。固定切分不单独指定这些 scale 时，18次方案不能通过。因此改进主要来自让算术本身装进片段，而非减少搜索状态数量。

完整 18 次自动计划保留 513 次 MulCC（每次 ReLU 27次，共19次）、1709次 Rotate。所有 Boot 都是 Runtime level5/s≈40 输入、level30/s40 输出。末端输出为 Q5/s≈40，可以用掉不再需要为下一次 Boot 保留的 Q。

[自动18结果](budget25/automatic/manual34/result.json)、[固定18结果](budget25/full18-input34/result.json)、[初始Q34复核](budget25/input-prefix-v2/results.json)、[P2=s50候选](budget25/automatic/manual34-p2-50/result.json)。

## 为什么初始要 Q34

手动首层接收27个已做 im2col 的密文，在 Q32 输入后用1Q得到 Q31。当前编译器图接收一个打包密文，首层包括旋转和掩码；本次阶段约束下需要3Q。两者的原始输入接口不同。

本次完整18次方案使用同一 Q50 链的更长初始前缀：

```text
初始 Q34/s40 → 打包 stem 3Q → Q31/s40
Q31/s40 → ReLU 22Q → Q9/s40 → 后续卷积 3Q → Q6/s40 → Boot → Q31/s40
```

`--input-level=33` 只设置最初输入。Boot 输出仍是 level30，不会在每次 Boot 后凭空多给3Q。检查点分别记录初始锚点和 Boot 出口。

另截取原图 stem 之后的算术，从 Q31/s40 开始，**不提供 Boot schedule 也得到18次**。这是去掉首层的子图，不能称为完整输入推理，但能在相同 ReLU 入口条件下比较网络主体。

初始 Q31、Q32、Q33 的固定18位置方案均失败；同一局部 greedy 分别给出50、47、47次。这是该阶段约束和局部刷新选择的结果，不是这些输入前缀下的数学下限。Q31 的指定19次方案已经通过，说明 greedy 本身仍有浪费。此次没有改写输入为手动27密文 im2col 接口，也没有证明18为全局最优。

## Scale 扫描与精度

扫描 P1∈{40,40.1,41,44,48,50}、P2∈{40,41,44,48,50,60}，尾乘出口 s40，系数编码精度至少40 bit；36组均通过编译。消耗分别为22Q两组、23Q九组、24Q十二组、25Q十三组，未找到21Q。这是有限搜索结果。

两组22Q为 P1=s40.1，P2=s44或s50。前者有59个 Rescale 操作，后者55个；**操作总数与关键路径 Q 消耗不同**。把整网19次 P2 出口均改为s50，Rescale 从1218降至1142，Boot仍为18；它是另一个算术 scale 候选，不应标为逐项完全复制手动版。

GPU 测试使用 188Server 物理 GPU2、Q50/P25 手动 fixture、32768 slots，对比计划对应的独立明文算术参考，阈值1e−4。目前已验证：

|路径|最大绝对误差|结果|
|---|---:|---|
|P1=s40.1/P2=s44 的单次 ReLU，Q31→Q9|7.16743e−7|PASS|
|打包 stem，Q34→Q31|4.50228e−6|PASS|
|ReLU → 真实 Boot|3.70497e−5|PASS|
|ReLU → 真实 Boot → ReLU|2.52909e−5|PASS|
|P1=s40.1/P2=s50 的单次 ReLU，Q31→Q9|8.96771e−7|PASS|

Boot 测试的参考包含手动 profile `arcsine_degree=0` 的正弦近似映射。相对理想恒等 Boot 的最大偏差约0.00454，不能将上表执行误差误认为恒等映射误差。见[单ReLU](budget25/export/one-relu-manual-gpu-report-summary.json)、[stem](budget25/export/stem-gpu-gpu-report-summary.json)、[ReLU→Boot](budget25/export/boot22-relu-boot-gpu-report-summary.json)、[Boot后继续ReLU](budget25/export/boot22-relu-boot-relu-gpu-report-summary.json)、[P2=s50](budget25/export/p1-40.1-p2-50-gpu-report-summary.json)。

完整 ResNet20 的18次结果是编译和 RuntimePlan 校验，尚未执行整网密态分类，也没有测量整网 GPU 延迟或精度。

## 实现与回归

主要修改：

- [BootstrapInputAlignment.h](../../third_party/ckks-runtime/third_party/dacapo/include/hecate/Support/BootstrapInputAlignment.h)：规划与物理转换共享按实际素数计算的入口对齐。
- [Common.cpp](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/Common.cpp) 及 greedy/depth-dp/DaCapo 候选检查：按值判断能否 Boot；初始 level 与 Boot 重置坐标分别处理。
- [TargetScalePlanning.cpp](../../third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/TargetScalePlanning.cpp)：允许共享分支做编码1+Rescale对齐；保留嵌套明文乘法的高 scale；旋转/取负保留自然 scale；刷新线性分支时优先刷新共享源；加入显式阶段 scale 约束。
- [生成脚本与用法](../../third_party/ckks-runtime/integrations/dacapo/README.md)：增加初始level选项，记录实验性 scale 约束和适用范围。

15个全预算/初始前缀检查、11个 target 检查、25个实际素数规划检查、9个手动 Boot 对齐检查、18个 native 产物检查，以及540个旧 threshold 图检查均通过；后者为495份可执行计划和45个预期拒绝。Ruff、Python语法和 Git whitespace 检查通过。没有扩大数值容差来让失败候选过关。

本次实验收尾时，多状态搜索原任务仍在远端运行，已超过14小时、单核满载、RSS约5.9GB，尚未给出最终规划结果。之后已按用户要求停止该任务，并将单状态搜索替换为默认实现，见[后续记录](single-state.md)。此次两个单状态对照约193秒、峰值约1.29GB；带阶段约束的局部规划约9～14秒。它们的候选和优化目标不同，不能把时间差当作同等搜索语义的纯性能提升。

## 复现产物

原始输入 bytecode SHA256：`d29b1693f705976e75aa02b146251040811418ded92ed2dfe3ce2e6e21350816`。完整自动18 RuntimePlan SHA256：`77b399e32e9288a5c96f3812b3529de72b02ec257363bc6fe1cdbecfa7e18a28`。

[阶段约束](budget25/manual-cuts/manual-stem-goals.json)、[固定18位置，仅作对照](budget25/manual-cuts/manual18.json)、[36组扫描](budget25/relu-grid/results.json)、[复现脚本](budget25/scripts/poseidon-budget25-auto.py)、[远端离线构建脚本](budget25/build/remote-build.py)、[产物清单与校验和](budget25/manifest.json)。自动18命令明确不含 `--native-target-bootstrap-schedule`；完整命令在结果JSON中。

完整原始图、RuntimePlan、bundle、全槽位GPU报告和日志保留在：

```text
188Server:/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/budget25/
```

本地只保存较小的约束、汇总、日志及复现脚本，不复制GB级模型数据。约束中的opid依赖上述原始图及前端优化，换图后必须重新匹配。
