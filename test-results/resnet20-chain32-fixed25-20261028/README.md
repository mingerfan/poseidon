# 固定 25Q 的 GPU32 模数链实验

2026-10-28，`experiment/native-scale-planning`。本轮只调整工作区的实际模数与顺序，保持 threshold scale 规则。尚未找到 optimized 或 greedy 达到 18 次 Boot 的模数链；这是一轮有限搜索的结果，不是不存在这种链的证明。

用户确认保持 Boot **Q6/s40 → Q31/s40**，每次 25Q。主比较初始输入为 Q31/s40；另外测试 Q34 初始前缀，只增加首次输入的余量，Boot 出口与后续预算不变。N=65536，完整 Q50/P25、底部 6 个 Q、Boot 上方 19 个 Q 与 P 均保留。每个 Q/P 都是互异的实际素数，满足 `q ≡ 1 (mod 131072)`，位宽均不超过 32。报告中的 s 均指 `log2(scale)`；Qk 表示 k 个素数，对应 RuntimePlan 的 level=k−1。

## 整网结果

使用同一原始 ResNet20 图、waterline=40、同一不含 target 的编译器，分别运行 greedy 与严格单前缀 DaCapo optimized。没有提供目标 scale 或 Boot schedule。下表均为完整模型重新编译的结果。

共生成6421组参数标签，对应6224条不同的实际链，完成6602次阶段诊断（6557次无Boot成功、45次明确失败）。其中挑选23组候选标签做52次整网编译，包含重复链对照和Q34补充对照；Q31的optimized结果为16组27次、7组28次，最低27次。

|工作链|工作区实际 log₂ 总预算|greedy，初始 Q31|optimized，初始 Q31|
|---|---:|---:|---:|
|原手动链|755.466086|182|27|
|只重排原来的素数，`perm-11`|755.466086|145|27|
|全部工作素数为 32 bit，`all32`|799.419848|154|27|
|较省 ReLU 的混合链，`mix-30-6-1`|761.536454|172|27|
|本轮 greedy 最低链，`mix-30-20-2`|789.535902|118|27|

all32 比原链多约 43.954 个实际模数 bit，虽然每个素数仍≤32 bit，并非相同总模数预算。实验同时包含原素数的精确重排、总预算不超过原链的候选，避免把容量增加与顺序改善混为一谈。

初始 Q34 的三个补充对照：原链 184/27，all32 150/27，`mix-30-20-2` 114/27，斜线前后分别为 greedy/optimized。较大的初始前缀没有改善 optimized 的 27 次结果。完整候选、精确模数、命令、计时、计划 SHA256 与 C++ 校验结果保存在 [full-results.csv](evidence/full-results.csv)、[full-results.json](evidence/full-results.json) 和 [candidates](evidence/candidates/) 中。

## 模数链确实影响 rescale，但局部节省没有延续到整个区间

编译器当前按 `s − log2(q_current) >= 40` 决定乘法之后是否继续 rescale。相加时向高 scale 的一侧对齐，不在相加之后主动 rescale。系数从默认 s=40 编码开始，再做局部吸收；没有倒推整棵多项式树的系数编码 scale。

用同一真实单 ReLU 及其后一个真实卷积测量最长路径，得到：

|链|ReLU 消耗 Q|ReLU 出口 s|单 ReLU 加 Boot 入口对齐|ReLU＋后续卷积消耗 Q|再加入口对齐|
|---|---:|---:|---:|---:|---:|
|原链|25|44.845572|27|27|29|
|all32|25|42.935388|27|27|29|
|mix-30-6-1|24|40.033666|25|26|28|
|mix-30-9-1|23|62.948451|25|26|28|

这些深度探针临时把算术下界降到 Q1，并去掉 Boot 输入 scale 合同，以测量“不插 Boot 会用多少 Q”。它们不是可作为整网结果的 25Q 合法计划。所有整网次数均使用原始生产合同，并逐次验证 Boot 输入 level5/s40、输出 level30/s40。

原链在单 ReLU 上的 P1/P2/P3/最终乘法累计消耗为 6/14/23/25Q。改链可以把完整 ReLU 缩到 23Q，但输出的高 scale 仍需要额外 Q 对齐。筛选到的无 Boot“ReLU＋卷积”最少消耗 26Q，加入口对齐为 28Q，超过固定的 25Q。

另一组实验专门把较小素数放在 P1 前部。N=65536 时，2²² 以下可用的 NTT 素数只有 786433、1179649、2752513。前两个相乘的 log₂ 约为 39.755，使默认输入40＋系数40的线性乘积有机会降到约40.245；而只除一个≤32bit素数时会留下至少48。但其他分支也受同一条链约束：本轮把 P1 出口压到约44.702时已消耗8Q，原链则是6Q/s60.069。后续阶段没有保留这一局部优势。该说明只解释此次定向尝试，不是全网下界证明。

原链 optimized 的 27 次 Boot 分布为：P1 边界附近9次、P2边界附近9次、P3边界附近1次、ReLU输入7次、ReLU之间1次。all32仍为19次多项式内部边界附近＋8次入口刷新，部分位置变化。混合链也改变了切分点，却没有减少总数。结构定位使用不变的 8/8/10/1 个 MulCC 分组及 `x*(P3+0.5)`，不是凭源代码行号推测；精确记录见各候选的 `boot-boundaries.json`。

因此，这轮证据支持：**模数链是局部浪费的一个来源，但当前 scale 传播与对齐规则也在限制结果；只调链尚不能恢复手动版的 22Q ReLU＋3Q卷积安排。** 另外 optimized 是每个切分点只保留一份前缀、按估算延迟选取的启发式，不能把27解释成全局最优。

## GPU 数值检查

在 188Server 的物理 GPU2 上，对同一组输入执行真实加密、完整单 ReLU 和解密；不含 Boot 或卷积。参考值为相同多项式的明文结果，不包含多项式近似 ReLU 本身的误差。

|编译器使用的链|最大绝对误差|门限|通过|
|---|---:|---:|---|
|原手动链|0.033630654|0.0001|否|
|all32|0.082825814|0.0001|否|
|mix-30-20-2|0.062025809|0.0001|否|

“原手动链”这一行执行的是编译器计划，不是手动算子实现。此前已经定位过低 scale 的 ×1 明文编码舍入问题，见[前次误差分析](../resnet20-scale-trace-20261027/variable-chain32.md)。本轮未修改该规则。这里的计划结构和实际素数 scale 等式全部通过，不代表密态精度合格；不能把118或27次当作已验证可用的完整密态推理方案。新链对应的真实 Boot 参数及完整网络数值尚未验证。详见 [gpu-results.json](evidence/gpu-results.json)。

## 代码与验证

按本轮要求，从当前 framework 路径移除 target 实现、CLI、专属属性和测试，撤回只为 target 保留结构的改写限制。保留实际素数驱动的 threshold、按需 Boot 输入对齐、25Q预算、独立初始前缀和严格单前缀 optimized。预算测试已改为不依赖 target 的用例，并明确检查已删除选项被拒绝。

目标移除后的607项检查通过：540项图检查包含489份有效计划、45个预算无解和6个已记录的单前缀剪枝拒绝；另有25项实际模数、9项手动Boot对齐、18项native产物、15项预算及删除选项检查。6个剪枝拒绝不能算作物理不可行，详细背景见[单状态审计](../resnet20-scale-trace-20261027/single-state.md)。日志保存在 [checks](evidence/checks/)。

本轮完整计划均通过 C++ RuntimePlan 验证；每个 Rescale 重新按实际 Q 核对浮点 scale 方程。测试的编译器 SHA256 为 `395e6fb9178bb673a197f51de79c2964da7ddb0fc76a7f0dbea7c2b9d7eb7060`。修改的 C++/TableGen/CMake 文件与远端构建覆盖层逐文件匹配，见 [source-hashes.json](source-hashes.json)。本轮改动未 commit/push。

## 复现与文件说明

所有编译和 GPU 测试都在188Server运行。本机仅保存小型结果。远端完整模型、bytecode、RuntimePlan、明文bundle与slot数组保留在：

```text
/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment/chain32-fixed25-20261028/
```

[summary.json](evidence/summary.json) 记录最终样本数和限制；[candidate-manifest.json.gz](evidence/candidate-manifest.json.gz) 记录全部候选的精确素数；[diagnostic-results.json.gz](evidence/diagnostic-results.json.gz) 保存阶段、无Boot深度、对齐余量和失败结果，不能与整网次数混用。

本目录 Python 是实验脚本，不进入编译器。`chain32_probe.py` 使用上述远端固定数据路径；将本目录脚本放在同一个远端目录后运行。输出阶段目录必须不存在，避免覆盖原始证据。例如对一个尚未编译整网的新候选分别执行：

```bash
python3 chain32_probe.py CANDIDATE full --strategy greedy
python3 chain32_probe.py CANDIDATE full --strategy optimized
```

`screen.py` 运行原链重排、同预算、近32bit和混合链筛选；`small_primes.py` 搜索小素数位置；`prefix_design.py`、`prefix_small_grid.py`、`prefix_continuous.py` 分别做P1定向试验。`full_batch.py` 与 `full_selected.py` 对选中的候选运行两种整网规划，`gpu_check.py` 执行三个单ReLU检查，`collect.py` 收集轻量证据。候选生成脚本用于新实验目录，原结果不可直接覆盖。

计时只代表当前服务器上的编译耗时，不是GPU推理时间。`peak_child_rss_kib` 来自进程累计子进程高水位，不能作为独立用例的精确RSS比较。
