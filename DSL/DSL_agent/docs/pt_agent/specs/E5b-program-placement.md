# E5b：按语句和阶段放置自举（2026-11-06，接 E5a）

## 0　目的与依据

E5a 让 Agent 端用 depth-dp 计划放置自举，再交给上游自带的 pars 编译。depth-dp 速度快，但放得差：
- E4c：DaCapo 自带的 ResNet-20（SiLU）上，optimized 19 次，depth-dp 85 次；
- 团队 M2：Qwen RMSNorm 小图上 optimized 3 次、depth-dp 17 次
  （`/home/lhohy/Code/Poseidon/test-results/qwen-dacapo-188-20261010/optimized-fixed-rmsnorm-comparison.json`）。

DaCapo 放得好，是因为它在活跃密文少的程序点上同步自举所有活跃密文，并跳过还有余量的值（DaCapo 论文 §6、§8.4）。
它慢，是因为候选点要靠整图分析找出来，旁路检测为 O(N²)。Orion 只在层边界放置，所以快（ResNet-20 放置 1.94 s），
但它进不到非线性模块内部。

PT-DSL 程序本身给出了这些候选点：语句边界、复合算子内部的阶段边界（softmax 的每一轮、norm 的 rsqrt 等）。本轮在这些点上
做动态规划，代价按现有延迟表精确计算，再做一遍跳过修剪，最后和 depth-dp 计划比较，保留预测延迟低的那个。
计划沿用 E5a 的替换表格式，编译沿用 E5a 的 pars 通路。API 0，冻结程序正文、提示词、精调规则都不变。

本轮依赖 E5a 的实现（record/plan 两种 RefreshBackend 策略、替换表、导出包里的计划文件、pars 编译入口），排在 E5a 之后执行。

## 1　阶段标签

- 后端基类新增空操作 `stage(name)`。Mock、Count、Dag、CostDag、Emit 后端都忽略它，trace 出的 IR 和所有计数不变。
- `lower()` 在处理每条语句前设置语句序号（`program.ops` 的下标，宏展开后的顺序）。
- 内核在阶段开始处调用 `b.stage(名称)`。同一条语句内，阶段序号按名称第一次出现的先后编号，所以逐 token 循环里
  重复出现的同名阶段得到同一个序号。至少标出：
  - `transformer.softmax`：平移与缩放、exp 近似、初次归一化（求和、求逆、相乘）、每一轮平方归一化的平方、求和、求逆、相乘
    （名称带轮次，例如 `round2.inverse`）；
  - `transformer.normalize`：均值、中心化、方差、rsqrt 近似、缩放；
  - `approx.newton`：每次迭代；
  - 其余内核（多项式求值、linear、matmul、conv 等）不标，整条语句为一个阶段。
- `RefreshBackend(policy="record")` 给每个新产生的内部节点记一个标签 `(语句序号, 阶段序号, 阶段内深度)`。
  阶段内深度：操作数来自同一阶段时取它们阶段内深度的最大值，来自更早阶段时取 0，乘法再加 1。
  标签按字典序排列；生产者的标签不大于使用者的标签（写成测试）。
- 导出包中的内核源码会多出 `b.stage` 调用；compiler 模式的包 trace 出的 IR 必须与改动前逐个运算相同（写成测试）。

## 2　按阶段切分的放置（stage-cut）

记 `L_out = boot_out_level`。一个值的 level 低于 `L_out`，称为"用过"。level 的推进规则与 `RefreshBackend` 相同。
本项目当前配置（qwen48、R121 规则）下自举后的归一化层数为 0，与这套规则一致；计划器读到的归一化层数不为 0 时停止并报告，
本轮不处理这种情况。

- **切点**：相邻两个不同标签之间。在切点 c 处，所有在 c 之前产生、在 c 之后被使用、并且用过的密文都做自举；
  没用过的直接越过切点（DaCapo 的旁路）。
- **状态**：在这个规则下，切点之后所有活跃密文的 level 都是 `L_out`，所以只要知道上一个切点，就能完全确定之后的 level。
  动态规划的状态就是上一个切点。
- **转移 a → b**：从切点 a 出发，按标签顺序推进 level（与 `RefreshBackend` 的规则相同：乘法结果为操作数 level 的最小值减 1，
  不得低于 `boot_in_level`）。出现不可行就停止，不再延伸 b。转移代价 = 段内各运算按 level 查延迟表的时间
  + 在 b 处需要自举的密文数 × 自举延迟。延迟表、键名和 level 换算与 `placement.depth_dp` 相同。
- **目标**：从程序开始到输出的总预测延迟最小。程序输入 level 为 `input_level`。输出之前不额外自举。
- **复杂度**：每个起点只推进一个窗口内的节点，总工作量约为（标签数 × 一个窗口内的节点数）。记录 `labels`、`cuts`、
  `dp_s`。BERT/Qwen 超过 600 s 时停止该变体，记录原因，其余照常。
- **输出**：选中的切点，以及每个切点上自举的值和读取自举结果的使用者，换算成 E5a 的替换表。

## 3　跳过修剪（对任何计划都适用）

输入一份计划（切点及每个切点自举的值）。依次尝试删掉一处自举：被删的值在该切点之后、它下一次被自举之前的使用者，
改为读取原值，只在受影响的范围内重新推进 level。仍然可行、并且总预测延迟下降，就保留删除。
顺序：按"删掉后受影响的节点数"从少到多；最多做 3 遍，或一遍中没有删除就停。记录 `removed`、`prune_s`。

## 4　选择与语法

- header 新增 `refresh="agent", placement="program-plan"`。计划器算出四个候选：depth-dp、depth-dp + 修剪、stage-cut、
  stage-cut + 修剪，保留总预测延迟最低的一个，写进计划文件（字段 `variant`），四个候选的自举数、延迟和用时都记录下来。
- 选中计划的应用、代价估计、Mock 和导出与 E5a 的 `depth-dp-plan` 完全相同：`RefreshBackend(policy="plan")`，
  估计出的自举数必须等于计划的自举数，否则报错。
- 已有取值（`depth-dp-plan`、惰性 `refresh="agent"`、`refresh="compiler"`）的行为不变。

## 5　实验

结果目录：`/home/lhohy/pt_agent_results/E5b-program-placement-<开跑日期>/`。

### 5.1　6 个冻结程序（与 E5a 相同的程序和改写方式，header 改为 `program-plan`）

每个程序：
1. 计划：四个候选的自举数、预测延迟、用时；选中的变体。
2. 一致性：`plan` 策略下 `estimate` 的自举数和延迟等于计划；2 个校准样本上 Mock 与 compiler 模式逐位相同。
3. 导出、trace、pars 编译（E5a 的入口和限制）：编译后的自举数、步数、编译时间、峰值 RSS，计划延迟（`cost.plan_latency`）。
4. 清理（同 E5a）。

### 5.2　规则基线，只做计划

同样改写 header，只做 §5.1 的第 1、2 步，不编译：
- BERT R0、R1：`/home/lhohy/pt_agent_results/A1-bert-prepare-v2-20261028T135014-01b9/{R0,R1}/program.ptdsl`，
  模型目录 `/home/lhohy/pt_agent_results/A1-bert-prepare-v2-a620-20261030/model`；
- Qwen R0、R1：`/home/lhohy/pt_agent_results/A1-qwen-prepare-v2-20261028T135355-6529/{R0,R1}/program.ptdsl`，
  模型目录为同一目录下的 `model`；
- ResNet-20 R6c：`/home/lhohy/pt_agent_results/r8e7-reconfirm-20261022T122558/baselines/R6c/program.ptdsl`。

读取方式与 `A1-r0r1-r121-20261102/run.sh`、`A1-optimized-20261102/cnn_driver.py` 相同。

### 5.3　小程序上与 optimized 对照（主体完成后做）

- 从 Qwen 冻结程序截取开头到第一个 `rmsnorm` 的语句，输出改为该 `rmsnorm` 的结果；BERT 同样截到第一个 `layernorm`。
- 每个小程序编译三次：编译器 optimized、编译器 depth-dp、`program-plan` 加 pars。记录自举数和编译时间。
- typecheck 不接受改过的输出时，只在实验驱动里构造这两个程序，不改 typecheck；optimized 崩溃或超时如实记录。

### 5.4　汇总

E5b.md 和 E5b.json：

| 程序 | Earth op | depth-dp | depth-dp+修剪 | stage-cut | stage-cut+修剪 | 选中 | 编译后自举 / pars 编译 s | 计划总用时 |

每格写自举数和预测延迟。另列：规则基线在选中放置下的自举数，以及与 Agent 程序的差值（动机二"交给编译器的程序更优"的数据）；
§5.3 的三方对照。

## 6　测试

- 阶段标签：生产者标签不大于使用者；逐 token 循环中同名阶段序号相同；compiler 模式包的 trace IR 与改动前逐个运算相同。
- stage-cut：在可以穷举的小 DAG 上（不超过 12 个切点），动态规划的结果等于穷举所有切点子集的最优值。
- 修剪：删除后的计划仍可行，预测延迟不升。
- 选择：四个候选里取最低；`plan` 策略的估计等于计划。
- E5a 和已有测试全部照常通过。

## 7　不要做

- 不改提示词、反馈、精调规则、能力注册表和冻结程序正文；不调用 API。
- 不改编译器、上游脚本、前端；不提高限制；失败不重试；不删历史结果；不提交、不推送。
- 本轮不做按语句缓存代价、宏的跨层复用、新的 Agent 会话，以及在精调中使用新放置，这些放到 E5c。

## 8　完成标准

- §1–§4 的实现和 §6 的测试通过。
- §5.1 的 6 个程序各有 REPORT.json（完成，或写明失败原因）；§5.2 的 5 个基线有计划记录；§5.3 有记录或写明未做的原因。
- E5b.md、E5b.json 齐全；registry 中 round 为 `E5b`；最后一次残留进程检查为 0。
- STATUS.md 新增 E5b 一行，"与 spec 不一致之处"逐条列出。Nix 独占，串行。
