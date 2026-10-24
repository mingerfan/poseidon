# U3：CNN 算子（池化、depthwise、concat、展平全连接、独立 BN），覆盖 DaCapo 的 CNN 基准

PLAN §4.B 的 U3。本轮分两部分，本 spec 是 **U3a**：
- **U3a**（本 spec）：kernel、lowering、PT-DSL、规则规划器、代价模型、张量参考和 P2 包。验证只用规则规划器生成的程序，**不调用 LLM**。
- **U3b**（下一轮另写）：用新提示词版本，在冻结协议下让 Agent 为这些模型生成程序。

本轮的约束：
- 新增的代码、实验和测试不读取报告集（既有回归测试按 AGENTS.md 的例外执行）；
- 不修改编译器和运行时；
- 不下载、不安装：DaCapo 的权重已获批准，在本机 `PT_DACAPO_UPSTREAM/examples/data/` 中；
- **现有程序发出的运算序列逐字不变**。

## 0. 目标模型与所需算子

DaCapo 上游的 `python/poly/poly/models/` 中的 4 个 CIFAR 模型，以及 `resnet20_relu_model`。每个模型都有 SiLU+AvgPool 和 ReLU+MaxPool 两种权重（MobileNet 的两种权重都没有池化差别）：

| 模型 | 现在缺少的算子 |
|---|---|
| AlexNet | AvgPool/MaxPool（3×3，步长 2，padding 1）；5×5 卷积；展平 4×4×256 后接全连接 4096→2048→1024→10 |
| VGG16 | AvgPool/MaxPool（2×2，步长 2）；全连接后接 BatchNorm1d |
| MobileNet | depthwise 卷积（groups=C，步长 1 和 2，3×3）；AvgPool(2) |
| SqueezeNet | Fire 模块中的通道 concat；AvgPool/MaxPool（3×3，步长 2，padding 1）；AdaptiveAvgPool(1) |
| ResNet20（DaCapo 的 relu 权重） | 无（用于验证 U1 入口和现有 kernel 可以直接用于另一份权重） |

**输入样本与数据划分**（2026-10-24 修订，解决与 AGENTS.md 数据划分的冲突）：
- 这些模型在 CIFAR-10 上评测，所以**沿用项目固定的 CIFAR 划分，不走 U1 的通用划分**：
  - 校准：训练集的冻结选图，与 4 个开发模型相同；
  - 留出验证：test[9000:10000]；
  - 确认：test[8000:9000]；
  - 报告：test[0:8000]。
- 数据由现有的内部样本源 `legacy_cifar`（`data/samples.py` 中的 `LegacyCifar`）提供。它的归一化是 ImageNet 均值/方差，与 DaCapo 测试脚本（`examples/tests/*.py`）的 `ToTensor + Normalize` 完全一致；输入形状按图的输入形状 reshape 成 (N,3,32,32)。
- 模型仍然从 U1 的 `--model` 入口进入（`dacapo_builders.py`）。样本源通过实验脚本的内部接口注入，与 run.py 对冻结模型的做法相同，不新增 CLI 参数。
- **本轮不读取报告集**：§5.5 的 acc_drop 只在留出集 test[9000:10000] 上计算；确认集只在需要时使用；报告集到 U3b 的最终报告阶段才读取。
- U1 的通用划分规则只用于用户提供的非 CIFAR 样本。用 CIFAR 测试通用路径时，只能使用训练集图像（U1 用的是 train[0:223]）或 test[8000:10000]。

**模型入口**：新增 `pt_agent/models/dacapo_builders.py`。每个 callable 从 `PT_DACAPO_UPSTREAM` 导入 DaCapo 原始的模型类，加载对应权重（只读，`torch.load(..., weights_only=True)`；如果权重文件不能用 weights_only 方式加载，停下来汇报），然后返回 `nn.Module`。之后一律走 U1 的 CLI：`pt-agent --model pt_agent/models/dacapo_builders.py:<name> --inputs <npz>`。

## 1. PT-DSL 扩展（`dsl/ops_spec.py`、`dsl/parser.py`、`dsl/printer.py`、`dsl/typecheck.py`）

1. `conv2d` 的 `groups` 允许取 1 或 C_in。`groups=C_in` 且 `C_out = C_in` 时是 depthwise 卷积。其他分组数在 check 阶段报"暂不支持"。
2. `avgpool(x, k, stride, pad)`：语义同 PyTorch 的 `count_include_pad=True`，除数恒为 k²。
3. **新增** `maxpool(x, k, stride, pad, method, deg | degrees, domain)`：语义同 PyTorch 的 MaxPool2d，padding 位置视为 −∞。近似参数用来近似窗口内比较时的 `relu(a−b)`（§2.3），写法与 `act` 相同，因此**它和 act 一样是 Agent 与精调可以调节的近似点**。
4. `concat(a, b, ..., axis=1)`：沿通道拼接。
5. `batchnorm` 作为独立运算：用于不紧跟在 conv 或 linear 之后的 BN。
6. `flatten` 后面紧跟 `linear` 的组合允许 H、W 大于 1（§2.5）。

每个新语法都给出 parser、printer 和 typecheck 的往返测试。

## 2. kernel（新增 `kernels/cnn.py`；原有 kernel 文件不修改）

所有 kernel 都只用 `Backend` 的现有接口，并且同时支持单块（`MuxLayout`）和多块（`MultiMuxLayout`）输入。

### 2.1 depthwise 卷积

- 输出通道 c 只依赖输入通道 c，不需要跨通道求和。做法：对每个 kernel 偏移 (i1, i2) 旋转一次输入，乘以按通道排布的权重向量（零填充的边界用有效性掩码），再累加。
- 步长 1 时输出布局与输入相同。步长 2 时输出布局按 `mux_layout(n, C, H/2, W/2, 2k)` 重新排布，投影方式与 `conv2d_mux` 的 Step 4 相同。
- BN 折叠与 bias 的处理和普通卷积相同。
- 多块时按块独立计算，不需要块间求和。
- 旋转数应约等于 k² 加上投影所需的旋转，**不随通道数线性增长**。测试中用 Count 后端断言这一点，并与"用普通 conv 加分块对角权重"的方法对比，旋转和乘法都应当更少。

### 2.2 AvgPool

用 §2.1 的 depthwise kernel 表示：每个通道的权重都是 1/k²，padding 位置为 0（与 count_include_pad 一致）。`AdaptiveAvgPool(1)` 映射为现有的 `mean`（全局平均）。

### 2.3 MaxPool

- 用 `max(a, b) = b + relu(a − b)` 两两合并：先在行方向对窗口内的 k 个位置合并，再在列方向合并（k=2 时 2 次，k=3 时 4 次 relu），然后与 §2.1 相同地处理步长。
- `relu` 用 maxpool 运算自己的近似参数（method、deg/degrees、domain），使用现有的 `act_cheb` 或 `act_minimax`。
- **padding**：窗口超出图像的位置，用掩码让它不参与比较：`m = x + mask·relu(nb − x)`，掩码在越界位置为 0。乘掩码会多消耗一个 level，代价模型如实计入。
- **近似点的校准**：maxpool 内部每一次 relu 的输入（相邻位置的差值）都要作为校准和验证的观测点。observer 对每个 maxpool 运算汇总所有差值的 min、max 和 max_u，用于 domain 的推断（规则规划器）以及 feasible 判据中的 max_u，口径与 act 相同。

### 2.4 通道 concat

**零代价拼接**：结果是各输入分块列表按顺序拼起来，得到 `MultiMuxLayout`。为此允许各块的通道数不同，即放宽 U2 中"按 g 均匀切块"的规则：块可以是任意通道数，但必须是合法的单块 `MuxLayout`，且 H、W、k 相同。不做数据搬移。

下游的 conv（U2 的块矩阵 conv）天然支持任意分块；输出按 U2 的规则重新紧凑分块。如果 concat 的结果直接被 add、act 或 maxpool 使用，这些运算本来就逐块进行。

### 2.5 展平后接全连接（`flatten` → `linear`）

不做数据重排，而是**把展平合并进线性层的权重**：在 mux 布局中，按每个 (c, y, x) 所在的槽位重新排列 W 的列（副本和未使用的槽位为 0）。等价于对"槽位向量"做一次线性变换。

- 输入是满长度 n 的密文时，`linear_flat` 的复制技巧不适用。需要新增"全长输入"的矩阵乘：旋转本来就是模 n 循环，不需要复制，对角线按 n 计算，然后用 SumSlots 折叠到输出维度。diag 和 bsgs 两种算法都要支持。
- 多块输入时，按 U2 的块矩阵求和。
- 只有当 flatten 的输入 H = W = 1 时，才继续走原来的 relayout 加 `linear_flat` 路径，保证现有程序不变。

### 2.6 独立 BatchNorm

lowering 时先尝试折叠进前一个 conv 或 linear（如果 BN 是它唯一的消费者）。不能折叠时，表示为按通道的仿射变换：乘一个明文向量，再加一个明文向量。这会多消耗一个 level，代价模型如实计入。VGG16 的 `fc_2 → bn_1` 必须走折叠路径。

## 3. 周边同步

1. **lowering**（`lower.py`）：新运算分派到 §2；原有分支不变。
2. **代价模型、DAG、放置**：不需要改算法，只需要正确统计新 kernel 发出的运算。
3. **张量参考解释器**（`tensor_sim.py`）：
   - 实现所有新运算的张量语义，maxpool 的近似版本使用同样的 `relu(a−b)` 多项式；
   - exact 模式下就是 PyTorch 的 max_pool2d；
   - observer 对 maxpool 的差值统计口径与 Mock 相同。
4. **校准与验证**（`agent_tools.py`、`run.py` 的 calibrate 阶段）：maxpool 作为近似点参与 rows、worst_acts、max_u 和精调。
5. **精调**（`planners/tuner.py`）：只在近似点的枚举中把 maxpool 视为与 act 相同的近似点（定义域对齐、阶数阶梯）。精调的参数、阈值和流程都不变。
6. **规则规划器**（`planners/rule.py`）：
   - 新运算一律用通用规则生成：maxpool 的近似参数与 act 的规则相同（deg 127，余量 1.5，用校准得到的差值范围）；
   - avgpool 和 adaptive avgpool 直接映射；
   - depthwise 卷积对应 `groups=C`；
   - concat 直接映射；
   - 展平后接 linear 使用 §2.5。
7. **P2 包**：`ops.py` 模板和包内 core 加入 §2 的 kernel（复制 `kernels/cnn.py`），interfaces 支持新布局。
8. **提示词**：新增版本 **r8-v5**，由注册表自动生成，包括新运算及其可调参数（maxpool 的近似参数）。**r8-v4 保持不变**，冻结的 4 个模型及 R9b 仍然用 r8-v4。版本由运行参数选择，默认值仍是 r8-v4。U3b 用 r8-v5。

## 4. U2 遗留项：mlp@4096 的延迟误差（7.3%）

自举数和旋转数都对得上，误差来自其他运算。在多块 linear 路径上，比较代价模型和 RuntimePlan 中各类运算（mul_cp、rescale、add、mod_switch、relinearize）的**数量和 level 分布**，找出差异来源，在汇报中说明。

- 如果是代价模型对某种运算的 level 估计错误，按自写错误例外修正，修正方法必须通用；修正后 6 个 U2 变体的自举数仍然全部精确，延迟误差重新报告。
- 如果是编译器做了代价模型没有模拟的优化（例如 CSE 合并了相同的编码），汇报现象，不修改代价模型。

## 5. 验收与测试（新增 `tests/test_u3_*.py`）

1. **现有程序不变**：P2 的 7 个程序、4 个冻结程序，以及 U2 的 6 个多块变体，`EmitBackend.render()` 文本和常量与 5412df2b 完全相同；4 个冻结模型的 validate（mock、tensor）、estimate、confirm 逐项相同；r8-v4 生成的提示词文本与原来逐字相同。
2. **逐运算等价**：每种新运算（depthwise 步长 1/2、avgpool 的三种配置、maxpool 2×2/s2 与 3×3/s2/p1、concat 2 路/3 路、展平 4×4 后接 linear 的 diag 和 bsgs、独立 BN 和折叠 BN），在单块和强制多块下，Mock 与张量参考满足 `|Δ| ≤ 1e-9·(1+|ref|)`；exact 模式下与 PyTorch 满足 `|Δ| ≤ 1e-9·(1+|ref|)`。
3. **maxpool 近似**：在 CIFAR 校准样本上，用规则规划器的参数时，maxpool 的输出与 exact maxpool 的最大误差和 max_u 写入报告；某个差值超出 domain 时，按发散处理，与 act 的口径相同。
4. **depthwise 的效率**：按 §2.1 的要求用 Count 后端断言。
5. **9 个模型变体的端到端**（AlexNet、VGG16、SqueezeNet 各取 SiLU+Avg 与 ReLU+Max 两种，MobileNet 取 SiLU，再加上 DaCapo 的 resnet20_relu 和 MobileNet ReLU），每个都通过 U1 的 CLI，使用规则规划器：
   - 导入、生成程序、check 都通过；
   - 在 20 张留出图上，Mock 与张量参考满足 §5.2 的容差，预测类别一致；
   - 用张量参考在留出集 test[9000:10000] 上计算 acc_drop（相对明文模型），写入报告，只用于记录，规则规划器不据此调参；
   - 代价模型预测的自举数、旋转数和延迟，以及密文数；
   - **编译**：常量不超过 1.5 GiB、并且能通过 trusted 运行器的内存限制的，实际编译，比较预测与编译的自举数（要求相等）以及延迟误差；超出的，生成服务器作业并标注"本机未编译"，不放宽资源限制；
   - 每次编译完成并记录后，立即调用 `toolchain.prune`。
6. **覆盖率**：重新生成 U1 的覆盖率表，加入以上模型；CNN 模型的"可以 lowering 比例"应为 100%，没有达到的要列出原因。
7. 既有 435 项测试全部通过。

## 6. 汇报

- 新运算的实现说明，包括 maxpool 的 relu 次数与 level 消耗、depthwise 的旋转统计、展平全连接的权重重排方式；
- §4 的诊断结论；
- §5.5 的大表：每个模型的密文数、自举数（预测与编译）、旋转数、预测延迟、延迟误差、张量参考的 acc_drop、编译与否以及原因；
- §5.6 的覆盖率表；
- STATUS.md 中更新 U3 一行，注明本轮是 U3a。

## 7. 不要做

- 不修改原有 kernel 文件的函数行为，不修改编译器，不放宽资源限制。
- 不调用 LLM，不修改 r8-v4 提示词，不修改精调和确认的参数。
- 不下载、不安装；DaCapo 权重只读，不复制进 git。
- 不做空间分块（单个通道 H·W > n）和 LLM 相关的算子。

## 8. 补充：旧测试迁移口径（Claude 于 2026-10-24 复核）

以下 4 项旧测试的调整，是本 spec 新增能力的直接结果，**明确授权**修改。每项都在汇报中列出原断言和新断言。

1. **`test_r1_parser`**：PT-DSL 的算子数从 26 改为 27，原因是新增了 `maxpool`。如果测试列举了算子名单，把 `maxpool` 加入名单。其他断言不变。
2. **`test_r8b_generic`**：完整注册表检查同时覆盖新增的能力（`maxpool` 及其近似参数、`conv2d` 的 `groups=C_in`、`avgpool`、`concat`、独立的 `batchnorm`、展平接 linear）。"每个注册的算子都有 kernel、代价和参考语义"这类一致性要求保持不变，只是检查范围扩大。r8-v4 是冻结版本，它的能力文本仍然按原来的注册表子集生成，逐字不变，由 §5.1 的测试保证。
3. **`test_r8a2_aten`** 和 **`test_u1_semantics`**：已支持的 MaxPool（kernel 2 或 3，stride 等于 kernel 或 2，padding 0 或 1，`dilation=1`，`ceil_mode=False`）的断言，从"check 阶段报告没有 lowering"改为"可以 lowering，并且 Mock 与参考执行一致"。同时**新增或保留**对未实现变体的拒绝断言：
   - `ceil_mode=True`、`dilation≠1`、`return_indices` 被下游使用的情况，在 check 阶段报告"暂不支持"，带 op id 和行号；
   - 转置卷积仍然在导入时拒绝；
   - R3 的 FX 前端及其测试保持原来的边界，一律不改。
4. **r8-v4 的提示词文本和默认版本保持不变**，相关的旧测试不做任何修改。

同类情况的一般规则（以后各轮同样适用）：

- 本轮 spec **明确新增了某项能力**，而旧测试断言的是"该能力不存在"（例如算子数量、注册表清单、某个算子没有 lowering）时，可以按新能力更新断言，并在汇报中逐项列出。
- 其他改变旧断言含义或期望值的情况，仍然按 AGENTS.md 停下来汇报。

完成上述迁移后，继续 U3a 的其余验收（§5.1、§5.3、§5.5–§5.7）。

## 9. 补充：BN 折叠规则与 torchvision 导入（Claude 于 2026-10-24 复核两个停止项）

### 9.1 更正 §2.6：BN 的折叠规则

上游 `VGG16.py` 第 117–119 行实际是 `fc_2 → mish → bn_1 → fc_3`，BN 在激活之后。§2.6 中"VGG16 的 `fc_2 → bn_1` 必须走折叠路径"是 Claude 读错了源码，**作废**。

模型语义一律按原模型保留：激活的位置不变，不交换运算顺序。§2.6 的规则改为以下通用规则，按顺序尝试：

1. **折叠进前一个 conv 或 linear**：BN 的输入来自 conv 或 linear，而且 BN 是它唯一的消费者。这就是现有规则。
2. **折叠进后一个 linear**：BN 的输出唯一的消费者是 linear。此时 `W·(γ⊙x + β) + b = (W·diag γ)·x + (W·β + b)`，严格等价，不消耗 level。VGG16 的 `bn_1 → fc_3` 走这一条。
   - 后一个是 conv 时，只有 `padding=0` 才严格等价（零填充的边界会让 β 的贡献不一致）。所以只在 padding=0 时才折进后一个 conv，否则走第 3 条。
3. **独立仿射**：按通道先乘一个明文向量，再加一个明文向量，多消耗一个 level，代价模型如实计入。

测试：
- 第 2 条要有单元测试：Mock 与张量参考、exact 与 PyTorch 都满足 §5.2 的容差；
- padding≠0 的 conv 前接 BN 时，走第 3 条，并有断言；
- VGG16 端到端的结果中，`bn_1` 被折进 `fc_3`，计入汇报。

### 9.2 DaCapo 模型文件中未使用的 torchvision 导入

上游的 AlexNet、VGG16、MobileNet、SqueezeNet 模型文件第 6 行是 `from torchvision import datasets, transforms`，模型类和 forward 都没有使用它们；ResNet.py 没有这一行。torchvision 在两个环境中都没有安装，**本轮不安装**。

授权在 `dacapo_builders.py` 中做如下**通用**处理：

- 导入模型文件之前，如果 `torchvision` 无法导入，就在 `sys.modules` 中临时放入一个**桩模块**：`torchvision`、`torchvision.datasets`、`torchvision.transforms`。桩模块的任何属性访问都抛出明确的错误："torchvision 未安装，此桩只用于满足未被使用的导入"。导入完成后，把这些桩从 `sys.modules` 中移除。
- 不修改第三方文件的文本，也不在内存中改写源码：只提供桩，比过滤导入语句的改动更小。模型类、forward 和权重都保持原样。
- 如果某个模型在构造或 forward 中实际用到了 torchvision，桩会抛出上面的错误。这时停下来汇报，不安装 torchvision，也不绕开。
- 测试：
  - 4 个模型在桩下都能构造，加载权重，并完成 forward；
  - 在构造或 forward 中访问桩的属性时，抛出上述错误；
  - 构造完成后，`sys.modules` 中不残留桩。

这只是测试工具（`dacapo_builders.py`）中的处理，不进入 U1 的通用 PyTorch 入口：用户的模型文件导入失败时，仍然直接报错。

完成 9.1、9.2 后，继续 U3a 的剩余验收。

## 10. 补充：大模型的内存占用（Claude 于 2026-10-24 复核 AlexNet 的资源停止项）

**现象**：AlexNet 的首图 Mock 在 4 GiB 的地址空间限制内，`flatten_linear` 申请 2048×16384 的 float64 矩阵（256 MiB）失败；SqueezeNet ReLU 在第 320 张留出图时也出现过一次 MemoryError（之后续跑完成）。AlexNet 的常量估计约 3.6 GB。

**判定**：这是本轮自写 kernel 和 Mock 的实现问题，不需要放宽资源。**4 GiB 的限制保持不变。** 按 AGENTS.md 的自写错误例外，做以下两处**通用**修正：

### 10.1 对角线与掩码逐条生成

- `flatten_linear`、U2 的多块 linear，以及全长输入的矩阵乘，不能一次性构造稠密的对角线矩阵或权重重排矩阵。每次只生成一条对角线（长度 n），交给 `b.const` 后即可释放；权重重排用索引映射按需取列。
- 其他 kernel 中，凡是先构造"全部对角线"或"全部掩码"大数组的，同样改为逐条生成。
- 发出的运算序列、常量的值和顺序都必须不变：17 个旧程序的 Emit 文本和常量逐字相同；已完成的 5 个模型变体的 Count 统计相同。

### 10.2 Mock 不保留常量数据

`MockBackend` 增加"常量流式"模式，并**默认开启**：
- `const(vec)` 每次都返回带有本次数据的 `_Plain`；
- 用于去重统计的表只保存哈希（统计"不同常量数"的 `distinct_consts` 照旧正确）；
- 常量数据在使用后随引用释放。

运算结果与现在逐位相同，因为每次传入的数据本来就相同。

Emit、Count、DAG、Hecate 等后端不变：Emit 仍要保存全部常量，大模型照旧走服务器作业。

验证：
- 4 个冻结模型的 validate（mock）与修改前逐项相同；
- 已完成的 5 个变体中任取 2 个，20 图的 Mock 输出与之前 `np.array_equal`；
- 记录 AlexNet 在修改前后的 Mock 峰值 RSS。

### 10.3 继续

修正后，继续 AlexNet（两种）和 VGG16（两种）的 §5.5 验收，包括 VGG16 中 `bn_1` 折进 `fc_3` 的端到端证明、§5.6 的完整覆盖率表，以及既有 435 项完整回归。

- 如果在 4 GiB 内仍然无法完成 20 图的 Mock，停下来汇报峰值和瓶颈，不放宽限制。
- 常量超过 1.5 GiB 的模型照旧生成服务器作业，标注"本机未编译"。代价模型的保真度以本机已编译的模型为准（目前 383/383、282/282、1122/1122）。
- 服务器作业的执行不属于 U3a。
