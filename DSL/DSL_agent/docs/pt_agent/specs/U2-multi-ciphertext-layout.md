# U2：多密文布局（单个张量跨多个密文）

PLAN §4.B 的 U2。本轮的约束：
- 不调用 LLM；新增的代码、实验和测试不读取报告集（既有回归测试按 AGENTS.md 的例外执行）；不修改编译器和运行时。
- **现有的单密文程序发出的运算序列逐字不变**：单密文是多密文的特例，走原来的代码路径。

## 0. 背景

- 现在所有布局都假设一个张量放在**一个**密文里：`MuxLayout(n, C, H, W, k)` 要求 `used = T·k²·H·W ≤ n`；`FlatLayout(n, size)` 要求 linear 的 `2·Ip ≤ n`。超过就抛出 `"tensor does not fit in a single ciphertext"`。
- U1 的覆盖率表显示，中间张量超出 16384 个槽位的模型，即使算子都支持也编不出来。例如 CIFAR 尺度的 VGG16/AlexNet（第一层 64×32×32 = 65536），以及 U4 的 LLM FFN。
- **本轮范围**：
  - 卷积类张量**按通道分块**放在多个密文中；
  - 向量**按长度分块**。
- **不在本轮范围**：
  - 单个通道的空间尺寸超过一个密文（H·W > n，例如 224×224 的输入）需要空间分块，涉及跨密文的卷积边界，记为限制，以后再做；
  - LLM 的 token 布局放在 U4，到时在本轮的多密文框架上扩展。

## 1. 布局对象（`kernels/layout.py` 扩展，原有类与函数不变）

1. **`MultiMuxLayout`**：一组 `(通道起止, MuxLayout)`。
   - 每个分块都是一个合法的单密文 `MuxLayout(n, C_i, H, W, k)`，只包含自己那部分通道，各自有自己的 T、p（副本数）和 block，各块的 H、W、k 相同。
   - **分块规则**（确定性，与模型无关）：每块容纳 `g = n // (k²·H·W)` 个通道组（每组 k² 个通道），即 `C_block = g·k²`。按通道顺序依次切分，最后一块可以不满。
   - `g = 0`（单个组都放不下）时，抛出 `KernelError("spatial tiling not supported: k²·H·W > n")`。
   - 如果 `C ≤ C_block`，**返回原来的单密文 `MuxLayout`，不是只有一块的 `MultiMuxLayout`**。这样单密文程序的路径和运算完全不变。
2. **`MultiFlatLayout`**：长度为 S 的向量切成长度为 `B = n/2` 的块（最后一块可以较短），每块是一个 `FlatLayout(n, size_i)`，各块都满足 R8f2 的尾槽为 0 的不变量。`S ≤ B` 时返回原来的 `FlatLayout`。
3. `pack_*`、`unpack_*`、`index_grid` 都提供对应的多块版本；打包（客户端）输出一个密文列表。

## 2. 值的表示与 lowering（`lower.py`）

- 环境中的值改为"密文列表加布局"。单密文时仍然是单个密文值，现有路径一行不变。
- `_pack_layout` 按 header 的 slots 和输入形状，选择单块或多块布局。
- `program.params` 的每个参数在多块时占用多个密文输入，按块的顺序编号。`pack_inputs` 和 `decode_outputs` 相应支持多块。
- 每种运算的多块实现见 §3。如果某个运算的输入和输出布局都是单块，**必须调用原来的单块 kernel**，不能走多块路径。

## 3. kernel（新增 `kernels/multi.py`，原有 kernel 文件不修改）

| 运算 | 多块实现 |
|---|---|
| act | 逐块调用原来的 act（cheb 或 minimax），每块一个密文；flat 块各自按 R8f2 的规则清零尾槽 |
| add | 两边的分块必须完全相同，逐块相加；不同时抛出 `KernelError("residual layouts must match")` |
| conv2d | 输出按 §1 的规则分块。第 j 个输出块 `= Σ_i conv(x_i, W[j 块, i 块])`，按 Step 1–6 的数学含义计算。**每个输入块的平移（Step 1 的旋转）只计算一次**，被所有输出块共用。不能对每对 (i, j) 各调用一次单块 conv，否则旋转会重复：编译器的 CSE 会删掉重复，代价模型却算了两遍，两边就对不上。部分和在输出块的布局中累加，Step 5 的副本填充和 Step 6 的偏置在累加之后对每个输出块只做一次。BN 折叠和步长（1、2）的语义与单块相同 |
| 下采样捷径 | 与单块相同，用恒等 1×1 步长 2 的卷积（加通道补零）表示，走上面的多块 conv |
| mean（全局平均池化） | 逐块调用原来的 `mean_mux`，得到每块的通道均值（位于槽 [0, C_i)，尾槽为 0）。如果总通道数 `C ≤ n/2`，就把各块分别旋转到通道偏移位置后相加，合并成一个 `FlatLayout`；否则输出 `MultiFlatLayout` |
| linear | 输入、输出都可以是多块。按块矩阵计算 `y_j = Σ_i W[j, i]·x_i`，每个块对是一次与 `linear_flat` 相同算法（diag 或 bsgs）的矩阵乘，在 `y_j` 的布局中累加，最后对每个输出块只做一次掩码和偏置。**放宽单块限制 `Op ≤ Ip`**：块对的矩阵先补零成方阵（以输入、输出中较大的 2 的幂为边长）再计算，满足 `2·边长 ≤ n`。这条放宽**只用于多块路径**，单块路径仍是原来的 `linear_flat` |
| relayout | 多块 mux 到多块 flat：只支持 H = W = 1（全局池化之后的张量）的情况，等价于按通道合并；其他情况抛出 `KernelError` 并说明 |
| refresh | 逐块 |

- 所有多块 kernel 只用 `Backend` 的现有接口（加、乘、旋转、常量、自举），因此 Mock、Count、DAG、Emit、Hecate 以及 P2 包的各种后端都自动支持。
- 共用平移的规则也适用于 linear 的多块路径：同一个输入块的旋转只做一次，被所有输出块共用。

## 4. 周边模块的同步

1. **代价模型与放置**（`cost.py`、`placement.py`）：只需要支持多输入、多输出的 DAG；不能修改 depth-dp 的算法。
2. **张量参考解释器**（`tensor_sim.py`）：语义不变，它在张量层计算，不关心布局。observer 的统计口径需要和 Mock 一致：取所有块的有效元素的 min、max 和 max_u。
3. **验证与 observer**（`agent_tools.py`）：act 的 observer 对多块值取所有块的有效槽合并统计；`decode_outputs` 和 `pack_inputs` 支持多块。
4. **P2 包导出**：
   - `ops.py` 模板加入多块运算（复制 `kernels/multi.py`，与 lowering 的逐条一致性由 P2 的测试 A 覆盖）；
   - `interfaces/default.json` 中，每个输入、输出列出它占用的全部密文下标和每块的布局；
   - `hc.func` 的签名按密文数生成（`"c,c,c,..."`）；
   - `prepare_inputs.py` 和 `unpack_outputs.py` 支持多块。
5. **CLI 和 ModelInfo**：`inputs` 中记录每个输入的密文数和布局摘要，`info` 阶段打印出来。
6. **提示词不变**（r8-v4）。Agent 看不到分块细节，分块带来的代价通过代价模型反映出来。

## 5. 测试与验收（新增 `tests/test_u2_*.py`）

1. **单密文逐字不变**：P2 的 7 个程序加上 4 个冻结程序，修改后用 `EmitBackend.render()` 得到的文本和常量，与修改前完全相同。修改前的文本由 git 中 125676bd 版本的代码生成，或者作为测试夹具保存。冻结 4 个模型在 20 张留出图上的 validate（mock、tensor）、estimate 和 confirm 逐项相同。
2. **布局单元测试**：`MultiMuxLayout` 和 `MultiFlatLayout` 的分块规则、pack/unpack 往返、边界情况（恰好放满、多一个通道、最后一块很小、`g = 0` 时报错）。
3. **逐运算等价**：每种多块运算各构造一个小例子，强制产生 2 块和 3 块以上，用 Mock 加多块 lowering 与张量参考解释器比较，满足 `|Δ| ≤ 1e-9·(1+|ref|)`。至少覆盖：
   - 输入单块、输出多块的 conv；输入多块、输出单块的 conv；多块到多块的 conv，以及步长 2；
   - add；
   - 多块 mean 合并成单块，以及保持多块；
   - linear 的四种组合（单/多块输入 × 单/多块输出），包括 Op > Ip 的情况；
   - cheb、minimax 和 square 三种 act 在多块 flat 上的尾槽不变量。
4. **共用平移**：对一个多块到多块的 conv，用 Count 后端统计的 rotate 次数，等于"输入块数 × 平移数 + 求和与投影所需的旋转"。并且与"对每对块各调用一次单块 conv"相比，旋转更少。
5. **真实模型强制多块**：在 resnet20、cnn_plain、mlp 的冻结程序上，只改 header 的 slots（8192 和 4096），其余不变：
   - ResNet20 在 slots=4096 时，第一层 16×32×32 = 16384 分成 4 块，后面各层随之变化；
   - mlp 在 slots=4096 时，3072 维的输入分成 2 块。
   
   对每个变体：
   - Mock 加多块 lowering 在 20 张留出图上与张量参考解释器满足 `|Δ| ≤ 1e-9·(1+|ref|)`，预测的类别与 slots=16384 的原程序一致；
   - `estimate` 的自举数等于实际编译的自举数，编译走 P2 包路径。也就是在多块情况下，代价模型对放置的复现仍然精确；延迟误差与 E6 同口径（中位数、最大值）报告；
   - 记录每个变体的密文数、自举数、旋转数和预测延迟，与 slots=16384 对比，说明分块的代价。
6. **P2 包**：上述多块变体都能导出包；包内的 `run_numpy.py` 与 Mock 完全相同；`compile_model.py` 编译通过。
7. **超出范围时明确报错**：H·W > n 的输入（例如 3×224×224 配合 slots=16384）在 check 阶段报告 `spatial tiling not supported`，带 op id 和行号，导入不受影响。
8. 既有 419 项测试（373 + 46）全部通过。

## 6. 汇报

- 改动文件清单；
- §5 各项结果；
- §5.5 的对比表：每个变体的密文数、自举数（预测与编译）、旋转数、预测延迟，以及延迟误差；
- 多块路径中 linear 放宽 `Op ≤ Ip` 的实现说明；
- STATUS.md 中更新 U2 一行。

## 7. 不要做

- 不修改原有单块 kernel 文件（`conv.py`、`pool.py`、`linear.py`、`poly.py`、`slots.py`）的函数行为；只能在 `layout.py` 中新增，不能改原有类和函数。
- 不修改编译器、提示词、精调参数、确认规则。
- 不做空间分块和 LLM 的 token 布局。
- 不调用 LLM；不下载、不安装。
