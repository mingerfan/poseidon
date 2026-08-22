# GPU 自举正向优化记录

## 1. 文档目的与记录范围

本文档专门记录已经通过正确性检查和性能对比、确认具有正向收益的 GPU 自举优化。记录重点不是具体代码改动，而是说明原始瓶颈、优化依据、计算与访存方式的变化、复现该思路需要满足的条件，以及最终观察到的性能提升。

编号从此前对自举优化所做的“六点/七点总结”之后开始。更早完成的动态 rescale、EvalMod 计算顺序调整、relin/rescale x2 融合以及最初的 P=9 Hybrid KeySwitch 专用化等工作视为本文档的起始基线，不再重复记录。

本文档只收录正向优化。计算正确但性能下降的实验，例如先物化两个 decomposition digit、再执行成对 Key-MAC 的方案，不进入本记录。后续经过验证的新优化应继续按照数字顺序追加。

除非单独注明，主要验证参数为：

- GPU：NVIDIA Tesla V100-SXM2 32 GB
- 环维数：`N = 65536`
- 自举模数：最多 34 个 Q 模数、9 个 P 模数，使用 32-bit RNS word
- 线性变换：Double-Hoist
- EvalMod：59 阶多项式、2 次 double-angle、动态 rescale
- 性能测试：1 次 warmup 加 1 次正式测试；重要结果额外使用 Nsight Systems 验证累计 kernel 时间
- 正确性：C2S、EvalMod、S2C 的分阶段 CPU/GPU 对比均通过后，优化路径才允许成为默认实现

需要注意，nsys 会引入插桩开销，因此 nsys 时间只能与其他 nsys 结果比较；不带 nsys 的 CUDA-event 时间用于表示实际 release 性能。

## 2. 正向优化历史

### 1. 将 P=9 coefficient-major ModUp 分块迁移到 Double-Hoist 分解路径

**原始瓶颈。** Double-Hoist C2S/S2C 会反复分解密文 digit，并把 digit 从局部 Q 基扩展到完整 QP 基。通用 ModUp 实现基本以单个目标模数为单位完成转换，相邻目标模数会重复读取同一组源系数和 decomposition 常量。在 `P = 9` 时，每个 digit 都要生成大量 Q/P residue，这部分重复访存相当明显。

**优化思路。** 将 ModUp 改成 coefficient-major 的目标模数分块。一个 thread block 负责一段连续系数，并同时计算最多 8 行目标模数；源 digit 的 residue 被读取后，在寄存器中复用于多个目标模数。固定的 `P = 9` 使循环边界以及最后一个不完整分块可以在编译期确定，从而降低动态索引开销。

这里的关键是：warp 内线程必须继续沿 coefficient 方向排列，而 8 行分块沿目标模数方向展开。这样源读取和结果写回都是连续的。如果交换这两个维度，写回将以 `N` 为跨度，反而会破坏显存合并访问。

**性能结果。** 该修改是 Double-Hoist P=9 专用化阶段的正向组成部分，并取代了验证参数下的通用 ModUp。第 1～4 项在当时是逐步测试的，但没有为每个中间状态都保留独立 release 结果；四项组合使 nsys 下完整自举从约 `184.35 ms` 降到约 `161.95 ms`，合计降低约 `22.4 ms`，即 `12.2%`。

**适用范围。** 当前专用路径面向 `N = 65536, P = 9`，其他参数仍保留通用 fallback。

### 2. Double-Hoist digit 的完整 QP Four-step NTT

**原始瓶颈。** ModUp 之后，每个有效 Q/P limb 都要执行长度 65536 的 forward NTT。普通的一维大 NTT 需要多轮 global-memory 写回和重新读取，而且会发射较多粒度很小的 stage kernel。该开销还会乘上 decomposition 数量以及 C2S/S2C 的旋转数量。

**优化思路。** 将长度 65536 的 NTT 拆成两个粗粒度 phase。每个 phase 在寄存器和 shared memory 中连续完成多级 radix butterfly，两个 phase 之间使用预先设计的转置布局。ModUp 直接输出第一阶段所需要的布局，因此不再单独执行一次全量 QP transpose。属于当前 decomposition digit 的直接 Q limb 已经处于 NTT 域，可直接读取原密文，不重复计算。

这一优化不改变 NTT 数学过程，也没有减少需要变换的 limb 数量；收益来自每次 global load/store 之间完成更多 butterfly，并用两个粗粒度 phase 代替多次细粒度 stage。复现时 Q 和 P 必须共享相同的 phase 边界和转置规则，否则新增的 QP 全局转置会抵消大部分收益。

**性能结果。** 完整 QP Four-step 路径带来了稳定的完整自举提升，因此成为 `N = 65536` 的默认路径。它与第 1、3、4 项共同贡献了约 `184.35 ms -> 161.95 ms` 的 nsys 阶段性下降，并明显减少了通用 NTT stage kernel 的发射数量。

**适用范围。** 当前优化实现针对 `N = 65536`；其他 N 使用通用 NTT。

### 3. Double-Hoist ModDown 中的 P 基一次性预加权

**原始瓶颈。** Hybrid ModDown 从 P 基转换回每个 Q 模数时，每个 P residue 都需要乘以对应的 punctured-product inverse。如果把这个因子放在 P→Q 累加内部计算，同一个 P residue 会针对所有目标 Q 模数重复乘权。在 `P = 9` 时，这部分工作量近似与 `P × Q × N` 成正比。

**优化思路。** 在 P→Q 转换之前，先对两个密文分量的每个 P residue 统一乘一次只依赖该 P limb 的权重。后续转换 kernel 直接读取预加权结果，仅执行与目标 Q 模数相关的 multiply-accumulate。这样，权重乘法由 `P × Q × N` 级别降为 `P × N`，而数学上不可省略的 P→Q 累加保持不变。

该方法成立的原因是预加权因子只依赖源 P limb，与目标 Q limb 无关。必须保证权重只应用一次：如果预处理和转换矩阵中同时包含该因子，基转换结果会错误。

**性能结果。** 最终 nsys 中独立 P 预加权 kernel 累计约 `1.1 ms`，但它移除了 P→Q kernel 内更大规模的重复模乘，因此整体为正向，并作为 Double-Hoist 专用化的一部分默认开启。

**适用范围。** 当前默认专用实现针对 `P = 9`。接口需要携带“输入是否已预加权”的状态，防止 generic path 重复处理。

### 4. P=9 P→Q 转换与 Four-step forward NTT 衔接

**原始瓶颈。** 直接实现 ModDown 时，会先把完整的 P→Q 系数结果写入 global memory，然后再为每个 Q limb 执行 forward NTT。转换结果会立即被 NTT 消费，因此这次完整 Q 数组的写回和重读没有算法价值。同时，通用转换会动态遍历 P，而当前生产配置固定为 9 个 P limb。

**优化思路。** 对 9 个 P limb 专门展开 P→Q 转换，并让转换 kernel 直接产生 Four-step NTT 第一 phase 所需的排列。kernel 对 9 个预加权 P residue 完成目标 Q residue 的累加，同时执行或衔接第一阶段变换；公共的第二 phase 再完成两个密文分量的 NTT。使用低 shared-memory 的第一阶段实现，保证 GPU 上能够同时驻留足够多 block。

可复现的核心不是简单地把“基转换”和“NTT”写在同一个函数中，而是 producer-consumer layout fusion：转换结果必须直接生成第一 NTT phase 的 tiled layout，避免以普通 coefficient-major 全量数组落地后再读取。

**性能结果。** P=9 专用转换加 Four-step NTT 持续快于仅采用 row-tiled 的过渡版本，因此成为默认路径。在最终 nsys 中，其第一阶段 kernel 的 83 次调用累计约 `13.94 ms`，虽然仍是当前最大热点，但已经显著低于此前的通用转换加 NTT 链路。第 1～4 项合计带来约 `22.4 ms`、即 `12.2%` 的完整自举 nsys 收益。

**适用范围。** 当前路径要求 `N = 65536, P = 9`。其他参数使用 row-tiled 或完全通用的 fallback。

### 5. QP plaintext MAC 中跨 group 复用 baby ciphertext

**原始瓶颈。** BSGS 线性变换中，多个 giant-step group 会消费同一个 baby-step ciphertext tile。原始 QP plaintext multiply-accumulate 为每个 group 分配独立 block，因此不同 group 会从 global memory 重复读取相同的 baby ciphertext，尽管它们之间只有 plaintext diagonal 和目标 accumulator 不同。

**优化思路。** 将 4 个或 8 个输出 group 放进同一个 block。对于固定的密文分量、RNS limb 和 32 个连续 coefficient，baby-step 数据只加载一次并存入 shared memory；block 内不同 group row 分别读取自己的 plaintext diagonal，更新各自独立的 accumulator。根据 group 数量选择 4 行或 8 行，在复用率与 GPU block 并行度之间取得平衡。

该优化没有减少明密文乘法次数，只减少 baby ciphertext 的 global load 次数。每个 group 的 plaintext diagonal 仍然不同，因此此阶段还不能复用 diagonal。

**性能结果。** QP plaintext MAC kernel family 的累计时间由约 `19.109 ms` 降至 `18.754 ms`，减少约 `0.355 ms`，即 `1.8%`。完整自举收益较小但能够重复，因此保留为默认实现。

**适用范围。** 当 group 数量大于 1 且 baby tile 不超过 8 个元素时启用；其他情况回退到单 group 实现。

### 6. Fused NTT phase2 + Key-MAC 的别名信息与寄存器控制

**原始瓶颈。** EvalMod Hybrid KeySwitch 使用融合 kernel：先完成 Four-step QP NTT 的第二 phase，然后立即把 digit 与两份 relinearization key 相乘并累加。该 kernel 拥有较多输入输出指针。缺少别名信息时，编译器需要保守地假设 source、key 和 accumulator 可能互相重叠，从而产生不必要的 reload。仅增加别名信息虽然改善了调度，却一度把寄存器提高到 48 个，降低了 resident warp 数量。

**优化思路。** 对确定互不重叠的缓冲区增加 alias restriction，使编译器可以安全地保留中间值，并减少保守加载；随后根据 64-thread block 的实际形状增加 occupancy 约束，将最终寄存器控制到每线程 40 个。最终 cubin 中 `STACK=0`、`LOCAL=0`，没有发生寄存器 spill。

这两部分必须结合验证。单独增加 alias restriction 会使寄存器数升高；过度限制寄存器又可能把变量 spill 到 local memory。复现时必须检查生成二进制的 register、stack 和 local-memory 使用，而不能仅根据源码判断。

**性能结果。** 110 次 fused phase2+Key-MAC 调用的累计时间由 `11.205 ms` 降至 `10.627 ms`，提升约 `5.2%`。EvalMod 根据运行波动降低约 `1～2 ms`；完整自举 release 时间达到约 `156.01 ms`，相比此前约 `156.5 ms` 有小幅正向收益。

**适用范围。** occupancy 配置编译在 `N = 65536` 专用融合 kernel 中，不改变密码学运算和参数。

### 7. QP plaintext MAC 的 ciphertext component 融合

**原始瓶颈。** 完成第 5 项后，线性变换仍然分别为密文 `c0`、`c1` 发射两组 block。两个分量乘的是完全相同的 plaintext diagonal，因此每个 diagonal coefficient 都会从显存读取两次。nsys 显示，优化前 QP plaintext MAC family 累计约 `18.76 ms`，是当时完整自举中最大的单类 kernel 热点。

**优化思路。** 让一个线程同时负责 `c0`、`c1` 中相同的 `(group, RNS limb, coefficient)`。block 把两个分量的 baby-step tile 分别加载到两块 shared-memory plane；每个 plaintext diagonal coefficient 只读取一次，随后完成两次模乘并更新两个 accumulator。grid 不再包含 component 维度，因此 block 数量约减半。最终 kernel 每线程使用 32 个寄存器和 2 KiB shared memory，没有 local-memory spill。

第 5 项是在多个 group 之间复用 baby ciphertext，本项是在两个 ciphertext component 之间复用 plaintext diagonal，两种复用发生在不同维度，因此可以同时生效。

**性能结果。** 这是此前六点/七点总结之后收益最大的一项优化：

- QP plaintext MAC 累计 kernel 时间：`18.76 ms -> 11.86 ms`，提升约 `36.8%`
- C2S release 时间：`59.45 ms -> 53.55 ms`，提升约 `9.9%`
- S2C release 时间：`23.81 ms -> 21.11 ms`，提升约 `11.3%`
- 完整自举 release 时间：`156.01 ms -> 148.96 ms`，减少约 `7.05 ms`，即 `4.5%`
- 最终完整自举 nsys 时间：约 `151.93 ms`

C2S、EvalMod、S2C 的分阶段 CPU/GPU 对比全部通过。该路径现已默认开启，可使用 `POSEIDON_DOUBLE_HOIST_QP_MAC_COMPONENT_FUSED=0` 进行回退对比。

**适用范围。** 两个密文分量必须使用相同的 plaintext diagonal 和 modulus layout；当前 Double-Hoist 线性变换满足该条件。如果某个算子对两个分量使用不同明文因子，则不能直接应用这一融合。

### 8. 59 阶 EvalMod BSGS remainder 链的延迟重线性化

**原始瓶颈。** 59 阶 Chebyshev 拟合在动态 rescale 配置下生成 8 个低阶 leaf，并使用 7 个 `Q·T_k+R` combine 节点组成平衡 BSGS 树。原实现中每个密文乘法都会立即把 size-3 结果重线性化为 size-2；即使某个 combine 的输出在父节点中只作为加法 remainder 使用，也会先完成一次完整的 P=9 Hybrid KeySwitch。随后父节点产生新的 size-3 乘积并再次重线性化，造成可以合并的 decomposition、ModUp、NTT、evaluation-key MAC 和 ModDown。

**优化思路。** 对只被父节点作为 remainder 消费、且不再参与密文乘法的 combine 输出保留 size-3 形式。父节点仍要求 quotient 是已经重线性化的 size-2 密文，因此 `quotient·T_k` 只产生 size-3；它与 raw remainder 按 `c0/c1/c2` 逐分量相加后，再对合并后的 `c2` 执行一次重线性化。当前树中，节点 1 的 relin 可合并到节点 2，节点 4 和节点 5 的 relin可沿 remainder 链共同合并到根节点 6，使 BSGS combine 的重线性化次数由 7 次降为 4 次，而密文乘法数、乘法深度、模数消耗和最终 scale 均保持不变。

该方法只沿 remainder 边传播 raw size-3 密文。quotient 边不能延迟，因为 size-3 quotient 再乘 size-2 basis 会产生 size-4 密文，并需要当前密钥结构不提供的高次 secret-key 重线性化。当前实现还要求被延迟节点到父节点之间不发生额外 rescale 或 scale plaintext 调整；不满足结构约束时保留原始路径。

**性能结果。** 在 V100、`N=65536, Q=34, P=9`、59 阶拟合、2 次 double-angle、动态 rescale、1 次 warmup 加 1 次测试下：

- BSGS tree combine：`23.276 ms -> 17.741 ms`，减少 `5.535 ms`，提升约 `23.8%`
- EvalMod（实部加虚部）：`74.744 ms -> 69.499 ms`，减少 `5.245 ms`，提升约 `7.0%`
- 完整自举：`149.326 ms -> 143.830 ms`，减少 `5.496 ms`，提升约 `3.7%`

分阶段 CPU/GPU 校验通过：EvalMod 最大误差约 `2.3e-13`，输出 Q 层级与 scale 和基线一致。完整自举对源消息的误差仍处于当前 N=65536 已知的约 `7e-3` 范围；基线同次配置约为 `8e-3`，本优化没有引入新的层级或精度退化。

**适用范围。** 当前默认仅对动态 rescale、Chebyshev、59 阶配置实际生成的 8-leaf/7-combine 计划启用。可通过 `POSEIDON_EVALMOD_LAZY_RELIN=0` 回退到逐节点立即重线性化路径。详细 trace 模式为了输出与 CPU size-2 中间节点可直接比较的快照，会额外重线性化 raw trace 副本，因此不能用于衡量该优化的正式性能。

### 9. WHET 风格的精确周期 QP 明文压缩与直接 MAC

**原始瓶颈。** Double-Hoist C2S/S2C 为每条 BSGS 对角线保存完整的 QP NTT 明文。在 `N=65536` 时，即使某些对角线在 NTT 域只包含很短的重复周期，原实现仍然为每个 Q/P limb 分配 `N` 个 residue，并在 plaintext MAC 中按完整数组读取。这不仅占用大量显存，也使同一周期值从 global memory 被重复加载。该问题不会增加明密文乘法次数，但会直接增加 QP MAC 的内存流量。

**优化思路。** 利用 DFT 对角线在 bit-reversed NTT 顺序中的精确周期，只保存一个最小的 2 次幂周期。对于完整 NTT coefficient 索引 `j`，压缩明文的读取位置为 `reverse_bits(j) & (period-1)`。Q 明文先按该规则检测最小周期；随后执行与原实现相同的精确 Q→P CRT 扩展，并要求所有 P residue 也满足同一周期。这里不做浮点量化或近似编码，压缩只是整数 residue 的无损重排。无法压缩、周期等于 `N` 的对角线继续保持普通 NTT 顺序，以避免 bit-reversed 的非合并显存访问。

QP plaintext MAC 增加独立的压缩布局 kernel 实例。默认完整布局仍使用原来的编译期路径，不承担周期读取分支；压缩实例为每条对角线读取周期元数据，并将 coefficient 映射到紧凑 Q/P 数组。周期为 32 或 64 等较小值时，相邻大量 coefficient 会复用相同的 compact cache line；周期为 `N` 时直接走连续读取。

**正确性验证。** 在 V100、`N=65536, Q=34, P=9` 下执行了三层检查：第一，压缩数组从 GPU 回读后，全部 Q residue 和 9 个 P residue 与完整 QP oracle 逐字一致；第二，完整和压缩路径最后一级、ModDown 之前的 Q-MAC 与 P-MAC 工作区逐字一致；第三，StC 输出以及 C2S 的实部、虚部密文逐字一致。因此该优化没有改变 scale、level、模数运算顺序或解密精度。

**存储与性能结果。** 当前 StC-first 实验调度中的线性变换明文存储为：

- 前置 StC：`560.75 MiB -> 111.121 MiB`，压缩 `5.05x`
- ModRaise 后 C2S：`1651.25 MiB -> 365.302 MiB`，压缩 `4.52x`
- 两者合计：`2212 MiB -> 476.423 MiB`，压缩约 `4.64x`

使用 5 次 warmup 加 5 次 CUDA-event 测试，完整与压缩路径在同一进程中 A/B：

- StC：`11.6672 ms -> 11.1486 ms`，减少 `0.5186 ms`，提升约 `4.45%`
- C2S：`53.7624 ms -> 51.8876 ms`，减少 `1.8748 ms`，提升约 `3.49%`

**适用范围与状态。** 当前结果来自独立的 `POSEIDON_BOOTSTRAP_COMPRESSED_QP_MAC_PROBE=1` 实验路径，尚未设为默认。下一步需要在原始生产顺序的完整自举中验证最终精度、整体时间和显存峰值；只有完整自举仍为正向，才允许默认启用。无法证明精确周期的明文必须回退到完整布局。

### 10. 22 阶 baby-4 EvalMod 的 remainder-chain lazy relinearization

**原始瓶颈。** 前置 StC 调度下的 22 阶、3 次 double-angle 实验使用 baby width 4，将多项式拆成 6 个最高三阶的 leaf，再通过 5 个 `Q·T_k+R` combine 节点恢复完整结果。虽然59阶路径已经实现 remainder-chain 延迟重线性化，但原实现用拟合阶数硬编码限制其只对58/59阶生效，导致22阶计算图的5个combine全部立即执行一次 P=9 Hybrid KeySwitch。计算图中两个内部combine输出只被各自父节点作为 remainder 消费，本不需要在加法前单独变成 size-2 密文。

**优化思路。** 将启用条件从“59阶专用”改为“经过验证的阶数加通用图条件检查”。对于只使用一次、仅位于父节点 remainder 边、并且边上不需要额外 rescale 或 plaintext scale 对齐的combine输出，保留其 size-3 形式。父节点的 size-3乘积与raw remainder逐分量相加，沿两条可延迟边继续传播，最后只在结果即将作为quotient参与后续乘法或成为根输出时执行relinearization。该baby-4树中共有2个可延迟中间输出，因此5次combine乘法保持不变，但combine KeySwitch由5次降为3次；加上5次basis和3次double-angle，EvalMod总KeySwitch由13次降为11次。

quotient边仍禁止传播size-3密文，因为size-3 quotient再乘size-2 basis会生成size-4密文，并需要当前relinearization key不支持的更高次秘密密钥分量。上传后的执行计划增加严格检查：基底必须恰好为 `T2,T3,T4,T8,T16`，必须生成6个leaf和5个combine，并且恰好有2个输出满足延迟条件。任何level、scale或树结构变化导致条件不成立时，路径保留立即relinearize语义或由实验断言拒绝。

**性能结果。** 在 V100、`N=65536, Q=34, P=9`、前置StC、22阶截断多项式、baby width 4、3次double-angle、动态rescale下，使用5次warmup加5次CUDA-event测量：

- EvalMod（实部加虚部）：`48.7348 ms -> 45.3534 ms`，减少 `3.3814 ms`，提升约 `6.94%`
- 完整前置StC自举：`112.215 ms -> 108.609 ms`，减少 `3.606 ms`，提升约 `3.21%`

GPU/CPU EvalMod最大误差保持在约 `2.1e-10`，最终GPU/CPU差异约 `8.6e-9`；EvalMod输入/输出仍为 `Q:30->15`，输出scale仍为约 `2^49.0684`。当前22阶系数来自59阶Chebyshev系数截断而不是独立Remez拟合，因此对源消息的约 `6.78` 误差仍由多项式逼近产生。本优化只验证同一错误多项式的GPU计算等价性，不宣称该参数已经满足自举精度要求。

**适用范围与状态。** 该优化现在是 `slim22_da3` 命名实验路径的默认行为，同时保留 `POSEIDON_EVALMOD_LAZY_RELIN=0` 回退。59阶默认路径继续使用相同的图条件机制和已有验证结果；生产 `dynamic32` 参数没有被22阶实验配置覆盖。

### 11. Giant Key-MAC 的 output-major QP 寄存器归约

**原始瓶颈。** Double-Hoist 已经将所有 giant rotation 的 Key-MAC 结果留在同一个 QP 基中，并在最后共享一次 outer ModDown，因此不存在逐个 group 重复 ModDown 的问题。但是旧 kernel 的线程网格仍包含 giant-group 维度：7 个非零 group 分别生成一个 size-2 QP ciphertext，`g=0` 的 identity group 也单独物化，随后 grouped outer ModDown 再从显存读取全部8组结果并求和。在第一层 `Q=6, P=9, N=65536` 下，当前32-bit `GpuWord` 使一个 QP ciphertext 为7.5 MiB；8组中间结果的写出和重新读取合计约120 MiB，而新路径只写入并读回一个7.5 MiB累加结果、约15 MiB流量，因此每次该矩阵变换约减少105 MiB global-memory traffic。

**优化思路。** 将 giant Key-MAC 从 `group × limb × coefficient` 改为 output-major 的 `limb × coefficient` 线程布局。每个线程独占最终密文两个分量中固定的 RNS residue，依次遍历7个非零 giant group：根据各自 Galois 元素计算输入位置，读取该组 hoisted digit 和预旋转 evaluation key，完成 Key-MAC，并在两个寄存器中及时模加。`g=0` 的 Q 密文以乘 `P mod q` 的方式直接初始化同一寄存器累加器，P 侧初始化为零。循环结束后每个线程只写一次 QP 结果，然后调用现有单 ciphertext outer ModDown。该设计不使用 atomic 或 shared memory，也不改变7次 decomposition、7次 Key-MAC以及最终1次ModDown的计算数量；收益完全来自避免多组 QP ciphertext 的物化与归约。

正确复现要求所有 giant group 位于同一 QP 基、level和scale，并由各自Galois key切换回同一个目标私钥。模加必须在循环内进行，不能把多个接近32-bit模数的乘积留到最后用64-bit整数一次累加。该路径只在 inverse-pre-rotated key 格式、存在 identity group且group数大于1时启用；否则继续使用原 grouped-buffer fallback。新 kernel 使用60个寄存器，旧 kernel为56个，二者均为 `STACK=0, LOCAL=0`，没有寄存器spill。

**正确性与性能结果。** 最终主测试采用后续优化工作的默认研究配置：V100、`N=65536, P=9`、前置StC、22阶截断Chebyshev、baby width 4、3次double-angle、动态rescale。为排除GPU0上的外部负载，关闭和开启路径都固定在同一张空闲V100上，并分别使用5次warmup加5次CUDA-event正式测量。低层StC保持 `Q:6->2`、rescale模式 `1,1,2` 和相同scale trace；22阶已知的多项式逼近误差在两条路径均为 `final source error=6.77884`、`polynomial approx=6.77809`，没有被本优化进一步放大。最终CPU/GPU差异从约 `9.16e-9` 变为 `8.26e-9`，属于同量级计算误差。性能结果为：

- 前置 StC：`11.7108 ms -> 11.3022 ms`，减少 `0.4086 ms`，提升约 `3.49%`
- ModRaise 后 C2S：`53.9440 ms -> 51.9374 ms`，减少 `2.0066 ms`，提升约 `3.72%`
- EvalMod：`44.7262 ms -> 45.2544 ms`；该阶段不经过线性变换融合，约 `0.53 ms` 差异视为运行波动
- 完整22阶前置-StC自举：`108.412 ms -> 107.054 ms`，减少 `1.358 ms`，提升约 `1.25%`

59阶、2次double-angle路径也做过独立的1+1交叉验证：前置StC约提升3.8%，C2S约提升2.1%，完整路径约提升1.1%。这组结果作为跨EvalMod配置的补充证据，不再作为本条目的主性能数字。

**适用范围与状态。** 该优化已成为 inverse-pre-rotated Double-Hoist giant阶段的默认实现。它不依赖固定 `P=9` 或固定 decomposition 数量，但当前性能结论来自 `N=65536, P=9`；其他参数仍受通用形状检查保护，并可设置 `POSEIDON_DOUBLE_HOIST_DIRECT_GIANT_ACCUMULATE=0` 回退到逐group物化和统一ModDown的旧路径。

### 12. QP plaintext-MAC 首 tile 直接初始化与 dnum 自适应分块

**原始瓶颈。** Double-Hoist 的 baby-step 结果按 tile 生成，并与各条对角明文相乘后累加到 giant-group QP accumulator。旧路径在每个线性变换 stage 开始时，先分别对完整 Q、P accumulator 执行一次 `cudaMemset`；第一个 plaintext-MAC kernel 随后又把这块全零数据从 HBM 完整读回，执行模加后再写回。对于默认 baby tile 4，每个 stage 的 accumulator 因而经历“清零写、第一 tile 读写、第二 tile 读写”共5倍 accumulator 大小的流量，其中最初两倍流量不承载任何有效信息。

**优化思路。** plaintext-MAC kernel 增加仅用于第一 tile 的初始化语义：每个输出线程从寄存器零值开始累加，并直接覆盖其唯一负责的 QP accumulator residue，而不是读取预先清零的 global-memory 值。后续 tile 继续读取并累加已有结果，因此模运算顺序和原路径完全一致。通用、一组分量一个线程的kernel，giant-group tiled kernel以及双密文分量融合kernel均实现同样语义；旧路径可通过环境变量恢复，便于参数外回归。

在此基础上，baby tile 不再全局固定为8。`dnum=(Q+P-1)/P` 等于1时，hoisted baby rotation只有一个decomposition digit，baby工作区较小而accumulator流量占比更高，因此将tile从4增到8，把两轮QP MAC合并成一轮；`dnum>1` 的高层C2S继续使用tile 4，避免更大的baby QP工作区和tile 8在实测中对C2S造成的轻微回退。这一选择同时保留显存预算检查。

**流量和launch变化。** 在本次 `N=65536, P=9`、三层StC和三层C2S参数下，首tile直接初始化去掉每个stage的两次整accumulator流量和Q/P各一次同步`cudaMemset`。按当前各层Q宽度累计，前置低层StC约少284 MiB、ModRaise后高层C2S约少844 MiB读写。低层StC进一步使用tile 8后，再少约284 MiB accumulator读写，并减少3层合计6次Q/P plaintext-MAC kernel launch；最终一轮StC+C2S总计约减少1.38 GiB HBM traffic。高层C2S保留tile 4，因为全局tile 8虽然把C2S accumulator流量进一步降低，但在V100上的单项时间从约52.10 ms回升到52.63 ms，推测增大的baby工作集和较低的有效缓存/占用收益抵消了流量节省。

**正确性与性能结果。** 测试固定为V100、`N=65536, Q=34, P=9`、前置StC、22阶截断Chebyshev、baby width 4、3次double-angle、动态rescale，采用2次warmup和5次CUDA-event正式测量，并在相反顺序下复测旧路径。旧路径两次稳定结果为：StC `11.3008/11.2912 ms`，C2S `54.2218/55.1638 ms`，完整流程 `107.111/107.000 ms`。最终自适应路径的代表结果为：

- 前置StC：`10.401 ms`，相对稳定旧基线提升约`7.9%～8.2%`
- ModRaise后C2S：`52.10～53.96 ms`，相对对应旧测量提升约`2.2%～3.9%`
- EvalMod：约`46.07 ms`，不经过该线性变换路径，没有归因于本优化
- 完整22阶前置-StC自举：`105.389～105.471 ms`，稳定减少`1.53～1.72 ms`，提升约`1.4%～1.6%`

StC层级轨迹保持`6->5->4->2`、C2S保持`34->33->32->30`，各stage scale与rescale数量不变；最终CPU/GPU差异约`8.44e-9`。22阶截断多项式对源消息的已知误差仍为`6.77884`，没有额外增加。全局tile 8的完整时间为`105.457 ms`，与tile 4几乎相同，但C2S单项回退，因此不作为默认策略。

**适用范围与状态。** 首tile直接初始化已默认启用，`dnum=1` 默认使用baby tile 8，其余形状继续服从全局baby tile（当前脚本为4）。设置`POSEIDON_DOUBLE_HOIST_QP_MAC_DIRECT_INIT=0`可恢复显式清零路径；设置`POSEIDON_GPU_DOUBLE_HOIST_DNUM1_BABY_TILE=0`可关闭dnum=1覆盖并恢复全局tile选择。两项默认状态会由测试脚本末尾的warning明确打印。

### 13. dnum=1 Baby KeySwitch 专用化与 c0 窄融合

**原始瓶颈。** 前置StC的三层Q宽度依次为6、5、4，而P固定为9，因此每层`dnum=ceil(Q/P)=1`。旧路径仍调用支持任意dnum的batch Key-MAC kernel：每个线程保留动态digit循环、通过指针表反复索引evaluation key，并使用48个寄存器。每个非零baby rotation完成Key-MAC后，Q侧还要启动独立的`add_lifted_galois_c0` kernel，重新读取Key-MAC输出和旋转后的`c0`，写回同一Q0分量。三层各有7个非零baby，合计21个独立add kernel。

**优化思路。** 为`dnum=1`增加无循环专用kernel：每个线程只读取唯一hoisted digit和唯一evaluation-key分量，直接产生两个密文分量。未融合版本的寄存器由48降至24，且无stack/local spill。在此专用kernel中进一步将旋转后的`c0*P mod q`加入第0分量，使Key-MAC最终写回时已经得到完整的lifted QP baby ciphertext。融合版本使用30个寄存器，仍可在V100上保持高驻留率；P侧继续使用24寄存器的专用kernel。plaintext-MAC保持为独立的32寄存器group-tiled kernel，以维持其访存合并和调度效率。

该边界不融合KeySwitch与plaintext-MAC本身。完整融合实验虽然删除约336 MiB baby QP物化和约66次launch，但Q kernel达到48个寄存器并产生64字节stack frame，StC由`10.401 ms`回升到`10.826 ms`；融合tile降为4后进一步恶化到`12.773 ms`。这说明V100上将evaluation-key读取、Galois索引和多个对角MAC串进一个长kernel会削弱延迟隐藏，不能仅按HBM字节数判断收益。

**访存与launch变化。** dnum=1专用化不改变evaluation-key乘法次数，只删除动态循环控制。c0窄融合在前置StC三层删除21个kernel launch，并避免Q0分量在add kernel中的一次读写。按`Q=6,5,4`、`N=65536`和7个非零baby计算，约减少52.5 MiB global-memory traffic。它不改变hoisted decomposition、baby QP存储、plaintext-MAC、inner/outer ModDown或rescale数量。

**正确性与性能结果。** 在V100、`N=65536, Q=34, P=9`、前置StC、22阶截断Chebyshev、3次double-angle路径下，采用2次warmup和5次CUDA-event正式测试：

- 上一阶段前置StC：约`10.401 ms`
- 仅dnum=1无循环专用化：`10.3354 ms`
- 专用化加c0窄融合：`10.1882～10.2192 ms`
- 完整22阶前置-StC自举：`105.389～105.471 ms -> 105.116 ms`，再减少约`0.27～0.36 ms`

StC层级仍为`6->5->4->2`，最终CPU/GPU差异约`8.63e-9`，22阶多项式的已知源误差仍为`6.77884`。作为边界对照，dnum=4高层C2S的通用Key-MAC+c0融合需要54个寄存器，理论驻留率从约62.5%降到50%；反向A/B仅有`52.300 -> 52.183 ms`、约0.22%的差异，低于运行波动，因此没有对C2S默认启用。

**适用范围与状态。** dnum=1无循环Key-MAC已作为通用精确形状专用化默认使用；c0窄融合仅在inverse-pre-rotated key格式、dnum=1的baby rotation上默认启用。设置`POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_C0=0`可恢复独立`add_lifted_galois_c0` kernel。dnum大于1继续使用原batch Key-MAC和独立c0路径。

### 14. 跨 giant group 的两阶段 Four-step INTT 批处理

**原始瓶颈。** QP plaintext-MAC得到所有giant-group accumulator后，当前Double-Hoist已经批量完成inner ModDown；但是每个非零giant group的第二个密文分量仍逐组调用一次outer hoisting。对于8个giant group（1个identity加7个非零rotation），每组首先对完整Q多项式执行一次N=65536 inverse NTT，再进行decomposition、ModUp和forward NTT。Cheddar/Four-step inverse NTT由两个kernel phase组成，因此每个线性变换stage需要为这一步发射`7×2=14`个kernel。三层StC或C2S各需要42次，完整前置StC自举中的两次线性变换合计84次；这些launch执行完全相同的模数表和变换，只是输入、输出batch不同。

**优化思路。** 将非零giant group作为第三个grid维度并入同一组Four-step INTT kernel。phase1使用`grid.z=7`读取连续Q密文batch中各组的`c1`，写入连续的coefficient-domain batch；phase2沿相同batch维度原位完成剩余变换和`N^{-1}`缩放。随后每组原有ModUp/forward-NTT继续读取自身的coefficient-domain切片，因此本阶段不改变decomposition数量、NTT蝶形数量、RNS顺序或evaluation-key运算。每个stage的outer-hoist INTT由14次launch降为2次，三层线性变换由42次降为6次，完整StC+C2S共减少72次kernel launch。

批处理只改变launch组织，不增加中间物化：新的连续coefficient batch替代各group分别持有的`source_intt_q`，总字数仍为`7×Q×N`。输入布局固定为`[group][component][Q][coefficient]`，因此source group stride精确为`2×Q×N`，destination stride为`Q×N`；把这两个stride从动态kernel参数改为由shape直接推导后，batched和单组kernel的资源完全一致：phase1均为34个寄存器，phase2均为56个寄存器，`STACK=0, LOCAL=0`，没有spill。

**正确性与性能结果。** 主验证参数为V100、`N=65536, Q=34, P=9`、前置StC、22阶截断Chebyshev、baby width 4、3次double-angle和动态rescale，使用2次warmup加5次CUDA-event测量。低层StC反向A/B结果为：

- 逐group INTT：`10.2032 ms`、`10.2194 ms`，平均约`10.211 ms`
- batched INTT：`10.0580 ms`、`10.1222 ms`，平均约`10.090 ms`
- 前置StC平均减少约`0.121 ms`，提升约`1.18%`

完整`slim22_da3`路径中，关闭批处理为`105.353 ms`；开启后的两次独立进程测量为`104.567 ms`和`104.722 ms`，平均`104.645 ms`，减少约`0.709 ms`、提升约`0.67%`。独立C2S stage profile在不同进程间存在约1ms级波动，因而不单独宣称C2S算子收益；完整计时和低层StC的重复A/B均保持正向。

所有层级和scale轨迹保持不变：StC仍为`6->5->4->2`，C2S仍为`34->33->32->30`，EvalMod仍为`30->15`。最终GPU/CPU差异约`8.34e-9～1.06e-8`；22阶截断多项式的已知源误差仍为`6.77884`，没有增加。

**适用范围与状态。** 该优化在Four-step NTT、`N=65536`、inverse-pre-rotated Double-Hoist、identity giant group位于第0组且其余group均为非零rotation时默认启用；任何shape或NTT算法不满足条件时自动回退逐group路径。设置`POSEIDON_DOUBLE_HOIST_BATCHED_GIANT_INTT=0`可以显式恢复旧实现。当前只批处理outer hoisting的inverse NTT；ModUp和forward NTT仍逐group发射，留给后续独立阶段验证。

### 15. 前置 StC 路径的 `[5,4,3,3]` C2S 分层与小矩阵 direct 计算

**原始问题。** `slim22_da3` 前置StC路径原来将 `logSlots=15` 的C2S FFT层固定合并为 `[5,5,5]`。对当前C2S编码器，三个矩阵实际包含 `[32,63,63]` 条非零对角线，后两层为了使用BSGS需要同时维护baby-step和giant-step中间结果。对仅合并3层FFT的矩阵，非零对角线只有15条；此时继续拆成两级BSGS的中间结构收益很小，反而会保留giant rotation、inner ModDown和outer hoisting调度。

**优化思路。** 新增独立的 `slim22_da3_c2s5433` 试验路径，将C2S合并层显式设为 `[5,4,3,3]`。5层和4层矩阵仍分别使用BSGS；两个3层矩阵将 `n1` 设为全部slot数，使15条对角线全部归入唯一的identity giant group。这两层只执行source hoisting、baby rotations、QP plaintext MAC和一次最终ModDown，不生成非零giant group，也不再进入giant-step KeySwitch。为避免直接路径被旧Double-Hoist外层强制执行 `ModDown -> lift to QP -> ModDown`，对“唯一identity group”形状增加了一次ModDown直达出口。

这里的“压缩”只表示通过改变FFT分层减少需要编码的对角明文数量，并不是WHET的周期性明文存储。该profile目前仍上传完整长度的QP plaintext；`GpuCompressedPlaintextQP`的bit-reversed periodic representation和compressed QP MAC仍属于单独probe，没有在本路径默认启用。因此本项可以与WHET式周期压缩继续叠加，两者不是同一个优化。

实际四层矩阵形状为：5层 `32 diagonals, 8 baby x 4 giant`；4层 `31 diagonals, 8 baby x 4 giant`；两个3层各为 `15 diagonals, 15 baby x 1 identity giant`。因此对角明文总数从158条减少到93条，减少约`41.1%`。按各stage实际active Q limb和 `P=9`估算，QP明文存储由 `6605` 个limb-polynomial降到 `3878`；在 `N=65536`、32-bit residue下约由 `1651.3 MiB` 降到 `969.5 MiB`，减少约 `681.8 MiB`。这一估算只计GPU QP对角明文，不将CPU编码副本或rotation key计入。

**层数与scale代价。** 当前动态rescale策略保持每个明文矩阵scale为 `2^45`。四层C2S的物理drop模式为 `1,1,2,1`，因此C2S从 `Q:34->29`，而原三层路径为 `1,1,2` 和 `Q:34->30`。新C2S输出scale为约 `2^68.015`，仍位于约定的 `[2^44,2^76)` 区间。后续22阶EvalMod会从实际C2S输出动态重建计划，新路径为 `Q:29->12`，旧路径为 `Q:30->15`。因此这是“明文存储和延迟换取更多物理Q prime”的独立权衡，不是无代价的默认替换。

**正确性与性能结果。** 测试固定为V100、`N=65536, Q=34, P=9`、前置StC、22阶Chebyshev截断、baby width 4、3次double-angle和动态rescale。两组独立1次warmup+1次CUDA-event测量都得到约 `40.8 ms` 的新C2S和 `55.4~55.6 ms` 的旧C2S，收益远大于此机器通常的测量波动。同一次完整A/B为：

- ModRaise后C2S：`55.592 ms -> 40.800 ms`，减少 `14.792 ms`，提升约 `26.6%`
- EvalMod：`48.013 ms -> 44.702 ms`，约提升 `6.9%`；这部分主要来自输入Q limb减少，不应解读为EvalMod计算图本身被优化
- 完整22阶前置StC自举：`107.946 ms -> 91.556 ms`，减少 `16.390 ms`，提升约 `15.2%`

C2S real/imag CPU-GPU最大误差为约 `1.18e-10/1.91e-10`，说明四层线性变换本身正确。新路径最终GPU/CPU delta约为 `9.61e-6`，高于旧路径的 `8.55e-9`，但仍小于当前 `1e-3` 算术容差；该变化与输出Q层数降低相符。22阶系数仍是59阶多项式的截断，因而两条路径的已知 `final source error=6.77884` 和 `polynomial approx=6.77809` 完全相同；本优化没有进一步放大多项式逼近误差。

**适用范围与状态。** 该优化仅由命名profile `slim22_da3_c2s5433` 启用；现有 `slim22_da3` 的 `[5,5,5]` C2S与生产 `dynamic32` 路径都不受影响。运行时会显式打印四层的diagonal、baby、giant和mode，并对 `[32,31,15,15]` 及两个direct identity-only plan进行强校验。在进一步解决额外Q prime消耗和最终GPU/CPU delta增大之前，不将它改为默认路径。

后续对C2S和22阶EvalMod做过真实prime联合搜索。将C2S尾部rescale primes调成 `31+23` bit并把末端double-angle边界调到30 bit，可把C2S/EvalMod总消耗从22枚降到21枚且两端scale都约为`2^45`，但C2S roundtrip误差从约`1e-10`增至`5.19e-6`，完整时间从`91.556 ms`回退到`92.388 ms`，source error也从`6.77884`增至`6.78436`。把尾部两枚prime均衡为 `27+27` bit后，roundtrip误差仍为`6.25e-6`；严格2次warmup加5次测量中，C2S为`39.785->40.667 ms`、EvalMod为`41.761->45.008 ms`、完整路径为`89.538->90.388 ms`，少一枚prime但性能反而回退。另一条保持C2S `34->29` 的21枚候选虽然恢复了C2S精度，却破坏CPU EvalMod的level对齐不变量。所有21枚候选均未设为默认，也不计入正向性能演进；当前链仍保留22枚总消耗。

进一步把搜索边界严格限定为前置StC和ModRaise之后的 `C2S(4个逻辑深度)+EvalMod(约8个逻辑深度)`，并在N=65536真实prime上联合搜索 `q[13..33]`。搜索可以得到 `C2S 34->28, EvalMod 28->14` 的20枚物理prime方案，但其最终scale为 `2^75.069`，几乎贴住动态scale合法区间上沿。完整1次warmup加1次测量中，CPU/GPU最终差为`9.56e-10`、HE算术误差为`8.21e-10`，说明计算本身正确；但C2S/EvalMod/完整路径分别为`40.582/43.879/90.735 ms`，没有性能收益。将最高保留边界prime改为30 bit后，调度会再做一次物理rescale，得到 `28->13` 和最终 `2^45.073`；该版本CPU/GPU最终差为`9.75e-10`，完整路径为`91.270 ms`。因此第14枚表面保留的Q prime实际承载约30 bit尚未清偿的scale，若输出需要回到本路径原生`2^45`工作scale，它不是可继续使用的净计算层。

**默认选择。** 虽然归一化21枚方案相对此前最快的22枚链有约`1.3～1.7 ms`的局部回退，但它把可直接继续计算的输出Q数量由12提高到13；在包含多次自举的真实电路中，这一层裕度可能延后下一次自举，从全局上抵消局部延迟。因此将 `34->28->13`、最终scale约`2^45`的链设为命名profile `slim22_da3_c2s5433` 的默认配置。取消显式chain覆盖后的默认路径复测得到C2S `40.784 ms`、EvalMod `44.107 ms`、完整路径`90.833 ms`，输出scale为`2^45.0732`，最终CPU/GPU差为`1.08e-9`。20枚高scale方案不设为默认，生产profile `dynamic32`及其他自举路径不受影响；显式设置`POSEIDON_BOOTSTRAP_Q_BIT_CHAIN`仍可覆盖该选择。

### 16. EvalMod 动态 leaf 的目标层 CAccum 与同步 D2D 消除

**原始瓶颈。** 22阶动态EvalMod的6个leaf虽然已经允许“先累加、后统一处理”，但运行时仍按单项执行：每个Chebyshev basis先与对应系数明文做独立PMult，生成完整size-2 ciphertext；不同basis位于不同Q层时，再通过同步DtoD复制把较高Q前缀物化到leaf目标层；随后以完整ciphertext Add链累加。常数项和Chebyshev校正使用out-of-place `add_plain/sub_plain`，为了只修改`c0`却仍需把未改变的`c1`（size-3时还包括`c2`）复制到新buffer。因此这部分既有重复HBM traversal，也在CPU线程上产生阻塞式`cudaMemcpy`。

**优化思路。** 动态leaf的ModDrop只删除Q后缀，不改变保留prime上的NTT residue。于是无需先复制出低层ciphertext：新路径直接把各basis和系数明文的底层相同Q前缀作为只读view，在leaf最终`output_q_count`上计算。每个kernel同时处理`c0/c1`，在寄存器中合并最多4个`basis_i * plaintext_i`，直接写入最终leaf accumulator；6个leaf的常数系数也融合进各自第一批CAccum，对`c0`加常数而不触碰`c1`。这相当于把“PMult物化→Q前缀DtoD→多遍Add”改成一次目标层流式归约，并保持所有basis原buffer可供后续节点复用。

对EvalMod中其他只改变`c0`的常数加减，新增原地路径：`c0`通过modular add/sub kernel更新，`c1/c2`保留在原地址，不再复制。输入准备直接写入`T1`持有的buffer，去掉`source→scratch→T1`的一次完整中转；动态rescale规划为0次时直接复用原对象；double-angle无需rescale的分支也用所有权移动代替DtoD。这里没有把memcpy伪装成copy kernel，减少的是实际global-memory字节和同步API。

**计算图与访存变化。** 在`N=65536, Q=34, P=9`、`slim22_da3_c2s5433`、baby width 4、3次double-angle、EvalMod `Q:28->13`下，leaf区间的GPU kernel由96个降为12个：原先64个component级PMult、20个two-component Add和12个单component Add被12个two-component四项CAccum替代。leaf GPU kernel累计时间由`3.348 ms`降为`1.397 ms`。整个EvalMod kernel数由925降为841，恰好减少84次；basis、BSGS combine和double-angle的kernel数分别保持`346/292/184`，证明改动只压缩leaf而没有删错计算图。

完整自举capture中的同步DtoD由200次、`963.641 MB`、`3.113 ms`降为104次、`350.749 MB`、`1.215 ms`。结合未变化的线性变换区间，EvalMod自身的同步DtoD约由134次降为38次，减少96次；完整capture减少约`612.9 MB`读取/写回载荷和约`1.90 ms`GPU memcpy时间。剩余38次主要属于仍需保留独立输入的basis/combine level alignment，并非本轮leaf物化。

**正确性与性能结果。** 同一V100和同一二进制使用`POSEIDON_EVALMOD_D2D_FREE_DATAFLOW=0/1`，各执行1次warmup加3次CUDA-event测量：

- 回退物化路径：EvalMod `43.816 ms`，完整前置StC自举 `88.995 ms`
- 目标层CAccum路径：EvalMod `38.323 ms`，完整前置StC自举 `84.121 ms`
- EvalMod减少`5.493 ms`、提升`12.54%`；完整路径减少`4.874 ms`、提升`5.48%`

EvalMod real/imag CPU-GPU最大差保持在约`2.43e-11/2.26e-11`，完整CPU-GPU差约`1.02e-9`；输入输出层仍为`28->13`，最终scale仍为`2^45.0732`。22阶截断路径固有的`final source error=6.78436`和`polynomial approx=6.78361`没有变化，因此本优化没有增加已有的多项式逼近误差。

**适用范围与状态。** 新数据流默认启用。动态leaf只有在所有非零项均为NTT form、basis与plaintext至少覆盖共同目标Q前缀、乘积scale一致且输出为两分量时才进入四项CAccum；不满足条件自动回退逐项求值。原地常数加减同样要求parms、shape、NTT form和scale严格匹配。设置`POSEIDON_EVALMOD_D2D_FREE_DATAFLOW=0`可整体恢复旧leaf物化、out-of-place明文加减和同步D2D，便于回归；`POSEIDON_EVALMOD_CACCUM_LEAF=0`可只关闭leaf CAccum。最终报告为`profiles/bootstrap22_leaf_d2dfree_v100_20260820.nsys-rep`，报告属于测试产物，不提交仓库。

### 17. EvalMod BSGS combine 的 last-use 零拷贝 ModDrop

**原始瓶颈。** 第16项消除leaf物化后，nsys仍显示每个实部/虚部EvalMod的BSGS combine区间包含多次同步DtoD。它们来自动态计算图的level alignment：当quotient、Chebyshev basis、临时product或remainder的active Q数量不同时，旧实现会重新分配目标ciphertext，把保留的Q前缀逐component复制到新buffer，再立即进入乘法或加法。这里的`drop_modulus`不是CKKS rescale，也不进行模运算；它只丢弃RNS链末端limb，因此复制出的内容与原buffer的低Q前缀逐bit相同。

**优化思路。** 为多项式combine DAG预先统计每个node的消费次数，并记录每个basis degree的最后一个combine消费者。当一个node只剩当前这一次消费、一个basis到达最后一次消费，或者对象本身就是当前combine产生的独占临时量时，不再物化Q前缀，而是保留原GPU allocation和每个ciphertext component的物理起始地址，仅把`parms_id/q_count`以及shard的逻辑`limb_count`缩到目标Q。后续kernel原本就根据每个component的shard指针和逻辑limb数访问数据，物理stride中未使用的尾部padding不会被读取。这样没有用copy kernel替代memcpy，也没有改变任何residue，而是直接删除整次HBM读写。

该路径只处理单个full-Q shard、`P=0`、相同degree且目标level为源Q前缀的纯ModDrop。仍会被其他DAG节点复用的quotient/remainder、尚未到最后消费者的basis、布局不满足条件的对象，以及开启逐节点EvalMod trace时，全部自动回退原`drop_modulus`复制路径。真实rescale仍执行原有模约减kernel，不能也没有被此优化绕过。

**访存与调用变化。** 在V100、`N=65536, Q=34, P=9`、`slim22_da3_c2s5433`、degree-22/baby-4/DA3和EvalMod `Q:28->13`下，两个EvalMod分支的BSGS combine各删除6个GPU memory op，区间GPU op由`159->153`；密码学kernel数量和乘法/relinearization图保持不变。完整自举capture的DtoD由104次、`350.749 MB`、`1.215 ms`降为92次、`287.834 MB`、`0.999 ms`，即实际减少12次同步复制、`62.915 MB`传输和约`0.216 ms`memcpy engine占用。combine投影时长由实/虚分支约`6.145/6.143 ms`降为`5.833/5.846 ms`，合计减少约`0.61 ms`；其余差额主要来自同步API及临时allocation/materialization不再阻塞host dataflow。该结果说明收益确实来自预期的combine level alignment，而非其他阶段波动。

**正确性与性能结果。** 同一Release二进制固定GPU并仅切换`POSEIDON_EVALMOD_ZERO_COPY_MODDROP=0/1`，两组均执行1次warmup加3次CUDA-event测量：

- 复制Q前缀：EvalMod `40.009 ms`，完整前置StC自举 `85.976 ms`
- last-use逻辑缩短：EvalMod `39.325 ms`，完整前置StC自举 `84.619 ms`
- EvalMod减少`0.684 ms`、提升约`1.71%`；完整路径减少`1.357 ms`、提升约`1.58%`

两组输出层均为`Q:28->13`，最终scale均为`2^45.0732`；新路径EvalMod real/imag CPU-GPU最大差约为`2.23e-11/2.46e-11`，完整CPU-GPU差约为`9.68e-10`。已知的degree-22截断`final source error=6.78436`和`polynomial approx=6.78361`不变。因此该修改没有改变层数、scale或近似精度。

**适用范围与状态。** 该优化默认启用，当前实现首先覆盖实际热点的动态lazy-relinearization combine路径；不满足严格shape和生命周期条件时透明回退。设置`POSEIDON_EVALMOD_ZERO_COPY_MODDROP=0`可恢复复制式Q前缀物化，便于回归。最终nsys报告为`profiles/bootstrap22_zero_copy_moddrop_v100_20260820.nsys-rep`，报告属于测试产物，不提交仓库。

### 18. 共享 basis/combine operand 的只读 Q-prefix view

**原始瓶颈。** 第17项只能原地缩短已经到最后一次使用的对象；T1/T2等Chebyshev basis以及部分BSGS quotient/basis仍会被后续DAG节点复用，不能修改其`q_count`。当一个共享高层ciphertext要和低层operand相乘时，旧路径因此仍需分配低层临时buffer，并按component同步复制其Q前缀。最终nsys定位到每个实部/虚部EvalMod的basis generation和BSGS combine分别保留2个这样的`drop_modulus`调用，也就是每个分支8个component级DtoD memory op。

**优化思路。** 新路径不修改共享ciphertext的所有权和元数据，而是在单次乘法调用内构造临时只读view：view继承原component的GPU起始指针，只把本次调用可见的`parms_id/q_count/shard.limb_count`限制到双方共同的低Q前缀。乘法输出直接按该目标level分配，底层two-component ciphertext multiply仍读取完全相同的NTT residues并启动原kernel。调用结束后view销毁，原高层basis仍保持完整，可继续服务其他leaf/combine节点。

只有两个输入均为NTT form、size-2、`P=0`、相同degree/device、单个full-Q shard且目标是双方共同Q前缀时才使用该路径；否则回退复制式ModDrop。对于basis correction中已经独占的output/PMult临时量，仍采用第17项的原地逻辑缩短。该设计没有浅拷贝GPU内存所有者，也没有让长期对象持有外部指针，因此不引入悬空引用或双重释放风险。

**计算图与访存变化。** 在V100、`N=65536, Q=34, P=9`、`slim22_da3_c2s5433`、degree-22/baby-4/DA3和EvalMod `Q:28->13`下，basis generation每个实/虚分支的GPU op由`177->173`，BSGS combine由`153->149`；两阶段各删除4个memory op，密码学kernel数完全不变。完整自举DtoD由92次、`287.834 MB`、`0.999 ms`降为76次、`185.074 MB`、`0.691 ms`，即删除16次同步复制、`102.760 MB`传输和约`0.308 ms`memcpy engine占用。`multiply_outer_components`仍为26次、`multiply_cross_component`仍为12次，且两份nsys报告中它们的累计GPU时间几乎相同，证明优化没有退化到另一套慢乘法kernel。

**正确性与性能结果。** 第一次1次warmup加3次测量受到机器速度漂移影响，ON比OFF慢约`0.269 ms`；同一次nsys中连未修改的double-angle也整体慢约3%～4%，因此没有据此直接设为默认。随后反转测试顺序并把两组都提高到2次warmup加5次CUDA-event测量：

- Q-prefix view关闭：EvalMod `39.1506 ms`，完整前置StC自举 `83.9174 ms`
- Q-prefix view开启：EvalMod `37.9392 ms`，完整前置StC自举 `83.5088 ms`
- EvalMod减少`1.2114 ms`、提升约`3.09%`；完整路径减少`0.4086 ms`、提升约`0.49%`

实验组EvalMod real/imag CPU-GPU最大差约为`2.28e-11/2.42e-11`，完整CPU-GPU差约为`9.29e-10`；输出仍为`Q:28->13`和scale `2^45.0732`。已知的degree-22截断近似误差仍为`6.78436`，没有新增数值误差。相邻两次测试的StC/C2S本身仍有约1%量级漂移，所以完整路径收益应保守理解；EvalMod局部收益同时得到稳定复测和结构性nsys计数支持。

**适用范围与状态。** 该优化默认启用，设置`POSEIDON_EVALMOD_Q_PREFIX_VIEWS=0`可恢复共享operand的复制式level alignment。它依赖第17项的安全shape检查，逐节点EvalMod trace模式会自动回退复制路径。最终nsys报告为`profiles/bootstrap22_qprefix_probe_v100_20260820.nsys-rep`，报告属于测试产物，不提交仓库。

### 19. Baby KeySwitch 与 QP plaintext MAC 的寄存器直达融合

**原始瓶颈。** Double-Hoist虽然已经把一次source decomposition复用于全部baby rotations，但旧的baby阶段仍以完整QP ciphertext作为KeySwitch与明文矩阵乘之间的接口。每个baby rotation先分别在Q、P基上执行evaluation-key MAC，将两个ciphertext component完整写入`baby_tile`；随后的QP plaintext-MAC kernel再把它读回，与各giant group对应的diagonal相乘并累加。高性能库的nsys中出现`fused_kswitch_mac_ptx`一类主kernel，而本库融合前的C2S则由`pre_rotated_keymul_batch_kernel`和`qp_plain_mul_accumulate*`两族独立热点构成，这说明两条dataflow在中间结果物化上仍有实质差异。

**优化思路。** 新kernel以一个QP limb和一个coefficient为线程粒度，在寄存器中完成当前baby的digit×evaluation-key乘加，并立即把得到的两个KeySwitch component乘上plaintext diagonal，再累加到对应giant-group accumulator。KeySwitch结果不再成为全局可见的baby ciphertext。对于同一个baby被多个giant group消费的情况，evaluation-key MAC只计算一次，随后在寄存器中对最多4组不同diagonal分别做模乘；这同时实现了“KeySwitch→plaintext MAC直达”和跨group复用，而不是简单把两个旧kernel机械拼接。identity baby的原密文分量以及Q侧需要补回的lifted Galois `c0`也进入同一dataflow，避免额外的完整ciphertext补写。

当前专用路径覆盖`N=65536`、inverse-pre-rotated Galois key、baby tile不超过8且单tile消费的giant group不超过4的形状；Q与P基各启动一个融合kernel，保持各自模数和Montgomery参数不变。重复的`(group,baby)`映射、缺失key/pointer table或不支持的shape会自动回退旧路径，因此其他degree、key layout和矩阵分组不会被强制套用该实现。

**访存与调用变化。** 对前置StC、C2S `[5,4,3,3]`路径，四个C2S stage的输入Q数分别为`34/33/32/30`，P数均为9，baby数分别为`8/8/15/15`。旧接口需要物化

`2 * N * 4 bytes * [8*(34+9) + 8*(33+9) + 15*(32+9) + 15*(30+9)] = 985,661,440 bytes`

的baby QP ciphertext；其后plaintext MAC至少还要把这些数据读回一次。因此融合消除了约`0.99 GB`写回和`0.99 GB`重新读取，即约`1.97 GB`的HBM流量，不包括随之删除的独立identity/c0补写。nsys的C2S区间中，旧路径的86次batch Key-MAC、24次独立plaintext MAC、43次lifted-c0 add和4次identity lift，变为24次融合kernel以及3次不满足融合形状的残余操作。C2S kernel实例由`454`降至`324`，减少130次、即`28.6%`；C2S累计GPU kernel时间由`40.242 ms`降至`33.851 ms`，减少`6.391 ms`、即`15.9%`。

**正确性与release性能结果。** 同一Release二进制固定在同一张V100，仅切换`POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_PLAIN_MAC=0/1`，两组均使用2次warmup加5次CUDA-event正式测量：

- 旧物化路径：C2S `40.2544 ms`，完整前置StC自举 `83.6054 ms`
- 融合直达路径：C2S `35.1346 ms`，完整前置StC自举 `78.7710 ms`
- C2S减少`5.1198 ms`、提升约`12.72%`；完整路径减少`4.8344 ms`、提升约`5.78%`

融合路径的C2S real/imag CPU-GPU最大差均约`8.56e-7`，roundtrip差约`4.28e-7`；EvalMod仍为`Q:28->13`、输出scale `2^45.0732`，最终CPU-GPU差约`1.10e-9`。已知degree-22截断多项式误差仍为`6.78436`，说明该优化只改变GPU dataflow，没有改变明文矩阵、RNS层级或算法语义。

**适用范围与状态。** 该融合已经成为满足上述shape检查时的默认路径，设置`POSEIDON_DOUBLE_HOIST_FUSED_BABY_KEYSWITCH_PLAIN_MAC=0`可恢复旧的baby-QP物化实现。验证报告为`profiles/bootstrap22_fused_baby_ks_ptxt_v100_20260821.nsys-rep`，报告和派生文本均属于测试产物，不提交仓库。

### 20. 融合 Baby KeySwitch/Plaintext-MAC 的占用率感知 block 形状

**原始瓶颈。** 第19项删除QP中间物化后，融合kernel成为C2S第一热点。对最终V100 cubin做静态资源检查表明，`GroupTile=4`的P/Q实例分别使用66和72个32-bit寄存器，每线程`STACK/LOCAL=0`，不存在寄存器spill；问题是原先每block 256 threads导致寄存器驻留受限。以较重的Q实例为例，每block需要`256*72=18,432`个寄存器，V100每SM的65,536个寄存器只能容纳3个block，即24个warp、理论occupancy 37.5%。

**优化思路。** 将该融合kernel单独改成128-thread block，不改变其他KeySwitch、NTT或plaintext-MAC kernel。Q实例每block寄存器需求降为9,216，可同时驻留7个block，也就是28个warp、理论occupancy 43.75%；P实例也从3个256-thread block提高到7个128-thread block。更高的并发warp用于掩盖evaluation-key与diagonal的HBM读取延迟。该变化只把相同的`q_count*N`或`p_count*N`线程划分为更多block，线程到`(limb, coefficient)`的映射、模乘顺序、giant-group累加顺序和kernel发射次数均保持不变。

64-thread block在66/72寄存器条件下也只能达到约43.75%的理论occupancy，但会进一步增加block数量和调度边界，因此没有在缺乏额外occupancy收益的情况下继续缩小。保留`64/128/256`三个合法值只是为了跨GPU架构复测；V100默认选择128。

**正确性与性能结果。** 同一Release二进制固定在同一张V100，仅切换`POSEIDON_DOUBLE_HOIST_FUSED_BABY_BLOCK_SIZE=256/128`，两组均使用2次warmup加5次CUDA-event测量：

- 256 threads：C2S `35.8998 ms`，完整前置StC自举 `79.7402 ms`
- 128 threads：C2S `35.7774 ms`，完整前置StC自举 `78.6352 ms`
- C2S减少`0.1224 ms`、提升约`0.34%`；完整时间差包含EvalMod和运行状态波动，不把全部`1.105 ms`归因于本项

独立nsys结构对比进一步确认了局部收益。四类融合kernel的累计GPU时间由`12.6421 ms`降至`12.5098 ms`，减少`0.1322 ms`、即`1.05%`；C2S全部kernel累计时间由`33.851 ms`降至`33.703 ms`，减少`0.148 ms`，kernel实例数均为324。`GroupTile=4`两类实例合计减少约`0.200 ms`，`GroupTile=1`合计增加约`0.068 ms`，净结果仍与Release C2S的`0.122 ms`下降一致。

128-thread路径的C2S CPU/GPU real/imag最大差约`8.36e-7`，roundtrip差约`4.17e-7`；最终仍为`Q:28->13`、scale `2^45.0732`，CPU/GPU差约`1.08e-9`，degree-22已知近似误差仍为`6.78436`。默认值已改为128；设置`POSEIDON_DOUBLE_HOIST_FUSED_BABY_BLOCK_SIZE=256`可恢复旧launch形状。验证报告为`profiles/bootstrap22_fused_baby_block128_v100_20260821.nsys-rep`。

### 21. `[5,4,3,3]` C2S 的 full-baby 单层批处理

**对比报告揭示的瓶颈。** 使用与先进库报告相同版本的 Nsight Systems 解析器重新导出原始数据后，先进库完整自举只有339次kernel发射，其中CtS为109次；本库第20项之后仍有1492次完整发射和324次CtS发射。两者在CtS中的核心kernel都已经是KeySwitch与plaintext MAC融合形式，但融合粒度不同：先进库四层CtS各只有一次`fused_kswitch_mac_ptxt`，本库的`fused_baby_keyswitch_plain_accumulate`仍发射24次。当前`[5,4,3,3]`矩阵的baby数为`[8,8,15,15]`，全局tile为4时需要`[2,2,4,4]`轮，每轮又分别启动Q和P kernel，因而恰好产生24次发射。这证明剩余差距不是“是否融合”，而是融合后仍按baby tile多次读写同一个giant accumulator。

**优化思路。** 将融合kernel的host参数和device参数容量由8个baby扩展到16个，并将该命名路径的默认tile设为15。四个C2S stage现在都能在一个tile内处理全部baby step：每个线程在寄存器中依次完成本层全部baby的decomposition-digit/evaluation-key MAC，立即乘对应diagonal，并持续累加同一组输出；直到本层结束才写回giant-group QP accumulator。Q与P仍因模数表和输出地址不同而各使用一个kernel，因此四层合计从24次降为8次，而不是宣称已经达到先进库的4次。

该变化不减少baby rotation、evaluation-key模乘或diagonal模乘的数学次数；它消除的是8轮额外tile边界。按四层的`(Q+P,group_count)=(43,4),(42,4),(41,1),(39,1)`、`N=65536`和两个密文分量估算，合并后的单层累加约避免580 MiB giant accumulator HBM读写，并减少16次kernel launch。容量扩展后的实际V100 cubin保持原资源形状：最重的Q/group-4实例仍为72个寄存器，P/group-4仍为66个寄存器，`STACK=0, LOCAL=0`；更大的参数表进入约885字节kernel constant argument区，没有引入寄存器spill。

**正确性与release性能结果。** 严格A/B固定在同一张空闲V100（GPU 3），两组均使用2次warmup加5次CUDA-event正式测量；测试期间另一个GPU存在外部满载任务，因此没有混用其结果：

- tile 4：C2S `33.9138 ms`，完整前置StC自举 `78.1568 ms`
- full-baby tile 15：C2S `32.7668 ms`，完整前置StC自举 `74.8890 ms`
- C2S减少`1.1470 ms`、提升约`3.38%`

完整时间观测下降`3.2678 ms`，但其中EvalMod也从`38.3096 ms`波动到`37.1622 ms`，所以本项只保守归因得到CtS局部测量支持的`1.1470 ms`。tile 15路径的C2S real/imag CPU-GPU最大差约`8.42e-7`，roundtrip差约`4.20e-7`；最终CPU-GPU差约`9.37e-10`，输出仍为`Q:28->13`和scale `2^45.0732`。degree-22截断多项式原有的`final source error=6.78436`没有被本项改变。

**nsys验证。** 新报告中CtS融合kernel由24次、`12.5098 ms`降为8次、`10.1976 ms`；CtS全部kernel由324次、`33.7028 ms`降为308次、`31.0351 ms`。完整capture由1492次、`80.3882 ms`降为1476次、`76.5703 ms`。StC仍为315次，EvalMod仍为841次，说明launch变化准确落在目标C2S路径，没有把其他阶段的波动冒充结构收益。

**适用范围与状态。** full-baby tile 15只成为`slim22_da3_c2s5433`与`slim22_direct_da3_c2s5433`两条`[5,4,3,3]`实验路径的脚本默认值；其他profile继续使用tile 4，低层`dnum=1`路径继续使用独立的tile 8覆盖。用户显式设置`POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE`时始终优先。最终验证报告为`profiles/bootstrap22_full_baby15_v100_gpu3_20260821.nsys-rep`，报告及派生文件属于测试产物，不提交仓库。

## 3. 累计性能演进

下表用于观察总体趋势，不应被视为完全相同环境下的一组严格单变量实验。nsys 数据包含 profiler 开销，release 数据来自 CUDA event。

| 阶段 | 完整自举时间 | 测量方式 | 主要贡献 |
|---|---:|---|---|
| 此前六点/七点总结后的起始状态 | 约 180～184 ms | 历史 release/nsys | 已有 EvalMod 和 P=9 KeySwitch 优化 |
| Double-Hoist P=9 ModUp/NTT/ModDown 专用化 | 约 162 ms | nsys | 第 1～4 项 |
| QP plaintext MAC group 分块及其他已保留调整 | 约 156.5 ms | release | 第 5 项及配套路径 |
| Fused phase2+Key-MAC 编译与资源调优 | 156.01 ms | release | 第 6 项 |
| QP plaintext MAC ciphertext-component 融合 | 148.96 ms | release | 第 7 项 |
| 59 阶 BSGS remainder-chain lazy relin | 143.83 ms | release | 第 8 项 |
| 前置 StC、22 阶 baby-4 lazy relin 实验 | 108.61 ms | release，5+5 | 第 10 项；多项式精度尚不合格 |
| Giant Key-MAC output-major QP 归约 | 107.05 ms | release，22阶路径5+5 | 第 11 项；相对同GPU同配置基线降低约 1.36 ms |
| Giant-source Four-step INTT 批处理 | 104.65 ms | release，22阶路径2+5 | 第 14 项；相对同配置逐group路径平均降低约 0.71 ms |
| `[5,4,3,3]` C2S与3层direct矩阵 | 91.56 ms | release，22阶独立路径1+1 A/B | 第 15 项；C2S减少约14.79 ms，但额外消耗Q prime |
| `[5,4,3,3]` 归一化Q链默认值 | 90.83～91.27 ms | release，22阶独立路径1+1 | 输出由12枚提高到13枚Q，最终scale约`2^45.073`；以约1.3～1.7 ms换取一层可用裕度 |
| EvalMod目标层leaf CAccum与D2D消除 | 84.12 ms | release，22阶独立路径1+3 A/B | EvalMod `43.82->38.32 ms`；leaf kernel `96->12`，完整DtoD `200->104` |
| EvalMod combine last-use零拷贝ModDrop | 84.62 ms | release，22阶独立路径1+3 A/B | EvalMod `40.01->39.33 ms`；完整DtoD `104->92`，减少`62.915 MB` |
| EvalMod共享operand只读Q-prefix view | 83.51 ms | release，22阶独立路径2+5 A/B | EvalMod `39.15->37.94 ms`；完整DtoD `92->76`，减少`102.760 MB` |
| Baby KeySwitch→QP plaintext MAC寄存器直达融合 | 78.77 ms | release，22阶独立路径2+5 A/B | C2S `40.25->35.13 ms`；C2S kernel `454->324`，消除约`1.97 GB`中间HBM流量 |
| 融合Baby kernel改为128-thread block | 78.64 ms | release，22阶独立路径2+5 A/B | C2S `35.90->35.78 ms`；融合kernel累计时间减少约`1.05%` |
| `[5,4,3,3]` C2S full-baby tile 15 | 74.89 ms | release，空闲V100、22阶路径2+5 A/B | C2S `33.91->32.77 ms`；融合kernel `24->8`，CtS nsys kernel累计`33.70->31.04 ms` |

当前22阶前置StC实验路径的最新 Nsight Systems 报告为 `bootstrap22_full_baby15_v100_gpu3_20260821.nsys-rep`。这些 profiler 报告属于测试产物，不需要与本文档一起提交到仓库。

## 4. 后续记录规范

以后每个新增条目至少应包含：

1. 观察到的瓶颈以及对应参数范围。
2. 从计算量、访存、同步或 kernel 发射结构角度描述优化原理。
3. 正确复现该优化必须满足的条件。
4. 已执行的正确性检查。
5. 算子和完整自举的前后性能，无法得到单项数据时应明确说明是组合收益。
6. 是否成为默认路径，以及如何选择 fallback。

只有在收益能够重复，或者累计 kernel 测量足够稳定时，才将实验加入本文档。计算正确但性能下降的实验应保留在实验记录或 profiler 结果中，而不是写入正向优化历史。
