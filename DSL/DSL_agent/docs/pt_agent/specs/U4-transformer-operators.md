# U4：Transformer/LLM 算子与 token 布局

PLAN §4.B 的 U4，在 U5 之后。分两个里程碑，codex 按顺序完成，每完成一个就在 STATUS 中记录阶段结果，然后继续下一个：
- **U4a**：token 布局与通用运算，以 **BERT-tiny**（M-FAC/bert-tiny-finetuned-sst2，已下载）端到端为验收。这也是 PLAN 中 11/02 检查点的达标条件。
- **U4b**：解码器结构（RMSNorm、RoPE、SwiGLU、GQA、因果掩码、大词表 lm_head），以 **tiny Qwen2**（onnx-internal-testing/tiny-random-Qwen2ForCausalLM，已下载）端到端为验收。

本轮约束：
- 不调用 LLM（Agent 在 LLM 模型上的运行属于之后的论文实验）；
- 新增的代码、实验和测试不读取报告集（既有回归测试按 AGENTS.md 的例外执行）；
- 不修改编译器和运行时；不下载、不安装；
- **现有程序（CNN、MLP，以及 U2、U3、U5 的全部程序）发出的运算序列逐字不变**；r8-v4/v5/v6 的提示词不变。

## 0. 现状（U1 覆盖率）

- BERT-tiny 可以 lowering 的比例为 78.6%，缺少 layernorm、matmul、permute、softmax。
- tiny Qwen2 为 73.1%，缺少 rsqrt 等 act、broadcast、cast、compare/where/slice_scatter（掩码构造）、matmul、permute、softmax。
- U5 已提供 `exp`、`inv`、`rsqrt` 的近似（remez、newton），U2 提供了多块 flat 布局，U3 的 linear 支持多块。

## 1. 布局（`kernels/layout.py` 中新增，原有类不改）

1. **`TokenLayout(n, L, D)`**：序列长度为 L、每个 token 宽度为 D 的张量 (1, L, D)。**每个 token 一个密文**，向量放在前缀槽位 [0, D)，尾槽为 0（沿用 FlatLayout 的不变量，R8f2）。D > n/2 时，每个 token 按 U2 的 MultiFlat 规则分块。
   - head 划分 (1, L, H, dh) 只是槽位区间的解释（head h 占 [h·dh, (h+1)·dh)），不搬动数据。
   - 之所以选择这种布局，是因为它与团队已验证的 Qwen 包一致，正确性最直接。把多个 token 打包进一个密文以降低代价，作为以后的优化，本轮不做。
2. **`ScoreLayout(n, L, H, Lk)`**：注意力分数 (1, H, L, Lk)。**每个 query token 一个密文**，H 个 head 的 Lk 个分数依次排在 [h·Lk, (h+1)·Lk)。

## 2. 运算与 lowering（新增 `kernels/transformer.py`；lowering 分派）

除非另有说明，都只用 Backend 的现有接口，并且对 L 个 token 逐个处理。

| 运算 | 语义与 lowering |
|---|---|
| linear（作用于最后一维） | 每个 token 调用现有的 linear kernel，权重相同；不同 token 的明文常量自然去重。另外新增 **"高矩阵"变体**：在 `Op ≫ Ip` 时，把输入复制填满槽位，按 Ip 条对角线计算，常量为 Ip × ⌈Op/n⌉ 个，代价为 O(Ip) 次旋转。U3 的多块 linear 会补成方阵，大词表时不可行，所以需要这个变体。现有 linear 路径的选择规则不变；只有在原路径不可行（常量或补方后的规模超过阈值）时，才使用高矩阵变体。阈值写在 config 中 |
| add、mul（逐元素）、act | 逐 token 进行；激活使用 U5 的全部方法 |
| layernorm（最后一维） | 均值 = 槽位求和 × (1/D)，广播回 D 个槽；中心化；方差 = 中心化后平方的均值；`r = rsqrt(var + eps)`，使用 U5 的 newton/remez，**近似参数作为 Agent 和精调的旋钮**，写法与 act 相同，定义域来自校准时观察到的 `var + eps` 的范围；`y = centered · r · γ + β`（γ、β 是明文） |
| rmsnorm | `r = rsqrt(mean(x²) + eps)`，`y = x · r · γ`，近似参数同上 |
| softmax（最后一维，ScoreLayout 或 TokenLayout） | 对每个 head 块：`e = exp(s)`（U5 方法，定义域来自校准的分数范围）；乘以掩码（§3）；`S = 块内求和`；`inv = inv(S)`（U5 的 newton/remez，定义域来自校准的 S 范围）；`p = e · broadcast(inv)`。**exp 和 inv 各自是一个近似点**，参数写法同 act。不使用加密的 max 减法。如果模型中有公开的常数平移，照常保留 |
| matmul，模式 A：`Q · Kᵀ` | Q、K 都是 TokenLayout（可按 head 解释）。对每个 query token i、每个 head h：`score[h, i, j] = Σ_d Q[i, h·dh + d] · K[j, kvmap(h)·dh + d]`，做法是逐元素乘，然后用 log₂(dh) 次旋转在 head 块内求和，再把结果放到 ScoreLayout 中该 head 的第 j 个槽，用掩码加旋转。`kvmap` 由图中 GQA 的 expand/reshape 推出：q head h 使用 kv head ⌊h/(H/Hkv)⌋，不复制数据。缩放因子 1/√dh 并入 Q 的投影权重，或者作为明文乘法；能并入时必须并入 |
| matmul，模式 B：`P · V` | P 为 ScoreLayout，V 为 TokenLayout。对每个 query i、head h：`out[i, h·dh:(h+1)·dh] = Σ_j P[h, i, j] · V[j, kvmap(h)·dh : …]`。P[h,i,j] 通过掩码和旋转广播到 dh 个槽，乘以 V_j 后累加。输出为 TokenLayout |
| permute、reshape、expand | 只支持"head 拆分与合并"（(1,L,D) ↔ (1,L,H,dh) ↔ (1,H,L,dh)）、`Kᵀ` 的转置，以及 GQA 的 KV expand。它们只改变布局的解释，不发出运算。其他情况在 check 阶段报"暂不支持"，带 op id 和行号 |
| 最后一维内的静态重排（slice、neg、concat） | 通用规则：结果 = Σ_{offset} rotate(x, offset) ⊙ mask_offset，其中 offset 是不同的位移，掩码里的值取 0、1 或 −1（以覆盖 neg）。RoPE 的 rotate_half 只需要 2 个位移。只在单个 token 向量内部、并且索引映射是静态的时候使用；否则报"暂不支持" |
| 位置相关的常数乘加（RoPE 的 `x·cos + rotate_half(x)·sin`） | cos、sin 按 token 位置是不同的明文常量；用逐 token 的明文乘法与上一行的重排组合，不需要专门的 RoPE 运算 |
| cast | float32/float64 之间的转换在明文模拟和 CKKS 中都没有意义，lowering 中视为恒等。张量引擎按 float64 计算。在报告中说明，HF 的 RMSNorm 在 float32 下的中间舍入不会被复现 |

**不依赖特定模型的写法**：
- 以上每个运算都按通用语义实现，没有针对 BERT 或 Qwen 的特殊代码路径。
- 遇到语义相同但写法不同的图（例如 matmul 和 bmm 的不同组合、permute 的顺序），由前端规范化成同一种形式，不在 lowering 中逐一识别。

## 3. 掩码（通用规则）

softmax 的输入常常是 `s + M`，其中 M 是加性掩码，取 0 或很大的负数（因果掩码，以及来自 attention_mask 的补齐掩码）。CKKS 中无法表示 −∞。规则如下：

1. **识别**：在 softmax 的输入中，把只依赖于常量和 kind=mask 输入的加数分离出来作为 M（数据输入无关的部分）。要求 M 关于掩码输入是逐元素仿射的，即 `M = a + b·m`，a、b 都是常量，而且对所有二值 m，M 的取值都属于 {0} ∪ (−∞, −1e4]。不满足就报"暂不支持的掩码形式"。
2. **转换**：`softmax(s + M) = normalize(exp(s) ⊙ w)`，其中 `w = [M == 0]`，它是关于 m 的逐元素仿射式（常量部分为静态的 0/1）。
   - 静态部分（因果掩码）：被排除的 (i, j) **不计算分数**，与团队 Qwen 包的做法相同，同时节省运算；
   - 依赖掩码输入的部分：用加密的 mask 与 exp 的结果相乘（密文×密文），mask 按 key 维广播。
3. exp 的定义域只按**未被屏蔽的**分数校准。

## 4. 序列输入与输出（修订 U1 中 causal LM 的输出约定）

- 补齐、截断、长度选择沿用 U1 的规则。attention_mask 作为加密输入（kind=mask）。
- **修订**：U1 规定 causal LM 输出全部 L 个位置的 logits，由客户端选择最后一个真实位置。这对大词表的代价太高：tiny Qwen2 的词表是 151936，每个位置约 19 个密文的 lm_head。**改为**：客户端额外提供一个加密的 one-hot 选择向量 `sel`（kind=mask，长度 L，最后一个真实位置为 1），服务器计算 `h_last = Σ_t sel_t · h_t`（密文×密文，L 次乘法），然后**只做一次** lm_head。
  - 服务器仍然不知道真实长度，满足 U1 不暴露长度的要求；
  - `lm_next_token` 指标的口径不变；
  - ModelInfo 中记录这个输入；
  - 客户端的打包与解包按此修改。
- 文本分类模型（BERT）的池化使用 CLS（位置 0），位置是静态的，不需要选择向量。

## 5. 校准、验证、精调、提示词与规则规划器

1. **近似点**：layernorm/rmsnorm 的 rsqrt，softmax 的 exp 和 inv，以及 act，都是近似点，统一参与：
   - 校准时观察输入范围；
   - 验证时统计 max_u 和 rows；
   - 精调时对定义域对齐，并沿各方法的阶梯降阶（U5 §2.7）。
2. **张量引擎**：实现以上全部运算的张量语义，使用相同的系数、相同的运算顺序和相同的掩码规则；对补齐位置的处理与 Mock 一致。
3. **规则规划器**：
   - act 照旧（cheb deg 127，余量 1.5）；
   - exp 用 remez deg 15，定义域为校准范围 × 1.5；
   - inv 和 rsqrt 用 newton（deg0=7，iters=2），定义域为校准范围 × 1.5（下界不小于校准下界的 1/1.5，并且保持 > 0）。
   
   这些只是对照用的固定规则。
4. **提示词**：新增 r8-v7，包含 token 布局、新运算及其近似点参数、掩码规则和代价提示。r8-v4 到 r8-v6 不变，默认值不变。
5. **P2 包与 DSL 管理模式**：新运算复制进包内的 core；新布局写入 interfaces（每个 token 的密文下标，以及 mask 和 sel 输入）。用 U4a 的程序跑一次 manage=dsl，要求与 compiler 模式的 RuntimePlan 相同（S1 的流程）。

## 6. 输入样本

两个模型都使用仓库已有的 Markdown 文本，由测试工具**切分成句子**（按句号、问号、换行切分，每句不超过 24 个英文单词或等长的中文），作为用户样本。这些样本在 U1 的通用划分中使用（≥ 223 条），长度 L 由 U1 的规则自动选择。BERT 没有标签，所以使用一致率指标；Qwen 使用 `lm_next_token`。

## 7. 验收与测试（新增 `tests/test_u4_*.py`）

**U4a**：
1. **现有程序不变**：U5 验收中的全部程序，Emit 文本和常量与 62ecf194 完全相同；4 个冻结模型的验证、代价和确认逐项相同；r8-v4/v5/v6 逐字相同。
2. **逐运算等价**：linear（含高矩阵变体）、layernorm、softmax（含静态掩码和加密掩码）、matmul 模式 A 和 B（MHA 与 GQA 的 kvmap）、head 拆分与合并、静态重排（含 neg）。在单块和多块（D > n/2）下，Mock 与张量引擎满足 `|Δ| ≤ 1e-9·(1+|ref|)`；exact 模式与 PyTorch 一致（float64，`≤ 1e-9·(1+|ref|)`）。
3. **掩码规则**：HF 的因果掩码和补齐掩码都被正确识别和转换；构造一个不满足仿射或取值条件的掩码，报"暂不支持"。
4. **BERT-tiny 端到端**（规则规划器）：
   - 经 U1 CLI 导入，lowering 比例达到 100%；
   - 在 20 个留出样本上，Mock 与张量引擎满足第 2 条的容差，预测类别一致；
   - 留出集上与明文模型的一致率写入报告（只记录）；
   - 代价模型给出密文数、自举数、旋转数和预测延迟；
   - 本机能编译的（常量 ≤ 1.5 GiB，且符合 trusted 运行器的限制），预测与编译的自举数相等；否则打包为服务器作业；
   - P2 包导出，manage=dsl 的 RuntimePlan 与 compiler 模式一致。

**U4b**：
5. RMSNorm、静态重排实现的 rotate_half 加位置常量（RoPE）、SwiGLU、GQA、因果掩码加补齐掩码、选择向量，以及高矩阵 lm_head，各自的逐运算等价，口径同第 2 条。
6. **tiny Qwen2 端到端**：要求与第 4 条相同；指标为 lm_next_token。另外验证：选择向量路径得到的最后位置 logits，与"输出全部位置后由客户端选择"在 exact 模式下逐元素相同。
7. 覆盖率：重新生成 U1 的覆盖率表，BERT-tiny 和 tiny Qwen2 的可 lowering 比例都应为 100%；没有达到的，逐个列出原因。
8. 既有回归全部通过（545 项加上本轮新增）。

## 8. 汇报

- U4a、U4b 各自的改动文件和测试结果；
- 两个模型的端到端表：L、每个张量的密文数、自举数（预测与编译）、旋转数、预测延迟、一致率或 next-token 一致率、编译与否；
- 代价的分解：注意力（模式 A、B 和 softmax）、norm、FFN、lm_head 各占多少，作为以后 token 打包优化的依据；
- 覆盖率表；
- STATUS.md 新增 U4 一行，并分别标注 U4a 和 U4b 的完成情况。

## 9. 不要做

- 不修改现有 kernel 的行为、编译器、r8-v4 至 r8-v6 的提示词和冻结模块的判据。
- 不做 token 打包优化，不做 KV cache 与多步解码，不做加密的 max。
- 不调用 LLM，不读取报告集，不下载、不安装，不放宽资源限制。

## 10. 补充：DSL 字段、高矩阵 linear 的切换规则、旧测试迁移（Claude 于 2026-10-26）

### 10.1 PT-DSL 字段与新增算子

**原则**：近似点的方法和定义域**没有默认值**，都必须写明。这与 act 相同，避免隐藏的选择；规则规划器和 Agent 负责填写。各方法需要的参数与 U5 相同（cheb/remez：`deg`；newton：`deg0` 和 `iters`；composite：`degrees` 和 `eps`）。

1. **新参数类型 `Mask[...]`**：值为 0/1 的加密输入，按 flat 打包，例如 `am: Mask[1, L]`、`sel: Mask[1, L]`。与 `Cipher[...]` 一样占用密文输入，在 ModelInfo 中的 kind 为 mask。
2. **`layernorm(x, w=W[...], b=W[...], eps=<float>, rsqrt_method=..., rsqrt_domain=(lo, hi), rsqrt_<参数>...)`**：layernorm 在 ops_spec 中已经存在，这里扩展近似字段。`eps` 取自图，必须写出。
3. **新增 `rmsnorm(x, w=W[...], eps=<float>, rsqrt_method=..., rsqrt_domain=..., rsqrt_<参数>...)`**。
4. **`softmax(s, axis=-1, keep="all" | "causal", mask=<Mask 值> | None, exp_method=..., exp_domain=..., exp_<参数>..., inv_method=..., inv_domain=..., inv_<参数>...)`**：softmax 已存在，这里扩展字段。
   - `keep="causal"`：静态因果，被排除的 (i, j) 不计算；
   - `mask`：加密的 key 掩码 m（形状 (1, Lk)），保留权重 `w[i, j] = keep[i, j] · m[j]`；
   - §3 的识别只接受能写成 `keep_static[i, j] · m[j]` 的形式，覆盖 HF 的因果掩码加补齐掩码。其他仿射形式在 check 阶段报"暂不支持的掩码形式"。
   - `exp_domain` 只按未被屏蔽的分数校准；`inv_domain` 按求和结果 S 校准。
5. **`matmul(a, b, transpose_b=<bool>, heads=H, kv_heads=Hkv, scale=<float | None>)`**：matmul 已存在，这里扩展字段。`transpose_b=True` 时为模式 A（Q·Kᵀ），输出 ScoreLayout；`transpose_b=False`，并且 a 是 ScoreLayout 时，为模式 B（P·V）。
   - 前端规范化时，把 head 拆分与合并的 reshape/permute、Kᵀ 的转置和 GQA 的 expand **折叠进 matmul 的属性**。PT-DSL 中不出现这些纯布局运算；无法折叠的在 check 阶段报"暂不支持"。
   - `scale` 能并入 Q 投影时，由前端并入，并在 PT-DSL 中写 `scale=None`。
6. **新增 `shuffle(x, index=<int tuple>, sign=<int tuple>)`**：最后一维内的静态重排，`y[k] = sign[k] · x[index[k]]`，sign 取 ±1，index 长度等于 D。RoPE 的 rotate_half、静态的 slice 加 concat 都由前端规范化为 shuffle。
7. **新增 `select(h, sel)`**：`h_last = Σ_t sel[t] · h[t]`，输入为 TokenLayout，输出为 (1, 1, D)。
8. **逐 token 的常量乘法**：沿用已有的 `mul(x, W[...])`，W 中的常量形状为 (1, L, D)，例如 RoPE 的 cos、sin。不新增算子。

所以新增的 DSL 算子是 **`rmsnorm`、`shuffle`、`select`**（3 个）；扩展字段的是 `layernorm`、`softmax`、`matmul`；新增参数类型 `Mask`。每一项都给出 parser、printer 和 typecheck 的往返测试与非法参数测试。

### 10.2 高矩阵 linear 的切换规则

- 配置字段：`pt_agent/config.py` 中的 `LINEAR_TALL_MIN_RATIO = 4`。
- 度量：`ratio = O / Ip`，其中 O 是输出维度，`Ip` 是输入维度 I 向上取到 2 的幂。
- **规则**（确定性的，不依赖"原路径是否可行"的判断）：当输入是**单块**（`2·Ip ≤ n`）并且 `ratio ≥ LINEAR_TALL_MIN_RATIO` 时，使用高矩阵变体；否则走原有路径（U2、U3 的规则不变）。现有程序中没有满足这个条件的 linear，由 §7.1 的 Emit 不变性测试来保证；如果有，停下来汇报。
- **输出与 MultiFlat 的衔接**：高矩阵变体的输出按 **n/2 一块**写入，与 U2 的 MultiFlat 规则相同：第 c 块包含输出 `[c·n/2, (c+1)·n/2)`，尾槽 `[n/2, n)` 为 0，由权重掩码保证。块数为 `⌈O/(n/2)⌉`。最后一块可以较短，尾槽同样为 0。O ≤ n/2 时输出就是单块 FlatLayout。下游运算按 MultiFlat 处理，不需要特殊路径。
- 计算方式：输入 x 平铺填满 n 个槽（log₂(n/Ip) 次旋转），然后对每个输出块 c：`y_c = Σ_{d<Ip} rotate(x_rep, d) ⊙ w_{c,d}`，其中 `w_{c,d}[j] = W[c·n/2 + j, (j + d) mod Ip]`（j < n/2；越界的输出行和尾槽都为 0）。同一个输入块的旋转只做一次，被所有输出块共用。偏置与掩码按块加入。
- 测试：
  - O/Ip 恰好为 3、4、5 时的切换边界；
  - O 不是 n/2 的整数倍；
  - 块尾槽严格为 0；
  - Count 统计的旋转数为 O(Ip + log n)，不随块数线性增长；
  - 与张量引擎和 PyTorch 一致。

### 10.3 旧测试迁移（同时保持 r8-v4 到 r8-v6 的文本不变）

按 U3 §8 的一般规则（spec 明确新增的能力，可以更新"能力不存在"或计数类断言），并做以下具体安排：

1. **`test_r1_parser`**：算子数从 27 改为 **30**（新增 rmsnorm、shuffle、select），如有列举名单就同步加入。其余断言不变。
2. **`test_r8b_generic`**：目前断言"r8-v5 的注册表等于全部 HANDLED_OPS"并写死 15 个算子。改为：
   - **r8-v5 的注册表等于一份冻结的 15 个算子快照**（就是现在写死的那 15 个），r8-v4 和 r8-v6 同样各自对照冻结快照，以保证这些版本的提示词文本逐字不变；
   - **"完整性"检查转到最新版本**：断言 r8-v7 的注册表等于当前全部 HANDLED_OPS，并且每个算子都有 kernel、代价和参考语义。
   
   两条断言都在汇报中列出原文与新文。
3. **提示词文本**：r8-v4、r8-v5、r8-v6 由各自的冻结快照生成，加上逐字比较的测试（与修改前的文本相同）。新增能力只出现在 r8-v7。
4. 其他旧测试的断言如果需要改变含义，仍然停下来汇报。

完成后从 U4a 开始实现。

## 11. 补充：BERT 张量执行出现非有限值后的逐点诊断与可调整范围（Claude 于 2026-10-26）

**现象**：U4a 的 BERT 在 20 条留出样本上做张量执行时出现非有限值；Mock 对照尚未开始。第一个 softmax 的 inv 定义域为 (0.0053, 335746)，规则规划器固定使用 newton deg0=7、iters=2。

**初步判断**（需要用数据确认）：
- 求和 `S = Σ exp(s)` 的动态范围达到约 6×10⁷：分数跨度约 20，经过 exp 后放大到 e²⁰ 量级。
- 7 阶初值多项式在这样宽的区间上逼近 1/x，相对误差远大于 1，而 Newton 求倒数要收敛，需要 `|1 − x·y0| < 1`，于是迭代发散。
- layernorm 的 rsqrt 可能有同类问题（方差的范围很宽）。

### 11.1 逐点数值诊断（只用校准样本和留出样本，不读报告集）

新增 `experiments/u4_point_diagnosis.py`，在张量引擎上运行，并在每一步都同时计算 exact 版本作为对照。

1. **逐近似点表**（按程序顺序列出全部近似点：act、exp、inv、rsqrt）：
   - 观察到的输入范围与定义域；max_u；输入分布的分位数（p0、p1、p50、p99、p100）；
   - 在**实际输入**上，近似输出相对 exact 函数的最大绝对误差和最大相对误差；
   - 前面各点都换成 exact、只保留该点近似时，下游层输出与 exact 模型的相对误差（单点影响）；
   - 第一个出现非有限值或相对误差超过 1 的点（首个失真点）。
2. **softmax 专项**：对每个 softmax，按行统计：
   - 行最大分数、行内跨度；
   - S 的分位数、`S_max / S_min`；
   - 在当前 exp 定义域下的 S 范围。
   
   对 inv，给出把相对误差降到 1e-3 所需的 newton 迭代次数（按 U5 的误差模型）：分别在当前定义域，以及在 §11.2 第 1、2 条调整之后的定义域上计算。
3. **norm 专项**：对每个 layernorm，统计 `var + eps` 的分位数和比值；rsqrt 的统计口径与 inv 相同。

### 11.2 可调整的范围（只限以下几类；都是通用规则，不针对 BERT 或某个位置做特殊处理）

1. **softmax 的公开平移**：softmax 新增属性 `shift=<float>`（公开常数，默认 0）。语义为 `softmax(s) = softmax(s − shift)`，这是严格等价变换。规则规划器取 `shift = 该 softmax 校准分数的上界`（每个 softmax 运算一个常数，不按行变化）。这样 exp 的定义域变为 `(lo − shift, (hi − shift)·margin 后的上界]`，在校准范围内 exp 的输出不超过 1，`S ≤ Lk`（留出样本略超出校准上界时只会稍大于 1，由 margin 覆盖）。`shift` 不是近似字段，默认 0 不改变既有程序。
2. **inv 的公开缩放**：inv 近似点允许 `inv_scale=<float>`（公开常数）。语义为 `inv(S) = inv(S / scale) / scale`，严格等价。规则规划器取 `scale = 校准 S 的上界`，使输入落在 `(S_min/scale, 1]`。
3. **规则规划器中 inv 和 rsqrt 的默认参数**可以按 11.1 的数据修订：
   - newton 的 `deg0 ∈ {7, 15, 31}`、`iters ∈ {1, …, 5}`，或者改用 remez，阶数遵循 U5 的阶梯；
   - 选择规则必须写成通用公式：根据定义域比值和目标相对误差 1e-3，选出满足要求、乘法深度最小的组合；
   - 同一规则用于 layernorm 和 rmsnorm 的 rsqrt。
4. **exp 的默认参数**：remez 的阶数可以在 {15, 31, 63} 中，按目标误差用同样的通用公式选择。
5. 以上改动都只影响**规则规划器的默认值和 DSL 的新增可选属性**，原有字段的语义不变。新属性在 r8-v7 的提示词中说明，r8-v4 到 r8-v6 不变。

**不允许**：
- 用其他函数替换 softmax 或 norm 的语义（例如二次型 softmax、ReLU 归一化）；
- 引入加密的 max 或比较运算；
- 放宽数值一致性容差或精度判据；
- 针对特定层、特定样本或特定模型写规则。

### 11.3 判定与接续

- 如果 11.2 的调整使所有近似点的实际输入都落在定义域内，并且在 20 条留出样本上没有非有限值，就继续 U4a 的 §7.4：先做 Mock 对照，再做后续验收。汇报中给出调整前后的逐点表。
- 如果调整之后，某个 inv 或 rsqrt 仍然需要过大的深度：按 11.1 的计算，达到 1e-3 相对误差需要 iters > 5，或者在定义域上 remez 阶数需要 > 63。这时**停下来汇报**该点的分布和所需深度，由 Claude 和用户决定下一步（例如逐行统计量、分段方法等），不自行引入超出 11.2 的手段。
- 原有的测试授权（§7、§10.3）继续有效。

### 11.4 补充：搜索中遇到未收敛候选（Claude 于 2026-10-26，对应 `u4-points-choices-20261026T123231747385`）

- Remez 交换算法在 50 轮内没有收敛的候选（包括 newton 的 deg0 初值），一律判为**不可行候选**，不使用它的系数，不重试，也不放宽收敛判据。这与 U5 中 composite 搜索排除未收敛候选的口径相同。
- 候选表中记录 `feasible:false, reason:"nonconvergent"`，以及 method、deg（或 deg0/iters）、iterations、max|err|、|E|。异常发生时必须记录正在计算的候选；这次 STOP 中缺少 `failed_degree`，需要补上。
- 一个候选不收敛，不影响同一点的其他候选，也不影响其他点：搜索继续。某个点的全部候选都不可行时，该点标为 `status:"infeasible"`，并按 §11.1 第 2 条的误差模型给出所需的 iters 和阶数。全部点搜索完、逐点表完整之后，再按 §11.3 判定。**不因第一个不可行点就中止整张表。**

### 11.5 补充：定义域覆盖

1. 定义域**只由校准样本决定**（这次 `choices_from:"calibration only"` 是正确的）。留出样本只用于判定，不能用来改变定义域、margin 或方法选择。
2. 留出样本的实际输入超出定义域时，记为该点的**覆盖失败**：记录 outside 个数、max_u、超出的最大值，以及对应的样本和行号。不做截断（clamp），也不按留出数据事后扩大 margin。该点的规划因此判为不通过，并进入 §11.3 的停止汇报。
3. 如果某个近似点的输入存在**解析界**（由前面运算的数学性质保证，而不是由数据统计得出），优先使用解析界，校准范围只用来核对。
4. margin 的默认值（目前在正值输入上为 ×1.5、/1.5）只能通过事先确定的通用规则修改。修改之后，最终验收要换用此前没有参与选择的确认样本，不能仍用这 20 条留出样本。

**本次的结论**：按 §11.4 和 §11.5，第一个 softmax 的 inv（定义域比值约 6.3×10⁷）和第二个 softmax 的 inv（比值约 2.3×10⁵，并且有覆盖失败）都会判为 infeasible，触发 §11.3 停止。下一步由 §11.6 规定。

### 11.6 softmax 的"缩放—反复平方—归一化"形式（用户于 2026-10-26 选定）

**原因**：§11.2 的公开平移是对整个运算取一个常数，不能消除各行最大分数之间的差别。实测平移后 S/scale 的比值仍有约 2.8×10⁷（第一个 softmax）和 10⁵（第二个 softmax）。按 1/x 的逼近理论，比值 10⁵ 时 63 阶多项式的相对误差大于 1，Newton 从最优常数初值出发需要约 21 次迭代，超出 §11.3 的上限。因此需要改变 softmax 的**计算形式**，但不改变它的语义。

**恒等式**（严格成立，属于通用变换，与模型无关）：

    softmax(s) = N^k( softmax((s − shift) / 2^k) ),   其中 N(p) = p² / Σp²

masked 位置在第一步就是 0，之后平方仍是 0；causal 的静态排除不变。

**DSL**：softmax 新增可选属性 `steps=<int ≥ 0>`，默认 0。默认值的行为与现在完全一致，既有程序的 Emit 输出逐字不变。`steps > 0` 时必须同时给出 `step_inv_method`、`step_inv_domain` 和 `step_inv_*`（与 §10 一样，近似字段没有默认值）。这 k 步共用同一个近似，诊断时按 `::step_inv[i]` 分步记录。r8-v7 的提示词需要说明这个属性；r8-v4 到 r8-v6 不变。

**lowering**（Mock 与张量引擎的结构相同）：

1. `t = (s − shift) · 2^-k`。这是公开常数，并入前一个运算的缩放。
2. `e = exp(t) · mask`；`S0 = Σe`；`p = e · broadcast(inv(S0 / scale) / scale)`。
3. 重复 k 次：`q = p · p`；`T = Σq`；`p = q · broadcast(step_inv(T))`。

**规则规划器**（通用公式）：

- 设 `[a0, b0]` 是 k=0 时、经过平移和 margin 之后的 exp 定义域。取最小的 k，使 `(b0 − a0) / 2^k ≤ 1`；若 k > 8，判为不可行并停下。本次两个 softmax 的宽度约为 29.7 和 22.3，因此 k=5。
- exp 的定义域取 `[a0/2^k, b0/2^k]`。目标相对误差为 `1e-3 / 2^k`：逐元素的误差每平方一次会翻倍。按 U5 的阶梯选出满足目标且深度最小的方法。
- 第一步 inv 的定义域采用**解析界**（§11.5 第 3 条）：`S0 ∈ [e^a, n·e^b]`，其中 `[a, b]` 是 exp 的定义域，n 为 key 数的上界（Lk）。`scale` 取这个上界，所以这个点的覆盖由 exp 的覆盖保证，不再单独依赖校准统计。比值不超过 `e·Lk`。
- `step_inv` 的定义域也采用解析界：因为 Σp ≈ 1，所以 `Σp² ∈ [1/n, 1]`。加上正值输入的 margin，取 `[1/(1.5·Lk), 1.5]`。Lk=40 时比值为 90。目标相对误差为 1e-3。说明：每行乘的是同一个标量，所以中间步骤的 inv 误差会在下一步被归一化抵消，只有最后一步的误差会留下来。按 1e-3 统一选择偏保守，但更简单。
- 校准统计仍然要记录，用来核对解析界（校准的实际输入应当落在解析界内；若不在，说明实现有误，停下）。

**判定**：
- 先跑 exact 版本：`steps=k` 与 `steps=0` 的输出差在数学容差 1e-12+1e-12|ref| 之内。
- 再重跑 §11.1 的逐点诊断，并给出新旧对照表，包括每个 softmax 增加的乘法深度。
- 然后按 §11.3、§11.4、§11.5 判定。通过后接续 U4a 的 §7.4（Mock 对照、编译、自举数）。
- §11.3 的上限（单点 iters ≤ 5、阶数 ≤ 63）仍然适用于每个近似点。

**测试**：
- 恒等式的单元测试：随机分数，k=0..6，覆盖 mask 和 causal 两种情况；
- `steps=0` 时，冻结程序和既有程序的 Emit 输出逐字不变；
- 用解析界算出的定义域覆盖校准实际输入的测试；
- `test_r1_parser` 增加 `steps` 和 `step_inv_*` 字段的用例。

### 11.7 补充：k 的上限与宽分数域（Claude 于 2026-10-27，对应 `u4-qwen-prepare-20261027T064721360265`）

**现象**：tiny Qwen2 两个 softmax 平移并加 margin 后，exp 定义域宽度为 606.66 和 385.52。按 §11.6 的规则需要 k=10 和 k=9，超出上限 8，于是在 `squared_softmax_adjustments` 停止。k=8 时缩放后的宽度为 2.370 和 1.506。

**原因**：§11.6 的"缩放后宽度 ≤ 1"只是为了让 exp 容易逼近而设的经验目标，并不是可行性条件。真正的约束是：在既定阶梯内，exp 和第一步 inv 能找到满足误差目标的可行候选（§11.3、§11.4）。

**修订规则**（通用公式，替换 §11.6 规则规划器的第一条，其余各条不变）：

1. `k0 = max(0, ceil(log2(b0 − a0)))`，即原规则的 k。取 `k = min(k0, 8)`。**k 不超过 8**，不提高上限。
2. exp 定义域仍为 `[a0/2^k, b0/2^k]`，宽度可以大于 1。目标仍为 `1e-3 / 2^k`，仍按 U5 和 §11.2 第 4 条的阶梯（remez {15, 31, 63}）选择满足目标且深度最小的方法。
3. 第一步 inv 仍用解析界，`scale = n·e^b`，定义域为 `[e^(a−b)/n, 1]`。比值是 `n·e^(宽度)`，不再保证不超过 `e·Lk`。仍按 §11.2 第 3 条的阶梯选择：newton deg0 ∈ {7, 15, 31}、iters 1–5，或 remez。
4. step_inv 的定义域、目标和规则都不变。
5. 只要某个点在 k=8 时没有可行候选，就按 §11.4 记为 `status:"infeasible"`，把整张表算完，再按 §11.3 停下汇报，同时给出所需的阶数或 iters。**不改用 k > 8，不扩大阶梯，不改变 margin。**
6. 当 k0 ≤ 8 时，本规则与 §11.6 完全相同。因此 BERT（k0=5）的程序文本、成本以及 U4a 的全部结果都不受影响。

**只读诊断**：只记录，不影响规则，用于汇报和论文中的局限说明。它只用校准样本和 exact 引擎，在 n15 和 n42 两处，把 exp 输入（平移之前）的 min/max 分成三类分别统计：

- query 和 key 都有效；
- key 被 padding mask 掉；
- query 位于 padding 位置。

这张表用来说明宽度来自哪一类。即使宽度主要来自 masked 项，本轮也**不**引入 exp 之前的掩码替换或其他新的变换。这类做法记为后续 Agent 优化的候选。

**测试与回归**：

- 规则单元测试，覆盖三种宽度：`k0 ≤ 8` 时与 §11.6 的结果相同；`k0 = 9, 10` 时取 k=8，exp 和 inv 的定义域按上面的公式计算；无可行候选时报告 infeasible 而不抛出未分类异常；
- exact 恒等式测试补充 k=8、缩放后宽度大于 1 的情况，容差为数学容差 1e-12+1e-12|ref|；
- BERT 的规则程序文本逐字不变（用 U4a 已保存的 DSL 比较）；
- 若 r8-v7 提示词中写了"宽度 ≤ 1"，同步改为本规则；r8-v4 到 r8-v6 不变；
- 已通过的 U4a 结果不重跑，只运行受本改动影响的单元测试组，以及 U4 和 parser 测试。

**接续**：规划成功后，按 STATUS 中列出的 U4b 未完成项依次继续：

1. RMSNorm、RoPE、SwiGLU、GQA 因果加 padding、最后位置选择、高矩阵 head 的逐算子验证；
2. Qwen 完整规则程序，20 条留出样本的 Mock 和 tensor 验证，next-token 一致率；
3. 最后位置选择向量与全位置 exact logits 的比较；
4. P2 导出，compiler 和 managed 两种模式编译，完整 RuntimePlan 比较；
5. 成本分解（含 lm_head）。

逐点表中要标出 k0、实际 k、缩放后宽度、第一步 inv 的比值，以及所选方法和深度。IR 的大小和执行路线按 §7.5、§7.6 和 §7.9.5（运行 managed 之前先报告大小）。资源限制、容差、测试授权都不变。

### 11.8 补充：按相对误差拟合，以及 exp 阶梯的口径（Claude 于 2026-10-27，对应 `u4-qwen-wide-20261027T070852687880`；拟合方式由用户选定）

**现象**：k=8 时，n15 的第一步 inv 定义域比值为 342。全部 19 个候选都不可行：直接 remez 63 阶的相对误差为 0.187；newton 三种初值的相对误差为 80、34、6，都在收敛区之外。

**原因**：`approx.remez` 最小化的是**绝对误差**，而 §11.2 的目标是**相对误差**。对 1/x 和 1/√x，绝对误差最优的多项式在小 x 一端的相对误差远大于相对误差最优的多项式：

- 按相对 minimax 理论，比值 342 时 63 阶约为 2e-3；31 阶初值约为 0.06，newton 2 次即可达到 1e-3。
- 实测分别为 0.187 和 5.96。

这是拟合口径与目标不一致造成的，属于通用问题，不针对 Qwen。

**1. exp 阶梯的口径（更正 §11.7 第 2 条）**：exp 的候选阶梯就是现行实现中 `exp_ladder=True` 使用的 U5 LADDER 中 ≤ 63 的各阶，**包括 7 阶**。§11.7 中写的 {15, 31, 63} 是笔误。现行选择器不改，BERT 两处的 exp 取 7 阶是正确的。

**2. 相对误差 Remez**（`kernels/approx.py`，通用实现）：

- `_exchange` 增加可选的权函数 w，求解 `p(r_j) + (−1)^j·E/w(r_j) = f(r_j)`；误差函数为 `w·(f − p)`；极值搜索、交换、收敛判据、50 轮上限的规则都不变。
- 相对拟合取 `w = 1/f`，适用于 exp、inv、rsqrt。
- 不传权函数时行为与现在完全相同。`remez(fn, deg, domain)` 的默认结果保持数值相同（`array_equal`），U5、S1 和既有程序不受影响。
- 未收敛的候选按 §11.4 判为不可行。

**3. DSL**：近似字段新增可选的 `<role>_fit ∈ {"absolute", "relative"}`。它适用于 exp、inv、rsqrt 和 step_inv 的 remez 方法，以及 newton 的初值，默认值是 `"absolute"`。

- `fitted_coefficients` 按 fit 分派。Mock、张量引擎、Emit、P2 都经由它取系数，所以结构一致。
- 默认值下，冻结程序、既有程序以及 r8-v4 到 r8-v6 的 Emit 和提示词逐字不变。
- r8-v7 的提示词说明这个字段。
- 在 parser 中增加对应用例。

**4. 规则规划器**：

- 每个 exp、inv、rsqrt、step_inv 点的候选集合是 {absolute, relative} × 原有的方法和阶梯，iters 和阶数的上限不变。
- 选择键为 `(depth, relative_error, method, deg/deg0, iters, fit)`，即现有的键后面加上 fit。
- 相对误差的评估方式不变，仍用同一组诊断网格、驻点和端点。
- 这是对所有 transformer 规则点的通用规则。act 的 cheb127 和 U5 的其他路径不在范围内。

**5. 判定**：仍按 §11.3、§11.4、§11.5 和 §11.7 第 5 条执行。若在 k=8 下仍有点不可行，就把整张表算完，然后停下汇报。

**6. BERT 重新规划（用户已接受）**：

- 用新规则重新生成 BERT 的规则程序，并逐点给出新旧选择对照（方法、fit、深度）。
- 如果程序文本有变化，在同一组 20 条留出样本上重做 Mock 和 tensor 验证（定义域只来自校准样本，选择过程没有使用留出样本，所以留出样本仍可用于判定）。
- 然后重做：成本预测，compiler 和 managed 两种模式编译，预测、compiler、managed 三方的自举数和切点一致性检查，完整 RuntimePlan 比较。
- 给出新旧两版的自举数、深度和延迟对照。旧的 U4a 证据保留，作为对照。
- 运行 managed 之前先报告 IR 大小。

**7. 测试与回归**：

- 加权 Remez 的单元测试：比值 342 的 inv，以及一个 rsqrt 定义域上，相对拟合的误差不大于绝对拟合，并且接近理论界；
- 不传权函数时，系数与现在相同；
- `steps=0` 和默认 fit 时，22 个旧程序、4 个冻结模型和 r8-v4/v5/v6 的检查结果保持；
- 依次运行 U4 加 parser、main、U5、S1、packaging 各组；tensor 组只有在 main 组涉及的 kernel 测试出现差异时才重跑。

**8. 接续**：Qwen 规划成功后，按 §11.7 的"接续"一节完成 U4b。逐点表增加 fit 一列。资源限制、容差、测试授权都不变。

## 7.5 补充：BERT 前端 trace 超过 8 GiB 单文件限制后的接续（Claude 于 2026-10-26，对应 `u4-bert-native-20261026T133137097241`）

**现象**：BERT 的 Mock 和张量 20 条验证已通过。compiler 模式下，DaCapo 前端以 MLIR 文本格式写到 8,589,934,592 字节时触及单文件限制，编译调用为 0。managed 模式**没有运行**：脚本在 compiler 模式失败后就退出了。

**估算**：程序规模为 681,273 个运算，其中 mul_cp 186,496 个，去重后的常量 2,503 个（共 328 MB 的 float64），boot 37,462 次，max_depth 323。作为对照，ResNet20 的文本 IR 约为 0.99 GB，对应 6,208 个常量。BERT 的常量更少，文本 IR 却超过 8 GiB，说明前端文本中的体积主要来自常量被重复写出（或运算本身的文本），而不是去重后的常量数据。去重是在 hecate-opt 的 cse 中完成的，不在前端写出之前。这个推断需要第 2 步的数据确认。

**资源限制保持不变**：可信运行器为 6 GiB 内存、8 GiB 单文件、1 h、空闲磁盘不少于 10 GiB；服务器作业沿用同样的限制。本节**不批准**扩大任何资源。问题通过缩小 IR 体积来解决，不靠放宽限制。

### 7.5.1 执行顺序（都在本机的可信运行器上进行；仍然无法完成的再打包服务器作业）

1. **先跑 managed 模式**。原包和原命令不变，`manager.emit_raw` → 前端预处理序列 → `mlir_read` → 管理 → 编译尾段。managed 的原始 IR 由我们自己的 Emit 生成，每个常量只写一次，体积应当接近"去重常量 + 运算文本"。
   - 记录 raw、preprocessed、managed 三份 IR 的大小（≥256 MiB 时按 S1 的规则 gzip），以及各阶段的峰值 RSS 和耗时。
   - 能编译时，核对 RuntimePlan 的自举数与 `boots_pred=37462` 是否一致，再按 §7.4 的口径完成验收。
   - **U4a 的 11/02 检查点以 managed 模式的结果为准。**
2. **compiler 模式改用 MLIR bytecode**。包里已有 `trace_model.py --bytecode` 选项（`hc.save(..., bytecode=True)`）。bytecode 用二进制存放数值，并对相同的属性去重，因此重复写出的常量只占一份空间。这是**导出格式的选择**，不修改 DaCapo，也不修改 hecate-opt。
   - 如果 `compile_model.py` 不接受 `.mlirbc`，就按通用方式补上输入判别：后缀为 `.mlirbc` 时直接交给 hecate-opt（MLIR 能自动识别 bytecode），其余文本和 gzip 的规则不变。这个改动对所有包都生效，既有 7 个 P2 包和 S1 的程序默认仍用文本，行为不变。
   - 编译通过后，与 managed 模式比较 RuntimePlan，口径与 S1 相同。
3. **只有 bytecode 仍超过 8 GiB 时，才做体积诊断**。把前端的文本输出写到一个命名管道（FIFO），由一个只计数、不落盘的读取进程统计：
   - 各 op 名的个数和字节数；
   - `earth.constant`（或前端的常量表示）的个数，以及单个常量的平均字节数；
   - 去重后的常量个数（按"包内常量名"或常量在 trace 中首次出现的位置判别，不做内容哈希校验）。
   
   FIFO 只用于这项计数诊断，**不能**作为绕过单文件限制、把超大 IR 喂给 hecate-opt 的通道。诊断完成后停下来汇报，由 Claude 和用户决定下一步。
4. 第 1、2 步中任何一步超出内存或 1 h 限制：停下来汇报超限的阶段（Python 管理、preprocess、depth-dp、编译尾段中的哪一个）、峰值和耗时，以及已经产出的 IR 大小。不重试，不改限制。

### 7.5.2 服务器作业

- 两份作业**按 7.5.1 更新后再打包**：compiler 作业改为 `--bytecode`，managed 作业不变。限制仍然是 6 GiB、8 GiB、3600 s、空闲磁盘 10 GiB，脚本中的 `ulimit` 保持原值。
- 只有本机因为磁盘或耗时以外的环境原因无法运行时，才交给服务器作业执行；不能用服务器作业来规避资源限制。

### 7.5.3 汇报（此外按 §7.4 的口径）

- 两种模式各自的 IR 体积（文本和 bytecode）、峰值 RSS、各阶段耗时，以及是否编译通过；
- RuntimePlan 的自举数，以及与 `boots_pred` 的一致性；
- 观察项（不阻塞本节）：37,462 次 boot、预测延迟约 9,725 s，主要来自 TokenLayout（每个 token 一个密文）乘以 softmax 平方步骤带来的深度。只记录按运算类别统计的深度和 boot 分布，**本轮不做优化**。这将作为 Agent 精调和论文局限的输入。

## 7.6 补充：在原资源限制下得到共享常量的 managed IR（Claude 于 2026-10-26）

**现象**：`manage.earth_emit.emit_raw` 按每次明文使用各建一个常量节点，并写出完整载荷。raw 文本达到 54,064,749,752 字节（gzip 后为 215 MB）。按现行规则，交给 hecate-opt 的临时文件需要约 64.8 GB 空闲空间，实际只有 13.8 GB。预处理尚未开始。§7.5 对体积的预期有误：§7.5 假设 managed 的 raw 中常量只写一次，这是我的错误。

**DaCapo 侧的事实**（只读源码确认，不做修改）：
- `PrivatizeConstant.cpp` 为每个 HEScaleOp 的每个常量操作数复制一个 ConstantOp，复制时保留原常量的 `getLoc()`、`getValue()` 和 `rms_var`。在内存中，这些副本的 DenseElementsAttr 是同一个已去重的属性；但以文本打印时，每个副本都会写出完整载荷。所以预处理输出如果用文本，体积会重新膨胀到与 raw 相近。
- `ElideConstant.cpp` 对每个 ConstantOp 执行 `dyn_cast<DenseElementsAttr>`，因此常量必须是 dense 属性，不能改用 `dense_resource`。
- `hecate-opt`（`tools/optimizer.cpp`）支持 `--emit-bytecode`，输入端能自动识别 bytecode。

**资源限制保持不变。** 以下改动都是通用的，作用于全部 managed 程序，不针对 BERT。

### 7.6.1 raw：用属性别名共享载荷

- `emit_raw` 和 `emit` 在模块开头为**每个不同的载荷**写一次 MLIR 属性别名，例如 `#ptk17 = dense<"0x…"> : tensor<16384xf64>`；每个常量 op 写成 `value = #ptk17`。这是 MLIR 文本的标准语法，解析器会把它还原成同一个 DenseElementsAttr，语义与内联载荷完全相同。
- 载荷的身份沿用包里已有的常量去重（`Backend.const` 的 `k` 名）。不另外做哈希校验。
- raw 中常量节点的个数和位置**保持不变**（仍然每次使用一个），只是载荷不再重复写出。因此 hecate-opt 看到的模块与现在完全相同，预处理的结果也不会改变。
- 每个常量的位置信息写成 NameLoc，同时保留原有的归属：`loc("ptdsl-const-k17"("ptdsl-op-5":3:0))`。

### 7.6.2 预处理：输出 bytecode，再读取结构文本

1. `hecate-opt` 运行前端预处理序列（不变），加上 `--emit-bytecode`，输出 `preprocessed.mlirbc`。bytecode 对属性去重，体积约等于"不同载荷 + 运算"。
2. 再调用一次 `hecate-opt`（不加任何 pass），读入 bytecode，用 `--mlir-elide-elementsattrs-if-larger=1` 打印**结构文本**：运算、类型、`rms_var` 和位置都完整保留，只是省略 dense 载荷。结构文本 ≥256 MiB 时按 S1 的规则 gzip。
3. `mlir_read` 新增结构模式：常量的载荷按位置中的 `ptdsl-const-k*` 从包的常量表取回（内存中共享同一个数组）。逐个核对：
   - 位置中没有 `ptdsl-const-k*` 的常量（例如由 canonicalize 新生成的常量）→ 停下汇报，**不猜测载荷**；
   - 用取回的载荷按 profile 重新计算 `rms_var`，与 IR 中打印的值逐位比较，不一致 → 停下。
   
   这两项核对保证载荷恢复正确，并不放宽任何判据。
4. 如果这个构建的 `hecate-opt` 不接受 `--emit-bytecode` 或 `--mlir-elide-elementsattrs-if-larger`：先用 `--help` 确认，然后停下汇报。不改 DaCapo，也不另找替代工具。

### 7.6.3 managed IR 与编译尾段

- 管理器输出的 managed IR 同样使用 §7.6.1 的别名：每个常量 op 按自己的类型（level/scale）写出，载荷则引用别名。
- **写出之前先做规模预估**，并记入报告：常量 op 的个数，以及不同 (载荷, 类型, rms_var) 组合的个数。后者可以近似 `ElideConstant` 写出的 `.cst` 的大小，约为组合数 × slots × 8 字节。预估 `.cst` 超过 8 GiB → 不运行编译尾段，停下汇报。
- 编译尾段的命令不变。产物中任何一个文件超过 8 GiB、内存超过 6 GiB 或超时 → 停下汇报超限的阶段，不重试。

### 7.6.4 回归（先做，全部通过后才能跑 BERT）

- S1 的 20 个程序和 U3 的 managed 程序，用新的别名、NameLoc、bytecode 与结构文本这条路径，RuntimePlan 与当前 S1 的基线**口径相同、结果一致**（19/20 的那个已知差异保持原样）。若有不一致 → 停下，并报告首个差异出现在哪个阶段。
- 其中较小的 3 个程序：旧的文本路径与新的结构模式所读出的图逐节点相同，包括类型、边、`rms_var` 和载荷。
- `emit_raw` 的体积单元测试：同一个载荷被使用 N 次时，文本中只出现一次载荷。

### 7.6.5 执行顺序

1. **compiler 模式先按 §7.5 第 2 步用 `--bytecode` 跑**。这一步不需要新代码，可以马上执行；前端直接从内存写出 bytecode，常量属性已经去重。
2. 实现 §7.6.1–§7.6.3，通过 §7.6.4 的回归后，再跑 BERT 的 managed 模式。
3. **U4a 的 11/02 检查点**：compiler 或 managed 任一模式编译通过，并按 §7.4 与张量参考一致即可。两种模式都通过时，再按 S1 的口径比较两者的 RuntimePlan。此条取代 §7.5.1 第 1 步中"以 managed 为准"的说法。
4. FIFO 诊断（§7.5 第 3 步）只在 compiler 的 bytecode 仍然超限时才做。

## 7.7 补充：自举计数差异（预测 37,462 次，编译 15,399 次）的诊断与接续（Claude 于 2026-10-26，对应 `u4-bert-compiler76-20261026T144521066801`）

**现象**：compiler 模式改用 bytecode 后，trace 和编译都以 0 退出（trace 48 s，编译 20 s，峰值 RSS < 1 GiB）。编译日志和 RuntimePlan 都记录 15,399 次自举、110 个 depth-dp 边界；成本模型预测 37,462 次、107 个 barrier。

**初步判断**（需要用数据确认）：
- RuntimePlan 中各类运算的个数约为我们 DAG 计数的一半：rotate 72,051 对 145,591，mul_cp 100,188 对 186,496，add_cc+add_cp 153,944 对 add+sub 277,500。
- DaCapo 前端在 save 之前运行 `cse, canonicalize, privatize-constant, canonicalize`。我们的 kernel 在 TokenLayout 下会重复构造结构上完全相同的计算，例如对同一个密文做相同步长的旋转、对同一个 K 的重复展开。cse 会把这些合并掉，而成本模型（`cost.py` 在原始 DagBackend 上放置自举）没有这一步。
- S1 中 ResNet20 的差异属于同一类问题（缺少前端 cse 预处理），当时只在 managed 路径上修正，没有修正成本模型。之前冻结的模型能精确吻合，是因为它们的图中几乎没有可合并的重复计算。

### 7.7.1 诊断范围（只读，不调用 LLM，不用报告集，资源限制不变）

1. **重新 trace 一次**（bytecode，约 50 s），用 `hecate-opt` 不加 pass、以 `--mlir-elide-elementsattrs-if-larger=1` 输出结构文本（≥256 MiB 时 gzip）。按 op 类别统计个数，并按位置中的 `ptdsl-op` 归并到 PT-DSL 的每一行，与我们 DAG 的逐行计数对照。
2. **在我们的 DAG 上做与 MLIR CSE 语义一致的值编号**：op 种类、操作数 id（顺序有意义，不做交换律归一化）、属性（旋转步长等）、常量身份（包内的 `k` 名）全部相同，即视为同一个值。重新统计个数，并重新运行 `placement.depth_dp`。
3. **用 S1 的 `mlir_read` 读入第 1 步的结构文本**（只需要结构；depth-dp 不需要常量载荷），在这张图上运行 S1 的 depth-dp 移植，统计自举数。
4. **对照表**：

   | 图 | 运算个数（分类别） | depth-dp 边界数 | 自举数 |
   |---|---|---|---|
   | 原始 DAG | | | |
   | 值编号后的 DAG | | | |
   | 前端预处理后的图 | | | |
   | RuntimePlan | | | |

   判定：
   - 如果值编号后的 DAG 与 RuntimePlan 的自举数相同，原因确认为缺少 cse。
   - 如果仍有差距，再找出差距来自 canonicalize 的哪一类改写：对照第 1 步与第 2 步的逐行计数，给出第一个计数不同的 PT-DSL 行，以及该行内不同的 op 类别。**只诊断，不扩展修正**，停下汇报。
5. 给出逐行的去重率（原始个数 / 去重后个数），并标出去重最多的 kernel（预计是 matmul mode A/B 和 softmax 的 broadcast/sum）。这一项作为观察，本轮不改 kernel。

### 7.7.2 允许的修正（仅当第 4 步确认原因是缺少 cse）

- 在成本模型使用的 DAG 上加入**值编号**，语义与 7.7.1 第 2 步相同。可以在 `DagBackend` 记录运算时直接复用相同的节点（hash-consing），也可以在放置自举之前单独做一遍。Mock 和张量引擎的数值语义不受影响：被合并的节点值本来就相同。
- **对冻结与既有结果的影响**：对 4 个冻结开发模型、U3 的 9 个程序和 P2 的 7 个程序，分别比较加入值编号前后成本模型的输出（自举数、延迟、boot_nodes、statement_costs）。
  - 全部相同：对所有路径启用。
  - 有任何不同：只对 r8-v7 和 U4 的程序启用，r8-v4 到 r8-v6 的路径保持原样；报告中列出哪些程序发生了变化、变化了多少。**这些变化不回写冻结结果。**
- **Agent 的反馈**：r8-v7 的 evaluate 反馈中的自举数和延迟随之改变。这是预期的修正，不需要改提示词文本。
- **测试**：
  - 值编号的单元测试：相同的旋转、相同的 mul_cp、相同常量、不同步长，以及交换操作数顺序（不合并）；
  - BERT：值编号后的预测自举数等于 15,399，boot 边界数与编译日志的 110 一致；
  - 上述回归比较。

### 7.7.3 §7.4 判定与 U4a 检查点

- 7.7.2 完成后，compiler 模式的 BERT 判为 §7.4 **自举数一致**，前提是预测值等于 RuntimePlan 中的值。RuntimePlan 已经保存，**不需要重新编译**。
- 至此，U4a 的 11/02 检查点（"1 个 Transformer 能完成 lowering 和编译，并与张量参考一致"）以 compiler 模式的结果满足。在 STATUS 中写明：编译的是 compiler 模式，自举数 15,399；CKKS 实际执行不在检查点范围内。
- 如果 7.7.1 判定原因不止于 cse，则检查点只记为"编译通过，自举数的预测存在 X 次差异，原因在诊断中"，不算完全满足，停下汇报。

### 7.7.4 §7.6.1–§7.6.4 的接续

- 7.7.3 完成后，再继续 §7.6.1–§7.6.4（属性别名、bytecode 预处理、结构模式、回归）和 BERT 的 managed 模式，顺序不变。
- 新增一项交叉核对：managed 路径读入前端预处理后的图，在其上运行 S1 的 depth-dp 移植，所得自举数应当与 compiler 模式的 15,399 相同（与 S1 的口径一致）。若不同 → 停下汇报，并给出第一个不同的边界。
- 7.7.1 第 1 步中重新 trace 产生的 bytecode 和结构文本，在 §7.6.2 实现之前保留；用完后按既有规则清理。

## 7.8 补充：成本模型复现前端的规范化（减法表示、值编号、输出可达性）（Claude 于 2026-10-26，对应 `u4-cse-diagnosis-20261026T150825422995`）

**诊断结论**（§7.7.1 已完成）：

| 图 | 自举数 |
|---|---|
| 原始 DAG | 37,462 |
| 只做值编号 | 28,982 |
| 前端预处理后的图 = RuntimePlan | 15,399 |

- 共有 278,343 个算术节点不能到达最终输出。只统计可达节点，并把 sub 改成前端的表示之后，每一行的加法、两类乘法和旋转个数都与前端一致。
- 死代码的主要来源：最后一层只有 CLS token 进入 pooler，其余 39 个 token 在最后一层的计算都不可达。例如第 22 行的 linear，mul_cp 5,160 个中只有 129 个可达。
- 减法的表示：`frontend.cpp:208-225` 中，`sub(x, c)` 若 c 是常量，就构造取负后的**新常量**，再做 `add`；否则构造 `negate(c)`，再做 `add`。这正好解释了前端多出的 860 个 neg，以及第 7 行 layernorm 中 sub 与 add/neg 计数的差异。
- 这些都是前端固定的、与模型无关的规范化，因此修正是通用的。

### 7.8.1 实现（只改成本模型使用的 DAG；Mock、张量引擎、Emit、包导出不变）

在 `placement.depth_dp` 之前，对 DagBackend 记录下的图依次做以下三步，得到"规范化图"。成本模型中由 DAG 运算决定的输出都在规范化图上计算：自举、延迟、boot_nodes、statement_costs、hotspots。

**`ciphertext_counts` 不变**（2026-10-26 统一口径）：它表示每个 PT-DSL 值按张量形状和布局声明的密文块数，是布局属性，不是 DAG 统计。例如 BERT 第 22 行仍记 40，即使规范化图中只有 1 个该行输出参与最终结果。本轮**不新增**活跃密文数，不建立 PT-DSL 输出到 DAG 节点的映射，`lower.py` 的观察接口也不在修改范围内。死代码的规模只用 §7.8.1 末尾所说的比例记录在 STATUS 中（可达与不可达节点数，以及按行统计的不可达节点数），这些数字都由规范化前后的 DAG 直接算出。

1. **减法表示**（按前端的写法，在记录运算时或规范化时完成均可）：
   - `sub_cp(x, k)`：替换为 `add_cp(x, k')`，其中 `k' = Backend.const(−k 的载荷)`，沿用包内已有的常量去重；Python 标量按同样方式取负。
   - `sub_cc(x, y)`：替换为 `neg(y)` 后接 `add_cc(x, ·)`。`neg` 按成本表中的 `negate_single` 计价，不消耗深度。
   - 在 IEEE 754 中 `a + (−b)` 与 `a − b` 的结果逐位相同，所以数值语义不变（Mock 和张量引擎本来也不使用这张图）。
2. **值编号**：与 §7.7.2 的定义相同，在减法改写之后进行，使相同的 `neg(y)` 和相同的取负常量能够被合并。保留第一次出现的节点。
3. **输出可达性（DCE）**：从函数的输出反向遍历，删除不可达的节点，并保持其余节点原来的顺序。所有算术运算都没有副作用，所以可以直接删除。

此外，归属规则是：合并后的节点归属第一次出现时的 PT-DSL 语句（与 MLIR CSE 保留第一个 op 的位置一致）；被删除的节点不计入任何语句。

不做的事：
- 不复现 canonicalize 的其他改写。诊断表明本例中不存在其他改写；如果回归中发现差异，就停下汇报，不再扩展。
- 不改 kernel 来消除死代码或重复计算（例如只计算 CLS token 的最后一层）。这留给 Agent 或后续工作，本轮只在 STATUS 中记录死代码和重复计算的比例。

### 7.8.2 回归

1. **单元测试**：
   - `sub_cp` → 取负常量 + `add_cp`，包括 Python 标量的情况，且取负常量会被去重；
   - `sub_cc` → `neg` + `add_cc`，并且同一个 y 的两次减法共享同一个 `neg`；
   - DCE 删除不可达的节点，保留输出及其祖先，节点顺序不变；
   - 在小图上手工计算深度和自举数进行核对。
2. **BERT**：规范化图上的预测自举数为 15,399，depth-dp 边界数为 110，`max_depth` 为 323；各类运算个数与前端图相同（add、mul_cp、mul_cc、rot、neg）。若有不同 → 停下，给出第一个不同的行。
3. **既有程序**：
   - 比较 4 个冻结开发模型、U3 的 9 个程序、P2 的 7 个程序和 S1 的 20 个程序，在修正前后成本模型的输出。
   - 凡是已有真实 RuntimePlan 的程序，修正后的预测自举数必须仍然等于 RuntimePlan 中的值。原来相等、修正后变得不等 → 停下。
   - 启用范围的规则与 §7.7.2 相同：全部相同就对所有路径启用；有任何不同，就只对 r8-v7 和 U4 启用，并列出发生变化的程序。冻结结果不回写。
4. **managed 路径的一致性核对**（只读）：检查 S1 的 `emit_raw` 中 `sub` 的表示是否与前端相同（前端从不写出 `earth.sub`）。只记录结论，**本轮不修改 S1**。如果两者不同，在 §7.6.4 的回归中一并报告；若 S1 已知的那 1 个 RuntimePlan 差异可能由此引起，也一并说明。

### 7.8.3 接续

- 7.8.2 全部通过后，按 §7.7.3 判定：compiler 模式的 BERT 自举数一致，U4a 的 11/02 检查点满足。RuntimePlan 已经保存，不需要重新编译。
- 然后按 §7.7.4 接续 §7.6.1–§7.6.4 和 BERT 的 managed 模式。
- 不调用 LLM，资源限制不变。原有测试授权继续有效。

## 7.9 补充：BERT managed 的 Python 管理阶段超时（Claude 于 2026-10-27，对应 `u4-bert-managed76-20261027T030724861581`）

**现象**：共享载荷之后，raw 降到 793 MB（gzip 12.9 MB），预处理、结构读取和 depth-dp 都很快完成，其中 depth-dp 只用了 0.97 s。但随后的 Python 管理阶段在 3600 s 限制内没有返回。SqueezeNet-SiLU（约 5 万个运算、282 次自举）可以正常完成。

**代码审读确认的热点**：`manage/greedy.py` 的 `manage()` 在主循环之后，为每个 bootstrap 插入输入端的 modswitch，写法是

    for n in list(result):
        if n.kind == 'bootstrap':
            n.args[0] = refine(n.args[0], n)
            created = result.pop()
            result.insert(result.index(n), created)

`result.index(n)` 每次都线性扫描，而且是逐字段的 dataclass 相等比较；`list.insert` 每次还要移动其后的全部元素。BERT 约有 15,399 个 bootstrap、约 10⁶ 个节点，需要上百亿次 Python 级操作。SqueezeNet 的 bootstrap 只有 282 个，所以问题没有暴露。

### 7.9.1 诊断（标准库，不安装任何工具，资源限制不变）

1. 在 `manage()` 和后续步骤中加入**分段计时**，写入 `trace-stages.json`：
   - schedule、depth_dp、主循环、bootstrap 输入端 modswitch 的插入（即上面的循环）、`early_modswitch`、`check`、managed 发射，每段单独计时。
2. 分段计时始终记录。诊断运行时另外用 `faulthandler.dump_traceback_later(600, repeat=True)` 每 10 分钟输出一次 Python 栈。先把 BERT 的管理阶段单独运行一次，上限为 20 分钟，读入已保存的 `preprocessed.mlirbc` 和结构文本，**不重新 trace**，以确认热点。如果热点不在上述循环中，同样记录下来。
3. 规模外推：在 SqueezeNet-SiLU 和 BERT 上，给出各段耗时与节点数、bootstrap 数的关系。

### 7.9.2 允许的修正（只限算法复杂度，输出必须完全相同）

- 只能改动 `manage/` 下的 `greedy.py`、`check.py`、`earth_emit.py`、`mlir_read.py`，并且**只做不改变输出的复杂度修正**：
  - **上述插入循环**改为线性重建：按原来的顺序对每个 bootstrap 调用 `refine`（保持 `next_id` 的分配顺序），把新建的 modswitch 记在 `bootstrap id → modswitch` 的映射中，然后一次遍历 `result`，在每个 bootstrap 之前放入它的 modswitch。最终的节点顺序、id、类型和属性都与原写法相同。
  - 7.9.1 定位到的其他平方级写法（线性 `index`/`remove`/`insert`、重复的全表扫描），按同样的原则改为映射或一次遍历。
- **不允许**：改变放置、EarlyModswitch 或类型推导的语义；改变 depth-dp；跳过 `check`；为了速度跳过或合并任何阶段；放宽资源限制。

### 7.9.3 验证

1. **新旧实现逐节点相同**：在随机小图（种子固定，包含多个 bootstrap、共享输入和输出）以及 S1 的 3 个小程序上，比较新旧 `manage()` 的完整 Function：节点顺序、id、kind、args、type、attrs、outputs、cuts、boundaries。旧实现作为测试中的参照保留。
2. **回归**：§7.6.4 的 19 个本机成员中，选取 bootstrap 最多的 3 个（例如 MobileNet、SqueezeNet、ResNet20），用新实现得到的 managed IR 与上一轮结果逐节点相同，RuntimePlan 与 S1 的口径一致。其余成员不必重跑。
3. **BERT managed**：在原有的 3600 s 限制内完成管理、按 §7.6.3 做规模预估、编译尾段和 RuntimePlan 比较。managed 的自举数应为 15,399，并且全部 110 个切点与 compiler 模式相同。
4. 如果修正后仍然超时或超出内存，停下来汇报各段耗时，不重试，不改限制。

### 7.9.4 接续

- 7.9.3 完成后，继续 U4a 剩余的兼容性回归（旧程序 Emit/常量、四个冻结模型、r8-v4/v5/v6 逐字对照、既有 545 项测试），然后进入 U4b。
- **11/02 检查点**已经由 compiler 模式满足（§7.8.3），不受本节影响。
- 原有测试授权继续有效，不调用 LLM。

### 7.9.5 补充：M3 磁盘准入停止后的接续（Claude 于 2026-10-27，对应 `u4-manage-m3-20261027T044209517968`）

**现象**：复杂度修正已经通过（4/4，旧 BERT 插入段从 676 s 以上未返回降到线性）。M3 第 1 例完整通过；第 2 例 R6c 的完整 Function 相同、check 也通过，但 managed IR 解压后为 1,640,287,632 字节，超过 1 GiB，需要走临时文件路线。按"临时文件大小 + 10 GiB"的准入规则，需要 12,377,705,872 字节空闲，当时只有 11,885,137,920，差 470 MiB。第 3 例尚未启动。

**规则不变**：10 GiB 的空闲余量、1 GiB 的 stdin 上限、6 GiB 内存和 8 GiB 单文件限制都保持原值。

**接续**：
1. **先重新测量空闲空间**。2026-10-27 04:58 时为 12,709,011,456 字节，已经满足 R6c 的需求。只要满足准入，就继续 R6c 的归一化、编译尾段和完整 RuntimePlan 比较，然后跑第 3 例（u3a-resnet20_relu）。每例完成后按原 prune 规则立即清理已消费的中间文件，再开始下一例。
2. **BERT managed 运行之前**，先给出 managed IR 解压后的大小（按 §7.6.1 的别名写法估算或实测）：
   - ≤1 GiB：走 stdin，不需要临时文件；
   - >1 GiB：按原准入规则检查。
3. 任何一例不满足准入，就停下汇报所需空间和实际空闲空间，并执行下面第 4 步的清单整理，不自行删除。
4. **磁盘清单（只读，不删除）**：在 `~/poseidon-work/platforms/aarch64-linux/results/`（约 58 GB）和 `~/pt_agent_results/`（约 13 GB）中，按目录列出大小和最后修改时间，并标出以下几类：
   - 被现行 pt_agent 配置、P2 包或服务器作业脚本引用的目录（例如 `agent-current-dacapo-build-r1712`），**必须保留**；
   - 被 STATUS、PLAN 或冻结结果（tag `pt-agent-freeze-v1`、`r8e7-reconfirm-*`）引用的目录，保留；
   - 其余目录列为**候选清理项**，给出合计大小。
   
   清单写入 `DISK_INVENTORY.json`，并在 STATUS 中给出摘要。**是否删除由用户逐项批准**，未经批准不删除任何文件。
5. 之后按 §7.9.3 第 3 步完成 BERT managed，再按 §7.9.4 继续。

## 7.10 补充：Qwen managed 预检在管理阶段内存不足（Claude 于 2026-10-27，对应 `u4-section118-stopped-20261027T084721590783`）

**现象**：Qwen 的 compiler 模式已通过（54752 次自举、138 个切点）。随后在大小预检中调用 `manage(recovered)`，在 `early_modswitch` 内发生 MemoryError（普通 Python 进程 RLIMIT_AS 4 GiB），进程运行 229 秒后退出。BERT 同一路径可以通过，Qwen 的计划规模约为 BERT 的 2.1 倍（1,090,571 步对 522,418 步）。

**初步判断**（需要用数据确认）：这是内存占用的问题，不是算法问题。

- `u4_relative_acceptance.preflight` 调用 manage 时，预处理图 `function`、`front` 和 `recovered` 仍然被引用，同时还要再构建一份管理后的图。
- `manage/ir.py` 中的 `Node` 是普通 dataclass，没有 `__slots__`，每个节点都带一个实例字典。
- `early_modswitch` 又为全部节点额外建立了 `nodes`、`users`、`before` 三张表，最后用递归生成器重建节点列表。

### 7.10.1 诊断（标准库，不安装工具，资源限制不变）

- `timed_stage` 在每个阶段增加两项记录：进入和退出时的 RSS（`/proc/self/status` 的 VmRSS 和 VmHWM），以及 Node 实例数。
- 诊断运行时另外用 `tracemalloc` 在每个阶段结束时取快照，记录按行号排序的前 20 项和总量。tracemalloc 只用于诊断运行，不在验收运行中开启。
- 在 BERT 上完整跑一次，得到各阶段的峰值和每节点字节数。然后在 Qwen 上用同样的 4 GiB 限制跑到失败点，给出逐阶段的内存表，以及失败时仍然存活的大对象（哪几份图、哪些表）。

### 7.10.2 允许的修正（只限内存占用，输出必须完全相同）

范围是 `manage/{ir,greedy,check,earth_emit,mlir_read}.py` 和实验入口 `experiments/u4_relative_acceptance.py`、`u4_bert_managed.py`、`u4_managed_payloads.py`：

1. 及时释放不再使用的引用。例如在调用 manage 之前释放预处理图和 `front`，只保留后续比较真正需要的计数和切点；各阶段的临时表在用完后立即释放。
2. `Node`、`Type` 改用 `__slots__`（可用 `dataclass(slots=True)`，取决于 Nix 环境中的 Python 版本）。冻结的 `Type` 可以复用相同的实例。会被修改的 `attrs` 字典**不能**共享。
3. `early_modswitch` 的递归 `expanded` 改为显式栈的迭代实现，输出顺序不变。`users`、`before` 只为确实存在的项建立条目。
4. 能用生成器的地方不先建立完整列表。

**不允许**：改变管理算法、遍历顺序、编号或任何输出；把数据移到磁盘数据库或 mmap 等新的外部存储方案；改变 4/6 GiB、3600 秒、8 GiB 的限制，或把 Python 管理进程归到 6 GiB 的可信运行器下；改用服务器作业来回避这一限制。

### 7.10.3 验证

- 修正前后，在 3 个回归程序和 BERT 上，管理后的 Function（节点序列、类型、attrs、切点、边界）完全相同。BERT managed 的完整 RuntimePlan 与已有的 §11.8 结果相同。
- 给出修正前后 BERT 和 Qwen 逐阶段的峰值内存对照。
- 运行 U4 加 parser 测试组和 S1 测试组。如果改动涉及 `ir.py`，再运行 main 组。

### 7.10.4 接续

1. 修正通过后，重做 Qwen 预检，先在对话中报告 managed IR 的大小和执行路线（≤1 GiB 走 stdin，否则走临时文件并按准入条件检查空闲空间）。
2. 然后进行 managed 编译，做预测、compiler、managed 三方的自举数和切点核对，以及完整的双模式 RuntimePlan 比较，完成 U4b。
3. 如果修正之后，峰值仍超过 4 GiB：停下汇报逐阶段的内存表和估计需要的内存，由用户决定是否另行批准资源。已经封装、但未执行的服务器作业保持不动。
4. 原测试授权和资源限制都不变。

### 7.10.5 补充：诊断的采集范围（Claude 于 2026-10-27，对应 `u4-memory710-stopped-20261027T091313390406`）

**现象**：修正前在 BERT 上开启 tracemalloc 做诊断，tracemalloc 自身的开销使进程在 `early_modswitch` 处达到 4 GiB 上限。此前不开 tracemalloc 时，BERT 可以通过。

**结论**：已采集到的数据足以确定修正方向，不再要求"修正前在 BERT 上开着 tracemalloc 完整跑一次"。

| 阶段 | 存活 Node | 退出时 RSS |
|---|---:|---:|
| recording | 853,535 | 约 2.06 GB |
| 预处理 | 455,365 | 约 3.15 GB |
| 管理 | 约 120 万（预处理图与管理后的图同时存活） | 约 4.0 GB（失败） |

由此可见：

- 每个存活节点约占 2.4–3.4 KB；
- 预处理图在整个管理阶段都没有释放；
- Qwen 的规模约是 BERT 的 2 倍，recording 阶段本身就会接近上限。

**调整后的诊断口径**：

1. VmRSS、VmHWM、VmSize 和 Node 实例数在每个阶段**始终记录**。它们开销很小，验收运行中也保留。
2. tracemalloc 只在小程序上运行（R6c 或 3 个回归程序中的一个），用来取得按行号排序的前 20 项和每节点字节数。BERT 和 Qwen 上不开 tracemalloc。
3. 已有的诊断记录保留，作为"修正前"的数据。

**§7.10.2 的范围补充**：

- 修正同样适用于 recording 阶段，包括 `RecordingBackend`、`RecordedExpr` 以及它们产生的节点或记录对象（`__slots__`、复用相同的冻结对象、及时释放）。
- 预处理完成后，只保留后续需要的计数和切点。recording 的结果在预处理用完之后立即释放。
- 其余条款和"不允许"各条都不变。

**接续**：直接进入 §7.10.2。完成 §7.10.3 的验证后，在不开 tracemalloc 的情况下，给出修正前后 BERT 的逐阶段内存对照，以及 Qwen 修正后的逐阶段内存。然后按 §7.10.4 执行：先报告 IR 大小，再进行 managed 编译、三方核对和完整 RuntimePlan 比较。修正后如果仍超过 4 GiB，就停下汇报，交给用户决定。资源限制和测试授权都不变。
