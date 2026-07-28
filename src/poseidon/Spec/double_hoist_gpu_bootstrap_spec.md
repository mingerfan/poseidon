# Poseidon GPU CKKS Bootstrapping Double-Hoist 实现规范

**文档状态**：Implementation Specification / 主实施纲领  
**目标代码基线**：`poseidon-lfy-26.7.21.zip`，分支语义以 2026-07-21 的双 30-bit GPU 高精度自举实现为准  
**目标算法**：Bossuat et al., EUROCRYPT 2021，Section 4.2–4.3，Algorithm 4 与 Algorithm 6  
**主要目标阶段**：`CoeffToSlot`，随后复用到 `SlotToCoeff`  
**不直接修改的阶段**：ModRaise、59 阶 EvalMod、多项式 BSGS、两次 double-angle  

---

## 1. 文档目的

本规范用于指导在当前 Poseidon GPU 框架中实现 **double-hoisted BSGS matrix–ciphertext multiplication**。实现完成后，C2S/S2C 的数学结果、模数链消耗、scale 调度和矩阵分解方式必须保持不变；优化只重组 rotation/KeySwitch 的数据流。

本规范重点解决四个问题：

1. 如何将当前“一次 rotation = 一次完整 HYBRID KeySwitch”的实现拆分为可复用阶段；
2. 如何共享 baby rotations 的 Decompose/ModUp，并延迟 inner/outer loop 的 ModDown；
3. 如何在 `Q∪P` 基中完成明文乘加，同时保持连续访存；
4. 如何控制 workspace、寄存器、kernel 粒度和 occupancy，避免为了减少 KeySwitch 反而引入新的 GPU 瓶颈。

本文使用以下规范词：

- **MUST**：正确性或架构要求，不应省略；
- **SHOULD**：推荐实现，只有经过测量证明不合适时才可改变；
- **MAY**：可选优化。

---

## 2. 当前 7.21 GPU 实现基线

### 2.1 当前线性变换调用链

当前 C2S/S2C 的 GPU 调用链为：

```text
GpuEvaluator::coeff_to_slot / slot_to_coeff
    └── GpuEvaluator::dft
          └── GpuEvaluator::multiply_by_diag_matrix_bsgs
                ├── rotate(source, baby_step)
                ├── multiply_plain(rotated_source, diagonal)
                ├── add(inner_sum, product)
                ├── rotate(inner_sum, giant_step)
                ├── add(result, rotated_inner_sum)
                └── rescale_many(...)
```

主要文件：

- `src/poseidon/gpu/gpu_evaluator.h`
- `src/poseidon/gpu/gpu_evaluator.cpp`
- `src/poseidon/gpu/gpu_linear_transform.h`
- `src/poseidon/gpu/gpu_uploader.cpp`

### 2.2 当前 rotation 的实现

`GpuEvaluator::rotate()` 当前执行：

```text
1. 为输出密文重新分配 Q-only c0/c1
2. 对 c0 做 Galois permutation
3. 对 c1 做 Galois permutation，物化 rotated_c1
4. 清零输出 c1
5. switch_key_hybrid_ciphertext(rotated_c1)
6. 输出完整 Q-basis rotation ciphertext
```

`switch_key_hybrid_ciphertext()` 当前是一个不可拆分的完整过程：

```text
INTT(switch_poly)
→ decomposition / ModUp Q→QP
→ forward NTT
→ evaluation-key multiply-accumulate
→ inverse NTT / ModDown P→Q / forward NTT
→ 加回 destination
```

主要文件：

- `src/poseidon/gpu/gpu_keyswitch_handler.h`
- `src/poseidon/gpu/gpu_keyswitch_handler.cpp`
- `src/poseidon/gpu/kernels/gpu_keyswitch_kernels.h`
- `src/poseidon/gpu/kernels/gpu_keyswitch_kernels.cu`

### 2.3 当前实现的主要重复工作

对同一 BSGS 矩阵，所有 baby rotations 的输入都是同一个 `source_ciphertext.c1`，但当前每个 baby rotation 都重复执行：

- `c1` 的 INTT；
- dnum decomposition；
- Q→P 基转换；
- decomposition digits 的 forward NTT；
- 单独的 ModDown；
- 完整 Q ciphertext 的物化与写回。

每个 giant rotation 又执行一次完整 KeySwitch。

因此当前实现虽然使用了 BSGS 减少 rotation 数量，但仍把 KeySwitch 当作黑盒，没有利用 KeySwitch 内部阶段可交换、可延迟的性质。

---

## 3. 目标算法与优化边界

### 3.1 KeySwitch 阶段记号

将一次 rotation 的主要成本写为：

```text
D = Decompose：INTT + dnum decomposition + ModUp + NTT
M = MultSum：decomposition digits × evaluation key，模乘加
R = ModDown：QP → Q
P = Permute：Galois automorphism
```

普通 rotation 近似为：

```text
D + M + R + P
```

其中 `D` 和 `R` 包含大量 NTT、INTT 和 RNS 基转换，通常远贵于 `M` 和 `P`。

### 3.2 经典 BSGS

设一个矩阵有 `n = n1 × n2` 条 diagonal：

- `n1`：baby-step 数；
- `n2`：giant-group 数。

经典 BSGS 的 rotation 成本约为：

```text
(n1 + n2) × (D + M + R + P)
```

### 3.3 Single hoisting

所有 baby rotations 共享一次 source `c1` 的 Decompose：

```text
D + n1 × (M + R + P)
+ n2 × (D + M + R + P)
```

### 3.4 Double hoisting

Double hoisting 继续延迟 ModDown：

```text
D + n1 × (M + P)
+ n2 × (D + M + R + P)
+ R
```

直观含义：

- baby loop：Decompose 只做一次；baby 结果保留在 QP，不逐个 ModDown；
- 每个 giant group：在 QP 中完成所有 baby plaintext multiply-accumulate，再做一次 inner ModDown；
- outer loop：giant rotations 的 KeySwitch correction 保留在 QP 中累加，最后只做一次 outer ModDown。

### 3.5 本优化不会改变的内容

Double hoisting **MUST NOT** 改变：

- C2S/S2C 的矩阵数；
- 每个矩阵的 diagonal 数和数学数值；
- C2S/S2C 每矩阵的 logical rescale 次数；
- 自举模数链和剩余 level；
- ciphertext scale；
- EvalMod 的 59 阶 Chebyshev、多项式 DAG 和两次二倍角；
- 最终解码结果和精度目标。

它只改变同一线性矩阵乘内部 rotation/KeySwitch 的执行顺序。

---

## 4. 数学数据流规范

### 4.1 Lifted QP 表示

定义 `LiftP(ct)`：把 Q-basis ciphertext `ct=(c0,c1)` 提升为一个尚未 ModDown 的 QP 对象：

```text
Q limbs: P mod qi × c0/c1
P limbs: 0
```

满足：

```text
ModDownP(LiftP(ct)) = ct
```

该对象不是普通 CKKS ciphertext，不允许被通用 evaluator API 当作 Q ciphertext 使用。

### 4.2 Hoisted decomposition

对 source ciphertext 的 `c1` 只执行一次：

```text
H = Hoist(c1)
  = {d0, d1, ..., d(dnum-1)}
```

每个 digit 都必须已经完成：

```text
coefficient-domain decomposition
→ Q/P ModUp
→ NTT
```

最终存储为 QP NTT-domain digit batch。

### 4.3 Lifted baby rotation

对于 baby step `b`：

- `b=0`：直接生成 `LiftP(source)`；
- `b≠0`：使用 hoisted digits 和 rotation key 做 key multiply，不立即 ModDown；加入 `P*c0`，再应用 Galois permutation，形成 lifted rotated ciphertext：

```text
BabyQP[b] = LiftedRotateQP(source, H, b)
```

必须满足：

```text
ModDownP(BabyQP[b]) ≈ Rotate(source, b)
```

误差只来自正常 KeySwitch，与经典 rotation 同阶。

### 4.4 Inner-loop 延迟 ModDown

当前 `gen_linear_transform_bsgs()` 已经对 diagonal 做了 giant-step 预旋转。因此 double-hoist path **MUST 复用当前 `plain_vec` 的 diagonal 语义**，不得再次改变 diagonal 旋转方向。

对 giant group `g`：

```text
GroupQP[g] = Σ_b BabyQP[b] × PlainQP[g+b]
InnerQ[g]  = ModDownP(GroupQP[g])
```

一个 giant group 内只执行一次 ModDown。

### 4.5 Outer-loop 延迟 ModDown

对每个 `InnerQ[g]`：

- `g=0`：直接 `LiftP(InnerQ[0])` 加入 `OuterAccumQP`；
- `g≠0`：对 `InnerQ[g].c1` 做 Decompose，执行无 ModDown giant KeySwitch，加入 `P*InnerQ[g].c0` 并 Galois permutation，得到 lifted giant rotation，累加到 `OuterAccumQP`。

全部 giant groups 完成后：

```text
ResultQ = ModDownP(OuterAccumQP)
```

最后按原矩阵配置执行：

```text
rescale_many(ResultQ, matrix_group.step())
```

### 4.6 完整伪代码

```cpp
HoistedDigits H = hoist_decompose_modup_ntt(source.c1);

zero(group_accum_qp[0..n2-1]);

for (baby tile T : baby_steps) {
    BabyQPTile babies = make_lifted_baby_rotations_qp(
        source.c0, source.c1, H, T, pre_rotated_galois_keys);

    qp_plain_mul_accumulate_groups(
        babies,
        matrix.double_hoist_plan,
        group_accum_qp);
}

zero(outer_accum_qp);
for (g = 0; g < n2; ++g) {
    InnerQ inner = moddown_qp_to_q(group_accum_qp[g]);

    if (giant_step[g] == 0) {
        lift_p_and_accumulate(inner, outer_accum_qp);
    } else {
        HoistedDigits Hg = hoist_decompose_modup_ntt(inner.c1);
        LiftedQP rotated = keyswitch_no_moddown_and_permute(
            inner.c0, Hg, giant_step[g], pre_rotated_galois_keys);
        add_qp(outer_accum_qp, rotated);
    }
}

CiphertextQ result = moddown_qp_to_q(outer_accum_qp);
rescale_many(result, destination, rescale_count);
```

---

## 5. 总体架构设计

### 5.1 实现策略

实现必须分阶段推进，不能直接替换经典路径：

```text
Classic BSGS（现有，永久保留）
    ↓
Single-hoist correctness path
    ↓
Pre-rotated key path
    ↓
Inner delayed ModDown
    ↓
Outer delayed ModDown
    ↓
GPU-specific BSGS ratio tuning
```

### 5.2 Feature flag

必须保留运行时回滚：

```text
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=classic
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=single_hoist
POSEIDON_GPU_LINEAR_TRANSFORM_MODE=double_hoist
```

默认值在 double-hoist 全部正确性和性能门槛通过前必须为 `classic`。

也可在 C++ 中增加：

```cpp
enum class GpuLinearTransformMode {
    ClassicBsgs,
    SingleHoistBsgs,
    DoubleHoistBsgs,
};
```

### 5.3 适用范围

第一版必须只支持：

- 单 GPU；
- 单 full-coefficient shard；
- size-2、NTT-form、Q-only 输入 ciphertext；
- HYBRID KeySwitch；
- C2S 矩阵；
- generic `q_count/p_count/dnum`，不能写死 `p_count=2`。

C2S 稳定后再启用 S2C。Conjugation 不走 double hoisting，因为只有一次 rotation，无法摊销 hoist 成本。

---

## 6. 数据结构设计

建议新增文件：

```text
src/poseidon/gpu/gpu_double_hoist.h
src/poseidon/gpu/gpu_double_hoist.cpp
src/poseidon/gpu/kernels/gpu_double_hoist_kernels.h
src/poseidon/gpu/kernels/gpu_double_hoist_kernels.cu
```

### 6.1 Hoisted decomposition

```cpp
struct GpuHoistedDecomposition
{
    parms_id_type parms_id{};
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t dnum = 0;

    // coefficient-domain source, reusable by setup/debug paths
    DeviceVector<GpuWord> source_intt_q;

    // Layout: [dnum][limb][coeff]
    DeviceVector<GpuWord> digits_q_ntt;
    DeviceVector<GpuWord> digits_p_ntt;
};
```

布局必须为：

```text
digits_q_ntt[((digit * q_count + limb) * N) + coeff]
digits_p_ntt[((digit * p_count + limb) * N) + coeff]
```

理由：同一个 warp 固定 `digit/limb`，访问连续 coeff；KeyMult kernel 对 dnum 循环时，warp 每次访问仍然连续。

### 6.2 QP ciphertext buffer

```cpp
struct GpuQPCiphertextBuffer
{
    int device_id = 0;
    std::size_t degree = 0;
    std::size_t q_count = 0;
    std::size_t p_count = 0;
    std::size_t batch_count = 0;

    // [batch][component][limb][coeff]
    DeviceVector<GpuWord> q;
    DeviceVector<GpuWord> p;
};
```

索引：

```text
Q: ((((batch * 2 + component) * q_count + limb) * N) + coeff)
P: ((((batch * 2 + component) * p_count + limb) * N) + coeff)
```

Q/P 分开保存，避免 kernel 内对 Q/P 模数表分支，也兼容当前 `modup_q/modup_p` 和 `accum_q/accum_p` 设计。

### 6.3 Double-hoist matrix plan

当前 hot path 使用 `std::map<int, GpuPlaintextData>` 和 `bsgs_index()` 动态组织。Double-hoist setup 必须生成一个紧凑计划：

```cpp
struct GpuDoubleHoistTerm
{
    std::uint32_t giant_index;
    std::uint32_t baby_index;
    std::uint32_t diagonal_id;
};

struct GpuDoubleHoistMatrixPlan
{
    std::uint32_t log_slots = 0;
    std::uint32_t n1 = 0;
    std::uint32_t n2 = 0;
    std::uint32_t rescale_count = 1;

    std::vector<int> baby_steps;
    std::vector<int> giant_steps;
    std::vector<std::uint32_t> group_term_offsets;
    std::vector<GpuDoubleHoistTerm> terms;

    // GPU pointer/id tables，setup 后不变
    DeviceVector<const GpuWord *> diagonal_q_ptrs;
    DeviceVector<const GpuWord *> diagonal_p_ptrs;
    DeviceVector<std::uint32_t> term_baby_indices;
    DeviceVector<std::uint32_t> group_term_offsets_device;
};
```

### 6.4 Persistent workspace

```cpp
struct GpuDoubleHoistWorkspace
{
    GpuHoistedDecomposition source_hoist;
    GpuHoistedDecomposition outer_hoist;

    GpuQPCiphertextBuffer baby_tile;
    GpuQPCiphertextBuffer group_accumulators;
    GpuQPCiphertextBuffer outer_accumulator;
    GpuQPCiphertextBuffer qp_temp;

    GpuCiphertextData inner_q;
    GpuCiphertextData result_q;

    DeviceVector<const GpuWord *> key0_ptrs;
    DeviceVector<const GpuWord *> key1_ptrs;
    DeviceVector<std::uint32_t> key_indices;

    std::size_t baby_tile_size = 4;
    std::size_t capacity_degree = 0;
    std::size_t capacity_q_count = 0;
    std::size_t capacity_p_count = 0;
    std::size_t capacity_dnum = 0;
    std::size_t capacity_n2 = 0;
};
```

Workspace 必须在 `GpuBootstrapWorkspace` 中持久化：

```cpp
GpuDoubleHoistWorkspace coeff_to_slot_double_hoist;
GpuDoubleHoistWorkspace slot_to_coeff_double_hoist;
```

不得在每个 rotation 或每个 matrix term 中重新分配。

---

## 7. Galois key 格式

### 7.1 为什么需要 pre-rotated keys

只共享 decomposition 但继续使用当前标准 Galois key 时，需要对每个 hoisted decomposition digit 分别做 Galois permutation。若 dnum 较大，这会增加 `dnum × QP` 的置换流量。

论文的优化 key format 把 inverse automorphism 预先施加到 rotation key 上，使 runtime 只需在 KeyMult 之后对两个输出 component 做一次 permutation。

生产 double-hoist path **SHOULD 使用 pre-rotated inverse Galois keys**。

### 7.2 GPU key metadata 修改

在 `gpu_key.h` 增加：

```cpp
enum class GpuGaloisKeyFormat : std::uint8_t
{
    Standard,
    InversePreRotated,
};

struct GpuKeyMeta
{
    ...
    GpuGaloisKeyFormat galois_format = GpuGaloisKeyFormat::Standard;
};
```

并在 `GpuEvaluationKeyData` 中保存：

```cpp
std::vector<std::uint32_t> galois_elts_by_key_index;
```

不能依赖隐含 `key_index → galois_elt` 算术映射；setup 必须从 CPU `GaloisKeys` 的真实索引关系记录。

### 7.3 上传阶段生成 pre-rotated keys

建议在 `GpuUploader` 新增：

```cpp
static GpuGaloisKeysData upload_double_hoist_galois_keys(
    const GaloisKeys &src,
    const GpuParameterData &params,
    int device_id);
```

执行：

```text
1. 按现有 upload_galois_keys 上传 standard key
2. 分配相同 [Q_storage | P] 布局的目标 key
3. 对每个 key_index、dnum、component：
   target = φ_inverse(galois_elt)(source)
4. 标记 format = InversePreRotated
```

Key 多项式已经在 NTT domain，使用 NTT-domain Galois permutation kernel。该转换在 untimed setup 完成。

第一版可保留 standard keys 作为 classic path；显存不足时可只保留 pre-rotated keys，并在 classic 模式下重新上传 standard keys。

---

## 8. QP diagonal plaintext

### 8.1 当前问题

当前 `GpuMatrixPlain::plain_vec` 只上传 Q-basis plaintext。Double hoisting 在 QP 中做 inner-loop PMult，因此每条 diagonal 必须同时提供 Q 和 P residue。

### 8.2 数据结构扩展

推荐新增：

```cpp
struct GpuMatrixPlainQP
{
    GpuMatrixPlain q_matrix;
    std::vector<GpuPlaintextData> qp_diagonals;
    GpuDoubleHoistMatrixPlan double_hoist_plan;
};
```

也可直接扩展 `GpuMatrixPlain`：

```cpp
std::map<int, GpuPlaintextData> plain_vec_qp;
std::optional<GpuDoubleHoistMatrixPlan> double_hoist_plan;
```

### 8.3 P limbs 的生成方式

推荐路线：**setup-time GPU base extension**。

```text
CPU 生成现有 Q NTT plaintext
→ 上传 Q plaintext
→ setup-time inverse NTT Q
→ exact Q→P base conversion
→ forward NTT P
→ 存成连续 [Q | P] GPU plaintext
```

必须是无损 RNS base extension，不能通过浮点重新编码。

应新增：

```cpp
GpuPlaintextData extend_plaintext_q_to_qp_ntt(
    const GpuPlaintextData &q_plain,
    const GpuLevelInfo &level_info,
    GpuPlaintextExtensionWorkspace &workspace);
```

该步骤不计入 timed path。

### 8.4 Scale 与 metadata

QP plaintext 的逻辑数值和 scale 必须与原 Q plaintext 完全相同：

```text
parms_id = 当前 Q level
q_count  = 当前 Q count
p_count  = key P count
scale    = 原 diagonal plaintext scale
is_ntt_form = true
```

P limbs 只用于延迟 ModDown，不改变 CKKS scale。

---

## 9. KeySwitchHandler API 重构

当前匿名 `HybridScratch` 和完整 `switch_key_hybrid_ciphertext()` 必须拆分。旧 API 保留，并由新阶段 API 组合实现，避免两套数学实现长期分叉。

### 9.1 新公开 workspace

将可复用 scratch 从 `.cpp` 匿名空间移到头文件或独立 internal header：

```cpp
struct GpuHybridKeySwitchWorkspace;
```

至少包含：

- source INTT；
- one-digit ModUp scratch；
- all-dnum hoisted Q/P buffers；
- Q/P component accumulators；
- four-step NTT scratch；
- pointer tables。

### 9.2 新阶段 API

```cpp
void hoist_decompose_modup_ntt(
    const GpuConstRNSPolyView &switch_poly_ntt,
    const GpuLevelInfo &level_info,
    GpuHoistedDecomposition &destination,
    GpuHybridKeySwitchWorkspace &workspace) const;

void keyswitch_multsum_no_moddown(
    const GpuHoistedDecomposition &hoisted,
    const GpuConstEvaluationKeyView &keys,
    const GpuEvaluationKeyData &key_storage,
    std::size_t key_index,
    GpuQPCiphertextBuffer &destination,
    std::size_t destination_batch,
    bool initialize) const;

void add_p_times_c0_qp(
    const GpuConstRNSPolyView &c0_q,
    GpuQPCiphertextBuffer &destination,
    std::size_t batch,
    const GpuLevelInfo &level_info) const;

void moddown_qp_ciphertext_to_q(
    const GpuQPCiphertextBuffer &source,
    std::size_t source_batch,
    GpuCiphertextData &destination,
    const GpuLevelInfo &level_info,
    GpuHybridKeySwitchWorkspace &workspace) const;
```

### 9.3 复用现有内核

优先复用当前实现中的：

- `inverse_ntt_switch_poly()`；
- `process_hybrid_decomposition_block()` 的 ModUp/NTT 部分；
- `launch_hybrid_multiply_accumulate_two_components()`；
- `launch_hybrid_apply_moddown_ntt()`；
- P→Q base conversion parameter tables。

当前 `#if 0` 的 all-dnum PAccum 实验代码可作为参考，但不得直接恢复未经验证的 dead code。应提取其“all decomposition digits 连续存储”的数据布局思想，并重新实现、单测。

---

## 10. Evaluator API 修改

### 10.1 新入口

在 `gpu_evaluator.h` 增加：

```cpp
void multiply_by_diag_matrix_bsgs_double_hoist(
    const GpuCiphertextData &source,
    const GpuMatrixPlainQP &matrix,
    const GpuGaloisKeysData &pre_rotated_galois_keys,
    std::uint32_t rescale_count,
    GpuDoubleHoistWorkspace &workspace,
    GpuCiphertextData &destination) const;
```

### 10.2 dft 调度

`GpuEvaluator::dft()` 增加 mode 和 workspace：

```cpp
void dft(
    ...,
    GpuLinearTransformMode mode,
    GpuDoubleHoistWorkspace *double_hoist_workspace,
    ...);
```

每个融合矩阵执行完成后仍立即按原计划 rescale。不能跨两个 DFT 矩阵延迟 rescale。

### 10.3 BootstrapData

在 `GpuBootstrapData` 增加：

```cpp
GpuLinearTransformMode linear_transform_mode;
GpuLinearMatrixGroupQP coeff_to_slot_matrix_qp;
GpuLinearMatrixGroupQP slot_to_coeff_matrix_qp;
```

classic mode 继续使用原字段，确保回归测试与逐阶段对比。

---

## 11. CUDA kernel 设计

建议新增以下内核，而不是构造一个超大 monolithic kernel。

### 11.1 Hoisted decomposition

```text
kernel A: inverse NTT source c1
kernel B: per-dnum ModUp + forward NTT → digits_q/p_ntt
```

可以继续复用当前 fused ModUp/NTT 和 four-step NTT 路径。

输出必须是所有 dnum 连续存储，供多个 baby key 使用。

### 11.2 Batched KeyMult without ModDown

```cpp
launch_hoisted_keymul_batch_no_moddown(
    out_q, out_p,
    hoisted_digits_q, hoisted_digits_p,
    key0_ptrs, key1_ptrs,
    batch_key_indices,
    tile_count,
    ...);
```

推荐 grid：

```text
grid.x = ceil(N / blockDim.x)
grid.y = active limb
ngrid.z = tile_count × component
```

每个 thread 处理一个固定 `(rotation, component, limb, coeff)`，只在 dnum 维度做短循环。

每次 modular product 必须先约减到当前 modulus，再用 64-bit 累加约减值。不要在寄存器中累加多个未约减的 60-bit product。

### 11.3 Add `P*c0` and identity lift

单独 kernel：

```text
Q limb: out += (P mod qi) × c0
P limb: 不增加
```

`baby_step=0` 的 identity lift：

```text
Q: P*c0, P*c1
P: 0, 0
```

### 11.4 QP Galois permutation

Production key path 在 KeyMult 后只对两个 QP components 做 permutation。

必须使用 out-of-place permutation：

```text
output address 连续
input address 按 permutation table gather
```

不要让一个 thread 同时处理多个 limb。grid.y 固定 limb，warp 内 lane 连续 coeff。

Permutation table 应预计算并存储：

- 若表可由位运算低成本生成，可现场计算；
- 若现场计算导致较高寄存器或整数指令开销，则存 `uint32_t[N]` read-only table；
- 不应为每个 limb 复制 permutation table。

### 11.5 QP plaintext multiply-accumulate

```cpp
launch_qp_plain_mul_accumulate_groups(
    group_accum_q/p,
    baby_tile_q/p,
    diagonal_q/p_ptrs,
    term_baby_indices,
    group_offsets,
    ...);
```

每个 thread 固定一个 `(group, component, limb, coeff)`，在 tile 内循环 baby terms：

```cpp
GpuWord acc = existing_acc;
for (term in group_tile_terms) {
    acc += mul_mod(baby[term.baby][...], plain[term.diag][...]);
    periodic_reduce(acc);
}
out = acc;
```

每个线程只保留一个或最多两个 accumulator。禁止将一个 giant group 的所有 term 放入寄存器数组。

### 11.6 ModDown

ModDown 应继续分成可控制寄存器的现有阶段，优先复用已验证 kernel。第一版不要把完整 `INTT(P)+BConv(P→Q)+add+NTT` 融合为一个大 kernel。

### 11.7 Outer accumulation

所有 outer lifted rotations直接加到一个 `outer_accumulator_qp`，避免物化 `n2` 个 outer rotation ciphertext。

---

## 12. 访存连续性规范

### 12.1 基本布局

整个实现 MUST 延续 Poseidon 当前的 limb-major 布局：

```text
[limb][coeff]
```

批量对象使用：

```text
[batch][component][limb][coeff]
```

最内层永远是 coeff。

### 12.2 Warp 映射

MUST 满足：

```text
一个 warp 固定：batch / component / limb
lane 0..31 对应连续 coeff
```

禁止 lane 跨 limb。禁止使用：

```text
linear_index → coeff 和 limb 混合取模
```

导致一个 warp 尾部跨越两个 limb 的映射。

### 12.3 Q/P 分离

Q 和 P 使用独立 buffer/launch 或至少独立 grid 区间。这样：

- 同一 warp 使用同一 modulus family；
- modulus、Barrett constants、NTT tables 连续；
- 避免 `if (limb < q_count)` 的 divergence；
- P 区很小，可单独选择 block 配置。

### 12.4 Key 布局

当前 key poly 是一块连续 `[Q_storage | P]`。Double-hoist keymul 必须继续零拷贝使用 active Q prefix：

```text
Q active prefix: [0, q_count)
P start: storage_q_count * N
```

不得为每个 q_count compact/复制完整 key。

### 12.5 Baby tiling

不要默认物化全部 `n1` 个 QP baby ciphertext。推荐：

```text
baby_tile_size = 4（初始值）
```

调度为：

```text
生成 T 个 baby QP
→ 更新所有 giant group QP accumulators
→ 复用 baby tile buffer
```

这样保留 hoisting 收益，同时避免 `n1 × QP ciphertext` 显存增长。

### 12.6 Pointer table

所有 diagonal/key pointer tables MUST 在 setup-time 上传。Timed path 禁止每次调用：

- 创建 `std::vector<const GpuWord*>`；
- host-to-device copy pointer arrays；
- 遍历 `std::map` 生成 schedule。

---

## 13. 寄存器与 occupancy 规范

### 13.1 设计原则

Double hoisting 会增加 QP 计算宽度，但不能通过一个超大 kernel 同时融合：

- dnum loop；
- 两个 component；
- Q/P；
- 多个 baby；
- 多个 giant group；
- permutation；
- PMult accumulation。

这种融合会造成寄存器数组、长 live range 和 spill。

### 13.2 Kernel 切分要求

MUST 采用以下粒度：

1. `hoist_decompose_modup_ntt`；
2. `keymul_batch_no_moddown`；
3. `add_p_c0 / identity_lift`；
4. `galois_permute_qp`；
5. `qp_plain_mul_accumulate`；
6. `moddown`。

允许合并 2+3，前提是 NCU 证明无 spill；第一版不合并 permutation。

### 13.3 寄存器门槛

推荐目标：

- `registers/thread ≤ 64`：理想；
- `64–80`：可接受，需确认 occupancy；
- `>96`：默认视为失败，除非性能测试证明收益；
- local-memory load/store 必须为 0 或可忽略。

禁止仅凭 `registers/thread` 单指标判断，必须结合：

```text
launch__occupancy_limit_registers
sm__warps_active.avg.pct_of_peak_sustained_active
l1tex local load/store
smsp issue active
```

### 13.4 Block size

初始候选：

```text
128 threads
256 threads
```

setup-time 或 benchmark-time 选择。不要直接写死 512 threads，因为 QP keymul 的寄存器和表访问可能限制 occupancy。

### 13.5 循环展开

- dnum 小且编译期未知：不完全展开；
- p_count=2 专用路径可有限模板化，但 generic path 必须保留；
- baby tile loop 建议 `T=2/4/8` 模板化生成三个 kernel 版本；
- 不要对最大 `n1=16/32` 全展开。

---

## 14. Workspace 内存预算

以开发中常见参数为例：

```text
N = 65536
q_count = 34
p_count = 5
dnum = ceil(34/5) = 7
GpuWord = 4 bytes
```

估算：

```text
一个 QP polynomial  = (34+5)×65536×4 ≈ 9.75 MiB
一个 QP ciphertext = 2×9.75 ≈ 19.5 MiB
hoisted digits      = 7×9.75 ≈ 68.25 MiB
baby tile T=4       ≈ 78 MiB
n2=4 group accum    ≈ 78 MiB
outer accumulator   ≈ 19.5 MiB
source INTT Q       ≈ 8.5 MiB
```

加上 ModDown/NTT 临时区，单矩阵 workspace 目标应控制在：

```text
250–400 MiB
```

MUST 支持配置：

```text
POSEIDON_GPU_DOUBLE_HOIST_MAX_WORKSPACE_MB
POSEIDON_GPU_DOUBLE_HOIST_BABY_TILE
```

如果估算超过预算：

1. 先减小 baby tile；
2. 再考虑逐 group accumulator；
3. 不得静默回退到频繁 allocate/free；
4. 可以显式回退 classic BSGS。

---

## 15. BSGS 参数重新选择

### 15.1 不能沿用 sqrt 分块假设

Classic BSGS 认为 baby/giant rotation 成本相似，最佳通常 `n1≈n2`。Double hoisting 后 baby side 更便宜，最佳倾向：

```text
n1 > n2
```

当前 `find_best_bsgs_ratio()` 依赖固定 `log_bsgs_ratio`，必须增加 GPU double-hoist cost model。

### 15.2 第一版策略

对每个矩阵枚举所有 power-of-two `n1`：

```text
1, 2, 4, 8, 16, ... slots
```

计算：

- 实际 baby step 数；
- 实际 giant group 数；
- diagonal term 数；
- workspace；
- 需要的 Galois key 数。

使用估算：

```text
Cost = D_source
     + nbaby × (M_baby + P_perm)
     + ngiant × (D_outer + M_outer + R_inner + P_perm)
     + R_outer
     + ndiag × PMult_QP
```

### 15.3 第二版 autotune

从单算子 benchmark 测得：

```text
T_decompose(q_count)
T_keymul_no_moddown(q_count,p_count,dnum)
T_moddown(q_count,p_count)
T_permute_qp(q_count,p_count)
T_pmult_qp(q_count,p_count)
```

setup-time 选择最小预测时间的 `n1`。结果写入 `GpuDoubleHoistMatrixPlan`，timed path 不再搜索。

### 15.4 Key 数量权衡

较大的 n1 可能增加需要的 baby rotation key 数。setup 必须输出：

```text
classic key count
double-hoist key count
key memory bytes
selected n1/n2
predicted operation count
```

不得只用 kernel 时间而忽略 key 显存。

---

## 16. 文件级修改清单

### 16.1 必须修改

| 文件 | 修改内容 |
|---|---|
| `gpu_evaluator.h/.cpp` | 新增 double-hoist matrix API；dft mode dispatch；接入 workspace |
| `gpu_keyswitch_handler.h/.cpp` | 拆分 Decompose、KeyMult-no-ModDown、ModDown；公开持久 workspace |
| `gpu_key.h/.cpp` | key format metadata；galois element map |
| `gpu_uploader.h/.cpp` | pre-rotated key 上传；QP matrix plaintext；double-hoist plan 生成 |
| `gpu_linear_transform.h` | QP matrix/plan 数据结构 |
| `gpu_evaluator.h` 的 `GpuBootstrapData` | C2S/S2C double-hoist setup objects |
| `gpu_evaluator.h` 的 `GpuBootstrapWorkspace` | C2S/S2C 持久 double-hoist workspace |
| `gpu_keyswitch_kernels.h/.cu` | 可拆分的 keymul、ModDown 接口；必要的 batch kernel |
| `gpu_elementwise_kernels.h/.cu` | QP PMult-accumulate、lift-P、QP add |
| bootstrapping test CMake | 新文件加入构建 |
| gpu_data_struct test CMake | 新文件加入构建 |
| mgpu CMake | 若公共 GPU evaluator 依赖新文件，加入 source list |

### 16.2 建议新增

| 文件 | 职责 |
|---|---|
| `gpu_double_hoist.h/.cpp` | 调度、plan、workspace 管理 |
| `gpu_double_hoist_kernels.h/.cu` | QP batch/permute/accumulate 专用内核 |
| `test_gpu_double_hoist.cpp` | 独立单元和性能测试 |

---

## 17. 分阶段实施计划

### Milestone 0：基线与测量冻结

- 保留 classic path；
- 固定 7.21 dual-30 正确性输入；
- warmup ≥ 3，iterations ≥ 10；
- 记录每个 C2S matrix 的 diagonal、n1/n2、rotate 数、q_count；
- NVTX 拆分 baby/giant/PMult/rescale。

**验收**：基线时间稳定，重复运行方差低于 3%。

### Milestone 1：KeySwitch workspace 持久化

暂不做 hoisting，仅把当前 `HybridScratch` 改为 reusable workspace。

**验收**：

- classic 结果不变；
- timed path 无 KeySwitch DeviceVector allocate/free；
- 性能不回退超过 3%。

### Milestone 2：Single hoisting，standard key correctness path

- source c1 Decompose 一次；
- 对 hoisted digits 做每-step permutation；
- 每个 baby 仍单独 ModDown；
- 主要用于验证 staged KeySwitch API。

**验收**：

- 每矩阵 source Decompose 次数从 `nbaby` 降为 1；
- 输出 scale/parms_id 与 classic 一致；
- decode 误差不劣于正常 KeySwitch 容差。

### Milestone 3：Pre-rotated Galois key

- setup 生成 inverse-pre-rotated keys；
- runtime 从“permute all digits”改为“KeyMult 后 permute 2 components”。

**验收**：

- permutation bytes 显著下降；
- single-hoist 比 standard-key hoist 更快；
- key format 校验错误时明确拒绝，不静默误用。

### Milestone 4：Inner delayed ModDown

- baby outputs 留在 QP；
- QP diagonal PMult-accumulate；
- 每 giant group 一次 ModDown。

**验收**：

- inner ModDown 数从 `nbaby` 降为 `ngiant`；
- QP PMult 无寄存器 spill；
- C2S 单矩阵快于 single-hoist。

### Milestone 5：Outer delayed ModDown

- giant rotation correction 在 QP 累加；
- 最终一次 outer ModDown。

**验收**：

- outer ModDown 从 `ngiant-1` 降为 1；
- 完整 Algorithm 6 数据流成立；
- C2S full stage 正确。

### Milestone 6：GPU BSGS ratio / tile autotune

- 枚举 n1；
- 选择 baby tile；
- 记录 workspace 与 key memory；
- 可选 CUDA Graph。

### Milestone 7：S2C 接入

C2S 稳定后复用同一框架到 S2C。因当前 S2C 时间较小，不能阻塞 C2S 首次落地。

---

## 18. 正确性测试

### 18.1 Stage-level tests

必须分别测试：

1. `hoist_decompose_modup_ntt` 与当前每次 KeySwitch 内 decomposition 输出一致；
2. `keyswitch_multsum_no_moddown + moddown` 与当前 `switch_key_hybrid_ciphertext` 一致；
3. pre-rotated key rotation 与 classic `rotate()` 解密结果一致；
4. `QP PMult + delayed ModDown` 与 `ModDown + Q PMult` 一致；
5. outer delayed ModDown 与逐 giant rotation 一致。

### 18.2 Matrix tests

对每个 C2S/S2C 融合矩阵：

```text
classic GPU BSGS
single-hoist GPU BSGS
double-hoist GPU BSGS
CPU BSGS reference
```

比较：

- `parms_id`；
- `q_count`；
- scale 相对误差；
- decoded max absolute error；
- decoded precision bits。

### 18.3 边界用例

- baby step 0；
- giant step 0；
- negative/wrapped rotation；
- 不是完整矩形的 sparse BSGS group；
- 32、63 diagonal；
- lower q_count matrix；
- generic `p_count=5`；
- q_count 不能整除 p_count 的 final dnum tail；
- repeated bootstrap calls；
- workspace 复用与扩容。

### 18.4 Full bootstrap

必须使用当前 7.21 的：

- 双 30-bit GPU path；
- 对应 CPU 高精度 reference；
- 59 阶多项式；
- 两次 double-angle。

Double hoist 不得改变输出 level 或 scale。

---

## 19. 性能与 NCU 分析规范

### 19.1 NVTX 区域

```text
double_hoist.matrix[k]
  hoist.source.intt
  hoist.source.modup_ntt
  baby_tile[t].keymul
  baby_tile[t].permute
  baby_tile[t].qp_pmult_accumulate
  inner_group[g].moddown
  outer_group[g].decompose
  outer_group[g].keymul
  outer_group[g].permute_accumulate
  outer.final_moddown
  matrix.rescale
```

### 19.2 必须记录的操作计数

每矩阵打印或结构化输出：

```text
source_decompose_count
outer_decompose_count
keymul_count
inner_moddown_count
outer_moddown_count
qp_pmult_count
permute_count
baby_tile_count
workspace_peak_bytes
```

理论目标：

```text
source Decompose = 1
outer Decompose  = nonzero giant groups
inner ModDown    = giant group count
outer ModDown    = 1
```

### 19.3 NCU 指标

KeyMult/PMult：

```text
dram__throughput
lts__t_bytes
smsp__sass_thread_inst_executed_op_integer
sm__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
local-memory load/store
```

Permutation：

```text
global load/store sectors per request
L2 hit rate
memory throughput
```

ModDown：

```text
NTT occupancy
BConv memory efficiency
register spill
```

---

## 20. 性能验收门槛

Double hoist 默认启用前必须满足：

1. **正确性**：完整 bootstrap 精度不低于 classic 路径的统计容差；
2. **level/scale**：完全一致；
3. **稳定性**：连续 100 次调用无错误、无显存增长；
4. **C2S 性能**：double-hoist 相比 classic 至少提升 25%；
5. **增量价值**：double-hoist 相比 single-hoist 至少提升 10%，否则保留 single-hoist；
6. **端到端**：完整 bootstrap 至少提升 10%；
7. **寄存器**：关键新 kernel 无明显 local-memory spill；
8. **workspace**：默认配置不超过设定预算，且失败时明确回退。

上述数值是工程 gate，不是论文承诺。若实测证明 H100 上最佳收益不同，可更新门槛，但必须保留对比数据。

---

## 21. 主要风险与处理

### 21.1 QP PMult 成本抵消 ModDown 收益

Double hoisting 将 `2×n` component PMult 从 Q 扩展到 QP。若 PMult 已是带宽瓶颈，收益可能降低。

处理：先实现 single hoist并分解计时；Inner delayed ModDown 必须独立开关。

### 21.2 Pre-rotated key 生成错误

Key 的 automorphism 方向错误会得到“格式正确但解密错误”的结果。

处理：为每个 rotation step 单独验证：

```text
classic rotate(step)
vs
pre-rotated-key staged rotate(step)
```

### 21.3 Final dnum tail

`q_count % p_count != 0` 时最后一个 digit limb 数不足。所有 hoisted layout 和 keymul kernel必须显式使用 `decomp_limb_count`，不能假设完整 p_count。

### 21.4 Workspace 过大

使用 baby tiling和 `n2` group accumulator；支持预算检测与 classic fallback。

### 21.5 Kernel 过度融合

任何试图把 KeyMult、Permute、PMult、group reduction 合并为单 kernel 的优化必须在基础版本完成后单独评估。默认拒绝以高寄存器和低 occupancy 换 launch 数减少。

### 21.6 Current dead experimental code

`gpu_keyswitch_handler.cpp` 中 `#if 0` 的 all-dnum PAccum 代码与当前 `HybridScratch` 已不一致。不得简单取消 `#if 0`。只可复用算法思路和 kernel，重新定义数据结构并单测。

---

## 22. 明确不做的事情

本次 double-hoist 工作不包括：

- 改成 WHET fg-CtS；
- 改变五层矩阵融合；
- plaintext compression；
- Intermediate ModRaise；
- 改 EvalMod 59 阶多项式；
- 将 real/imag EvalMod 双流并行；
- fused two-prime rescale；
- 多 GPU double hoisting。

这些可以作为后续独立优化，不能混入本次 correctness bring-up。

---

## 23. 最终实施检查表

### 数据与 setup

- [ ] 增加 double-hoist mode 与 classic fallback
- [ ] 生成紧凑 BSGS matrix plan
- [ ] 上传 QP diagonal plaintext
- [ ] 生成/上传 inverse-pre-rotated Galois keys
- [ ] 保存 key index 与 galois element 映射
- [ ] setup-time 选择 n1/n2 和 baby tile

### KeySwitch

- [ ] HybridScratch 改为 persistent workspace
- [ ] Decompose/ModUp/NTT 独立 API
- [ ] KeyMult-no-ModDown 独立 API
- [ ] generic QP ModDown API
- [ ] identity lift 和 `P*c0` kernel
- [ ] QP Galois permutation

### Matrix multiply

- [ ] baby tile generation
- [ ] group QP accumulators
- [ ] QP plaintext fused multiply-accumulate
- [ ] inner one-ModDown-per-group
- [ ] outer QP accumulator
- [ ] final one outer ModDown
- [ ] original rescale_many 保持不变

### GPU 性能

- [ ] warp 固定 limb，lane 连续 coeff
- [ ] Q/P 分离布局
- [ ] timed path 无 pointer-table upload
- [ ] timed path 无 DeviceVector allocate/free
- [ ] 无 local-memory spill
- [ ] 128/256 block size 对比
- [ ] baby tile 2/4/8 对比
- [ ] n1 候选自动评估

### 验证

- [ ] staged KeySwitch 等价测试
- [ ] per-rotation pre-key 测试
- [ ] per-matrix classic/single/double 对比
- [ ] C2S full stage
- [ ] S2C full stage
- [ ] dual-30 full bootstrap
- [ ] 100 次 workspace reuse soak test
- [ ] NCU/NSYS 报告归档

---

## 24. 实施优先顺序结论

推荐严格按以下顺序落地：

```text
1. 持久化 KeySwitch workspace
2. 拆分 KeySwitch 阶段 API
3. Single hoisting correctness
4. Pre-rotated Galois keys
5. Inner-loop delayed ModDown + QP PMult
6. Outer-loop delayed ModDown
7. GPU-specific n1/n2 与 baby tile 调优
8. 接入完整 C2S
9. 接入 S2C
10. CUDA Graph / 更激进 kernel fusion
```

其中第 1–4 步必须先完成。没有稳定的 staged KeySwitch 和 pre-rotated key，直接实现 QP inner loop 会让错误难以定位。

Double hoisting 的最终目标不是简单减少 `rotate()` 调用次数，而是让矩阵乘中的多个 rotation **不再各自承担完整 Decompose 和 ModDown**。在当前 Poseidon GPU 架构中，正确的落地方式是把 KeySwitch 由黑盒算子重构为可调度的数据流，并通过持久 workspace、QP 连续布局和受控 kernel 粒度实现该数据流。
