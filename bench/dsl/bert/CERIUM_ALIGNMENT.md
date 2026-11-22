# Cerium 单 block 静态对齐审计

本审计针对已经编译、执行并归档的 **BERT-base batch1 / tokens128 / hidden768 /
12 heads×64 / FFN3072 / one block**，保留 Softmax、两次 LN、GELU、残差及末尾刷新。
本次仅本地 CPU 静态读取，无 SSH、GPU、运行时导入或重跑。没有完整层数值等价声明。

## 计数单位：107 logical refresh，不是 83 个单CT Boot

```
单CT逻辑 refresh = 18 b33 + 41 b35_scalar + 24 vec2 × 2 = 107
实际 packed core / ModRaise 次数 = 18 + 41 + 24 = 83
```

一次逻辑 refresh 指一条独立输入 CT 流得到一条保留输出 CT 流，适合与当前
Poseidon native `Boot` 单CT输入/单CT输出口径比较。**107 不是 Cerium 实际执行
107 个 bootstrap core。** vec2 会先把两条流合成一个复数 CT，执行一个 core，再拆成
两个 real CT；当前不做该配对的 Poseidon 1→1 Boot 需按两条流计数。
若将来补齐复数 pack/Boot/split，可分享 core，不能再把此逻辑数当物理调用数。

`bootstrap.py:450–490` 的 vec2 wrapper 实际执行：

```python
x = BreakVector(x)
assert len(x) == 2
x = x[0] + x[1]  # 第二lane在调用前已经乘i
outputs = CeriumFunctionCall(..., {"x": x})
z = MakeVector([outputs["z0"], outputs["z1"]])
```

max 用 `zeroPt5_*_im=0.5j`，sum 用 `softmax_one_im=one*1j`：
`Enc(a),Enc(b) → Enc(a+i b) → one complex refresh → Enc(a),Enc(b)`。
标量 b35 wrapper 只保留 z0；不能因为 core 有 z0/z1 就一律乘2。
EvalMod 内部的 real/imag 支路也不另计 refresh。

| 阶段 | b33 | b35 scalar | vec2 core | 单CT逻辑 refresh |
|---|---:|---:|---:|---:|
| QKV | 0 | 0 | 0 | 0 |
| QK后 | 0 | 6 | 0 | 6 |
| Softmax（3个两CT函数） | 0 | 24 | 24 | 72 |
| AV | 0 | 0 | 0 | 0 |
| O+attention残差后 | 3 | 0 | 0 | 3 |
| attLN | 0 | 4 | 0 | 4 |
| FFNup后 | 12 | 0 | 0 | 12 |
| GELU | 0 | 0 | 0 | 0 |
| down+残差后 | 3 | 0 | 0 | 3 |
| ffnLN | 0 | 4 | 0 | 4 |
| 最终refresh | 0 | 3 | 0 | 3 |
| **总计** | **18** | **41** | **24** | **107** |

每个 Softmax：7个 max vec2 + 2个 exp scalar + 1个 sum vec2 +
6个 inverse-sqrt scalar = **16 core / 24 CT refresh**。两个 LN 各是 exp 初值1次 +
Newton刷新3次。**GELU 的6次 vec2函数调用只是算术batch，内部没有 Boot。**

### 可复核证据，不是手填调用统计

```bash
python3 -B bench/dsl/bert/audit_cerium.py --check \
  test-results/bert-a800-20261115/cerium-audit.json
# 不写文件；不导入 Cerium/Torch/CUDA。省略 --check 则将重新生成的 JSON 打到 stdout。
```

脚本递归展开 `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/main_1gpu_1/functions0`，从真实 `.function_args` 的 z/z0/z1
绑定、`.outputs` 中非 `[X]` 保留输出判断 scalar/vec2；不根据函数名中的 `vec` 猜数量。
JSON 保留调用点行号、展开多重数、DAG哈希及逐阶段结果。
compiled `functions0` 与对应生成 `partition_0/ll.functions` 均作字节一致性核对。

已核验 main 二进制：
- 路径：`test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/main_1gpu_1/function_map_0.so`
- SHA-256：`45452728af86d587f7ee8cfdb18861817258db3d212785c35dd41eff3e790ed1`
- 与 `test-results/bert-a800-20261114/cerium/ARTIFACT_MANIFEST.json:839–857` 相符。

## Packing、真实 active Q 与 scale

`p_i` 是下文有序 prime 池第 i 项。**Cerium L 是 active Q prime个数**：
`Q_L=∏p[0:L]`，不是 Poseidon 的 level编号；移植应比较 q_count。
真实 scale 使用实际编码参数和 rescale 除数推导，不能拿 DSL 名义28/56/84替代。

| 缩写 | 实际scale公式 | log2 |
|---|---|---:|
| H | p12 p11 p10 | 83.064739826 |
| Q（V同） | p8 p7 p6 | 82.985248744 |
| K | p8 p5 p4 | 82.961890579 |
| A | p15 p14 | 55.414710020 |
| P（probability scale，非special modulus） | p7 p6 p5 | 82.959414058 |
| R | p4 p3 p2 | 82.890737611 |
| B | p17 p16 | 55.437918663 |
| U | p20 p19 / 1000 | 45.493452712 |
| D | p2 B / 1024 | 73.058138496 |
| T | p12 p11 / 256 | 47.379991198 |

| 阶段 | shape / CT布局 | 输入 (activeQ,scale) → 中间/输出 | DSL名义logscale | refresh |
|---|---|---|---|---:|
| QKV | 128×768，3 hidden CT → Q/K/V各6 CT；每CT两head×128×64，半槽0 | (13,H) → (9,Q/K/Q) | 84→84 | 0 |
| QK | 12×128×128，6 CT；两head循环对角分块/CT | (9,Q),(9,K) → (4,A/1024) → (16,A) | 84→56→56 | 6 |
| Softmax | 同6 score CT；3次vec2函数 | (16,A) → (8,P) | 56→84 | 72 |
| AV | 概率6 CT + V6 CT → context6 CT | (8,P),(9,Q) → (5,R) | 84→84 | 0 |
| stacking | 每两context合并成[128,256]，6→3 CT | (5,R) → (5,R) | 84 | 0 |
| O+残差 | 128×768，3 CT，加原H残差 | (5,R),(13,H) → O(3,D) → 残差(2,B/1024) → (18,B) | 84→84→56→56 | 3 |
| attLN | 同3 hidden CT；variance/isqrt仅1 CT按token广播 | (18,B) → (5,R) | 56→84 | 4 |
| FFNup | 128×768→128×3072，3→12 CT，每CT[128,256] | (5,R) → (2,U) → (21,U) | 84→56→56 | 12 |
| GELU | 同12 CT；6个两CT算术batch | (21,U) → (5,R) | 56→84 | 0 |
| down+残差 | 128×3072→128×768，12→3 CT，加3个attLN残差CT | (5,R) → down(3,D) → 残差(2,B/1024) → (18,B) | 84→84→56→56 | 3 |
| ffnLN | 同3 hidden CT；raw输出后wrapper另rescale | (18,B) → raw(3,p2 T) → (2,T) | 56→84→56 | 4 |
| 最终refresh | 输出3 CT，[128,256] | (2,T) → b35裁剪(13,T) → 乘p10×256 → (13,H) | 56→56→84 | 3 |

索引公式：
- `H_c[256*t+f]=X[t,256*c+f]`，c<3、t<128、f<256。
- `HEAD_ORDER=(0,2,1,3,4,6,5,7,8,10,9,11)`；Q/K/V第n个CT在
  `256*t+128*k+f` 存 head `HEAD_ORDER[2*n+k]`，k<2、f<64，其余半槽0。
- `Score_n[256*t+128*k+d]=dot(Q_h[t,:],K_h[(t+d)%128,:])/40`；
  runner mask的键token正是`(i+j)%128`。它不是普通row-major score矩阵。
- AV后 `Context[2*c] + rotate(Context[2*c+1],-64)` 恢复4个连续head组成的256-feature行。
- FFN/GELU保持12个[128,256] CT；不是永久把每两hidden CT压成一个复数CT。

Softmax内部：max每轮 L16→L2→Boot L16（Boot前scale A/4096）；最终max归约L14。
exp L14→L2→Boot裁至L13→最后平方L11；乘mask后的e保留L9。
sum归约L7→modswitch L2→vec2 Boot L14（前scale p13p12/8192，后乘8192）。
isqrt的三轮刷新输出L10、L10、L11；平方得inverse L9，乘e后rescale输出L8。
GELU入口U由编码scale1000的scalar调整到主多项式尺度，不能把名义scale56当实际2^56。

**已发现的debug注释错误：**
`test-results/bert-a800-20261114/cerium/fullblock/bert_runner.py:601` 的 `attention_residual_output_bs` 写成B，刷新前实际为
**B/1024**。由真实PT可得 `R×(B/1024)/(p4 p3)=p2 B/1024`，再rescale除p2；
Boot后的scalar编码scale1024才恢复B。上述表格使用推导值，不机械抄此debug OutScale。

## 模数池、special P与bootstrap参数

**64项28-bit prime是复用池，不是64 active Q，也不是固定Q52/P12分割。**
普通active Q用池前缀；special modulus用按操作选出的后缀：
`P_e=∏p[64-e:64]`。e取决于level、keyswitch类型及分解，非统一dnum。
同一prime可在一个阶段作Q、另一个阶段作P。

| 操作 | active Q / 临时域 | extension e | special-P池索引 |
|---|---:|---:|---|
| b33高层rotation（dnum5） | 52 | 11 | 53..63 |
| b35高层rotation（dnum4） | 49 | 13 | 51..63 |
| L13线性rotation（dnum1） | 13 | 13 | 51..63 |
| L30 relin（dnum1） | 30 | 30 | 34..63 |
| toEphemeral | 2 | 1 | 63 |
| bootstrap换回dense key | 临时53或50 | 2 | 62,63 |

- N65536、slots32768；名义SCALE28，双prime逻辑rescale。
- 第一物理prime `p0=204865537`；bootstrap ground modulus为**两prime乘积**
  `p0*p1=42131001064292353`（约55.225731714bit），不是一个55-bit prime。
- b33强制输入Q2；稳定raise Q52，backend换key临时用Q53含p52再rescale；core输出Q21。
  available19→Q21，available16→裁Q18；prefix名`b33_16`不代表core只输出16层。
- b35输入Q2；稳定raise Q49，临时Q50含p49；core的z0/z1各输出Q16。
  wrapper实际裁至 **availableLevels+2**；最终available11→Q13。
- Hanki degree30、DA/HK_r3、K16、ε=1/256；b33 C2S三段×2-rescale，
  b35四段×2-rescale；S2C三段×1-rescale及real projection，非单纯degree参数。
- 主secret Hamming weight **32768**；**h32仅ephemeral secret**，不要当主secret稀疏度。

真prime有序池，0-based；完整数值也保存在JSON `he_parameters.ordered_prime_pool`：

```
0–15: 204865537,205651969,206307329,207880193,209059841,210370561,211025921,211812353,214171649,215482369,215744513,216137729,216924161,217317377,218628097,219676673
16–31: 220594177,221249537,222035969,222167041,222953473,223215617,224002049,224133121,225574913,228065281,228458497,228720641,230424577,230686721,230817793,231473153
32–47: 232390657,232652801,234356737,235798529,236584961,236716033,239337473,239861761,240648193,241827841,244842497,244973569,245235713,245760001,246415361,249561089
48–63: 253100033,253493249,254279681,256376833,256770049,257949697,258605057,260571137,260702209,261488641,261881857,263323649,263454721,264634369,265420801,268042241
```

## 与当前Poseidon新方案仅部分对齐

当前紧凑方案 **3 hidden CT / 6 score CT** 对齐了布局规模；双prime物理lowerer及
**native56 / d30DA3K16候选**对齐部分scale调度/参数选择，**不是原样Cerium profile**。
这不是这些候选已GPU验证、数值等价或性能更优的声明。

`bench/dsl/bert/cerium_packing.py:1–28` 明确保留当前模型自己的 encrypted mean
subtraction、exp4、3步reciprocal、GELU4、rsqrt3；Cerium却是sign/max reduction、
6步exp、6步inverse-sqrt及分段GELU。不能仅凭packing或Boot次数宣称非线性已对齐。
`bench/dsl/bert/cerium_lowering.py:1–7` 也明确它不是Cerium compiler。
候选degree/DA/K相同仍不保证Hanki系数/ε、ground双prime、DFT分组、P/key策略、
real/imag配对、实际scale或输出activeQ相同。

| 能力 | 当前标准RuntimePlan边界 |
|---|---|
| QKV/O/up/down实数linear，QK/AV CCmatmul | 可用现有Rotate/Add/Mul/Relin/Rescale/ModSwitch lower；这些源码本身不需要conjugate或complex PT。融合性能另论。 |
| period256/128 real PT | 显式tile到32768 real slots后可语义复现；当前短数组encode不能假定自动周期重复。 |
| 压缩encoded PT与remapableBase | 展开PT+静态绑定可算，但不保留Cerium压缩布局/重用性能；同等压缩/换绑需新增Prepare/存储/API支持。原生kernel真实用`i>>7`寻址。 |
| 非线性中的conjugate | 标准ComputeKind无Conjugate；理想real消息可把x+conj(x)重写为2x，但不保持imaginary-noise投影，非精确原图。 |
| complex PT、vec2 pack/split | 当前Plain/EncodePayload/Prepare接口仅real；需暴露complex与Conjugate。底层GPU已有conjugate，CPU encoder已有complex，不代表标准Runtime图已能调用。 |
| native Boot profile | 现有Boot可承载单CT刷新；当前Boot目标scale约定与Cerium可变prime乘积scale不同。已有MulCP可做真实归一化，不能当作原样搬图。 |

ApiCall callback是后端扩展入口，不等于标准图已具有上述可移植原语。

## 原生源码与runtime DAG证据

以下为真实文件路径（相对Poseidon仓库根；`../reference`是兄弟源码树）：
- `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/main_1gpu_1/functions0:17–521,1319,1970–2663,2810–2894`：
  顶层实际调用，末尾三条b35仍在。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/softmax_128x128_vec/functions0`：
  max vec2行21899/42962/63878/85928/107369/128369/149873；
  exp scalar154325/154346；sum vec2 159155；
  isqrt scalar161129/161150/163439/165476/166736/168794。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/layernorm_att_1gpu/functions0:8711,11105,13142,15179` 与
  `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/layernorm_ffn_1gpu/functions0`相同行号：各4个b35。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/bootstrap_32K_33lvl_t1/functions0:10483`：2poly×21limbs=1CT；
  `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/compiled/bootstrap_32K_35lvl_t1_vec_1/functions0:10248–10253`：4poly×16limbs=2CT。
- `test-results/bert-a800-20261114/cerium/fullblock/bootstrap.py:298–544,688–808`：core、wrapper、ground modulus、Hanki及真实PT scale。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_cerium.py:97–138,534–656,1119–1200,1507–1779,1942–1986,2197–2214,2519–2565`：
  QK/AV/linear/LN/GELU及全部刷新位置。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_params_torch.py:96–250`：head order及真实weight/mask/affine编码scale；
  `test-results/bert-a800-20261114/cerium/fullblock/bert_runner.py:51–68,414–652,727–751`：packing/OutScale及secret权重。
- `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/bootstrap_32K_33lvl_t1/partition_0/inputs0:2–12`、
  `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/bootstrap_32K_35lvl_t1_vec_1/partition_0/inputs0:2–12`：真Q/P索引；
  `test-results/bert-a800-20261114/cerium/fullblock/bert_outputs/attention_qkv_1gpu/partition_0/kernels_0.cu:1149–1150`：压缩PT寻址。
- `../reference/cerium/cerium/compiler/backend/keyswitch_digits.cc:36–88,871–896`、
  `compiler_keyswitch.cc:198–250`、`terms.h:625–726`（后三者同backend目录）：动态P与分解；
  `../reference/cerium/cerium/compiler/backend/compiler_rescale.cc:101–213`：临时额外Q prime。
- `third_party/ckks-runtime/runtime/plan.hpp:30–69`、
  `src/poseidon/runtime_api/poseidon_gpu_api.cpp:2161–2229`：标准算子/payload和单CT Boot契约；
  `src/poseidon/gpu/gpu_evaluator.h:753–761`：底层GPU conjugate；
  `src/poseidon/runtime_api/gpu_plaintext_encoder.h:10–27`：real-only GPU Prepare接口。

机器可读产物：`test-results/bert-a800-20261115/cerium-audit.json`。
其中统计、prime池和二进制/DAG哈希由CPU脚本重取；阶段数值契约是源码审计注记，
不是声称能从低层DAG的函数名自动推导全部scale。
