# Qwen2.5 0.5B：R071 对应 24 层模型数据分析

更新日期：2026-10-20。分析对象：本目录 `Qwen25_24Layer_R071_DSL`；默认路径为 **24 个 Decoder Block、两个 token 顺序 decode**。本报告区分源码统计、条件估算、父版本实测和本版本实测，供编译/GPU 测试接入使用。

## 1. 当前结论与验证状态

24 层结构、独立权重/KV、完整词表输出和编译入口已经准备；沿用 R071 的算子、非线性近似、native Boot 合约、显式释放和 CPU 明文缓存优化。

| 项目 | 当前证据 |
|---|---|
| 24 层、真实宽度配置及权重 schema | 已检查 |
| 算术内核与已发布 R071 的一致性 | 19 个继承文件哈希检查通过 |
| 小尺寸 24 层四条算术路径 | 通过，最大绝对误差约 8.972×10⁻⁸ |
| 真实尺寸 24 层明文计算 | 未运行 |
| 真实尺寸 24 层 Hecate/Dacapo 编译 | 未运行；仅编译命令 dry-run 通过 |
| 24 层各算子实际 level、明文编码 level、自举次数 | 待真实编译计划统计 |
| 24 层完整 CKKS 精度、运行时间、CPU 峰值 | 未实测 |
| GPU 显存、时间、精度 | 未实测 |
| 预训练权重的逐层非线性校准 | 未完成；现有配置用于合成数据测试 |

因此，本报告提供完整的**当前可确认数据**，不是“24 层完整推理已通过”的验收报告。不能把两层的 154 次自举、28.99 GiB RSS 或耗时乘以 12 当作本版本结果。

## 2. 版本及证据来源

- 来源两层版本：`Qwen25_2Layer_R071_DSL`，不是 081/082 候选。
- 本次 24 层源码首次发布：dsl `0df9868a285daf9a8839ada1738c3bbd16678232`；benchmark `6af8d28d61fb1dac36686f893f035d22ab2ef98a`。这些是报告新增前的源码基准提交。
- [配置](source/model/package/config/model.json)、[权重清单](source/model/package/config/weight_schema.json)、[来源哈希](source-binding.json)。
- [验证记录](evidence/validation.json)、[本报告静态统计](evidence/model-data-analysis.json)。
- [两层历史结果](evidence/parent_071/FINAL_ANALYSIS.md)、[继承布局的静态审计](evidence/parent_071/slot-layout-audit.json)。历史证据只用于解释来源。

## 3. 模型规模和完整计算流程

| 配置 | 数值 |
|---|---:|
| Decoder Block 数 | 24 |
| hidden size | 896 |
| FFN intermediate size | 4864 |
| Query heads / KV heads | 14 / 2 |
| head dimension | 64 |
| 每个 KV head 对应 Query head 数 | 7 |
| vocab size | 151936 |
| RMS epsilon / RoPE theta | 10⁻⁶ / 1000000 |
| 当前可见 token 范围 | 1 或 2 |
| CKKS 多项式次数 N / 槽数 | 65536 / 32768 |
| 默认 scale | 2⁴³ |

配置中的最大位置数 512 不代表已支持 512-token Attention。当前交付的非线性和接口只覆盖最多两个可见 token。

每个 Block：

`hidden → RMSNorm → Q/K/V 投影 → RoPE(Q,K) → 因果 GQA Attention → O 投影 → 残差 → RMSNorm → gate/up → SiLU(gate)×up → down → 残差`

24 层之后执行 final RMSNorm 和完整 151936 词表 LM head。Attention 包含 Softmax 近似；单个可见 key 时权重恒为 1，跳过 Softmax。最终 logits 不再执行 Softmax，解密后的选 token 由客户端完成。

默认 decode-pair 首先处理 token 0，建立 24 层 KV；再处理 token 1，使用这些缓存。源码中共 48 次单 token Block 调用，首 token 不请求 logits。编译器可以裁剪不影响最终 logits/KV 的运算，所以调用次数不等于编译后保留完整 Block 的次数，更不等于自举次数。

两个输入 embedding 都由调用者提供。这里的“两 token 顺序 decode”不是已实现客户端采样、自动生成下一 token 的完整服务流程。

## 4. 权重数量、磁盘大小与 0.5B 的关系

按本目录 schema 逐项相乘求和，不生成大数组：

| 单 Block 张量组 | 元素数 |
|---|---:|
| Q 权重 + bias | 802816 + 896 |
| K 权重 + bias | 114688 + 128 |
| V 权重 + bias | 114688 + 128 |
| O 权重 | 802816 |
| gate 权重 | 4358144 |
| up 权重 | 4358144 |
| down 权重 | 4358144 |
| 两个 RMSNorm 权重 | 1792 |
| 单 Block 合计 | **14912384** |

| 全模型张量组 | 元素数 |
|---|---:|
| 24 个 Block | 357897216 |
| final RMSNorm | 896 |
| embedding 表 | 136134656 |
| LM head | 136134656 |
| schema 合计：291 个具名张量 | **630167424** |

当前 fixture 分别保存 embedding 和 LM head，不合并这两份数组。若共享两者，只计一份时独立参数为 **494032768，约 0.494B**；但本包没有对预训练权重进行共享关系转换或校验。不能把 6.30 亿个存储元素误称为新增了网络层。

- 当前 fixture 的 float32 数组载荷：`630167424×4 = 2520669696` 字节，约 **2.52 GB / 2.348 GiB**；尚未生成，NPY 头和文件系统开销另计。
- 共享一份 embedding/LM head 时，同等维度 float16 载荷为 `494032768×2 = 988065536` 字节，约 0.988 GB。这是指定存储假设的计算，不是已下载权重文件的实测大小。
- 服务端 DSL 的输入是 embedding，查表在客户端进行；schema 中的 embedding 表不意味着整表都要参与服务端 CKKS 编码。
- 源码首次交付只有 59 文件、322523 字节；原 ZIP 为 95158 字节。本次增加分析文档和 JSON 后目录会变大，旧 ZIP 不包含这份新报告。

这些原始数组容量与 CKKS 编码容量、CPU RSS、GPU 显存是四种不同指标。

## 5. 输入、输出和 KV 数据量

| 路径 | 输入密文数 | 输出密文数 | 含义 |
|---|---:|---:|---|
| decode-pair | 2 | 29 | 两个 embedding，内部建立并使用缓存 |
| prefill 两 token | 2 | 29 | 一次传入两个已知 embedding |
| prefill 单 token | 1 | 29 | KV 每层只含一个 token |
| 外部缓存 decode | 25 | 29 | 一个新 embedding + 24 层已有 KV |

输入 embedding 前 896 槽有效，尾部清零。输出包含：

- 5 条 logits 密文：前四条各 32768 个值，最后一条 20864 个值，总计 151936。
- 24 条 KV 密文：每层一条；两 token 时前 512 槽有效，顺序 `K0,V0,K1,V1`，每段 128 维；单 token 时有效 256 槽。
- 两 token 全部输出有效值共 `151936+24×512=164224`；这不是 29 条密文的实际存储长度。

权重是明文常量，输入、logits 和 KV 在真实 CKKS 执行中都是密文。明文参考程序的 NumPy 数组不能称为加密输出。

## 6. 各子层槽位利用率

下表为每个 Block 相同的接口布局，按 32768 槽计算。

| 数据位置 | 有效槽 | 比例 |
|---|---:|---:|
| hidden、RMS 输出、残差 | 896 | 2.734375% |
| Q、RoPE-Q、Attention 输出、O 输出 | 896 | 2.734375% |
| 单 token K 或 V | 128 | 0.390625% |
| gate、up、SiLU、门控乘积 | 4864 | 14.84375% |
| down 输出 | 896 | 2.734375% |
| 每层最终两 token KV | 512 | 1.5625% |
| 最终五条 logits 平均 | 151936/5 | 92.734375% |

接口前缀较稀疏，但普通 Linear 会把输入复制成多行，在一条密文里同时做不同权重行的点积：896 维每批 36 行，4864 维每批 6 行。计入末批，权重乘法阶段的占用率为：Q/O 约 98%，K/V 87.5%，gate/up 97.7941%，down 88.6667%。LM head 使用 BSGS。

RMS 标量全槽广播是一个独立值的重复；Attention 的 head 分数也会复制到 V 的对应坐标。重复参与有效计算与独立 token 并行是不同口径。以上数据不是全程按时间或 level 加权的平均利用率。

## 7. 自举与 level 的当前安排

### 7.1 编译合约已知的部分

- `compiler-config.json` 的应用编译 level 范围为 **L1–L8**，自举规划上下界为 1 和 5，scale 指数 43。
- `operator-spec.json` 明确配置 native profile `cpu-bootstrap48-level1-to5`：输入 **L1、两分量**，输出 **L5、两分量、scale=2⁴³**。
- CPU 基线对应历史 48 阶真实自举及 native 回调，不是 22 阶版本；部署时仍需实际提供兼容 Runtime/核心库和回调。
- 算子规格同时描述更长的完整上下文（上界 29），与应用规划 L1–L8、Boot L1→L5 是不同层次。不能据此宣称每个模型张量都常驻完整上下文的最高 level。
- 编译工具开启自动自举规划；实际编译完成后还检查存在 native Boot、存在 Release，且没有 input reuse。只有 dry-run 时，这些编译产物检查尚未执行。

`operator-spec.json` 的 status 仍为 `placeholder`，延迟/噪声项多为 null，Boot latency 为 0。它们是合约中的未实测字段，**不是“自举耗时为零”或经过校准的 GPU 性能表**。默认合约 target 为 CPU，GPU 要提供对应规格重新编译。

### 7.2 各阶段应如何理解

| 阶段 | 可以确定的操作关系 | 本版本实际 level |
|---|---|---|
| Linear / QKV / LM head | 明文编码匹配乘法消费者；旋转、乘法、归约、mask 由具体布局决定 | 待编译统计 |
| RMSNorm | 平方、归约、逆平方根多项式、缩放；有连续乘法深度 | 待编译统计 |
| SiLU / SwiGLU | 近似多项式及 gate×up 引入密文乘法深度 | 待编译统计 |
| 双 token Attention | QK、Softmax 近似、与 V 组合及布局转换 | 待编译统计 |
| 残差 / KV 合并 | 加法输入需 level/scale 对齐，布局 mask 也有成本 | 待编译统计 |
| 每次已选 native Boot | 输入 L1 → 输出 L5 | 合约已确定；位置/次数未统计 |

rescale 通常降低 level；mod-switch 可把值对齐到较低 level；自举恢复可用深度。不能为每个 Decoder 层固定分配一个 level，因为同层内部的不同分支和中间值可能处于不同 level。

明文也不是全部在同一 level 编码。权重行、mask、常数根据消费者的 level/scale 编码，同一逻辑常量可能在不同 level 或缓存淘汰后重复 Encode。因此目前不能给出本 24 层的“某 level 有多少明文、多少 GiB”实测表；需要导出的 RuntimePlan 逐值统计。

外部缓存 decode 的浮点数据接口已检查，但跨调用 KV 的真实密文 level/scale 交接尚未验证。不能把 shape 兼容等同于两个 CKKS 计划能够直接拼接。

## 8. 内存分析：可计算的容量与尚缺的实测

### 8.1 单值系数容量

采用当前 CPU 计数约定，level L 表示 L+1 个有效模数，每系数 8 字节：

`容量字节 = 65536 × 8 × (L+1) × 分量数`

通常编码明文为一个多项式、重线性化后的密文为两个分量。乘法临时三分量、特殊模数、Boot 内部对象、分配器和密钥不能漏算或套用两分量值。

| level | 单个编码明文 | 两分量密文 | 29 个输出若全部处于此 level |
|---:|---:|---:|---:|
| L1 | 1 MiB | 2 MiB | 58 MiB |
| L2 | 1.5 MiB | 3 MiB | 87 MiB |
| L3 | 2 MiB | 4 MiB | 116 MiB |
| L4 | 2.5 MiB | 5 MiB | 145 MiB |
| L5 | 3 MiB | 6 MiB | 174 MiB |
| L6 | 3.5 MiB | 7 MiB | 203 MiB |
| L7 | 4 MiB | 8 MiB | 232 MiB |
| L8 | 4.5 MiB | 9 MiB | 261 MiB |

第三列输出合计是**指定统一 level 的条件估算**，不是已知输出 level 或实际峰值。若输出 level 不同，应逐条相加。槽位只使用 128 个也不会让一条密文缩小成上述容量的 128/32768。

### 8.2 当前继承的内存优化

1. 官方计划显式 Release，并使用 FinalOnly 保留最终结果。
2. CPU DFS 调度减少同时存活的图中间值。
3. 64 MiB 明文 Encode 根缓存，允许按需要重编码；它不是整个程序内存限制。
4. packed-row Linear、多行点积与完整槽位 LM head；KV 每层合并输出。
5. 没有混入后续候选的主动内存池回收、wide_baby16 或 native_naf 改动。

内存池保留空闲块时，Release 不保证操作系统 RSS 立即下降。CPU 优化计划要求 host placement，不能直接当作多 GPU 计划使用。

### 8.3 24 层尚不能给出的数值

| 指标 | 状态 |
|---|---|
| 释放模式图数据峰值 | 待真实编译和存活分析 |
| 全部保留 CT 系数容量 | 待逐 Value/level/分量统计 |
| 全部保留 PT 编码容量 | 待逐 Encode Value 统计，包含缓存重编码 |
| 公用计算密钥、自举矩阵和工作区 | 待匹配后端统计 |
| 整进程峰值 RSS / GPU 峰值显存 | 待对应设备实测 |
| 保留所有中间结果时真实进程峰值 | 未运行；静态容量不能代替 RSS |

完整进程由图数据、编码常量、密钥、自举工作区、原始权重、运行库及内存池等组成。部分计数可能重叠，必须统一所有权后才能相加。24 层串联时很多缓冲可复用，而最终 24 层 KV 要保留；峰值既不必线性增长，也不能直接沿用两层数字。

## 9. 计算量和时间口径

单 Block 七个线性权重矩阵合计 14909440 个元素，对一个 token 的普通 dense 参考约对应同数量的 MAC（乘加对）。两个 token 经过 24 层、末 token 做一次 LM head，未裁剪时线性 MAC 为：

`48 × 14909440 + 136134656 = 851787776`

这个统计只描述原始数学矩阵计算，不包含 Attention 的 QK/AV、非线性、RoPE、归约和布局操作，也没有扣除编译裁剪。它不是 CKKS 指令数、NTT 次数或运行时间预测。一次 CKKS 旋转可能包含密钥切换，不能和一次普通浮点 MAC 对比。

32 线程是继承 CPU 后端的建议测试设置，不表示任何时刻都占满 32 核。模型 DSL 本身不提供线程调度保证；GPU 也需要自己的编译与运行配置。

## 10. 已有精度证据

### 本 24 层版本：小尺寸浮点检查

保留 24 层，测试配置为 hidden=8、FFN=16、Q heads=4、KV heads=2、head_dim=2、vocab=19、slots=32。

| 路径 | Block 调用次数 | 小尺寸输入/输出 | 结果 |
|---|---:|---:|---|
| prefill 两 token | 24 次，每次包含两 token | 2 / 25 | 通过 |
| decode-pair | 48 次单 token | 2 / 25 | 通过 |
| prefill 单 token | 24 | 1 / 25 | 通过 |
| 外部缓存 decode | 24 | 25 / 25 | 通过 |

小尺寸只需要一个 logits 密文，因此输出为 25；真实词表需要五个，所以为 29。最大绝对误差 `8.971986953376643e-8`，不同推理路径的 logits/K/V 一致，输出尾部为零，传入缓存未修改。不是 CKKS 噪声测试，没有 GPU 或虚部精度结论。

非线性沿用 RMS degree10p2、pair13 Softmax 和 canonical_fold2 SiLU；配置中的 RMS 方差域 [0.5,2]、SiLU 域 [-2,2] 需在真实权重逐层检查。把两层 profile 复制到 24 层不等于验证 24 层误差累积。

### 父版本 R071：两层完整 CKKS 历史结果

| 指标 | 两层历史值，不能作为本 24 层结果 |
|---|---:|
| 真自举 | 154 |
| Runtime | 8114.139 秒，约 2.254 小时 |
| 峰值 RSS | 28.988235 GiB |
| 最大实部 / 虚部 / 尾部误差 | 0.0007017303 / 0.0006805129 / 0.0006578499 |
| 全部保留 CT / PT 静态系数容量 | 271.540039 / 13.805664 GiB |
| CT+PT 静态合计 | 285.345703 GiB |

这组结果是合成权重、两 Block、两个 token、CPU 32 线程条件下的验收。完整预训练模型和 GPU 仍需分别验证。

## 11. 后续补齐数据所需流程

1. 固定权重哈希、配置、输入样本和工具链 commit；先检查真实宽度的非线性输入区间。
2. 运行 trace 和官方编译，验证所有 Boot 为 native；保存 RuntimePlan、算子规格、编译命令及日志。
3. 使用对应目标的计划优化与 PlanVerifier；CPU 可用本包 DFS/cache64，GPU 需自己的设备放置方案。
4. 从实际计划导出：每阶段/每 level 的 CT/PT 数量及字节、Boot 位置/次数、Encode 次数、全保留系数载荷、释放模式存活峰值。
5. 完整 Mock 通过后，按资源准入执行 CKKS；记录实部、虚部、尾部误差、Runtime、CPU 用量和峰值内存。GPU 单独采集显存。

本报告的生成没有启动上述重计算。运行命令见 [README](README.md)，逐项静态数字及来源哈希见 [JSON](evidence/model-data-analysis.json)。
