# Qwen2.5 0.5B：R071 对应 24 层模型数据分析

更新日期：2026-10-20（已补真实尺寸编译统计）。分析对象：本目录 `Qwen25_24Layer_R071_DSL`；默认路径为 **24 个 Decoder Block、两个 token 顺序 decode**。本报告区分源码统计、条件估算、父版本实测和本版本实测，供编译/GPU 测试接入使用。

## 1. 当前结论与验证状态

24 层结构、独立权重/KV、完整词表输出和编译入口已经准备；沿用 R071 的算子、非线性近似、native Boot 合约、显式释放和 CPU 明文缓存优化。

| 项目 | 当前证据 |
|---|---|
| 24 层、真实宽度配置及权重 schema | 已检查 |
| 算术内核与已发布 R071 的一致性 | 19 个继承文件哈希检查通过 |
| 小尺寸 24 层四条算术路径 | 通过，最大绝对误差约 8.972×10⁻⁸ |
| 真实尺寸 24 层明文计算 | 独立 dense 参考和完整 Runtime Mock 已通过 |
| 真实尺寸 24 层 Hecate/Dacapo 编译 | 已通过；原始和 DFS/cache64 计划均经原生 PlanVerifier |
| 24 层实际 level、明文编码 level、自举次数 | 本次已统计，见第 7～8 节及明细文件 |
| 24 层完整 CKKS 精度、运行时间、CPU 峰值 | 未实测 |
| GPU 显存、时间、精度 | 未实测 |
| 预训练权重的逐层非线性校准 | 未完成；现有配置用于合成数据测试 |

因此，本报告提供源码数据和**本版本真实编译计划的统计**，不是“24 层完整推理已通过”的验收报告。不能把两层的 154 次自举、28.99 GiB RSS 或耗时乘以 12 当作本版本结果。

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

- 当前 fixture 的 float32 数组载荷：`630167424×4 = 2520669696` 字节，约 **2.52 GB / 2.348 GiB**；本次已生成同规格合成 fixture，NPY 头和文件系统开销另计。
- 共享一份 embedding/LM head 时，同等维度 float16 载荷为 `494032768×2 = 988065536` 字节，约 0.988 GB。这是指定存储假设的计算，不是已下载权重文件的实测大小。
- 服务端 DSL 的输入是 embedding，查表在客户端进行；schema 中的 embedding 表不意味着整表都要参与服务端 CKKS 编码。
- 源码首次交付只有 59 文件、322523 字节；原 ZIP 为 95158 字节。本次增加分析文档和 JSON 后目录会变大，旧 ZIP 不包含这份新报告。

这些原始数组容量与 CKKS 编码容量、CPU RSS、GPU 显存是四种不同指标。本次编译导出的原始常量 bundle 共 16815 个文件、4.093388 GiB，也不等于编码后的明文系数容量。

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

## 7. 24 层实际 level、自举及操作分布

**本节是这个 24 层版本的实际编译统计，不是两层乘以 12。** 使用冻结 CPU 合约，应用 L1～L8，native Boot L1→L5、scale 2⁴³。编译器完成自动自举插入；图中包含显式 Release，不复用输入缓冲。

本次实际 Value 只出现 L1～L5；合约允许的 L6～L8 没有出现在这份最终计划中。完整算子上下文的上界不能直接当成每条明文或密文的编码 level。

- 最终计划自举 **2370 次**，原始计划同为 2370 次；DFS/cache 调度不改变自举位置所对应的值和算术。
- 输入 level：`[5, 5]`。
- 29 个输出 level：`[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`，按五组 logits、24 层 KV 排列。
- 优化后 Encode 次数：**28937**；累计编码系数载荷 **67.641602 GiB**，含缓存淘汰后的重编码。

| 操作 | 原始计划次数 | 优化后次数 |
| --- | --- | --- |
| add_cc | 405218 | 405218 |
| add_cp | 1146 | 1146 |
| boot | 2370 | 2370 |
| encode | 19699 | 28937 |
| mod_switch | 2879 | 2879 |
| mul_cc | 1722 | 1722 |
| mul_cp | 73365 | 73365 |
| negate | 48 | 48 |
| release | 982683 | 991921 |
| relinearize | 1722 | 1722 |
| rescale | 75087 | 75087 |
| rotate | 399454 | 399454 |

### 7.1 优化后明文、密文的 level 分布

| 类型 / level / 分量 | 数量 | 系数 GiB |
| --- | --- | --- |
| ciphertext:L1:c2 | 70,486 | 137.667969 |
| ciphertext:L2:c2 | 265,580 | 778.066406 |
| ciphertext:L2:c3 | 746 | 3.278320 |
| ciphertext:L3:c2 | 523,738 | 2,045.851562 |
| ciphertext:L3:c3 | 579 | 3.392578 |
| ciphertext:L4:c2 | 95,796 | 467.753906 |
| ciphertext:L4:c3 | 329 | 2.409668 |
| ciphertext:L5:c2 | 5,691 | 33.345703 |
| ciphertext:L5:c3 | 68 | 0.597656 |
| plaintext:L1:c1 | 296 | 0.289062 |
| plaintext:L2:c1 | 959 | 1.404785 |
| plaintext:L3:c1 | 6,228 | 12.164062 |
| plaintext:L4:c1 | 18,575 | 45.349121 |
| plaintext:L5:c1 | 2,879 | 8.434570 |

同一个网络层内部会出现多个 level，不能给第 1～24 层各指定一个固定 level。所有操作的输入输出 level/scale 转换、逐阶段统计、29 条输出 level 和完整自举清单见 [COMPILED_LEVEL_DETAILS.md](COMPILED_LEVEL_DETAILS.md)，机器可读数据见 [compiled-analysis.json](evidence/compiled-analysis.json)。

### 7.2 来源归属的可靠性

诊断位置映射核验：`{"passed": true, "value_count": 982710, "labelled_values": 982710, "unlabelled_values": 0, "multiple_labels": 61418, "note": "Fused/shared locations are reported jointly; never duplicated across totals", "reencode_ids_resolved_from_identical_payload": 9238}`。追踪时只增加位置标签，不改变算术；编译后的 ValueId 必须与 RuntimePlan 完整匹配才接纳来源映射。编译器合并的共享值不会重复归入多个层；无可靠标签的值保留为 unattributed。

## 8. 明文、密文总量与内存

### 8.1 全部 Value 都保留时的系数容量

| 类型 | 原始计划数量 | 原始 GiB | DFS/cache64 数量 | DFS/cache64 GiB |
| --- | --- | --- | --- | --- |
| 编码明文 PT | 19,699 | 46.125488 | 28,937 | 67.641602 |
| 密文 CT | 963,013 | 3,472.363770 | 963,013 | 3,472.363770 |
| 合计 | 982,712 | 3,518.489258 | 991,950 | 3,540.005371 |

统计包含所有定义的 Value、不同 level 的编码、重编码产生的新 ValueId 和实际分量数。它描述忽略 Release、保留当前计划所有结果时的**图系数载荷**，不包含自举内部历史临时对象，也不是测得的进程 RSS／GPU 显存。它不等于只存储唯一权重一次的容量。

缓存淘汰后的重编码增加了累计 PT Value 数量及全保留载荷，同时减少了同时驻留的明文。密文总量累计计入每个旋转、加法、乘法、重缩放和自举输出；原始权重文件的大小不能直接表示这种累计容量。

公式：`65536 × 8 × (level+1) × components` 字节。明文通常一个分量，密文通常两个分量，乘法中间值按计划实际分量统计；槽位空闲不缩小整个多项式。

### 8.2 按 Release 执行的图数据峰值

| 指标 | 原始计划 MiB | DFS/cache64 MiB |
| --- | --- | --- |
| 同时存活 PT+CT 峰值 | 47,980.500 | 581.000 |
| PT 单独最大值 | 47,232.500 | 64.000 |
| CT 单独最大值 | 2,366.000 | 581.000 |

PT+CT 总峰值时，PT 为 0.000 MiB，CT 为 581.000 MiB。上表两个“单独最大值”可能出现在不同时间，不能直接相加当成总峰值。输入按调用者整个执行期间持有计入。

64 MiB 限制的是 Encode 根缓存；派生明文、密文、密钥、自举工作区、原始权重、执行计划和内存池均不受这个数值直接限制。当前模型沿用 R071 的显式 Release、DFS、cache64 和 packed-row/LM head 打包，未加入后续 082 的主动池回收或 NAF 后端。

### 8.3 编译与 Mock 的资源实测（不是 CKKS 推理）

| 阶段 | 秒 | 该子进程峰值 RSS GiB |
| --- | --- | --- |
| verify-delivery | 0.115 | 0.016148 |
| full24-mock | 89.792 | 0.634892 |
| optimize-schedule-cache | 139.717 | 8.671341 |
| build-mock | 13.281 | 0.468475 |
| verify-eager | 11.729 | 0.443939 |
| verify-optimized | 13.582 | 0.448860 |
| analyze-values-levels | 108.216 | 5.318684 |
| dump-locations | 11.079 | 9.004692 |
| compile-24layer | 96.887 | 9.054184 |
| trace-24layer | 193.702 | 17.881062 |

取得互斥锁后本次分析用时 1396.252 秒；此前排队 4834.817 秒。不同子进程峰值不能相加。

其中此前统计与修复阶段已用 715.735 秒；上表列出最终这次执行的各阶段，前序阶段明细保存在 JSON 的 previous_timings。源码位置补全前后，原始和优化后两份 RuntimePlan 均逐字节相同，常量 payload 也相同；只是增加可读来源标签，没有改变模型计算或自举安排。

**24 层真实 CKKS 推理峰值 RSS、保留全部中间结果的真实 RSS、GPU 显存、密钥和自举工作区总占用仍未实测。** 没有启动 24 层 CKKS 长测，因此本报告不提供这些设备实测值，也不把图系数容量冒充这些指标。

## 9. 完整 24 层 Mock 检查与证据

29 个输出全部通过，最大绝对误差 1.33912686051e-07，尾部误差 0；计划的 2370 个 Boot 位置均被 Mock 执行，其最大输入绝对值为 3.608604418。Mock 用明文执行同一 RuntimePlan，检查算术和布局；它没有 CKKS 噪声，不能当作真实自举精度通过。

编译器 SHA256：`6af174b19ef6594ac759f5d4bdfaf0d2366f4e6277499c470c531be58fefdbdc`。

原始计划 SHA256：`c5789e490cd9cb2dba44875c0137085595ba1cd4d0ee772263b2673b18581c7b`。

优化后计划 SHA256：`4f69c6ee95dc3cae5653800e70e9e857b047cdf6370e968f1e80ed62ac2a62b7`。

合成权重 manifest SHA256：`18e5a55ecb264ac930aa21e41dddb95ebeb6f10bec2580105b6fa4eb439efe0b`。

## 10. 验证边界与后续使用

本次统计仅覆盖默认 **24 层、2 token、decode-pair、CPU 合约**。prefill、外部缓存 decode、其他 token 长度、预训练权重或 GPU 编译规格可能生成不同计划，不能直接套用本表。真实权重还需逐层非线性校准，CPU/GPU 精度分别验收。

源码规模、接口与 slot 利用率见前文；本版本所有编译统计以 [compiled-analysis.json](evidence/compiled-analysis.json) 为准。父版本两层历史数据仍保留在 evidence/parent_071，仅用于来源比较，不参与本次 24 层统计。

