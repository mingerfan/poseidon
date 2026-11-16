# 显式prepared cyclic QK（单pair及单block GPU已验证）

实测更新：独立pair中位0.296522763s，全槽max误差1.57e-10；完整1block v4-QK
中位8.222786937s（Cerium135.71%），每run真实23Boot/27Linear/6QK，1warmup＋3测。
同窗无QK的algebra/shared版本9.107562999s；QK版本耗时降低9.7147%，整层精度未通过。
报告：`test-results/bert-a800-20261115/v4-a800/20261115T140945Z/FINAL-REPORT.md`。
这些测量不包含后续v5 polynomial或v6 AV集成；不从pair时间乘6推算整层。

`generate_aligned.py --prepared-qk` 在原DSL的cyclic QK位置发出标准V7
`bert.compact_qk` ApiCall；默认关闭，只允许cyclic encoder scope。支持与
`--algebra-rewrite --shared-rotation-rescale`组合。

这是**真实C++ prepared算术批处理，不是单个融合CUDA kernel**。接口、单独probe和
CPU测试位于[compact_attention/](compact_attention/README.md)，native adapter为
`prepared_qk_adapter.h`。`aligned_native/CMakeLists.txt`直接链接已有成熟库，不改核心GPU/API库。

## 新的opt-in输入packing（CPU＋GPU资源诊断已完成）

后续独立shared-or-unresolved窗口已实测：packedQKpair中位0.167475460s，baseline
0.297526603s；生产32768slot oracle误差分别5.99e-10/4.88e-10，真实64/128CC与其余计数吻合。
完整单block control6.315600019s→QKpack5.545887620s（Cerium91.53%）。
这是**非strict-clean诊断**，不是同负载因果加速/HE精度验收；详见
[GPU诊断与限制](packing_validation/DIAGNOSTIC_RESULTS.md)。

`--qk-pack-inputs`依赖`--prepared-qk`，并限cyclic/selective/q52-p11/encoder/L12。
公共IR必须证明Q/K的**理想message gap0**；raw密文输入布局未知直接拒绝，不decrypt检查，
CKKS gap噪声不被称为bitwise0。两次rewrite/registry、SHA、53个真实direct keys与
four-prime预算都接通。默认不开，不改变既有v1/七主产物或旧v6十产物的bytes。

| engine模式 | CC/relin | CP | R2调用 | direct rot/add |
|---|---:|---:|---:|---:|
| 默认baseline（上述GPU测量） |128/128|128|129|917/895|
| post-CC paired2（仅C++显式模式） |128/128|128|129|597/575|
| PRE-CC pack_inputs（新CLI） |64/64|128|65|531/513|

新模式把Q复制到64-lane gap、K相邻baby放入gap，然后一CC并行计算两baby；仍全部
128keys×双head×64features。两个hoist真正使用packed Q/K，串行深度仍L8→6→4，
不是因CC数减半就改为两prime。baseline与PRE-CC列后续已有独立GPU实际counter诊断核对；post-CC仍仅CPU验证。
完整T128 Graph独立dense oracle误差2.78e-15，N1024真实CPU加密pair误差4.52e-14；
后续生产N65536 GPU pair数值/计数诊断已通过，完整HE模型精度仍未通过。

### 版本与纯CPU门禁

仅非baseline manifest升级v2，新reader严格检查v1/v2、实际bool、互斥mode和能力必须
对应真实packed call；v1出现true-mode拒绝。旧c34 adapter完整构造器已用CPU VersionGate
确认正确SHA的v2会拒绝，不会忽略新字段静默执行baseline；不声称旧完整main在CUDA前拒绝。
新native将fixture/context/QK/key-plan检查放在async allocator之前；profiler整数from_chars
要求完整消费。独立review发现的3P2均修复并复核无剩余该范围P2，133项根测试与36项实际
native no-device gates通过。证据`qk-packed-inputs-integration/review-fixes/`。
早期手工TU/旧bencharchives的CPU gate不是coherent fresh build；其后AV集成已完成
正常coherent构建、GCC11/11.8目标重编及上述GPU资源诊断。早期packed-v1仍只留历史，
不能直接续用，所有新非baseline请求用v2能力门。

## 默认baseline保留的算术

每个T128双head pair仍执行128个真实CC products、128次relin、128个公开mask CP、
129次真实Rescale2、917个direct rotations和895次add；6pairs共768个CC products。
两个共享hoist批次分别是query giants和key babies。headreduce源每步变化，不能假装共享
同一个hoist。所有128个key、12个head均保留，输出仍为
`score[t,head,d]=dot(Q[t,head],K[(t+d)%T,head])/sqrt64`。

engine不接收私钥；adapter真实同步并保有两个GPU输入的Value owners、run返回前同步，
实际输出交标准Runtime验证。物理L8→L6→L4 scale从真实prime递推，mask encoding scale
按实际drop反推，**不覆盖密文metadata以伪装Rescale**。

## 本地CPU结果

- 98项Python测试通过：包含带脏gap的双head/非零giant独立dot oracle、原BSGS对照、
  两层tiny模型、algebra pass保留opaque节点、4-prime预算guard和边界SSA。
- 子模块CPU CTest 3/3通过，另有scalar-only和ASan/UBSan；真实CPU加密metadata测试
  decode误差约3.23e-14、scale误差0。详见
  `test-results/bert-a800-20261115/compact-attention-build/validation.json`。
- 本机GCC15/nvcc13.4完整native runner成功编译链接；本机成熟GPU API带NCCL/MPI，
  使用显式`BERT_PREBUILT_WITH_NCCL_MPI=ON`补齐只读库链接依赖。未重建核心库。
  后续已完成A800 GCC11/nvcc11.8兼容构建、真实GPU probe和上述整层实测；原CPU构建
  日志本身仍不构成GPU证据，正式证据属于各自独立snapshot。
- 默认CLI七主产物与冻结v3逐字节相同，含plan/spec/fixture/weights和完整bundle。

## 完整尺寸CPU计划

| T128，1block | algebra＋shared | 再加prepared QK |
|---|---:|---:|
| 普通/opaque compute调用 | 26285 | 12341 |
| native Boot计划数 | 23 | 23 |
| 普通Rotate | 7528 | 2026 |
| 普通MulCC/Relin | 1624/1624 | 856/856 |
| 普通MulCP | 4055 | 3287 |
| 普通Rescale调用 | 3393 | 2619 |
| prepared Linear/QK调用 | 27/0 | 27/6 |

**少掉的普通算术全部在6个QK callbacks里执行**：加入其内部768 MulCC/Relin、768 CP、
774 Rescale2、5502 Rotate和5370 Add后，实际基本算术数与左列相同，不把opaque压缩当
做消除计算。direct keys/hoisting、workspace复用与较少Runtime调度才是待测收益来源。

单层产物`test-results/bert-a800-20261115/algebra-shared-qk-v4-cpu/`，标准PlanVerifier通过，
28,339,200B原dense权重逐字节不变。

另已实际生成并验证**12层CPU计划**：72个独立dense矩阵、324个Linear＋72个QK调用，
各层直接消费前层3个Q13输出，重算每层23、全模型276次**计划**Boot；不等于GPU执行数。
本地生成12.304s、2.085GiB RSS，标准PlanVerifier通过；产物
`algebra-shared-qk-v4-full12-cpu/`。没有从这些计数外推GPU时间。
该未绑定12层CPU产物与远端冻结权重存在跨环境1ULP采样差异；仅作结构验证，
不能当作逐bit同权重A/B。正式新比较应使用[固定数据绑定](WEIGHT_BINDING.md)。

## 逐层可观察性

`layer_observability.py`附加导出`layer-boundaries.json`，记录真实物理value IDs、producer
ordinals、层间Q输入SSA、边界level/scale和区间计划Boot数。不加final outputs、不保留中间
CT、不改变release schedule；**不是逐层GPU timer，也不提供逐层HE误差**。

用户已长期授权继续有界GPU实验。v6 AV集成单层已测6.303876s（Cerium104.04%）；
原12层资产绑定的2→4→12另在独立c34窗口执行，以上新packing未混入，不需再次审批。
冻结v3完整12层119.297476s与这些候选严格分离；v4-QK仍没有12层GPU成绩。
