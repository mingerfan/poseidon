# BERT-base encoder DSL / A800 benchmark

研究用途；**不验证精度，不宣称密码学安全性**。输入是已完成 embedding 的隐藏向量，输出是 encoder hidden states，不含 embedding、pooler、分类头、dropout。

**跨机器构建：**见 [BUILDING.md](BUILDING.md)。默认从当前源码构建，不依赖私有目录或历史二进制；支持显式预编译库与无 CUDA 的 CPU 检查配置。历史实测材料、窗口脚本和大归档保留在本地，不是新 clone 自带输入。

## 最新实测（20261115–16）

**最新QK＋AV输入packing组合已实际跑完12层（1warmup＋3完整测）**：
online **64.858351042s**中位，Runtime wall65.483720380s；每run实际
276Boot/324Linear/72QK/72AV，真实跨层SSA，不是单层外推。
相对旧v6真实76.417356609s，耗时为**84.87%**（观察减少15.13%，不同窗/bin/key/pool，
非clean因果加速）。永久shared-or-unresolved diagnostic，不进正式测量表。
最后98304complex槽finite但max参考误差28789.56/RMS6105.33，**HE精度仍未通过**。
GPU及8锁独立证明已释放；Cerium12未测，无其百分比。
详见[最新完整12层结果与证据](full_inference_packed/RESULTS.md)。

**新增单block资源余量诊断（永久非strict-clean）**：

| 同新runner/同窗配置 | online中位 | /Cerium6.059s×100% |
|---|---:|---:|
|control-v6|6.315600019s|104.24%|
|QKpack|5.545887620s|91.53%|
|QKpack＋AVpartial/A2|**5.391346155s**|**88.98%**|

各1warmup＋3完整测，observed变化−12.19%/−14.63%；所有probe/内部实际计数通过，
但PID归属未决且可能共享负载，**不是clean/同负载因果或精度保持加速**。组合max参考误差
61.54（control5.57），精度未通过。只作diagnostic，不写正式表；
详见[诊断结果](packing_validation/DIAGNOSTIC_RESULTS.md)。后续
[同物料9次forward](multi_plan_validation/GPU_RESULTS.md)中，10对A/A解码f64逐bit相同，
B/B、C/C也重复；本次固定物料A/B/C参考RMS为1.469/8.560/1.187，仍不通过精度，
不能据某一次材料排序宣称普适更准。B/C仅各1个diagnostic计时样本。

历史N2native10.748670572s cold的官方wrapper失败仍保留；独立CPU修复后用户优先12层，
N4未启动即停止。最新packed12已完成如上，不回溯改旧失败或混入下面旧v6数据。

**v6完整12层真实串接encoder已完成**（T128，1warmup＋3完整测）：online中位
**76.417356609s**，Runtime wall中位77.037918938s；实际276Boot/324Linear/72QK/72AV
每run，72组独立dense矩阵、原远端12层data逐bit绑定。不是单层外推；无embedding/分类头。
2/4层另有完整cold诊断12.575209913/25.256381114s，不作warm稳态。
最后3CT/98,304complex槽有限，max参考误差27322，**精度未通过**；main2158.602s另计。
此前真实v3 12层119.297476332s是另一窗口，v6耗时为其64.06%，仅描述对照。
Cerium12尚无实测，不给其百分比。17:42:17UTC退出，17:44:22和主线18:05独立确认释放。
[12层v6结果/可迁移metadata证据](full_inference_v6/GPU_RESULTS.md)；
[历史v3证据](../../test-results/bert-a800-20261115/full12-v3/README.md)。

**最新v6完整单层A/B**（T128，1warmup＋3测，同新runner/同窗）：control-B
8.236990537s → **6.303876115s（Cerium的104.04%）**，耗时降低23.47%。
v6组合polynomial＋prepared AV，每run实际23Boot/27Linear/6QK/6AV，内部计数和单份共享
MaskBundle均吻合；wall中位6.490315884s，另有main565.975s/整native进程569.425s。输出全部有限但
max参考误差37.94，**精度未通过**。独立报告：
[`v6-a800/20261115T144308Z/FINAL-REPORT.md`](../../test-results/bert-a800-20261115/v6-a800/20261115T144308Z/FINAL-REPORT.md)。
原12层data的2→4→12实测亦已完成如上，不能由本单层时间推算。

此前v4单层A/B（同runner同窗）：等价改写＋共享Rescale
为 **9.107563s（Cerium的150.31%）**；再加prepared QK为
**8.222787s（135.71%）**，后者耗时降低9.71%。每轮实际23Boot/27Linear，QK版另6pairs、
全部内部实际计数吻合。输出有限但整层精度未通过。AV独立GPU probe亦已通过（不是整层成绩）。
详见[冻结v4正式报告](../../test-results/bert-a800-20261115/v4-a800/20261115T140945Z/FINAL-REPORT.md)。

此前v3：128 tokens、完整1个encoder block、1 warmup＋3测，在线中位 **9.8395s**，相对上一版52.1698s为 **5.30×**；Cerium为6.059s，新版仍慢 **1.624×**。每轮真实 **29次native Boot**，没有删除非线性或用私钥refresh。HE参数/近似不同，不能称等参数消融；整层精度未通过。

独立完整FFN-up新版 **140.879ms**，相对上一版4.617s约 **32.77×**；不能用该子层倍数代替整层5.30×。原始未优化prefix128现已完成单次cold：在线 **3087.961s**、实际1920次Boot；与新版warm中位的数值比约313.83×，但不是同采样/等参数的严格加速比。

详细样本和复现：[本轮报告](../../test-results/bert-a800-20261115/README.md)、[新路径](ALIGNED.md)。
后续优先级、真实12层串接与完整计时口径见 [优化/完整实测计划](OPTIMIZATION_PLAN.md)。

[等价算术/共享Rescale](EQUIVALENT_REWRITE.md)显式opt-in，单层实测23Boot；
[prepared QK](PREPARED_QK.md)已完成上述整层GPU测量，进一步合并调用粒度，
新packing仍覆盖全部dot products。QK/AV乘前packing＋masked reciprocal已完成150项
Python CPU测试、AV39/QK36无设备门禁、coherent全新正常构建和独立review；
现已取得上述新模式GPU诊断，正式统计口径与历史测量分开。
默认CLI七主产物及旧v6十产物逐字节不变，非baseline manifestv2避免旧reader静默降级。
这些CPU计数不等于GPU收益。
用户已长期授权自主继续测试；各轮使用独立冻结snapshot，每轮仍有硬截止/清理。
单次非零GPU利用率不再独自阻断；根据显存余量/可见进程准入，共享样本不得冒充clean A/B。
[v5多项式重排](POLYNOMIAL_REWRITE.md)与[v6 prepared AV](PREPARED_AV.md)组合已取得
上述新GPU成绩；未改写此前冻结结果。输入packing首次strict窗口因PID未决保守停止，
CPU未找到namespace桥接；后来独立资源诊断窗完成七stage如上。旧失败window不恢复或
改标签，不将UNKNOWN称他人任务，也不把新模式混入76.417s旧c34完整12层结果。
跨数值环境相同seed不保证float32权重逐bit相同；固定A/B新增
[`--data-from-plan`](WEIGHT_BINDING.md)，真实单层已校验输入和全部参数SHA。

## 上一轮实测（20261114）

同一张 A800、128 token、完整1个 encoder block、1 warmup+3正式样本：我方在线中位 **52.170s**，Cerium原生runtime **6.059s**；我方约慢 **8.61×**。不同HE参数/非线性近似，精度不作为验收。

同参数 FFN-up Rescale消融：独立Rescale **9.274s** → 积和共享Rescale **4.617s（2.01×）**；Cerium对应子层约0.281s。完整runtime wall还需计入重复权重初始化，不能只用online替代请求总耗时。

详细样本、内存边界与复现：[本地实验报告](../../test-results/bert-a800-20261114/README.md)。该目录是忽略的本地产物，不是新clone自带输入。

## 后续紧凑 packing / 模数链对齐

新路径见 [ALIGNED.md](ALIGNED.md)：3个 hidden CT、6个完整 cyclic score CT、真实融合 dense Linear，以及两prime/约56-bit物理调度。仍保留本模型非线性；不是 Cerium 原图的等参数复现。完整层加速来自上述真实GPU样本，不由计划Boot数或小矩阵计时外推。

[Cerium审计](CERIUM_ALIGNMENT.md)已确认每层 **107条单CT流刷新 / 83次packed Boot core**；原始prefix基线首个25分钟cold窗口未完成；第二窗已取得上述完整单cold结果，不能把它当作warm稳态中位数。

## 模型与来源

设计参考 `origin/benchmark:DSL/DSL_benchmark/llm_dsl/Qwen25_24Layer_R121_DSL/` 的 Hecate 算术接口、prefix packing、BSGS 线性层和显式 RuntimePlan 路径，但模型不是改名的 Qwen：

- BERT-base：hidden=768、12 heads ×64、FFN=3072，可配置层数；各层独立 dense 合成权重。
- 双向多头 Attention、Softmax、O 投影+残差+**减均值的 post-LayerNorm**、GELU FFN+残差+post-LayerNorm。
- 无 RoPE、GQA、RMSNorm、SwiGLU、因果 mask。
- 非线性明确采用 exp/GELU degree4、reciprocal Newton3、rsqrt degree3；不是校准后的预训练模型精度。

`model.py`：每 token 一个 prefix 密文，适合小 token 调试。`packed_model.py`：每密文8个4096-slot row；128 token 共16个输入/输出密文，所有 head/row 并行。Softmax 分数居中使用 `<Q,mean(K)> = mean(<Q,K>)`，避免同时保留全部128组 score 密文；仍计算所有128个 key offsets。

## 手工 scale / Boot

`backend.py` 是独立的表达式 frontend 和**显式物理 CKKS lowering**，直接生成标准 RuntimePlan，不调用自动 scale/Boot planner。也可由 `generate.py --emit-earth --emit-ckks` 导出相同算术和物理 checkpoint，再用 `compile.py` 通过正常 DaCapo 后端调度/导出。后者不重新规划 Boot。真实尺寸文本 checkpoint 很大，默认不导出。

- 根据 **实际 Q primes** 反推 canonical scales：`S[l]=(S[l-1]+log2(q[l]))/2`。
- 残差匹配通过真实 MulCP(1)+Rescale 完成，**不改 scale metadata 假装对齐**。
- MulCC 后显式 relinearize；BSGS 的兼容积和共享出口 Rescale；共享 masked source 在 baby rotations 前只 Rescale 一次。
- Native ModRaise 的整数 multiplier 必须 ≤2^53；canonical scale30 在该入口不够。因此 Boot 前用真实 MulCP(1, plaintext scale=2^16) 把密文 scale 提升到约46，再执行 native Boot；不是改 metadata。
- 仅折叠相邻 binary masks，避免把单个 baby-0 的 mask 合入 dense diagonal 后破坏其余项的 level 一致性。
- `stage` 保留全部阶段刷新；`budget` 等 level 耗尽才刷新；`selective` 只保留长区间/共享分支前沿的阶段刷新，另保留预算安全检查。全部是**真实 native GPU Boot**，没有 decrypt/re-encrypt。

## 构建与运行

原生 Hecate 入口为 `trace_hecate.py`（真正的 `hc.func` / `hc.save`），配置好已有的 `HECATE`、`HECATE_BUILD`、`PYTHONPATH` 后可执行：

```bash
python3 bench/bert_dsl/trace_hecate.py --layout packed --tokens 128 --layers 1 \
  --boot-policy none --output-dir NEW_NATIVE_TRACE
```

Adapter 在客户端取负公开系数，绕开当前前端将 `x-Plain` 错降为非法 `earth.negate(Plain)` 的问题；不改编译器模块。`--describe` 不需要 Hecate。

CPU 检查不需要 Hecate、Torch 或 GPU：

```bash
python3 -m unittest discover -s bench/bert_dsl -p 'test_*.py' -v
```

统一构建入口为 `CMakeLists.txt` / `build.sh`，默认从源码构建 GPU/API/Runtime 和 prepared Linear/QK/AV；`build_a800.sh` 仅作为本地兼容包装，不再自动 SSH 部署。工具链和 GPU 架构由标准 CMake 参数选择，显式预编译库模式仍可用。详见 [构建说明](BUILDING.md)；构建不会启动 GPU 测试。

先在目标 GPU **实际 probe** native profile，再据结果生成计划，不要猜 Boot 出口：

```bash
bert_native_runner --probe-profile PROBE_FIXTURE profile.json
python3 bench/bert_dsl/generate.py --layout packed --tokens 128 --layers 1 \
  --strategy selective --resident --profile-report profile.json --output-dir NEW_PLAN

POSEIDON_RUNTIME_ROTATION_KEY_POLICY=binary CUDA_VISIBLE_DEVICES=0 \
POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB=78000 \
bert_native_runner NEW_PLAN/model.runtime-plan.json NEW_PLAN/operator-spec.json \
  NEW_PLAN/fixture.json report.json --bundle NEW_PLAN/bundle \
  --warmup 1 --iterations 3 --allocator managed \
  --resident-plain on --resident-backend gpu --resident-cache host \
  --resident-budget-bytes 59055800320 --bundle-resident-bytes 4294967296 \
  --native-boot-workspace-bytes 4294967296
```

本次已 probe 的研究配置为 N65536、Q `[31]+[30]*31`、P `[31]*16`、default scale30、degree58/DA2 native Boot；实际出口 **L16/scale30**、输入范围 L2..31。计划先用真实乘法/Rescale将 Boot 出口归一到 canonical L15。`profile.json` 必须来自自己的兼容 builder/环境；不是任意同名 profile 都等价。Probe 只构建资源，不执行密文 Boot；入口 scale 与工作区仍须实测。上述显式驻留预算仅用于已核算完整峰值的独占 A800，不能把缓存准入当成整图显存保证。

`generate.py --scope ffn-up --layout packed --tokens 128 --input-level 5` 只运行 `128×768 → 128×3072` 投影（含 bias），可与 Cerium 的同形状 `ffn_up_1gpu` 子层比较，**不是完整 FFN 或完整 BERT**。

checkpoint 后端验证：

```bash
python3 bench/bert_dsl/compile.py --hecate-opt HECATE_OPT --ckks-config CONFIG \
  --input-dir PLAN_WITH_MANUAL_CKKS --output-dir NEW_COMPILED
```

## 测量边界

runner 保留标准 readers/PlanVerifier，并检查实际 native Boot 契约；GPU API 不接收 host-refresh 所需私钥。输出解密只作有限值/误差摘要，不决定成功。`completed` 表示执行完成，不表示精度通过。

- 主要指标：预热后、最终同步的 `online_execution_ms`。
- 另外保存 setup、initialization、整个 runtime wall、真实 native Boot 次数、密钥策略、明文驻留情况。
- GPU resident 准备采用真实 GPU Encode，再真实 D2H 得到 Host RNS 明文，没有 placeholder。`--resident-cache host` 只缓存 Host，原计划每轮初始化仍真实上传 Device PT；避免持久 GPU PT 与 FFT 临时量交错造成经典池碎片。`both` 则缓存两端。
- `--allocator async` 仅为实验诊断选项；本服务器试验在首轮出现长时间 CPU spin，未得到有效样本，不能作为已验证替代路径。
- resident cache 与 plan initialization 均不应算成纯 GPU 算术加速；无法驻留时需明确在线包含的 Encode/H2D。每轮完整 `runtime.run()` 仍执行初始化和 raw bundle 校验；不能仅重复当前计划的 execution（其临时量会释放）。
- `schedule.json` 的 memory sweep 只是 PT/CT payload 下界，**不含**密钥、Boot 常量/工作区、内部 evaluator 临时量和 allocator reserve。缓存会一直保留 PT，即使图里已经 Release；累计 RMM peak 也不是单独在线峰值或 driver 物理峰值。
- Cerium 固定128 token；不能拿真正2-token的 Poseidon 图与 pad 到128的 Cerium 图计算加速比。即使形状相同，两套非线性近似、HE参数、packing、CUDA Graph设置不同，也只能报告各自原生算法的系统延迟，不能称为等参数实现对比。

Tiny prefix/packed 两个原生 frontend 路径均已导出 MLIR bytecode，并用真实 `hecate-opt` 解析；本地没有 Torch，测试仅为前端未使用的 Tensor 类型检查提供临时 type-marker shim，没有模拟任何 Hecate 算术或 GPU 计算。

本轮日志、计划、profile、编译和计时报告放在忽略目录 `test-results/bert-a800-20261114/`，不把大权重、认证、密钥或测量生成物提交到源码。
