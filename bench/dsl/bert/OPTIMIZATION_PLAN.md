# 后续优化与完整实测计划

状态更新（2026-11-16）：最新QK＋AV输入packing组合**真实12层已完成**，
online中位64.858351042s、Runtime wall65.483720380s，1warmup＋3完整测，每run实际
276Boot/324Linear/72QK/72AV。与旧v6真实76.417356609s同config/input/全部192参数hash，
耗时84.873848%（观察减少15.126152%，不同window/bin/key/pool，仅描述性比较）。
永久shared-or-unresolved diagnostic、非formal/clean/精度保持结论；最终complex max28789.56/
RMS6105.33，HE精度未通过。GPU/8锁独立释放，当前任务完全结束，无后续GPU排队。
同权重历史v3 119.297476332s也只作跨窗口描述；Cerium12未测、无其百分比。
详见[实际12层证据](full_inference_packed/RESULTS.md)。
用户现长期授权自主继续实现/测试，无需逐项审批；每轮仍须
资源检查、独立watchdog/硬截止及清理；单次非零利用率不再单独阻断。
v4已实测A9.107563s/B(QK)8.222787s、每run23Boot；standalone AV亦过GPU oracle。
v6 polynomial＋AV已完成同窗实测：8.236991s→6.303876s，耗时降低23.47%，
为Cerium单block104.04%（非等HE/近似）。真实2/4cold为12.5752/25.2564s；
原12层权重串接实测如上。下一版QK/AV输入packing已CPU150tests/coherentbuild/review通过，
其后独立资源诊断窗已完成：control6.315600s/QKpack5.545888s/combined5.391346s，
C耗时为Cerium88.98%，但永久shared-or-unresolved诊断、非clean因果；组合误差明显恶化。
后续独立同物料9forward已完成：AA10对decoded-f64bitwise0，B/B、C/C重复；A/B/C参考RMS
1.469/8.560/1.187，仍精度不通过，非通用排序。多层首次75min预算预检BLOCK、N2native
cold10.748671s但官方wrapper失败、N4 prelease停止均保留；用户直接12层任务后独立45min
窗口已完成如上，不从单层推算、不复用旧预算。source/CPU发布v1失败与v2修复各自冻结。
同物料multi-plan已完成；QP仅CPU候选，publicRNS缓存维持BLOCKED_UNSUPPORTED，
均不混入这次实际12层测量，未自动启动下一GPU任务。
实际启动、结束和性能以各轮原始报告为准，不由授权或计划计数推定。

## 现有事实与目标

同A800、T128/H768/12×64/F3072、完整1 block：

| 配置 | 在线耗时 | 相对Cerium | 采样 |
|---|---:|---:|---|
| 原始prefix128 | 3087.960906s | 50964.9% | 0warmup＋1cold |
| 上一版packed | 52.169807s | 861.0% | 1warmup＋3测中位 |
| compact v3 | 9.839523s | 162.4% | 1warmup＋3测中位 |
| v4 A algebra/shared | 9.107563s | 150.31% | 1warmup＋3测中位 |
| v4 B 再加QK | 8.222787s | 135.71% | 与A同窗同runner |
| v6组合poly/AV | 6.303876s | 104.04% | 新同窗control-B8.236991s |
| Cerium | 6.059s | 100% | 1warmup＋3测中位 |

cold/warm及HE参数不同，不把原始/新版数值比313.83×称为严格稳态消融。
v3与Cerium还差约3.781s。v3六个Linear同步callback合计约0.978s（包含ready等待），
因此仅继续打磨FFN不能解释/消除全部差距。v3真实29次单流Boot，Cerium83 core/107流，
次数已更少但系统更慢，应分析每次Boot与其余计算/调度，而不是继续盲减次数。

## 一、先做低风险、可归因的改动

### A. 等价算术与scale调度

`cerium_packing.py:reciprocal`当前后两轮含`(d*r)*plain(-1)+2`。
全slot的−1乘法应lower为ciphertext Negate，不消费Rescale预算；**不能把带零尾的
负mask当成全slot−1**。

第一轮Newton可从`((-d/T)+2)/T`等价写成`2/T-d/(T*T)`，减少一次CP产品。
T128为2的幂，公开系数精确可表示；仍保留三轮Newton，不降低多项式阶数。
默认LN center=1时，全slot乘1也可消除；不能删除prefix/head/row清尾mask。

先跑独立plain oracle、physical scale守恒、PlanVerifier和Boot计数；重新调合法切点。
只报告实际GPU结果，不能把预计少几个Boot直接换算为毫秒。

### B. 普通rotation direct-key A/B

冻结v3 plan/fixture/HE参数与近似，仅将普通Runtime rotation policy从binary改为direct。
prepared Linear目前已经direct，不把其收益重复计算。记录实际key basis、keygen时间、
Host RSS、GPU key bytes及online差异；direct会增加setup/常驻内存，不是免费优化。
最新冻结v6普通244个rotations已CPU逐项证实全为signed power-of-two（±1..±128），
所以不再花GPU做该配置无分解收益的direct切换；prepared内部仍是另一范围。
证据`v6-a800/ordinary-rotation-static-audit.json`，不是GPU计时。

### C. 普通plaintext GPU常驻

当前prepared线性权重已GPU常驻，普通mask/系数仍Host RNS cache、每轮真实H2D。
当前tracked peak20.4GiB，值得单独测试`--resident-cache both`；这与旧宽布局几十GiB
常驻PT导致碎片的失败条件不同，但仍需真实验证，不能直接认定安全。
目标是减少Runtime wall11.618s与online9.840s之间的初始化/传输开销；
不能把wall改善错误记成online算术加速。仍需检查pool碎片、峰值与预算。

新增诊断基础设施（尚无其GPU结果）：`multi_plan/`一次复用真实key/input/完整Boot profile，
37项CPU mock与无device gate通过；`native_runner --cuda-profiler-iteration N`可只圈定一轮
Runtime（含init），setup不进入capture。正式summary拒绝instrumented样本。见[PROFILING.md](PROFILING.md)。

## 二、用profile决定中等规模工程

独立诊断运行加NVTX/层边界，使用Nsight Systems关联CPU提交、等待、H2D、GPU kernel；
必要时仅对最热kernel用Nsight Compute。只capture预热后的online窗口，避免把数分钟
setup录成巨型trace；先检查CUPTI/counter权限，受限时用正确stream上的CUDA events及
同步阶段wall诊断，不能拿CPU提交代替GPU时间。仪器化样本不作为正式性能结果。

分组至少包括QKV、QK、Softmax、AV、O、LN1、FFN-up、GELU、down、LN2、Boot。
Boot进一步拆normalization/encapsulation-ModRaise/C2S/EvalMod/S2C/输出处理。
CPU compute/Boot submission计时含等待，不能当作isolated GPU时间或据此断言CPU瓶颈。

### D. QK/AV prepared融合与rotation批量化

当前Linear仅27个调用，但Attention/非线性仍有大量普通Runtime原语，整个图约2.65万
compute/opaque调用。复用已验证的cyclic数学，优先减少QK/AV的逐算子分配、event、
校验、dispatch，以及重复ModUp/NTT；共享同源rotation的hoisting。

融合不能绕过真实Relin/Rescale，必须保持全部128 keys/12 heads和原数学近似。
新增macro须同时报告内部真实工作量，不能把macro数量当scalar operation数量。
v6已真实完成该步骤，普通compute/opaque仅1337，仍另有QK/AV/Linear内部工作。

下一版双head-gap QK packing：post-CC两产品共享headreduce已通过CPU，597/575
scalar rotations/adds，对原917/895；更强pre-CC packing把相邻baby放入空gap，旨在真正用
64CC覆盖原128keys，不降低数学工作。pre-CC和AVpartial+A2均已CPU全链路150tests通过，
AV通过masked reciprocal和consumer guard保持完整模型理想等价；现生产pair oracle和
真实64CC计数已GPU诊断通过，完整block诊断如上，不据静态计数外推时间/12层。
公开IR mask的ideal-zero-gap证明与CKKS噪声零值必须区分，原始输入不无证假定layout。

线性模块随后可按矩阵形状调BSGS baby/giant，而不是所有矩阵固定baby16。尤其FFN-up
与down输入/输出tile比例不同；需先泛化当前native helper固定256/16的校验和key-step
集合，并实测新增key/缓存空间的代价。该项只覆盖当前约1s的Linear callback，优先级
低于解释其余约8.9s；不预先宣称可相加的毫秒收益。

### E. 按level选P与profile联动

当前普通算子固定P11，低Q时可能明显大于必要P；Cerium按操作/level动态选P。
现有GPU底层支持level key/parameter views，但generic BERT profile并没有一个
可直接开启所有动态P的开关；需要检查/扩展public参数采用与eval-key绑定，不能简单
截断现有key的P limbs。

先微测真实旋转/Relin/Boot，再做完整block A/B。Q/P、dnum、噪声和安全预算都要报告。
Q49/P13等现有whole-context候选也必须实测：单Boot更便宜可能导致更多budget Boot，
不能只按Q少或单Boot快判断整层收益。

### F. 冷启动QP准备（独立路径，不混online）

`qp_startup/`仅CPU候选已完成：完整uploader object覆盖链接、host精确CRT264cases、
17项no-device gate，未替换主库。`qp_startup/target-prep/`另有A800GCC11/CUDA11.8
私有构建/ABI审计入口及23项mock，真实target flags/header/compiler/loader pins和原object
反汇编仍需目标worker提供，模板拒绝猜测。所有实际targetABI/GPU/Boot/时间均未测。
以后先全BERT公开常量fullCRT/ownerHash，再同binary freshprocess reference/optimized；
不把peer R20速度外推BERT，也不把setup加速包装成online推理收益。

### G. 公共Linear RNS离线缓存（小规模CPU原型已完成）

真实12层preparedLinear setup1427.935s，是aggregate；源码每term有N512编码、fullN
materialize/全Q exact压缩重建、独立H2D/D2H及同步，尚未量出各项占比。优先独立CPU
`public_rns_cache/`原型已完成：仅N1024/Q2-3/P2/单256²，3CTests、411codec checks及
3个真实小矩阵逐word/metadata往返是旧v1历史结果；其loaded-DSO路径在原子升级后可
误绑新inode的P2未获修复验收。严格actual mapping/heldFD候选在本机Btrfs出现fstat与maps
dev差异且map_files EPERM，保持failclosed并标UNSUPPORTED（当前CTest2/4，仅codec/schema
通过，producer正例被拒）。不使用inode-only/无subvolume证明bridge；该支线暂不promote，
不支持生产全模型或GPUloader。
首次生成仍保留全部逐residue验证/native审计，cache严格绑定
真实原W/bias/完整spec/QP与scale位型/encoder与compactABI/source库，后续只在外部
manifest pin和payload完整hash验证后流式复用公开PTwords。每次GPU上传仍须当前device
readback，不能从磁盘继承exact=true。不存input/keys/secret、不改online、不在本机生成
约21.3GB full12 cache，offline生成成本单列；当前无load性能或setup加速结果。

### H. CUDA Graph

若profile显示launch/CPU提交占比高，优先capture固定的Boot或QK/AV子图。
先将prepare与launch分离、预分配scratch并稳定地址；当前helper.run有stream同步和
分配，不能简单在外面套BeginCapture。正式比较用关闭profiler后的普通执行与graph执行。
复杂vec2 Boot配对/原样Cerium非线性迁移属于另一路：涉及complex PT、conjugation、
pack/split与profile支持，优先级低于当前低风险瓶颈验证。

## 三、完整实测：默认定义为12层encoder

默认目标：batch1、T128、hidden768、heads12×64、FFN3072、12个串接block，
每层独立dense weights。仍从embedding后的hidden输入，不含embedding/pooler/classifier；
如果目标是任务端到端，须把这些步骤及客户端/网络另行补齐，不能把encoder叫整条任务。

### 1. 真串接与资源预检

- 新候选先1→2→4→12层，不减tokens/维度、不省非线性/真实Boot。冻结v3已经真实
  完成12层1warmup＋3测；新v4/v5/AV不能借用该成绩冒充自身已验证。
- CPU12层预检应见3external/3final、72个不同矩阵/324个prepared calls，以及真实
  跨层SSA、独立weight hashes和各层Q13边界；Boot计数从计划重新审计，不把29×12
  当作已执行证据。
- 每层输入必须来自上一层GPU输出；不重新喂初始输入，不D2H后decrypt/re-encrypt。
- 保持每层Q13/canonical scale边界，审计全部SSA和独立权重绑定。
- 12层线性PT驻留粗算约1.776GB×12=21.317GB，仅是model PT估计；
  keys、参数表、Boot/linear workspace、普通PT、活跃ciphertext和碎片必须另外计入。
- 不把单层20.4GiB峰值乘12，也不把9.84s乘12当作实测。
- CPU plan/liveness＋实际setup admission后才运行；若需stream权重，将H2D纳入在线/请求wall。

### 2. 每层与整网计时

每层边界等待真实输出ready，记录stage/layer延迟、实际Boot次数、峰值和误差；
先提供1次诊断运行。现有runner只有总Boot/online和各Linear callback统计，须新增
layer→3个边界SSA/ordinal映射、每层实际Boot计数和同步边界timer。worker模式不支持
AllValuesAfterRun；逐层数值诊断须单独快照/下载，不把36个中间CT都作为正式final来
改变性能测试的liveness。正式性能运行关闭重profiler，1完整warmup＋至少5个完整样本，
报告原始数据、中位、范围；样本太少不宣称可靠p95。

分别记录：
1. 系统首启：读验计划、context、keys、profile、编码/上传、首次执行与解密；
2. 服务端online：完整12层，入口/出口真实同步；
3. Runtime/request wall：每轮setup/init、必需H2D、执行、D2H；
4. 客户端端到端：encode/encrypt、传输、服务端、回传、decrypt；说明网络是否真实测量。

不能从4轮进程总时间均摊出“首启”；缓存/上传计时边界必须两系统分别明确。

### 3. Cerium对照

`cerium_full12/`已完成CPU-only headless1/2/4/12单SSA wrapper、逐层真实冻结data绑定
exporter和future runner准备（25小型stdlib tests）；未export大权重、未native构建、未测GPU。
独立review要求cgroup有效限额和result provenance修复；metadata通过，根接口最终overlay
`review-fixes/root-interface-fix`取得conditional pass（94mock/20syntax），缺根接口例外限声明的
Linux6.8ABI；一手ABI和目标适用性尚未核实。普通root接口present走原严格检查。
**旧冻结prep不直接实跑，也不将conditional pass当target-ready**，实际export前须适用门检。
现有只读诊断记录target为Ubuntu5.15.0-88；历史packing host admission记录根memory.max/current
均present（limit266287972352B），所以缺根接口6.8例外不一定是该target阻塞；这不是新候选
实际fresh适用门检通过，不能替代将来检查。
源码内存筛查不是38.9GiB×12：固定keys/profile/workspace复用，
每层remap PT约1.279329GiB，12层条件静态小计46.40–46.78GiB；实际峰值/新main liveness
仍未知，无静态PT必超80GiB结论，也不保证fit。见该目录MEMORY.md。

源码审计已确认：reference Cerium原本就注册并串接12 blocks，归档实测副本才被裁成1。
共享子函数可以通过block0..11 remapable PT绑定不同weights，不需要复制12份helper。
必须恢复12层main与runner guard，生成/加载12份独立packed params，审计编译DAG的
block间H依赖；只加密最初3hidden CT一次，之后保持GPU hidden状态。
不是把现有1-block program独立跑12次，更不是将6.059s乘12。

**重要陷阱：**`reference/cerium/benchmarks/bert_base/bert_cerium.py:2558`在
`num_blocks==12`时自动加入pool/classifier、输出prediction。encoder-only对比必须
显式解耦head开关，并同步runner的数据加载/输出规则，仍输出最后3hidden CT。
`params_block12.pkl`是分类头，不是第12个encoder；encoder应为block0..11。
模型加载器还有`model.blocks`与packer读取`model.layers`的不一致，可沿已验证的
shared dense→原生packer路径逐层导出，不能只改一个block_num就假定全部可用。
两系统都固定同输入/权重/mask与相同token/layer数；HE/近似/Graph/P策略差异另列。
若12层Cerium暂不支持，先报告双方实际2/4层重叠配置，不能拿12层对单层。

### 4. 精度与结果标签

按用户当前要求，性能完整链可继续只诊断精度，但明确这是“全算子链性能实测”。
当前v3整层max参考误差6.41，不能宣称可用BERT推理。至少保留各层finite/实虚部/误差，
另以单Boot identity、算子plain-approx reference和真实模型reference逐级定位。
不改近似的性能消融与改近似/精度预算的实验分开。

## 实施顺序

冻结v3 → 计时/归因 → 等价算术＋direct-key＋GPU常驻单项A/B → 选出整层最优 →
1/2/4/12层真串接 → 同层数Cerium → 正式无profiler采样/完整wall/资源报告。

原始未优化cold已花约59min进程时间，不为“补一个中位数”盲目再跑数小时；
若需要严格稳态原始基线，应另给独占预算，明确补warmup和重复样本。
