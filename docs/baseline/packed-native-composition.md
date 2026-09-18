# Native 构造语法与多维 packed 接口组合

## 解决的问题

此前 native 函数/对象数组/循环契约仅允许 period4 常量、固定旋转步长和最多4个返回密文。
schema5 支持更大维度，但只允许直线型表达式。二者各自验证过，不意味着组合已经支持。

本轮新增独立任务 `hecate-periodic-packed-native-synthesis-v1`、检查契约
`hecate-periodic-packed-native-v1`，仍使用现有 `periodic-packed-v1` 执行 ABI。
明确选择方法：schema5 模型加 `--native-array-mutation`。旧 schema 与所有旧契约不变。
单独选旧 native-functions/arrays/loops 等中间版本仍拒绝与 schema5 静默混合。

新契约复用 native-v7 已有的类型检查、受限 AST 解释、真实 Hecate 装饰函数及 compiler inline：
helper c/p 类型、嵌套调用、对象数组、starred 参数、公开 range 循环、
标量增强赋值以及具有别名/视图语义的数组原位操作。
不是 eval/exec 候选 Python，不允许任意代码、I/O、网络或密文相关控制流。

## 布局和约束

- 输入为一个静态 rank1..4、1..256 元素张量。C-order 展平、补零到 P，再重复到16384 slots。
- P 属于4,8,16,32,64,128,256；只允许已有密钥覆盖的正二次幂旋转，最大 P/2。
- 公开常量为有限实数标量、length1 或 lengthP，最多256个名字。
- 最多16个输出密文；packed-prefix 仍只返回一个密文、由固定选择器还原逻辑 shape。
- 对象数组中的每个 cell 是整个 Expr，不是单个密文槽位。外层数组 rank<=4、cells<=16。
- 原位 ndarray 操作改变共享存储/视图，copy 不共享；标量 Expr 增强赋值则产生新 Expr。
- 原有 AST、展开工作量、深度和数据大小限制保留。候选 golden 工作量<=1024；
  整个 native module<=4096。允许形状不代表任意长程序都能通过深度/资源预算。
- 数值门限和 tc128 安全配置不变，不引入 bootstrap 或新 runtime。

数据路径：

`schema5/PyTorch reference → FX布局与公开常量 → Agent native Hecate程序 → 类型/布局检查
→ 沙箱内真实hc.func tracing → Dacapo编译/inline → HEVM/CST → SEAL CPU → 解密比较`。

## 冻结的六例与确定性基线

清单：`scripts/baseline/cases/packed-native-6-manifest.json`。
fixtures 在 `packed_native_cases.py` 中构造，明确不是 Agent；不会将答案发送给模型。

| 案例 | 输入shape | P | 主要组合 |
|---|---|---:|---|
| native-packed-0-tensor-affine-native-alias | [2,3,5] |32| c/p helper、starred参数、对象数组view原位更新 |
| native-packed-1-native-packed-loop-rank4 | [2,2,4,8] |128| 两次真实平方、public loop、标量增强赋值 |
| native-packed-2-tensor-linear-native-six-output | [2,8] |16| 真实跨slot归约、native数组返回6个密文 |
| native-packed-3-tensor-zero_rows-native-zero | [2,32] |64| 固定零权重行、zero_ct输入与native调用 |
| native-packed-4-ps-max256-batched | [2,1,8,16] |256| 带batch卷积、native返回16个密文 |
| native-packed-5-perm-nonsquare | [2,3] |8| packed转置、真实旋转和目标掩码 |

报告：`/home/lhy/poseidon-work/results/packed-native-goldens-nv5t6iva/report.json`。
SHA256：`a7ac662dd85eb2ecb7043d1a5f0cd7b4d63d81023a8260e6e81efa20e69c977d`。

6/6正确程序通过；24组输入、760个值，最大绝对误差
`1.2645769886798774e-07`。2个合法语法的错误程序真实执行后被拒绝：
错误alias改copy误差约0.0625；颠倒Linear输出误差约0.21875。
这验证数值门禁，不能把反例“成功拒绝”计为正确程序数。

基线运行0次API。原门限 `1e-5+1e-4*abs(reference)`；显式配置
`seal-cpu-eva-w45-v1`，默认W40不变；N32768、14个60bit模数、tc128检查。
已通过239项回归、0跳过，含历史密态证据与本批真实 native 计划/调用/原位操作证据。

## 复现与付费边界

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3650s python3 scripts/baseline/run_packed_model_goldens.py --native-packed
```

目标内已获付费授权时，六例的真实 Agent 命令：

```bash
timeout -k 10s 11000s python3 scripts/baseline/run_agent_batch.py \
  --deepseek --case-manifest scripts/baseline/cases/packed-native-6-manifest.json \
  --native-array-mutation --provider deepseek --model deepseek-flash --reasoning-effort high \
  --jobs 10 --api-timeout 1200 --max-tokens 384000 --provider-retries 3 \
  --stream --loopback-proxy-port 6478 --compiler-configuration seal-cpu-eva-w45-v1
```

最多24次生成、96次HTTP尝试，native执行并发2。只发送合成模型、公开权重、layout、
规则与受限诊断，不发送测试输入、reference、fixture源码或.env。
未给出人民币总费用上限；输出上限不是预先支出的token数。

Agent可以选择允许范围内的等价表达；此批不强制每个程序用齐全部构造。
必须检查原始响应和实际trace后才能报告采用了哪些语法，不能从“开了开关”推断逐项覆盖。
本批仍不证明完整上游DSL、任意维度、真实bootstrap或Poseidon GPU端到端。

## 真实 Agent 结果

报告：`/home/lhy/poseidon-work/results/agent-batch-aqu_6yng/report.json`。
SHA256：`f1bcbbf4d29e669292f005a04e3c5f1ce72f2c47a88d31db3d71db58888391b7`。

- 首次 parse 6/6；首次 check/compile/execute/数值正确 3/6；最终6/6。
- 实际10次官方 DeepSeek `deepseek-flash/high`生成，4轮修复，0网络重试。
- 每例平均修复0.667轮；4次失败都在static_check，没有失败的密态运行。
- 输入67320、输出88635，服务端总计155955 tokens，0缺失usage；不据此估算人民币账单。
- 24组密态输入、760个值；最大绝对误差 `5.849206681532716e-08`，
  按元素数加权MAE `6.665673772310008e-10`。完整误差指标和逐项输出保存在原始报告。

| 模型 | 首次 | 最终 | 生成调用 |
|---|---|---|---:|
| 三维广播仿射 | 拒绝：golden额外声明c0/c1公开参数，且与公开常量名冲突 | 通过 |2|
| 四维两次平方 | 通过 | 通过 |1|
| 六输出Linear | 通过 | 通过 |1|
| 零权重行Linear | 拒绝：裸rotate调用；首轮修复又错误改为hc.rotate | 通过 |3|
| 256元素带batch卷积 | 拒绝：不存在的hc.rotate调用 | 通过 |2|
| 非方阵转置 | 通过 | 通过 |1|

原始失败与修复均保留。零权重案例第二次修复才改为 `v.rotate(step)`；
没有改编译器来接受错误API，也没有手写替换 Agent 响应。

### 实际构造覆盖（不能从开关推断）

六输出Linear使用可达 `reduce_sum` helper，实际6次调用；
卷积使用可达 `neuron` helper，实际16次调用。共2/6程序、22次真实 native 调用。
另4个程序合法地选择直线型计算，只保留golden函数。

本批成功原始源码没有public loop、starred调用、对象数组构造或增强赋值。
因此：
- **已确认**：native helper与P16/P256、多输出真实Agent组合通过；
- **已确认**：手工/规则fixture的数组别名、循环、增强赋值等与不同P组合真实密态通过；
- **未完成**：新增packed接口下这些构造的逐项真实Agent生成覆盖，需要独立限定实现形式的测试。
不能声称本批已覆盖全部native语法，也不把fixture当Agent贡献。

只读独立审计：`/home/lhy/poseidon-work/results/agent-lineage-audit-tgqcgcvo/report.json`；
6/6通过，0新API、0新密态执行、0遗留私钥目录。

```bash
timeout -k 5s 230s python3 scripts/baseline/audit_agent_lineage.py \
  /home/lhy/poseidon-work/results/agent-batch-aqu_6yng/report.json
```

TLS证据：`deepseek-tls-probe-o2a9vfyd`。默认直连3/3连接超时，
6478代理3/3完成TLS验证并收到无凭据请求预期的401。实际批次用6478，0传输重试；
没有关闭证书校验。短探针不能保证未来长连接不会失败。

基线清理4859295742字节临时密钥，付费批次清理3787608370字节，合计8646904112字节，
约8.05GiB。随机密钥不可恢复，可重跑重新生成；保留原始响应、失败证据、编译与数值产物。
WSL可用约189GiB，未删除无关目录。

## 验证与仍需推进的目标

最终242项定向回归全部通过，0跳过，包含历史真实密态证据、新版规则/付费证据、
实际别名图对照、六种周期和非法密钥步长检查。
启动回归时曾误用不存在的测试模块名、漏传HECATE；这是测试命令配置问题，已修正。
新fixture初版假设常量origin保留gain/bias原名而触发StopIteration，已改为读取经验证译出AST的实际操作数；
不是放宽类型/数值门禁。这些失败未触发付费调用。

当前仍未支持任意维度/任意长模型、所有旧construction语法与packed的任意混用、完整上游高层helper、
真实bootstrap或Poseidon GPU编译产物执行。下一步优先补native数组/循环/增强赋值等的
packed逐项真实Agent生成覆盖，再扩大剩余输入/算子组合。整体目标保持active。
