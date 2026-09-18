# 跨计算图的 waterline 40/45 配对实验

后续：[联合配置身份与显式45验证](compiler-configuration.md) 已将JSON+waterline
纳入新请求契约，并重新验证循环golden；本实验和默认40保持不变。

本实验承接 [四槽求和精度诊断](hevm-waterline-precision.md)，不调用付费 API，
不重新生成候选，不改变默认 waterline40，不改变旧失败结果。
实验目的是检查增加编码 scale 对其他计算图的精度与编译安排的影响，
不是新增 Agent 语法覆盖或运行性能对比。

## 预先固定的范围

`scripts/baseline/probe_hevm_precision_graphs.py` 固定八个已有程序及其原报告 SHA256：

| 对象 | 来源 | 原始验证状态 |
|---|---|---|
| 密文平方后仿射 | 人工 native-array 算术 golden | 通过 |
| Linear(4,2) | 旧真实 Agent 候选 | 通过 |
| Linear(4,2) → square → Linear(2,2) | 旧真实 Agent 候选 | 通过 |
| Linear(4,2) → square → Linear(2,3) → square → Linear(3,2) | 旧真实 Agent 候选 | 通过 |
| fan-out 后合并 | 旧真实 Agent 候选 | 通过 |
| residual | 旧真实 Agent 候选 | 通过 |
| 8 神经元隐藏层 MLP | 人工 golden | 通过 |
| 四槽 rotate-and-sum | 原生公开循环人工 golden | 原始数值超差 |

固定三组新随机密钥；每组运行八个程序的 40、45 两种配置，共48次程序执行，
每次四组冻结输入，计划192组输入、576个输出值。保留所有失败，无自适应重试。
旧真实 Agent 程序的使用只说明源码来源；本次不增加真实 Agent 生成次数。

### 不变的变量

- 同一原始 Earth IR 与公开常量 CST，不重新 tracing，不手工重写算法。
- 原输入、权重、reference、输出选择器及逐项门限 `1e-5 + 1e-4*abs(reference)`。
- 原 compiler JSON、LLVM/MLIR18.1.2、SEAL4.0.0，以及现有 HEVM 执行二进制。
- N32768、14×60-bit 实际模数、tc128 安全检查。
- 每个 key trial 的全部16次执行共用同一套实际密钥文件，执行前后校验哈希；
  每次正常随机加密，不声称复用了同一个密文。
- 两种配置使用相同的 rotation key 集合 -3/-2/-1/1/2/3。
  对旧的仅含1/2密钥的案例，这是实验统一使用的超集，不冒称密钥策略与旧运行完全一致。

唯一配对配置差异是 compiler 的 `--waterline`。乘法图可能相应改变 result scale、
level、rescale 和常量编码安排，不能沿用纯求和实验中“指令完全相同”的结论。
每个产物由原 artifact gate 检查，真实密文元数据再与编译器记录逐项核对。

### reference 与隔离

逐项检查源报告、输入、权重、源码、IR、常量和编译产物的冻结哈希。
用模型描述构造 CPU float64 PyTorch reference，先与冻结权重完全比较；
同时通过独立图解释器的显式点积（`math.fsum`）复核原 reference。
它们不读取解密结果，核对容差为 `atol=rtol=1e-14`，不修改原 reference。

编译与执行使用既有 bubblewrap 隔离和可信 worker。runtime 只接收输入数组，
不挂载 reference。旧 payload 仅用于 I/O 和 layout，报告明确记录
`source_request_reused_for_io_only=true`；不将旧请求宣称为新精度配置下的 Agent 生成。
旧报告及默认 compiler profile 在实验结束后再次检查哈希。

## 已完成结果（2026-09-17）

**Confirmed fact：**16份编译产物均通过artifact检查；48次程序执行全部通过。
每个配置24次程序执行、96组输入、288个输出值；合计192组输入、576个输出。
无编译、执行或数值失败，0次API调用，未自适应补跑。

报告：`/home/lhy/poseidon-work/results/hevm-graph-precision-glskuy3w/report.json`

SHA256：`78ca3fd3faba35c64ee3755973f815280d20fa3e6dd9a547310e1192864caff0`

各行是三组新密钥、四组冻结输入上的最大绝对误差：

| 程序 | waterline40 | waterline45 |
|---|---:|---:|
| square | 1.080871236958103e-8 | 3.6318248408662157e-10 |
| Linear | 7.588534467473096e-9 | 2.7172064598346424e-10 |
| 两层MLP | 9.767544106864534e-9 | 2.92576532445743e-10 |
| 三层MLP | 5.851939183232346e-9 | 7.967865256475193e-11 |
| fan-out | 1.8996922479530554e-8 | 4.937596997933724e-10 |
| residual | 1.923134451686792e-8 | 6.160327803428345e-10 |
| wide MLP8 | 1.709150265760684e-8 | 1.0725905996711305e-9 |
| 四槽求和 | 7.56592754314056e-6 | 2.3637893575845226e-7 |

整体加权MAE：40为3.4140595219292e-7，45为1.0700369593236728e-8。
最大误差来自四槽求和；45相比40约降低32倍。
40本次全部通过不抹去原失败1.579301745589292e-5，也不证明40稳定。

### 编译差异：更低误差并非免费

下表 scale 均为log2(scale)，level为剩余data moduli数，来自实际HEVM；
每次运行的真实密文元数据已与之核对。多输出具有相同元数据的只显示一组。

| 程序 | result scale：40→45 | result level：40→45 | rescale指令数：40→45 |
|---|---|---|---|
| square | 60→75 | 2→2 | 1→1 |
| Linear | 80→90 | 2→2 | 0→0 |
| 两层MLP | 80→45 | 2→1 | 4→8 |
| 三层MLP | 80→75 | 2→2 | 10→16 |
| fan-out | 60→75 | 2→2 | 3→3 |
| residual | 80→45 | 2→1 | 2→3 |
| wide MLP8 | 80→45 | 2→1 | 16→32 |
| 四槽求和 | 40→45 | 1→1 | 0→0 |

**Evidence-based inference：**45是在当前固定模型范围内可用的精度候选；
其编译安排会影响modulus消耗、常量和运行工作量，不能只因误差更小就宣称性能优化。
上述指令数不是运行耗时实验，也不构成每条依赖路径的深度统计。

**Unconfirmed：**所有DSL组合、更深网络、所有输入或密钥均通过，以及实际性能收益。
本报告不作这些外推。

## 复现与审计

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 1850s python3 scripts/baseline/probe_hevm_precision_graphs.py
```

脚本无参数覆盖入口，使用已实现的 Nix/venv，不安装、不下载。
输出位于 `/home/lhy/poseidon-work/results/hevm-graph-precision-*`。
三套密钥各自在完成本组后按既有严格保留策略清理；其余证据保留。
`test_hevm_precision_graphs.py` 包含2项固定计划/失败计数单测，以及2项 opt-in
真实证据审计；证据审计使用 `POSEIDON_HEVM_GRAPH_PRECISION` 指定完整结果目录，
不重新执行 FHE、不调用 API。缺失证据时的 skip 不能算作通过。

## 解释边界

有限输入/密钥结果不构成全输入正确性或失败概率上界。精度提高不自动意味着
执行更快、level余量更大或所有图都受益。本轮不切换 production 配置。
下一门禁是为 JSON + waterline 建立联合版本身份，维护旧请求可复现性，
再验证完整生成/编译链；不得让 Agent 自行修改 scale、安全参数或门限。

4项新计划/真实证据检查以及已有159项回归共163项全部通过，无跳过；
包含旧真实付费结果、native构造、原始数值失败和此前精度诊断。
随后更新语义清单，仍保留历史单图实验和失败批次的原始记录。

三套临时密钥合计2,036,940,030 bytes（约1.90GiB）已清理，原随机私钥不可恢复；
可重新生成新的密钥，输入、权重、请求、IR、CST/HEVM、解密输出与日志全部保留。
清理后WSL文件系统可用约189.08GiB。

本轮新增实验脚本、审计测试、本文档，更新语义清单及测试与旧精度文档链接。
默认配置、生成器、compiler/runtime二进制和旧实验数据没有改变；未安装、commit、push或PR。
建议将这组独立实验和证据作为一个本地commit单元，等待明确提交授权。

目标仍为完整 DSL 语义支持，
不以本实验八个图通过替代尚缺的构造组合、高层 helper、真实 bootstrap 和新语法付费覆盖。
