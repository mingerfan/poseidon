# 固定统计量 BatchNorm：模型语义、真实 Agent 与密态验收

## 结论（2026-09-15）

新增 `batch_norm` 模型算子与受限PyTorch/FX推理转换。
六个人工golden及两个错误反例均得到预期的真实密态结果；
随后六个真实DeepSeek Agent程序全部首次通过。
当前模型能力账本新增BatchNorm后，132例累计证据覆盖14类已支持图算子。
这不是完整上游DSL、所有参数/shape组合、上游HE_BN packing helper或GPU的完成声明。

目标仍为完整支持项目DSL语义。本轮解决此前明确拒绝的一类真实模型语义，
不是仅增加同一仿射模型的Python语法拼写。

## What / Why：推理态BatchNorm如何变为FHE运算

对N,C,...布局中通道c的输入x：

`y = gamma[c] * (x - running_mean[c]) / sqrt(running_var[c] + eps) + beta[c]`

固定eval统计量和公开参数可以预先计算：

`G[c] = gamma[c] / sqrt(running_var[c] + eps)`  
`H[c] = beta[c] - G[c] * running_mean[c]`

密文部分只需要 `y = x * G + H`。
G/H均是公开系数；其计算不是密态sqrt/divide，也不是bootstrap。
这里BatchNorm的gamma/G不等于CKKS ciphertext.scale。

代码依据：
`third_party/dacapo/python/poly/poly/MPCB.py::abstractBN` 使用上述G/H关系；
`python/poly/poly/Func.py::HE_BN` 将G/H交给closure。
本地测试执行了所选真实上游纯函数，确认公式，但没有借此宣称其packing closure全部可用。

**Agent职责边界：** 确定性前端准备折叠后的公开常量及layout，
Agent收到原始模型描述和准备好的常量registry后生成Hecate程序。
因此本批证明的是这套闭环可处理BatchNorm，不证明LLM独立推导了公式，
也不将规则转换器的公开常量折叠包装成Agent研究贡献。

## How：接口与数据流

### 模型描述

schema2/3图节点增加：

```json
{
  "id": "norm",
  "op": "batch_norm",
  "inputs": ["x"],
  "running_mean": "mean",
  "running_var": "variance",
  "weight": "gamma",
  "bias": "beta",
  "eps": 0.25
}
```

mean、variance、gamma、beta命名公开常量数组，每通道一个数。
weight和bias可为null，分别采用gamma=1和beta=0。
没有training选项；额外字段拒绝。输出保持输入tensor shape，
当前模型输出仍需显式flatten到受限向量接口。

- `model_graph.py`：验证节点与常量，构造可信Torch模型；
  reference直接按原始公式计算，不调用系数折叠函数、不读取解密结果。
- `batch_norm_ops.py`：验证公开统计量/epsilon，计算G/H并按通道展开。
- `fx_to_hecate.py`：识别推理态BatchNorm模块及F.batch_norm，
  只为该函数解析明确的kwargs/defaults；沿用已有乘法/加法输出。
- `model_semantic_coverage.py`：记录输出可达的BatchNorm、epsilon和affine配置。
- 生成程序仍走受限AST → 真实Hecate → Dacapo → HEVM/CST → SEAL CPU，
  没有新增替代执行后端，也不执行Agent提供的任意Python。

### 类型、shape及禁止项

- 通道轴固定为axis1，即PyTorch N,C,...。不是现有unbatched Conv的axis0；
  不能在未对齐layout时静默把两者连接。
- 沿用现有四逻辑输入元素和period4 packing，没有扩大密态slot ABI。
- 本轮实际覆盖shape[2,2]与[1,2,2]；不声称完整BatchNorm2d/3d形状已可输入。
- shape[2,2]的G展开为[g0,g1,g0,g1]；
  shape[1,2,2]为[g0,g0,g1,g1]。这个区别由反例验证。
- 必须eval、固定running statistics、CPU float64公开参数；
  training和track_running_stats=False拒绝。
- 仅精确BatchNorm模块的num_batches_tracked允许只读int64标量状态；
  跟踪前后必须不变，不放宽其他整数模型状态。
- variance非负、variance+eps>0、eps有限且在[0,1]、系数有限且落在公开数值界内。
  非有限、错误通道数、超限折叠系数、非法引用及额外字段均拒绝。
- 全零gamma沿用已有显式zero_ct ABI，由可信客户端公钥加密零，
  不用透明密文、不做解密回填，也不把Enc(0)称作bootstrap。
- `HE_BN/HE_MPBN/HE_ConvBN`等上游高层函数尚未允许直接出现在Agent程序中。

## Evidence：确定性正反例

六个正确case：
`bn-batched`、`bn-spatial`、`bn-no-affine`、
`bn-zero-gamma`、`bn-polynomial`、`bn-linear`。

每个case四组固定输入；reference、Torch和规则生成程序在float64诊断中一致。
真实密态人工结果：6/6正确，2/2错误程序被数值比较拒绝。

| 人工程序 | 最大绝对误差 | 结果 |
|---|---:|---|
| bn-batched | 3.398595e-8 | 通过 |
| bn-spatial | 4.616806e-8 | 通过 |
| bn-no-affine | 5.768387e-9 | 通过 |
| bn-zero-gamma | 1.561667e-8 | 通过 |
| bn-polynomial | 1.913353e-8 | 通过 |
| bn-linear | 7.827303e-9 | 通过 |
| wrong-channel | 3.2500000014 | numerical_comparison拒绝 |
| wrong-epsilon | 0.2132316253 | numerical_comparison拒绝 |

wrong-channel把N,C排列错误地当成通道连续分块；
wrong-epsilon错误使用sqrt(var)+eps而非sqrt(var+eps)。
两者均通过静态检查/编译并真实执行，不以静态拒绝替代数值反例。

人工报告：
`/home/lhy/poseidon-work/results/batch-norm-goldens-iblqeema/report.json`  
SHA256：`31d44b763b2997a2d1934415c82f49a18baa860fba5d76088996cf708200f7d8`。

## 真实付费Agent结果

冻结清单：`scripts/baseline/cases/batch-norm-agent-6-manifest.json`  
SHA256：`6d9c32cb23173b73a64f0becad5cb4b843f2a9f478e4b8b9ab1455db225d4992`。

| 模型 | 覆盖点 | 请求数 | 最大绝对误差 |
|---|---|---:|---:|
| bn-batched | 两个batch、不同通道参数、负gamma | 1 | 1.102112e-8 |
| bn-spatial | N,C,L布局与通道广播 | 1 | 1.499503e-8 |
| bn-no-affine | gamma/beta均缺省 | 1 | 6.451034e-9 |
| bn-zero-gamma | 与x无关但仍为真实密文的常量结果 | 1 | 8.515790e-9 |
| bn-polynomial | BN→flatten→square | 1 | 7.413076e-8 |
| bn-linear | BN→flatten→Linear跨元素归约 | 1 | 1.023375e-8 |

首次parse/static/compile/execute/numerical均6/6；反馈修复0；
API请求6，传输重试0，没有未完成或最终失败。
四组输入共24次密态输入执行，88个输出值（Linear每组2值，其余每组4值）。
加权MAE=5.000288059920838e-9，最大绝对误差7.413076197337887e-8。
批次耗时74.09625244140625秒。

服务DeepSeek、模型deepseek-flash、high、输出上限384000、请求超时1200s、
最多3次传输重试及3轮反馈修复、API上限并发10（本批6）、native并发2、代理6478。
三轮无凭据TLS检测均HTTP401且ssl_verify_result=0。
从原有.env读取密钥，没有修改或输出密钥。
usage：prompt41568、completion23292、total64860，0次缺失usage；货币账单未查询。

保持SEAL4.0.0、degree32768、[60]*14、tc128、冻结门限
`abs(actual-reference)<=1e-5+1e-4*abs(reference)`。
没有GPU、HPU、多GPU性能或bootstrap测试，没有参数降级。

## 证据与历史域兼容

- 付费批次：
  `/home/lhy/poseidon-work/results/agent-batch-2p8i9_1j/report.json`  
  SHA256：`d7e618bd28d547d3843e60cef267b40e696bf9d9472f5e95ba240efaba16be3e`。
- BN独立审计：
  `/home/lhy/poseidon-work/results/batch-norm-agent-audit-nn6jbzym/report.json`  
  SHA256：`c6d8f75ee9eff85869f56c110664129b3f07e051071ab020ebdccd39716bf9a8`。
- 当前132例模型能力审计：
  `/home/lhy/poseidon-work/results/model-capability-audit-zcz0p8tp/report.json`  
  SHA256：`0d5327783f1d1f3642e3667973e184f7b1b5e0949270fe4fd2552a624bec04f1`。
- 132例=84个用户图描述案例+48个历史catalog案例。
  14类已开放模型图算子有真实证据，不等于14种底层FHE算子，也不等于全DSL。
  累计账本是异构历史实验，不报告为一次同配置132例成功率。
- 审计重新验证原始请求、模型、权重/输入、trace payload、compiler产物哈希，
  重新计算独立reference及保存解密数组的误差；不调用API、不重跑FHE。
- 老114/126例报告不改。其13算子历史域分析源码按原SHA256归档在
  `scripts/baseline/history/`，归档字节已校验；不执行这些归档来覆盖当前实现。
  新审计在旧域逐项一致，新域明确增加BatchNorm，不能把旧报告追认为新覆盖。
- `audit_agent_lineage.py`去除“辅助零只能出现在request-v5”的过窄硬编码，
  改由已有request_input_names验证后续版本的显式零ABI；保留物理输入数量、
  零值执行、密钥、安全参数和数值检查。
- 本轮最终回归441项：419通过、22条件跳过。真实BN正反例、新付费6例、
  132例累计、旧114/126例和前轮v20/v21/v22证据均显式启用复核。
  离线故障注入输出Provider retry不是额外付费。

本轮累计回收临时密钥9,505,720,140 bytes（约8.85GiB）：
人工8次5,431,840,080 bytes，付费6次4,073,880,060 bytes。
这些可再生密钥已删除；模型/权重/输入、原始响应、源码、IR、HEVM/CST、
解密值和日志仍保留，旧批次没有清理或覆盖。

## 复现与下一步

人工正反例（不调用API）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 2600s python3 scripts/baseline/run_batch_norm_goldens.py
```

付费命令（重复执行会再次计费；已通过案例无需重跑）：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_agent_batch.py --deepseek \
  --case-manifest scripts/baseline/cases/batch-norm-agent-6-manifest.json \
  --object-unary --provider deepseek --model deepseek-flash --reasoning-effort high \
  --max-tokens 384000 --api-timeout 1200 --provider-retries 3 \
  --loopback-proxy-port 6478 --jobs 10
```

离线审计：

```bash
timeout -k 3s 480s python3 scripts/baseline/audit_batch_norm_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-2p8i9_1j
```

相关测试：test_batch_norm、test_batch_norm_evidence、test_batch_norm_paid。
证据测试在固定Nix/venv中显式传入POSEIDON_BATCH_NORM_GOLDENS、
POSEIDON_BATCH_NORM_AGENT和POSEIDON_BATCH_NORM_CAPABILITIES；未传入会明确skip。

当前不提交、不推送、不创建PR，分支与既有修改保持。
下一阶段仍须逐项补上游helper/packing、模型组合及尚未支持的语义，
并分别记录“前端实现”“人工密态正确”“真实Agent正确”的证据层级。
