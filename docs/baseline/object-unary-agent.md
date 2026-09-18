# v22 对象数组一元运算：逐项真实 Agent 验收

## 结论与证据范围

2026-09-15，新启动的8例真实DeepSeek Agent测试最终 **8/8通过**。
首个候选7/8通过静态检查、编译、密态执行和数值比较；混合类型案例经过2轮反馈修复。
共10次付费请求、0次传输重试、0个未完成案例。12个去重观察项全部被实际执行，
并有有限输出影响证据；不是仅检查源码出现关键字。

链路是：公开模型描述 → DeepSeek生成Hecate源码 → 可信AST检查/构造展开 →
实际Hecate tracing → Dacapo Earth/CKKS → HEVM/CST →
upstream SEAL HEVM CPU真实加密执行 → 解密 → 独立reference差分。
**不是Poseidon GPU验收，不是所有上游DSL语义或所有输入的证明。**

本批8例是同一仿射模型 `y=0.5*x+x+0.375` 的8种指定实现形式，
不是8个模型家族，也不是自由生成或结构留出泛化实验。
人工golden只用于离线测试；答案、测试输入、reference数值和密钥不进入付费请求。
之前的v19/v20/v21结果按原版本保留，不追认为v22生成覆盖。

## 冻结配置

- 模型：DeepSeek `deepseek-flash`，high。
- API并发上限10；本批只有8例，最大有效案例并发8。native并发2。
- 输出预算384000 tokens；单请求硬超时1200s。
- 最多3次传输重试；初始生成加最多3轮诊断反馈修复，均保持有限上限。
- 命令级代理127.0.0.1:6478；三轮无凭据TLS预检都得到HTTP401、ssl_verify_result=0。
- 所有provider密钥仍从现有本地.env读取，本轮未修改或打印.env。
- LLVM/MLIR18.1.2，SEAL4.0.0，固定NumPy1.25.2。
- degree32768、modulus_bits=[60]*14、tc128；不降级安全参数。
- 数值门限 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`，未放宽。
- 没有bootstrap、decrypt-and-reencrypt、GPU或多GPU执行。

## 逐项结果

| 案例 | 要求的构造/类型语义 | HTTP请求 | 最终结果 | 最大绝对误差 |
|---|---|---:|---|---:|
| `ou-neg` | cipher对象数组一元负号 | 1 | 通过 | 1.772923e-8 |
| `ou-zero` | 0-D对象数组返回标量Expr | 1 | 通过 | 2.413689e-8 |
| `ou-view` | view取负与fresh结果独立 | 1 | 通过 | 7.587093e-9 |
| `ou-positive` | 公开一元正号复制与后续源修改 | 1 | 通过 | 1.166668e-8 |
| `ou-negative-call` | np.negative对象入口 | 1 | 通过 | 1.179364e-8 |
| `ou-positive-call` | np.positive公开对象入口 | 1 | 通过 | 1.283914e-8 |
| `ou-plain` | 已知构造Plain取负且保留类型 | 1 | 通过 | 9.985997e-9 |
| `ou-mixed` | 同一数组的cipher、数值、bool分别影响输出 | 3 | 通过 | 9.317551e-9 |

去重观察项12个：`operator.USub`、`operator.UAdd`、`call.USub`、`call.UAdd`、
`cipher_cells`、`plain_cells`、`public_cells`、`boolean_cells`、
`zero_dim`、`input_view`、`fresh.USub`、`fresh.UAdd`。
每个要求在审计feature_matrix中映射到具体通过案例；总计22个案例-要求观察记录。
这些计数不能与旧版本项数简单相加后宣称全DSL已覆盖。

### 覆盖检查的可信边界

- `object-unary-exercises-v1.json` 和8例manifest冻结模型与实现要求。
- `object_unary_exercises.py` 在可信normalizer实际执行处观察类型、shape、
  0-D返回、view存储关系，而非只扫描AST拼写。
- 整体结果扰动验证语法事件参与输出；对cipher、Plain、真正int/float、bool，
  分别进行按类型结果扰动，避免“只有cipher被使用，却宣称整个混合数组都覆盖”。
- `function_construction.normalize(...,_unary_probe=...)` 仅供可信有限诊断：
  验证span与类型、只改变该事件该类单元的结果。候选程序不能访问这个接口；
  正常展开和真实FHE执行完全不启用它。不执行候选Python，也不添加IO/网络权限。
- fresh正号用“别名替代复制”反事实；fresh负号用“原地取负并别名返回”反事实。
- 最终独立审计还要求混合类型影响记录属于同一个操作，
  view记录和fresh负号记录属于同一个操作，排除互不相关的拼凑。
- 探测输出改变是有限依赖证据，不是符号等价或所有输入的形式化证明。
- 五个含cipher取负的最终产物包含实际HEVM opcode2；
  positive/positive-call/plain三例没有该opcode，公开构造/常量折叠不冒充密态运算。
  例如negative-call生成了两次取负，虽然相互抵消，但命名入口真实执行、
  对输出有影响，实际产物仍含取负。这是语法覆盖，不是优化质量结论。

## 混合数组两次失败与修复

1. 第一次生成：
   `np.array([x,c0,c1,True],dtype=object)`。
   c0/c1是单元素公开数值数组，不是Python标量，与cipher/bool混合构造被
   当前受限构造规则拒绝：`Ragged object arrays not supported`。
2. 第二次使用np.empty再逐项赋值，但仍直接存c0/c1。
   静态观察没有要求的公开int/float单元：
   `Required typed unary operation not executed: public_cells`。
3. 第二轮修复将公开单元素数组显式提取成float，生成：

```python
@hc.func("c")
def golden(x):
    arr = np.empty(4, dtype=object)
    arr[0] = x
    arr[1] = float(c0)
    arr[2] = float(c1)
    arr[3] = True
    neg = -arr
    return neg[0] * neg[1] + neg[0] * neg[3] - neg[2]
```

此时neg各单元为 -x、-0.5、-0.375、-1，
返回 `(-x)*(-0.5)+(-x)*(-1)-(-0.375)=1.5*x+0.375`。
公开数组的float提取不是解密；cipher转换为float仍被拒绝。
原始两份失败response.txt与反馈均保存；失败发生在static_check，
没有为它们启动编译或密态执行。不是TLS问题，也没有人工替换最终答案。

## 统计

| 指标 | 结果 |
|---|---:|
| 首次JSON/Python语法解析 | 8/8 |
| 首次静态/编译/执行/数值通过 | 7/8（87.5%） |
| 最终通过 | 8/8（100%） |
| 语义反馈修复总轮数 / 每例平均 | 2 / 0.25 |
| API请求 / 传输重试 | 10 / 0 |
| prompt / completion / total tokens | 69191 / 96346 / 165537 |
| 缺失usage请求 | 0 |
| 实际密态输入组 / 输出标量比较 | 32 / 128 |
| 加权MAE | 4.8400323213269175e-9 |
| 最大绝对误差 | 2.41368863229674e-8 |
| 最大非零reference相对误差 | 7.124937491016164e-8 |
| cosine similarity | 约1（浮点计算可能稍大于1） |
| 批次耗时 | 152.90236258506775秒 |
| 静态失败尝试 | 2 |
| 最终基础设施或生成失败 | 0 |

Token数来自服务返回的usage；实际货币账单未查询，不猜测费用。

## 审计、清理与回归

- 批次：`/home/lhy/poseidon-work/results/agent-batch-l_nd65sn/report.json`
- 批次SHA256：`09577506c0fd22bcd09d6f9df73c805fbc0ba9d5bf3849c240c5eb3820558b6f`
- 独立审计：`/home/lhy/poseidon-work/results/object-unary-agent-audit-9egccgrf/report.json`
- 审计SHA256：`38f8135d5798d9d7415ca280ccacd6e1de96c59d4aa5a6d4237aef786f2b2f70`
- 审计重算模型/reference、请求/trace/产物哈希、静态覆盖和保存解密数组的差分；
  审计自身0次新API和0次新FHE执行。
- 本批可再生临时密钥清理5,431,840,080 bytes（约5.06GiB），残留密钥目录0。
  已删除密钥不可恢复，但可重新生成；保存的解密值和编译产物足够做离线审计。
- 原始模型/权重/输入、请求、10份API响应与反馈、最终源码、IR、HEVM/CST、
  解密结果及日志保留。未删除其他批次文件；终态无付费worker。
- 结束时WSL可用约189.17GiB。
- 本轮最终回归388项：369通过、19条件跳过；其中显式启用并重新检查
  v20/v21旧付费证据、v22人工密态正反例和本轮v22真实Agent证据。
  测试日志里的Provider retry来自离线故障注入，不是新付费调用。

## 复现

付费命令（再次执行将再次产生费用；旧成功批次无需重跑）：

```bash
cd '/mnt/d/Code Space/Poseidon'
PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_agent_batch.py \
  --deepseek --object-unary-exercises --provider deepseek \
  --model deepseek-flash --reasoning-effort high --max-tokens 384000 \
  --api-timeout 1200 --provider-retries 3 --loopback-proxy-port 6478 --jobs 10
```

只读离线审计（不会调用API或重跑FHE）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 480s python3 scripts/baseline/audit_object_unary_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-l_nd65sn
```

新的回归入口：`test_object_unary_exercises.py`、
`test_object_unary_paid_evidence.py`；
实际证据测试需要在固定Nix/venv中显式传入
`POSEIDON_V22_AGENT_BATCH=/home/lhy/poseidon-work/results/agent-batch-l_nd65sn`。
未传入则明确skip，不当成实际证据已通过。

## 本地变更与下一门禁

本轮只新增冻结清单/8例manifest、覆盖检查、batch与sandbox接线、
可信类型诊断、独立审计、测试和本文档。
顺带修正construction_exercises.descriptor对sc-案例遗漏的路由；
实际旧v21批次使用的是专用descriptor，旧报告未改且复核通过。

保留原分支与全部既有修改。没有安装、修改.env、提交、推送或创建PR。
完整DSL目标继续保持active：本批不覆盖所有类型×维度×语法组合、
上游高层helper/packing、全部FHE管理语义或Poseidon GPU执行。
下一步应按真实上游缺口扩展一个有界语义片段，先确定性正反例，再独立冻结付费覆盖；
同时把单构造测试与多构造组合/模型结构泛化分开统计，避免只扩大一个仿射模型的语法花样。
