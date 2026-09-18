# v20 对象数组算术：批量接线与真实 Agent 验收

## 本轮结论

新增 request v20 的普通批量模式 `--object-arithmetic` 和固定十例模式
`--object-arithmetic-exercises`。后者使用独立冻结规范和模型 manifest，
不会改变上一轮 v19 的 30 例/117 项清单。

**真实 DeepSeek 生成：首次 8/10，通过反馈修复后 10/10；13 次 API 调用。**
十例全部经实际 Hecate tracing、Dacapo Earth/CKKS 编译、HEVM/CST 与
SEAL CPU 加密执行、解密、独立 reference 数值比较。新定向清单的十个去重
typed observation 项均有执行与有限输出扰动证据。

本轮不是完整 DSL 验收：仅验证明确列出的对象算术形式，未证明所有输入、shape、
布局、语义组合或网络泛化。Poseidon GPU、bootstrap/upscale 等仍未验证。
十个观察项不是十个独立密态 opcode，也不能与旧117项简单相加声称语义总覆盖数。

## 固定模型与构造要求

- 8 个仿射案例：`y=0.5*x+x+0.375`。
- 2 个二次多项式案例：`y=0.5*x*x+x+0.375`。
- 输入/输出 shape 均为 [4]，公开固定权重、加密输入。
- 每例四组测试输入：零、有符号、固定种子随机、声明边界。
- 二次案例分别通过对象数组间密文乘法和重叠切片 `*=` 实现。
- 原始模型、权重、输入、reference、门限和安全参数未由 Agent 修改。

冻结资料：

- [每例语义要求](../../scripts/baseline/object-arithmetic-exercises-v1.json)
- [模型 manifest](../../scripts/baseline/cases/object-arithmetic-exercises-10-manifest.json)
- [执行与输出影响门禁](../../scripts/baseline/object_arithmetic_exercises.py)
- [离线真实证据审计](../../scripts/baseline/audit_object_arithmetic_batch.py)

请求只加入对应案例的公开要求，不包含测试输入、reference 数值、人工答案或密钥。
已有 provider 的请求白名单、请求哈希和 shared .env 凭据机制保持；
未为生成程序开放文件、网络或进程权限。

## 覆盖如何验证

可信解释器在真正发生对象数组运算时记录操作类型、源码位置、
输入/结果的公开存储 shape、左右存储是否重叠、Empty 和 cipher-array 情况。
只在 AST 中出现算术符号、普通标量运算或未执行代码不算对象数组覆盖。

随后使用有限扰动：

- 对普通算术/广播/重叠运算替换算术操作，要求仍合法展开并改变探测输出。
- 对明确要求别名的案例，将 AugAssign 替换成新数组绑定；要求输出改变，
  以排除“只用新绑定，没有使用共享引用”的程序。
- 对 Empty 案例，将 Empty 构造替换为公开零；要求输出改变。
- 只检查原候选的密态数值正确性；扰动程序是离线覆盖探针，不冒称额外 FHE 测试。

全部最终源码已复核。覆盖证据是有限的：
例如真实重叠不等于所有更新顺序都被当前案例区分；非交换运算、
逆向重叠、不同 shape 和所有别名边界还需后续测试。
当前没有以这个小集合证明整个 NumPy 或全部 Hecate。

| 去重观察项 | 最终有效案例 |
|---|---|
| `binary.Add` | `oa-zero-dimension` |
| `binary.Mult` | `oa-broadcast`, `oa-cipher-product` |
| `cipher_pair` | `oa-cipher-product` |
| `empty_left` | `oa-empty-subtract` |
| `inplace.Add` | `oa-overlap-add`, `oa-inplace-add-alias` |
| `inplace.Mult` | `oa-inplace-multiply-alias`, `oa-overlap-multiply` |
| `inplace.Sub` | `oa-empty-subtract`, `oa-inplace-subtract-alias`, `oa-overlap-subtract` |
| `overlap` | `oa-overlap-add`, `oa-overlap-multiply`, `oa-overlap-subtract` |
| `rank_broadcast` | `oa-broadcast` |
| `zero_dim` | `oa-zero-dimension` |

## 实测结果

| 指标 | 结果 |
|---|---:|
| 首次 JSON/Python parse | 10/10 |
| 首次 static/compile/execution/numerical | 8/10 |
| 修复后数值通过 | 10/10 |
| 平均反馈修复轮数 | 0.3 |
| 实际 API 调用 | 13 |
| 实际传输重试 | 0 |
| prompt tokens | 83051 |
| completion tokens | 148434 |
| total tokens | 231485 |
| 批次时间 | 165.3135 秒 |
| 密态输入组 | 40 |
| 输出标量比较 | 160 |
| 加权 MAE | 4.370716883642794e-9 |
| 最大绝对误差 | 1.9217175406538445e-8 |

配置为 DeepSeek `deepseek-flash`、high、输出上限384000 tokens、
单次1200秒超时、传输重试上限3、初始生成加最多3轮修复、API并发10、
native并发2，显式代理6478。运行前同代理三次TLS/证书检查均通过；
无认证探测返回401，没有由此冒称API key已验证。真实调用后认证成功由实际结果证明。
未查询或估算账单金额。

安全/数值参数保持：SEAL4.0.0、degree32768、[60]*14、tc128检查，
`abs(actual-reference) <= 1e-5+1e-4*abs(reference)`。
不改安全参数，不模拟bootstrap，不把CPU结果解释为GPU结果。

| 案例 ID | 调用次数 | 最大绝对误差 | 证据目录名 |
|---|---:|---:|---|
| `oa-broadcast` | 1 | 9.55663e-9 | `agent-deepseek-6lvdn78j` |
| `oa-inplace-multiply-alias` | 2 | 1.92172e-8 | `agent-deepseek-d_8ehq9m` |
| `oa-overlap-add` | 1 | 1.68233e-8 | `agent-deepseek-2kmxfk2j` |
| `oa-empty-subtract` | 1 | 1.04663e-8 | `agent-deepseek-2fei04ap` |
| `oa-cipher-product` | 1 | 1.32948e-8 | `agent-deepseek-n5s996t1` |
| `oa-zero-dimension` | 1 | 1.62959e-8 | `agent-deepseek-xek_s335` |
| `oa-inplace-subtract-alias` | 1 | 1.86423e-8 | `agent-deepseek-64aw5yv3` |
| `oa-inplace-add-alias` | 1 | 1.49900e-8 | `agent-deepseek-24cxvv6h` |
| `oa-overlap-multiply` | 3 | 1.05893e-8 | `agent-deepseek-5u1x9gwx` |
| `oa-overlap-subtract` | 1 | 1.69133e-8 | `agent-deepseek-hfmq6pbf` |

## 三次失败尝试与修复

全部为 static_check 失败，没有实际传输失败；原始响应和反馈均保留。

1. `oa-inplace-multiply-alias` 首次使用不存在的 `public_weight/public_bias`；
   请求实际提供 `c0/c1`。后续改为正确常量引用和显式公开数组转换。
2. `oa-overlap-multiply` 首次把 `c0/c0/c0` 产生的公开数组放入对象数组
   标量单元，触发单元类型限制。
3. 同一案例第二次使用 `float(c0)` 等数组到标量转换，当前契约不接受。
   第三次改为真正的 ciphertext object-array 乘法，通过。

后两次暴露的是当前受限值规则与更完整的 NumPy/DSL 数值转换之间的缺口，
不能简单说所有失败都只是模型能力差，更不能以 Agent 绕开这些路径后的成功
证明这些缺口已实现。公开数组/标量转换仍列为完整目标待办。

另有一个离线测试断言最初过宽：它禁止 JSON 任意位置出现 hecate_source，
而正常响应 schema 必须命名这个字段。已改为禁止请求顶层携带候选源码、
禁止包含人工答案，保留响应 schema。没有因此放宽生成执行权限。

## 证据、回归和清理

原始批次：
`/home/lhy/poseidon-work/results/agent-batch-6u_j0fug/report.json`

批次 SHA-256：
`c6113160a05538e50cd3f8a5c8a18e0bafca350ad2b22d2c01cb416fbcbd98e4`

最终离线审计：
`/home/lhy/poseidon-work/results/object-arithmetic-agent-audit-dhxpgqzz/report.json`

审计 SHA-256：
`35634ffb4f494f3e872ec227185f2ab7a816fe959077444d79f25641631badf9`

审计重新核对固定模型/规范、原始统计、冻结文件哈希、实际 trace payload、
typed event 和扰动证据、编译产物、SEAL安全参数与密钥验证记录；
并从固定输入独立重算两种模型公式，再比较保存的解密值。
该审计不读取凭据、不新增 API、不重新密态执行。

最终 **352项相关测试：333通过、19条件跳过、0失败**。
显式启用了上轮六个手工 v20 密态正反例和本轮十个真实 Agent 记录的审计，
它们不属于跳过项。旧 v19 付费记录重新审计仍为30/30、117/117。

本轮自动清理 **6,789,800,100 bytes（约6.32GiB）** 临时测试密钥，
原始响应、反馈、模型、源码、IR、HEVM/CST、解密输出保留。
旧随机密钥不可恢复，可重新生成新密钥；API key未改动。
本轮所有付费进程已结束。

为避免将“重跑”偷偷变成较弱的测试，普通 `--failed-from/--remaining-from`
现在会拒绝来自冻结构造批次的记录：不能丢掉逐项要求再宣称同一实验成功。
固定十例当前走fresh cohort；针对失败子集的未来重跑必须显式保留相同规范。
本次十例没有补批，也没有通过删除失败或放宽门限取得成功。

## 复现

离线审计，不产生API费用：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 500s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/audit_object_arithmetic_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-6u_j0fug
```

查看冻结批次，不读取凭据：

```bash
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py \
  --plan --object-arithmetic-exercises
```

以下仅记录本次付费配置；运行会产生新费用：

```bash
timeout -k 10s 24000s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_agent_batch.py --deepseek \
  --object-arithmetic-exercises --provider deepseek --model deepseek-flash \
  --reasoning-effort high --max-tokens 384000 --api-timeout 1200 \
  --provider-retries 3 --loopback-proxy-port 6478 --jobs 10
```

在现有固定 Nix/Python 环境、scripts/baseline目录设置
`POSEIDON_V20_AGENT_BATCH=/home/lhy/poseidon-work/results/agent-batch-6u_j0fug`
后，`python -m unittest test_object_arithmetic_exercises` 会启用真实付费证据审计。
未设置时明确跳过该证据测试，不把缺少证据算通过。

## 本地修改与下一步

修改：object_arrays.py / function_construction.py（仅可信观察钩子），
construction_exercises.py / candidate_contract.py（独立v20规范路由），
candidate_sandbox.py（只读模块），run_agent_batch.py（v20批量与重跑保护）。
新增：十例规范与manifest、object_arithmetic_exercises.py、
test_object_arithmetic_exercises.py、audit_object_arithmetic_batch.py和本文档。
既有C++/GPU/编译文件修改保留；无commit、push、分支或sparse checkout变更。

下一步应处理本轮真实暴露的公开数组/标量转换语义，并继续推进对象数组其他运算、
一般packing、真实helpers及独立backend门禁。不能以10例或旧117项重定义完整DSL目标。

