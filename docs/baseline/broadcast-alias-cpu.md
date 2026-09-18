# Broadcast、原生公开量减法与别名：真实 CPU 语义证据

2026-09-11。当前任务不再要求 Poseidon GPU 对接。本记录使用现有
Hecate → Dacapo → HEVM → SEAL CPU 链路，不新增执行后端。人工验证部分
无付费 API；后续 Agent 验证的调用与基础设施失败另节记录。

## 解决的问题

`x + [c]` 是否向每个 slot 加上 c，`x - w` 是否保持减法顺序，以及
`alias = x` 是否只绑定同一中间值而不改变计算，不能靠 Python 语法合法证明。
这批将对应的静态类型、常量表示、独立 reference、手写程序和密态结果连接起来。

实际代码边界：

- `model_graph.py` 检查数据图的 shape/broadcast，并以独立 Python 算术计算 reference。
- `fx_to_hecate.Emitter.binary` 对 packed 输入把 scalar 或 length1 公开量都导出成
  length1；Linear 后的 scalar-neuron 布局则导出 JSON scalar。原始 PyTorch
  rank 信息因此不一定保留为相同的 DSL 常量表示。
- `candidate_trace.py` 将 scalar 和 length1 都转为一元素 NumPy 数组，再通过
  白名单 AST 分派调用真实 Hecate；不执行生成的 Python 文本。
- `hecate_contract.py` 允许 cipher-left 的加、减、乘、取负和 fresh SSA 变量。
  `alias = x` 合法，但别名重绑定、`+=` 和将 public 常量伪装成密文临时量被拒绝。
- 编译器负责 scale、level、rescale 和 relinearization；密钥及输出 slot
  选择由固定 harness 负责，不由程序改变。

长度 2/3 的不匹配数组及嵌套 `[ [c] ]` 在这个输入 ABI 下拒绝；不推断任意
NumPy/PyTorch 广播或自动 repacking。本批并没有扩大 slot 数或输入 shape。

## 人工程序与独立 reference

案例是 `scripts/baseline/cases/broadcast-*.json`，用户权重直接来自该数据文件，
不通过案例名字查权重。人工源码位于 `scripts/baseline/golden_cases/broadcast/`。
`broadcast_fixtures.py` 声明每项期待的公开常量 registry 和结构分区。

| 数据图 | 对应语法证据 | 正确程序最大绝对误差 | 错误程序最大绝对误差 |
| --- | --- | ---: | ---: |
| x + [0.375] | add.length1、statement.alias | 1.00972e-8 | 0.750000005 |
| x × [-0.5] | multiply.length1 | 3.70705e-9 | 2.000000001 |
| x − [0.375] | subtract.length1 | 5.41091e-9 | 0.750000006 |
| x − [0.125,-0.25,0.5,0.75] | subtract.length4 | 1.17218e-8 | 1.500000007 |
| Linear(4,1)(x) − 0.375 | subtract.scalar、真实跨 slot 归约 | 2.75019e-9 | 0.750000003 |

第五项公开权重为 `[0.5,-0.25,0.125,0.75]`，无 bias；不是逐元素乘法代替 Linear。
每个正确程序都有一个故意使用错误运算/符号的配对程序。全部错误程序通过
语法、编译和密态执行门禁后，才在 numerical_comparison 被拒绝。

特别注意减法的常量来源：现有规则翻译器把图中的 `x-offset` 精确重写为
`x + (-offset)`，因此导出的常量 c 已经是负 offset。为单独验证原生减法，
golden 使用 `-(-x-c)`，代数上等于 `x+c`。未篡改 registry 来适应手写源码。
常量来源和负号不能凭变量名猜测；正确/错误版本的区别正在这里。

## 执行、审计和清理

执行报告：
`/home/lhy/poseidon-work/results/broadcast-golden-batch-cq9e4lbd/report.json`

SHA256：`55068edbd93b9f632a335d1307191a7be3cc0a86256b1db7f326efeb33730465`。

- 5 个正确 golden 通过；5 个数值反例正确拒绝，10/10 符合预期。
- 每项四组输入：零、有符号、seed=42 随机、范围边界，合计 40 组密态输入、
  136 个被比较输出值，其中正确程序 68 个，反例 68 个。
- SEAL 4.0.0、N=32768、14 个 60-bit key primes、tc128 检查保持不变。
- 门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 没有 bootstrap、明文替代或解密结果回填 reference。
- 测试重新检查冻结文件和 compiler artifact 哈希、真实 Hecate tracing 标记、
  实际密钥检查、原始四组输入与独立 reference，并从 decrypted.npy 重新比较。
- MAE、逐项绝对/非零相对误差和 cosine 都保存在单例 report 中；极小浮点舍入
  可能使 cosine 显示为 1.0000000000000002，不应据此声称相似度超过理论上限。
- 删除可再生临时密钥 2972830480 字节，约 2.77 GiB；零 private-keys 目录保留。
  保留全部模型、源码、IR、HEVM/CST 和解密结果。随机密钥原字节不可恢复，
  重跑时生成新密钥，不影响复现实验程序和数值门限。

复现（无网络/API，不安装依赖；使用已有固定 Nix 环境）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 3200s python3 scripts/baseline/run_broadcast_goldens.py
```

在已有固定 NumPy/Torch 环境下，将环境变量 `POSEIDON_BROADCAST_GOLDEN_REPORT`
设为上述报告，再运行 `python -m unittest test_broadcast_semantics -v`；
设置 `PYTHONPATH=scripts/baseline`。不设变量只运行静态/参考检查，不冒充执行通过。

## Agent 覆盖不能与人工覆盖混计

多批次审计已支持显式追加报告，并按实际 evidence 路径去重共享祖先：

```bash
timeout -k 3s 500s python3 scripts/baseline/audit_dsl_coverage.py \
  /home/lhy/poseidon-work/results/agent-batch-3s484qfx/report.json \
  --additional-report /home/lhy/poseidon-work/results/agent-batch-k1gloaa6/report.json
```

本轮报告：`/home/lhy/poseidon-work/results/dsl-grammar-audit-8b_ks3o0/report.json`。
SHA256：`fc3c1d234dc19dfeaac5008882e2b79a5fd47f191706e20f4aafb7ec4d685110`。

工具重新核对两组不同配置的 Agent 证据：96+6 个候选运行、408 组输入、
1176 个输出。仍是 26/32 个结构分区；报告保留每个 cohort 的模型族和数值覆盖，
不合并成单配置成功率。共享候选只计一次，同名但独立的案例仍有不同来源标识。

另外六个结构分区在本轮人工 golden 中获得了真实 FHE 证据。因此整个受限
结构清单的 32 项已有不同来源的执行样本，但 **Agent 覆盖仍为 26/32**，
且都不是全部程序组合、全部合法输入或整个上游 Hecate 的形式化验证。
非法控制流、bootstrap、任意高层 helper 等原先不支持的能力仍不支持。

本轮 Nix 联合检查 18/18 通过，无跳过，包含十个实际密态结果的重审计、
Torch/独立 reference 对照和 102 个 Agent 候选的源码/数值证据并集。
没有用这组有限结构测试把全部 goal 标记完成。

## 把规则真正发送给 Agent，并用同一组模型验证

`candidate_contract.SEMANTIC_GUIDANCE` 新增 schema=1 的结构化说明，涵盖公开量
broadcast、常量变换来源、SSA 别名、layout/输出选择、等价表达式和 FHE 边界。
`make_request` 将其副本纳入 request_id 哈希；provider 将它连同公开模型发给
DeepSeek，provider 和 candidate validator 都拒绝字段篡改、未知版本或额外说明。
不含参考答案、测试输入、私钥或规则转换器的程序；不扩展 AST/算子白名单。
历史没有此字段的请求继续按原规则/原哈希审计，不反写旧实验。

实测输入清单：`scripts/baseline/cases/broadcast-agent-manifest.json`，与人工
golden 使用同一份五个数据模型，未要求 Agent 模仿 golden 的写法。
模型为 deepseek-flash，high、384000 tokens、1200s、API workers=10，
本地 native workers=2，代理 6478；只有五例，不额外凑十个请求。

- 首批 `agent-batch-bnch9gul/report.json`：4/5 通过；1 例没有有效候选，
  在 HTTP 200 后读取正文阶段遭遇 TLS DECRYPTION_FAILED_OR_BAD_RECORD_MAC。
- 失败响应被完整拒绝，未转成 DSL；没有关闭证书校验或给 TLS 完整性失败
  增加自动重试。真实根因未能归到特定端或代理实现。
- 同一 Nix Python/OpenSSL 下重新做三轮无凭据测试：6478 三轮均验证证书/
  hostname、TLSv1.3 并完整收到预期 401 响应；默认直连三轮均建连超时。
  报告 `deepseek-tls-probe-_k_dnl6i/report.json`。短请求通过不证明长响应永不失败。
- 仅对失败例做新批次 `agent-batch-d1dbuvl6/report.json`：1/1 通过。
  无参数变化，原首批/失败诊断保留；不是覆盖原结果的“首次 5/5”。
- 五个成功候选都无需反馈修复，真实密态共 20 组输入、68 个输出，最大绝对
  误差 1.835669533045703e-8。独立重审计报告为
  `agent-lineage-audit-q24oeyp0/report.json`，累计 5/5、无密钥目录残留。
- 总计 6 次 API 请求，已知 usage 为 20775 tokens；TLS 失败的那次 usage
  未知，不能推断未计费。本批不是语义说明有效性的因果对照实验。
- 在线运行另清理临时密钥 1783698288 字节；加上人工验证本轮共清理
  4756528768 字节（约 4.43 GiB），原始模型、编译/解密证据均保留。

全部结果目录以上均相对 `/home/lhy/poseidon-work/results`。
最新三个 cohort 覆盖审计 `dsl-grammar-audit-o12wacw0/report.json`：
107 个独立候选运行、428 组密态输入、1244 个输出，Agent 覆盖 28/32。
SHA256 为 `5b449edda88a99c5c987b4cbb5fbeaeabfa035223b916482aee3694f5bda4797`。
新增的是 add.length1 和 multiply.length1；未观察到 statement.alias、
subtract.scalar、subtract.length1、subtract.length4。减法模型正确，并不
要求生成源码必须用减号；这四个结构项的人工证据不计作 Agent 覆盖。

`test_semantic_guidance` 验证新增说明实际进入请求、哈希不可变、旧请求兼容、
说明不扩大权限；设置 `POSEIDON_BROADCAST_AGENT_LINEAGE` 为最后一例的批次
报告后还会重审计五个真实候选与独立手算 reference、三队列结构覆盖。

最终回归记录：WSL 全量 386 项，308 通过、78 条件跳过、零失败。
固定 Nix 下显式开启本页人工/Agent 真实证据、两队列和三队列审计的联合
测试 23/23 通过，无跳过。全量回归中显示的 --jobs=11 和 --max-repairs=4
参数错误是有意设置的拒绝用例，不是本轮运行失败。
