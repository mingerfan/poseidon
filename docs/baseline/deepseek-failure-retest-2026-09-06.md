# DeepSeek Agent：14 个历史失败案例的独立诊断重测

## 结论与分母

- 日期：2026-09-06；新批次：`agent-batch-dznntui3`。
- 仅重测原 48 例实验中的 14 个失败案例：14 个全部尝试，1 个通过，13 个失败。
- 本次失败子集的首次及最终成功率为 **1/14 = 7.14%**，不是一次新的 48 例整体成功率。
- 原批次 `agent-batch-7r_d1p7f` 的 **34/48 = 70.83%** 保持不变，不将两批结果合并冒充首次成功率。
- 新增诊断确认：5 个响应的 8,192 个生成 token 全部用于 reasoning，最终回答为空；另外 8 个请求在 120 秒硬超时终止。
- 唯一成功案例 `linear-3` 通过真实 DeepSeek 生成、Hecate tracing、Dacapo 编译、SEAL HEVM CPU 密态执行和独立 float64 reference 数值比较。
- **本次仍不是 Poseidon GPU 验证**；未运行 bootstrap，未使用模拟执行或规则答案回退。

## 未改变的实验条件

- `deepseek-v4-pro`，thinking/high，`max_tokens=8192`，每次请求硬超时 120 秒。
- 最多三轮程序反馈修复；服务请求失败不隐式重试。本次每例恰好一次 API 请求，共 14 次，实际程序修复调用为 0。
- 两个案例并行；批次耗时 832.49 秒，约 13 分 52 秒。该耗时不是后端性能基准。
- 保持输入模型描述、提示请求、固定权重、四组测试输入和 reference 不变。
- 保持编译 profile、runtime 和安全参数不变；逐项门限仍为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 只增加响应诊断、失败子集选择及审计能力，没有修改 DSL 语义或放宽成功判定。

## 全部案例结果

| 案例 | 结果 | 响应/失败原因 | completion / reasoning token | 最终回答字符数 |
|---|---|---|---:|---:|
| linear-3 | 数值通过 | `stop` | 4881 / 4588 | 558 |
| mlp2-1 | 生成失败 | `length` | 8192 / 8192 | 0 |
| mlp2-2 | 生成失败 | `length` | 8192 / 8192 | 0 |
| mlp2-3 | 生成失败 | `length` | 8192 / 8192 | 0 |
| mlp2-4 | 生成失败 | `length` | 8192 / 8192 | 0 |
| mlp2-5 | 生成失败 | `length` | 8192 / 8192 | 0 |
| mlp3-2 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| mlp3-3 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| mlp3-4 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| mlp3-5 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| flatten_linear-2 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| flatten_linear-3 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| flatten_linear-4 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |
| flatten_linear-5 | 请求超时 | `transport_timeout` | 未收到 | 未收到 |

所有 13 个失败均在服务响应阶段终止，未进入候选程序解析、编译或密态执行。不能据此说这些案例生成了语义错误的 DSL；它们没有交付可验证的候选程序。

## 失败解释：事实、推断和未知

**Confirmed fact**：五个 `mlp2` 响应均为 `finish_reason=length`，completion 与 reasoning 都等于 8,192，最终 content 长度为零，refusal/tool-call 标志均为假。结合本次固定预算，可以确认这些新响应在输出 DSL 前耗尽了生成预算。`length` 和 reasoning token 字段的含义参见 [DeepSeek 官方 API 文档](https://api-docs.deepseek.com/api/create-chat-completion/)。

**Confirmed fact**：另外八个请求触发本地 120 秒硬超时，没有收到可用响应元数据，无法取得 finish reason 或 usage。分类为 provider infrastructure 是故障层分类，不等于已定位服务商内部故障。

**Evidence-based inference**：高思考强度与现有输出预算的组合，是五个已确认 length 案例值得单独验证的配置因素。相同配置下 `linear-3` 本次通过，也说明历史单次失败不等于该结构必然无法生成。

**Unconfirmed**：八次超时是由服务推理延迟、排队、网络波动还是其他原因导致；不能把这八次也归为 token 截断。重测结束后，无密钥 HTTPS 探针收到预期的 HTTP 401，说明检查时 DNS/TLS/HTTP 可达，但不能排除请求期间的瞬时问题；这也不是密钥失效证据，因为探针没有发送密钥。

旧批次未保存具体 finish reason 和失败 usage，因此本次发现不能追溯为原来 14 次失败的已确认原因。

## 数值与调用量

成功案例 `linear-3`：

- 四组输入均完成真实加密计算和解密，共 16 个 scalar 输出逐项比较通过。
- 按输出数量加权 MAE：`2.919418116541922e-9`。
- 最大绝对误差：`7.141071928229437e-9`。
- 最大非零 reference 相对误差：`1.3887844763083024e-7`。
- 这是固定测试输入上的工程证据，不是所有输入上的形式化等价证明。

14 次请求中，6 次收到 usage：prompt 7,913，completion 45,841，合计 **53,754 token**。其余 8 次超时的 usage 未知，因此这个数字不是完整账单用量，不能据此计算全部实际费用。

## 证据审计

`audit.json`：`status=passed`，`checked_cases=14`，`errors=[]`。审计检查：

- 子集恰好对应原报告的 14 个失败案例；原报告 SHA-256 未变。
- 每例 request ID 和冻结的 request.json 哈希与原实验相同。
- model、reasoning effort、max tokens、最大调用次数和 timeout 均与原实验相同。
- 权重、输入、reference 的实际数组数据与历史确定性基线一致。
- 编译 profile、runtime、门限、冻结文件和 artifact 哈希；批次中代码哈希一致。
- 重新计算的汇总指标与报告一致。

结果目录（WSL 原生盘）：`/home/lhy/poseidon-work/results/agent-batch-dznntui3`，包含 `report.json`、`audit.json`、`cases.csv`、`summary.md` 和各案例 evidence 指针。

原报告 SHA-256：`1a2a67f93f36521eea12ec0c2ca80c0b0969c0e501157fe371205d5015a4b892`。

## 本地实现与回归测试

- `deepseek_provider.py`：在严格完成性校验前记录安全的 finish reason、usage、文本长度和布尔标志。只记录长度，不保存 reasoning 正文、原始错误正文或密钥；未知结束原因不原样回显。
- `run_agent_batch.py`：增加 `--failed-from`，只选择已完成原批次的失败子集，并核对目录及生成配置，保留独立证据目录。
- `audit_agent_batch.py`：支持失败子集，比较原请求及配置，输出安全 provider diagnostics，保持原有指标口径。
- `test_deepseek_diagnostics.py`：新增八项离线测试，覆盖截断、缺失/非法 usage、未知响应内容泄漏防护及子集选择约束。
- WSL 普通 Python 回归：163 项，154 项通过、9 项因缺少该环境下的 Torch 跳过；这 9 项已在固定 Nix/PyTorch 环境中另外运行，全部通过。
- Windows provider 定向回归：37 项，36 项通过、1 项 Linux CA 专用测试跳过。
- 测试中出现 `--max-repairs 4` 参数错误是预期的拒绝非法参数测试，不是回归失败。

所有修改留在本地，未 commit、push 或创建 PR。原有用户修改保留。

## 复现命令

以下第一条会产生新的真实 API 调用和费用，不是读取已有报告；本轮已执行，不需重复运行：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_agent_batch.py --deepseek --jobs 2 \
  --failed-from /home/lhy/poseidon-work/results/agent-batch-7r_d1p7f/report.json
```

只审计新批次（无 API 调用，只在新批次中写派生报告）：

```bash
python3 scripts/baseline/audit_agent_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-dznntui3 \
  --baseline /home/lhy/poseidon-work/results/fx-batch-yw9qpdtx
```

## 下一实验建议（尚未执行）

优先建立单独的生成配置对照，验证降低思考强度是否能在原 token/时间预算内稳定交付 DSL，而不调整输入、reference、编译参数或数值门限。先用少量代表案例验证，再决定是否扩展到全目录；增加 token/timeout 应作为另一个明确标记的实验，不同时修改多个因素。新配置不能伪装成原配置的 `--failed-from` 重测。

本次诊断不支持继续广泛修复 WSL 环境，也不能宣称反馈修复能力已经通过真实实验。目标仍是稳定生成并验证正确程序，性能与 DSL 表达能力优化后置。
