# DeepSeek Agent：低思考强度、高预算的 13 例失败子集重测

## 结论

- 日期：2026-09-06；批次：`agent-batch-3kexhqeg`。
- 按用户要求降低思考强度、大幅提高生成预算及超时后，剩余 **13/13 案例最终通过**。
- **11/13 = 84.62% 首次通过**；另两例各经过一次真实诊断反馈修复后通过，最终 **100%**。
- 15 次真实 API 请求均返回 `finish_reason=stop`，无截断、超时、缺失 usage 或最终失败。
- 52 次真实加密输入执行、164 个 scalar 输出逐项比较通过；没有放宽数值门限。
- 审计：`status=passed`，`checked_cases=13`，`errors=[]`。
- 执行后端仍为 **upstream SEAL HEVM CPU**，不是 Poseidon GPU。本次不含 bootstrap、模拟执行或规则答案回退。

## 实验选择与配置

从上一批 `agent-batch-dznntui3` 中选择仍失败的 13 例。它本身是原 48 例实验的 14 例失败子集重测；其中 `linear-3` 上次已通过，本次不再调用。

| 设置 | 上次 | 本次 |
|---|---|---|
| 模型 | deepseek-v4-pro | 不变 |
| thinking | enabled | 不变 |
| reasoning_effort | high | low |
| max_tokens | 8,192 | 65,536（8 倍） |
| 单次 API 硬超时 | 120 秒 | 900 秒（7.5 倍） |
| 最大程序修复次数 | 3 | 不变 |
| 案例并行数 | 2 | 不变 |
| 数值门限 | `1e-5 + 1e-4*abs(reference)` | 不变 |

官方接口支持 `low` 思考强度，65,536 token 低于当前文档中的最大输出上限；参见 [思考模式](https://api-docs.deepseek.com/guides/thinking_mode/) 和 [模型限制](https://api-docs.deepseek.com/quick_start/pricing/)。本地实现仍保留有限上限，不改成无限等待或无限预算。

这是用户明确要求的**联合配置实验**，不是单因素消融。结果不能单独归因于降低思考强度、扩大 token 或延长超时中的某一项。

原实验结果保持独立：原 48 例首次成功率仍为 34/48；上次 14 例子集仍为 1/14。本次 13/13 不能包装成“新配置下完整 48 例首次成功率 100%”。三个历史批次合起来，每个已准备案例都至少有一次成功证据，但这不等于同一配置、同一批次的全量验证。

## 逐例结果

| 案例 | 首次通过 | 修复次数 | 最终结果 | MAE | 最大绝对误差 |
|---|---|---:|---|---:|---:|
| mlp2-1 | 否 | 1 | 通过 | 1.735225e-9 | 4.265342e-9 |
| mlp2-2 | 是 | 0 | 通过 | 2.885177e-9 | 7.885124e-9 |
| mlp2-3 | 否 | 1 | 通过 | 7.078422e-9 | 2.305532e-8 |
| mlp2-4 | 是 | 0 | 通过 | 1.313813e-8 | 3.426114e-8 |
| mlp2-5 | 是 | 0 | 通过 | 2.740430e-9 | 1.239935e-8 |
| mlp3-2 | 是 | 0 | 通过 | 1.676796e-9 | 7.748314e-9 |
| mlp3-3 | 是 | 0 | 通过 | 3.335196e-9 | 1.498801e-8 |
| mlp3-4 | 是 | 0 | 通过 | 9.155516e-9 | 3.035506e-8 |
| mlp3-5 | 是 | 0 | 通过 | 2.321252e-8 | 1.067251e-7 |
| flatten_linear-2 | 是 | 0 | 通过 | 3.271467e-9 | 9.746481e-9 |
| flatten_linear-3 | 是 | 0 | 通过 | 4.881948e-9 | 1.200112e-8 |
| flatten_linear-4 | 是 | 0 | 通过 | 2.236747e-9 | 6.688883e-9 |
| flatten_linear-5 | 是 | 0 | 通过 | 2.363102e-9 | 4.644509e-9 |

两层 MLP：首次 3/5，最终 5/5。三层 MLP：首次及最终 4/4。flatten→Linear：首次及最终 4/4。

每例固定四组输入：零、有符号、固定种子随机、声明范围边界。比较对象是独立 PyTorch CPU float64 reference，不是从解密结果回填的答案。

## 阶段指标与数值结果

| 指标 | 分子 / 本批全部 13 例 |
|---|---:|
| 首次响应 JSON 可解析 | 13/13 |
| 首次 Hecate 源码可解析 | 13/13 |
| 首次静态检查通过 | 11/13 |
| 首次编译通过 | 11/13 |
| 首次密态执行通过 | 11/13 |
| 首次数值正确 | 11/13 |
| 修复后最终数值正确 | 13/13 |

- 实际修复调用：2 次；所有案例平均修复轮数：`2/13 = 0.153846`。
- 真实密态执行：52 组输入；输出逐项比较：164 个 scalar，全部通过。
- 按输出数量加权 MAE：`6.109219879666512e-9`。
- 最大绝对误差：`1.0672513139908801e-7`，出现在 `mlp3-5`。
- 最大非零 reference 相对误差：`1.1599557672159477e-6`。
- 门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`；未更改安全参数或模型语义。
- 批次总耗时：682.776 秒，约 **11 分 23 秒**。包括生成、隔离准备、编译和执行，不是 FHE 后端性能指标。

## 真实反馈修复证据

`mlp2-1` 和 `mlp2-3` 的首个候选都被静态检查拒绝，诊断为：

```text
Left operand must be ciphertext; precompute public-only arithmetic
```

例如 `mlp2-1` 首个候选包含：

```python
z0 = c4 * h0 + c5 * h1 + c6
z1 = c7 * h0 + c8 * h1 + c9
```

反馈后的第二个候选改为：

```python
z0 = h0 * c4 + h1 * c5 + c6
z1 = h0 * c7 + h1 * c8 + c9
```

这里 `h0/h1` 是密文表达式、`c*` 是公开常量。实数乘法在数学上可交换，但当前允许的 DSL 子集要求密文作为左操作数。Agent 根据诊断调整表达方式；验证器规则没有为候选放宽。第二版完成了编译、密态执行和数值差分。

该证据确认两个真实的静态类型/表达规则反馈修复案例，不代表已经验证各种 compiler/runtime/数值失败的通用修复能力。

首个案例证据：`/home/lhy/poseidon-work/results/agent-deepseek-fnww2xw7`；第二个：`/home/lhy/poseidon-work/results/agent-deepseek-dwt1un6c`。分别保留 `attempt-00`、`attempt-01` 的候选和诊断。

## 服务响应与 token 用量

15/15 请求返回正常 `stop`，usage 全部完整：

- prompt tokens：20,930。
- completion tokens：96,372。
- total tokens：**117,302**。
- 单次最大 completion：**14,932**（`mlp3-5`），其中 reasoning 13,966。
- 本次有 4 个请求的 completion 超过旧 8,192 上限：`mlp2-2`、`mlp2-3` 首轮、`mlp3-3`、`mlp3-5`。

因此扩大预算确实容纳了超过旧上限的完整输出；但由于配置联合变化，不把它表述为严格的单因素因果证明。本批没有达到新 65,536 上限，也没有触发 900 秒超时。token 上限不是实际用量；账单金额需以服务商计费为准。

## 审计与边界

Confirmed fact：审计完成且无错误，确认：

1. 新子集精确对应上次仍失败的 13 例，历史选择链的源报告 SHA-256 未变。
2. 每例 request ID、冻结 request.json 哈希与历史请求相同。
3. 实际 provider 配置与本批声明一致；相对上批仅思考强度、token 上限、timeout 三项声明改变。
4. 权重、测试输入及 reference 的 NPZ 内容与确定性基线一致。
5. 编译 profile、runtime 二进制及数值门限与基线一致。
6. 冻结输入及 compiler artifacts 的哈希有效，运行期间各案例的源码哈希一致；汇总指标可重新计算。

Evidence-based inference：本次新配置解决了这个失败子集中的输出完整性问题，并使两个静态检查错误能够进入真实反馈修复环节。

Unconfirmed：新配置在全 48 例、更多随机重复、任意 PyTorch 模型、不同 shape 和未知模型族上的稳定成功率。当前输入仍是受限数据型模型目录，常量注册表和 layout 由已有工具准备，不是自由导入任意模型或独立 packing 搜索。有限测试不是形式化等价证明。

Poseidon 单 GPU执行仍不在本批验证范围内，不将 SEAL CPU 成功外推为 GPU 链路成功。

## 本轮实现与回归

- `deepseek_provider.py`：保留 high/8192/120 默认值，允许显式设置 low/65536/900，仍严格绑定到获准的请求配置；HTTP envelope 有界扩大至 8 MiB。
- `deepseek_http_worker.py`：响应容量上限同步扩大至 8 MiB；请求上限、TLS 校验、密钥 stdin 传递和错误脱敏保持不变。
- `run_candidate.py`：暴露并传递 `--reasoning-effort`，让进程总时限覆盖新 API 预算和有限修复轮次。
- `run_agent_batch.py`：批量透传新参数；改变历史生成配置要求显式 `--allow-config-change`；支持有深度限制、源哈希验证的失败子集选择链。
- `audit_agent_batch.py`：审核声明的配置差异，核对每例实际配置；生成报告不再硬编码旧超时和 token 数。
- `test_deepseek_expanded.py`：五项新离线测试覆盖默认值不变、有限边界、传输授权、CLI 参数传递和历史子集选择链。
- `test_deepseek_transport.py`：响应容量负向测试使用新的容量边界，未取消超限拒绝测试。
- `agent-quickstart.md`：补充高预算运行命令和历史结果边界。

测试：

- WSL Python：168 项中 159 项通过、9 项 Torch 专用测试跳过；这 9 项随后在固定 Nix/PyTorch 环境中全部通过。
- Windows provider 定向测试：42 项，40 项通过、2 项 Linux 专用测试跳过。
- 新测试开发时遇到 Windows 无 `fcntl` 和夹具传输对象尚未序列化两处错误，均在真实 API 调用前修正并通过回归；未改变生产代码的验证门禁来通过测试。
- `git diff --check` 无补丁空白错误；已有 CKKS CMake 文件的 CRLF/LF 提示不影响测试。

全部修改仅保留在本地 working tree，保留用户已有修改，未切换分支、commit、push 或创建 PR。

## 复现及结果位置

实际执行命令（会产生新的 API 费用，不需为查看本报告而重复运行）：

```bash
cd '/mnt/d/Code Space/Poseidon'
python3 scripts/baseline/run_agent_batch.py --deepseek --jobs 2 \
  --failed-from /home/lhy/poseidon-work/results/agent-batch-dznntui3/report.json \
  --allow-config-change --reasoning-effort low --max-tokens 65536 --api-timeout 900
```

批次证据：`/home/lhy/poseidon-work/results/agent-batch-3kexhqeg`，含 `report.json`、`audit.json`、`cases.csv`、`summary.md` 及逐例 evidence 路径。不要分享完整结果目录中的 private-keys。

独立审计，无 API 调用：

```bash
python3 scripts/baseline/audit_agent_batch.py \
  /home/lhy/poseidon-work/results/agent-batch-3kexhqeg \
  --baseline /home/lhy/poseidon-work/results/fx-batch-yw9qpdtx
```

下一项合理验证是用冻结的新配置对完整 48 例单独跑一轮，区分首次成功和修复成功，不用跨配置拼接代替全量结果。本轮只执行用户要求的剩余失败子集，没有自动增加全量付费调用。
