# 候选 Hecate 程序的隔离验证与反馈闭环

## 结论和证据边界

Confirmed（2026-09-05）：在已有 48 例规则转换对照之后，新增了可脚本调用的
候选响应接入、隔离 tracing/编译/密态执行和最多三轮反馈修复编排。
**当前接入的是文件回放，不是 LLM；尚无模型自主生成/修复的成功率。**

实际自测 `/home/lhy/poseidon-work/results/candidate-replay-7ommkr7q`：

| 响应 | 实际到达的关卡 | 结果 |
|---|---|---|
| 非法 JSON | response_parse | 拒绝；不 trace、不执行 |
| 将 Linear 的 rotate(1) 错写为 rotate(2) | 真实 trace/compile/encrypt/evaluate/decrypt/compare | 数值失败，最大绝对误差 0.6736366599782002 |
| 正确的既有规则答案 | 同一真实链路 | 通过；MAE 3.512786679593649e-9，最大绝对误差 8.880283147716383e-9 |

后两次各执行四组加密输入、比较八个输出值。容差仍为
`abs(error) <= 1e-5 + 1e-4*abs(reference)`，没有放宽。
SEAL 4.0.0、tc128、N=32768、14×60-bit 模数、waterline=40 不变。
没有 bootstrap、decrypt-and-reencrypt 替代计算、GPU 或新增 CPU HEVM 后端。

从保存的 JSON 响应文件经 `--replay` 独立入口再次完成同样的拒绝/数值失败/通过
序列：`candidate-replay-rm7lm1v_`。不能把这两个 scripted replay 的恢复结果
解释成 Agent 能读懂反馈，更不能外推 48 例的 Agent 正确率。

回归：宿主发现 104 项，95 项通过、9 项 Torch 测试在宿主跳过；这 9 项另在
固定 CPU Torch/Nix 环境重新执行，全部通过。新增候选合同/回放/隔离命令与
真实证据测试共 19 项，包含在上述宿主通过数中。旧 48 例和 golden 产物的
完整性/数值检查也在本轮回归通过，但未将旧产物检查说成重跑全部密态模型。

Inference：验证器可以识别“可编译、可执行但数学语义错误”的候选，并向未来
模型接口返回分层反馈。Unconfirmed：模型自主生成/修复、未见结构泛化、
任意 Python/PyTorch 安全导入、一般 packing/shape 支持、Poseidon GPU 路径。

## 所在层、输入输出和所有权

`受信 PyTorch model → 公开 request → provider response → 静态检查 →
隔离 Hecate frontend → 隔离 Dacapo compiler → artifact gate →
隔离原 SEAL runtime → 宿主独立差分 → feedback → 下一响应`

| 模块 | 责任 |
|---|---|
| `scripts/baseline/candidate_contract.py` | 固定请求 hash、响应字段校验、provider 接口、最多 1+3 次响应编排 |
| `scripts/baseline/candidate_trace.py` | 解释已验证的 AST 节点，构造真实 Hecate Expr；不 exec 候选源码 |
| `scripts/baseline/candidate_sandbox.py` | bubblewrap 最小挂载、隔离 namespace、清空环境、时间和资源上限 |
| `scripts/baseline/candidate_worker.py` | 可信隔离探针和对原 SEAL 执行器的调用 |
| `scripts/baseline/run_candidate.py` | 输入模型、独立 reference、公开 request、候选验证、日志和数值诊断 |
| `scripts/baseline/test_candidate_pipeline.py` | 合同/编排/命令门禁单测及 opt-in 真实产物回归 |

响应严格只允许：

```json
{"schema": 1, "request_id": "从 request.json 原样复制的 SHA-256", "hecate_source": "@hc.func(\"c\")\ndef golden(x):\n    ...\n"}
```

`request.json` 提供实际 FX 图、公开模型结构、常量数值和来源、固定 input/output
layout 与 selector、Hecate 子集规则和响应 schema。**不包含规则 DSL 答案、
测试输入、reference 输出或密钥。** 当前 request 的常量 registry 和 layout
仍由确定性转换器准备；因此这是固定布局的受限合成任务，不是 Agent 自主选择
packing，也不是一个独立于规则前端的通用模型 importer。

模型文件仍是既有 data-only catalog JSON，不导入用户提供的任意 `model.py`。
request、实际权重、reference/inputs 和规则答案 hash 由宿主持有；任何修复
都不能更新这个基准。规则答案只供本地对照及 `--self-test` 构造回放，不交给 provider。

接口 `generate(request, feedback_history) -> raw JSON string` 接收深拷贝的公开
数据；不能拿到 compiler callback、宿主路径、参考数组或 key 对象。
当前唯一实现是 `ReplayProvider`；不动态导入 provider 插件、不执行 provider
shell 命令、不读取 API 凭据，也没有 HTTP 模型客户端。

JSON 拒绝重复 key、NaN/Infinity、超大输入；候选只允许经过静态检查的直线 SSA
函数、`+`、`*`、rotate(1/2)，至多 256 次运算。参数、常量、layout 或 reference
覆盖字段直接拒绝；仅 JSON schema 合法不能证明函数语义正确。

结构化输出限制数据通道，但仍会产生语义错误，所以保留独立校验和最小权限执行。
这是参考 [OpenAI Docs: Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
和 [Safety in building agents](https://developers.openai.com/api/docs/guides/agent-builder-safety)
采用的边界原则，并未因此绑定 OpenAI 服务或宣称获得安全证明。

## 隔离如何工作、仍不保证什么

- 使用已经安装的 `/usr/bin/bwrap` 0.6.1，不安装或修复桌面/网络环境。
- 每个子进程独立 user/mount/PID/network namespace，清空环境、丢弃 capabilities，
  启用 die-with-parent/new-session。没有宿主网络接口、环境凭据或继承的任意 fd。
- 只读挂载现有 Nix store、固定 venv、必要依赖和逐个允许的可信 worker 文件；
  不挂载宿主根、完整源码仓库或 results 父目录。
- tracing/编译只得到公开 payload 和该次输出目录，不挂载私钥或 reference。
  runtime 才只读挂载本次密钥，接收只含 inputs 的数组；reference 仍在宿主比较。
- 只有 `/out` 对应的当前尝试产物目录可写，`/tmp` 是隔离临时空间。文件大小上限
  16 MiB/文件，地址空间 4 GiB，有限 fd/CPU 时间；外层也有硬超时。
- 实际探针确认：frontend 加载成功、私有 sentinel/源码/keys 不可见、环境清空、
  network/PID namespace 改变、仅 loopback、请求只读。探针失败即停，不降级裸跑。

AST 解释器调用与原 frontend 相同的 Hecate operator，最终仍产生 Earth/CKKS/
HEVM/CST，再走原 SEAL_HEVM；它不是 FHE 数值模拟器。上游 `expr.py` 使用 Python
调用栈生成位置，故新 artifact 的源码位置指向可信解释器，不再精确对应候选
源文件行；候选源码和诊断另存，未来可增加节点级 source map。

这不是虚拟机，也不是针对 Linux kernel/native compiler 任意漏洞的形式化安全
证明。未做 seccomp、cgroup 总磁盘配额或多租户攻击审计；不要把本实验入口部署为
公网任意 Python 执行服务。现有 artifact gate 本身也不是恶意二进制完整验证器：
这里只接本机隔离 frontend/compiler 的有界输出，并在隔离进程中执行。

## 反馈和失败统计

最多首次生成加三次修复，共四个响应。每次保存原始 response、源码、stage flags、
编译产物、实际执行结果、逐项 comparison 和 feedback。schema/源码语法、静态检查、
tracing、compiler、artifact gate、runtime、数值比较和 integrity 分层报告。

确认的解析/静态/差分错误记为 candidate；不能确定根因的 native 阶段错误记为
pipeline_unclassified，不仅凭退出码归咎于 Agent。基础设施/完整性失败不继续
消耗修复轮数。共享 preflight 失败发生在候选轮次之前。

feedback 提供有界错误说明和 aggregate 数值指标，不提供 reference/actual 逐项
数组。frontend/compiler 日志片段最多 4000 bytes，标为 `untrusted_tool_output`，
不得作为高优先级指令或自动执行的命令。runtime/key 日志不发给 provider。

`agent_calls=0`、`llm_generation_validated=false`、`agent_success_rate=null`。
首次/最终通过及 repairs_used 只描述当前回放会话；不是 seen/unseen 或 48 例
Agent 实验。测试中的模拟对象仅验证编排，实际密态证据由独立 opt-in 测试读取。

## 复现

在 `D:\Code Space\Poseidon` 的 Windows PowerShell 中：

```powershell
# 只导出公开请求，不执行生成/密态计算
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_candidate.py --case scripts/baseline/cases/linear-example.json --prepare
# 实际故障注入 + 真实密态差分（无 LLM）
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_candidate.py --case scripts/baseline/cases/linear-example.json --self-test
# 回放保存的 JSON 字符串数组；更换模型/请求将拒绝不匹配的 request_id
wsl.exe -d Ubuntu-22.04 --cd '/mnt/d/Code Space/Poseidon' -- env PYTHONDONTWRITEBYTECODE=1 python3 scripts/baseline/run_candidate.py --case scripts/baseline/cases/linear-example.json --replay /home/lhy/poseidon-work/results/candidate-replay-7ommkr7q/replay-responses.json
```

`--prepare` 单独入口已验证：`candidate-replay-88jmi2ti`，状态为
`request_prepared_not_generated`，没有生成/密态执行成功声明。
单独候选回归命令（WSL 源码根）：

```bash
timeout -k 3s 60s env PYTHONDONTWRITEBYTECODE=1 \
  POSEIDON_CANDIDATE_RESULTS=/home/lhy/poseidon-work/results/candidate-replay-7ommkr7q \
  python3 -m unittest discover -s scripts/baseline -p test_candidate_pipeline.py -v
```

完整回归沿用 `fx-rule-translator.md` 的命令，额外增加上面的
`POSEIDON_CANDIDATE_RESULTS`，pattern 保持 `test_*.py`。Torch 单测仍通过
`run_model_batch.py --unit-tests` 执行。

每次产生新的 native `candidate-replay-*` 目录，不覆盖历史文件。外层 900 秒，
trace/compile 各 60 秒，runtime 150 秒，再有 kill grace。**不要分享 private-keys
或整个结果目录**；只选择公开 request、诊断与需要发布的数值报告。

首个失败记录 `candidate-replay-yrzra061` 保留：顶层 `import hecate` 顺带导入
`runner.py`，其默认参数调用 `Path.home()`，在无宿主 home 的沙箱失败。
修复是仅加载固定可信 `expr.py`，未开放用户主目录、未修改上游 submodule。
后来将 frontend 加载移到 preflight，避免依赖加载失败被计成候选修复。

## 下一门禁与本地提交边界

真实 provider 接入前需要确认服务商、模型固定版本、费用上限及本机凭据方式。
不在聊天中粘贴密钥。接口通过不等于已验证 provider 拒绝/截断/超时/用量计费；
这些适配器测试和真实小批次实验仍待实现。之后才能测首次/反馈后 Agent 正确率，
并与既有规则转换器在同一 backend/固定布局上比较。

本轮新增上述六个脚本/测试和本文档，更新 compiler 状态文档；原规则转换器、
人工 golden、上游 Dacapo 及 native runtime 源码均未修改。所有源码仍在唯一
checkout，产物在 native results。没有安装、commit、push 或 PR。
经后续批准可分为“候选合同与隔离前端”“执行反馈、测试与文档”两组本地提交。
