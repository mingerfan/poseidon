# R7：LLM 规划器（工具循环）→ M1 第 1 条

前置：R6b 已完成（`cost.estimate` 可用）。目标日期：2026-10-27。

## 0. 背景与约定

- **目标**：DeepSeek 读取模型图摘要，生成 PT-DSL 程序。每轮生成后，用内部工具（parse/typecheck、lowering、精确模式结构检查、代价模型、验证集上的近似精度）评估这份程序，把结构化反馈交回给 LLM。最多迭代 6 轮，冻结**最好的可行版本**，然后交给 R6 已有的 mock/count/emit/compile/test 流程。规则规划器作为对照基线。
- **公平性约束**：
  - Agent 能看到的只有训练集上的校准数据和验证数据；
  - 真实编译结果、CKKS 结果和测试集结果**永远不**进入循环；
  - 测试集只在冻结之后，通过 `--final` 评测。
- **已核实的 LLM 调用方式**：
  - 凭证用已批准的加载器 `baseline/agent_credentials.load_api_key(repo_root)` 读取（读的是仓库根目录的 `.env`，不要打印）；
  - 网络请求复用 `baseline/deepseek_http_worker.py`：以 `[sys.executable, "-I", "-B", worker]` 启动子进程，标准输入写入 `json.dumps({"api_key", "body", "timeout", "provider": "deepseek", "loopback_proxy_port": 0})`，标准输出返回 `b"<状态码>\n" + 原始响应`，标准错误中以 `DIAG ` 开头的行是诊断信息。具体写法照抄 `baseline/deepseek_provider.HTTPSTransport.post` 中调用 worker 的那一段，不经过它的 validate。
  - 之前在 Nix 中实测过：直连（端口 0）可用，模型 `deepseek-flash`、关闭思考（reasoning_effort="none"）。42k 个 prompt token 约 15 秒返回。
  - 开启思考时，32k 的 max_tokens 曾被推理过程耗尽而没有产出代码（M01、M02），所以默认关闭思考。
- **API 预算**：全项目共 300 次。每一次调用（无论成功与否）都要在 `$PT_RESULTS/llm_calls.jsonl` 中追加一行，达到 300 次后拒绝调用。单次 agent 运行最多 6 次调用。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/llm.py                      新：§2
pt_agent/planners/prompts.py         新：§3（SYSTEM_PROMPT 逐字照抄；初始消息和反馈消息的构造）
pt_agent/planners/agent_tools.py     新：§4（AgentContext、evaluate）
pt_agent/planners/agent.py           新：§5（迭代循环、可行性判断与选择）
pt_agent/run.py                      改：--planner agent（§6）
pt_agent/tests/test_r7_agent.py      新：§7（离线，用假的 LLM，不调用 API）
../docs/pt_agent/STATUS.md           改
```

## 2. llm.py

```python
class LLMError(Exception): ...           # 只带固定的错误码和诊断信息，绝不包含 key 或请求原文

@dataclass
class LLMClient:
    budget_path: Path                    # $PT_RESULTS/llm_calls.jsonl，由外层作为参数传入
    model: str = "deepseek-flash"
    reasoning_effort: str = "none"
    max_tokens: int = 16384
    timeout: float = 600
    max_total_calls: int = 300

    def chat(self, messages: list[dict], *, purpose: str) -> dict:
        """1. 读取 budget_path 的行数，≥ max_total_calls 时抛 LLMError("budget_exhausted")。
        2. body = generation_protocol.Config(model=..., reasoning_effort=..., max_tokens=..., timeout_seconds=timeout).settings()
           再加上 "messages"。settings 里已经包含 response_format=json_object 和 thinking。
        3. 用 load_api_key(REPO) 取得 key，按 §0 的方式调用 worker。
        4. 无论结果如何，先往 budget_path 追加一行：
           {"ts", "purpose", "model", "status", "seconds", "prompt_tokens", "completion_tokens", "finish_reason", "error"}，
           其中不包含任何内容文本。
        5. 状态码不为 200 或 worker 失败时，抛 LLMError(固定码)；
           成功时返回 dict(content, finish_reason, usage, id, seconds)，content 取 choices[0].message.content。"""
```

`REPO = Path(__file__).resolve().parents[4]`。

## 3. planners/prompts.py

### 3.1 SYSTEM_PROMPT（逐字写入一个三引号字符串，版本号 `PROMPT_VERSION = "r7-v1"`）

````text
You are an FHE compiler engineer. You write PT-DSL programs that implement a given neural network on CKKS
homomorphic encryption. A trusted lowering turns PT-DSL into Hecate code; an unmodified compiler then inserts
rescale/modswitch/bootstrap. Your program must compute exactly the given network (except for the polynomial
activation approximations you choose) and should minimize the predicted encrypted latency.

PT-DSL syntax (a restricted Python subset; it is parsed, never executed):
- One entry function decorated with @fhe.program(slots=16384, refresh="compiler", placement="depth-dp").
  Parameters are annotated as Cipher[d0, d1, ...] with the network input shape.
- Statements: `name = op(inputs..., attr=value, ...)`, `return name`, and `with region("name", level_budget=k):`.
- Optional macros: top-level `def name(params): ... return t` without decorator; call them like ops.
  Macro parameters may receive tensors, numbers, strings and tuples of numbers. Use W[p + ".conv1.weight"] for names.
- No for/if/while, no arithmetic expressions, no nested calls. Names must not contain "__".
- Weights are referenced as W["<state_dict name>"]. Batch-norm parameters are referenced by their prefix.

Operators supported by the v1 lowering (anything else is rejected):
- pack(x, layout="mux", gap=1) for an image input [1,C,H,W]; pack(x, layout="flat") for a vector input [1,F].
  Every input must be packed exactly once before use.
- conv2d(x, w=W[...], b=W[...] or omitted, bn="<prefix>" or omitted, stride=1|2, pad=(k-1)//2, algo="mpcb")
  odd square kernel, dilation 1, groups 1. A BatchNorm that directly follows a conv and has no other consumer
  must be fused through bn="<prefix>" (it uses <prefix>.weight/.bias/.running_mean/.running_var). Stand-alone
  batchnorm is not supported.
- act(x, fn="relu"|"silu"|"gelu"|"sigmoid"|"tanh"|"exp", method="cheb", deg=D, domain=(lo, hi))
  Chebyshev approximation of degree D (1..255) on [lo, hi]. Every input value MUST lie inside [lo, hi]:
  outside the domain the polynomial explodes and the whole network diverges. It costs
  ceil(log2(D+1)) + 2 multiplicative levels. Higher D is more accurate and more expensive; a wider domain is
  safer and less accurate at the same D. You may choose D and domain separately for every act.
- add(a, b): element-wise, same shape and same layout.
- Downsampling shortcut: exactly the chain
  s = slice(x, axis=2, start=0, end=None, step=2); s = slice(s, axis=3, start=0, end=None, step=2);
  s = pad(s, pads=(0, 0, 0, 0, p, p)). The three statements must be consecutive in data flow.
- mean(x, axis=(2, 3)) followed by relayout(z, layout="flat") and then linear(z, w=W[...], b=W[...], algo="bsgs"|"diag").

Cost drivers: activations dominate the multiplicative depth and therefore the number of bootstraps;
rotations come from convolutions and pooling. The feedback reports predicted latency, bootstraps and depth.

Feasibility (all required): the program parses and type-checks; the exact-mode check matches the reference
network; on the validation images no activation input leaves its domain (max |u| <= 0.95, where u is the input
mapped to [-1, 1]) and the predicted class agrees with the exact network for at least 99% of the images.
Among feasible programs, lower predicted latency is better.

Answer with one JSON object: {"program": "<complete PT-DSL source>", "rationale": "<at most 80 words>"}.
Always return the complete program, never a diff.

Example (a different, small network; it shows the syntax, a macro, the shortcut chain and the head):
@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def tinynet(img: Cipher[1, 3, 16, 16]):
    x = pack(img, layout="mux", gap=1)
    h = conv2d(x, w=W["stem.weight"], bn="stem_bn", stride=1, pad=1, algo="mpcb")
    h = act(h, fn="relu", method="cheb", deg=63, domain=(-6.0, 9.0))
    h = down(h, "block1", 4, (-5.0, 7.0), (-6.5, 10.0))
    z = mean(h, axis=(2, 3))
    z = relayout(z, layout="flat")
    out = linear(z, w=W["head.weight"], b=W["head.bias"], algo="bsgs")
    return out


def down(x, p, cpad, d1, d2):
    t = conv2d(x, w=W[p + ".conv1.weight"], bn=p + ".bn1", stride=2, pad=1, algo="mpcb")
    t = act(t, fn="relu", method="cheb", deg=63, domain=d1)
    t = conv2d(t, w=W[p + ".conv2.weight"], bn=p + ".bn2", stride=1, pad=1, algo="mpcb")
    s = slice(x, axis=2, start=0, end=None, step=2)
    s = slice(s, axis=3, start=0, end=None, step=2)
    s = pad(s, pads=(0, 0, 0, 0, cpad, cpad))
    t = add(t, s)
    t = act(t, fn="relu", method="cheb", deg=31, domain=d2)
    return t
````

### 3.2 `initial_message(ctx) -> str`

返回一个 JSON 字符串（用 `json.dumps(..., separators=(",", ":"))`）：

```json
{"task": "Write the PT-DSL program for this network.",
 "network": "<graph.name>",
 "input": {"name": "x", "shape": [1, 3, 32, 32]},
 "nodes": [ ... ],
 "output": "<图输出张量名>"}
```

`nodes` 按 graph.nodes 的顺序，每项包含 `id`、`op`、`in`（输入张量名列表）、`out`、`shape`（输出形状），以及按 op 补充的字段：

| op | 补充的字段 |
|---|---|
| conv2d | `weight`、`bias`、`stride`、`padding` |
| batchnorm | `prefix`（BN 权重名去掉 ".weight" 后的部分）、`eps` |
| act | `fn`、`calib=[min, max]`（来自 1000 张校准图）、`needed_margin`（精确网络在验证集上需要的余量，见 §4） |
| slice | `axis`、`start`、`end`、`step` |
| pad | `pads` |
| mean | `axes` |
| linear | `weight`、`bias` |

### 3.3 `feedback_message(fb, iteration, best) -> str`

返回 `json.dumps({"iteration": k, "feedback": fb, "best_feasible_so_far": best或null, "next": 指令})`。指令分三种情况：

- 有错误时：`"Fix the errors and return the complete program."`；
- 可行时：`"The program is feasible. Reduce predicted latency (for example lower degrees or tighter but still safe domains where the data allows) while keeping all feasibility conditions. Return the complete program."`；
- 可行性条件不满足时：`"The program is not feasible. Fix the violated conditions and return the complete program."`。

## 4. planners/agent_tools.py

```python
@dataclass
class AgentContext:
    graph: Graph
    weights: dict
    calibration: dict
    weight_shapes: dict
    val_images: np.ndarray        # validation_set(500) 的图
    val_exact_logits: np.ndarray  # reference_forward 在这 500 张上的输出
    probe_images: np.ndarray      # val_images[:4]
    probe_act_inputs: list        # 参考网络中，各 act 节点（按节点顺序）的输入张量在 4 张探针图上的值
    needed_margin: dict           # 每个 act 输入张量在验证集精确网络上需要的余量（计算方法同 R6c §1.2 第 1 步）
    act_tensors: list             # 各 act 节点的输入张量名，按节点顺序

def build_context(graph, calibration, n_val=500) -> AgentContext

def evaluate(source: str, ctx: AgentContext) -> dict:
```

**evaluate 按顺序执行，每一步都把结果写进返回的 feedback 字典；某一步失败时，后续步骤不再执行：**

1. **parse**：抛出 PTDSLError 时，记 `{"stage": "parse", "error": str(e)}`。
2. **typecheck**：用 `weight_shapes`。有 error 时，记 `{"stage": "typecheck", "errors": [str(d) …]}`，最多 20 条。
3. **exact**：
   - 在 `MockBackend(16384, batch=4)` 上用 `act_mode="exact"` 执行 lower，observer 按顺序收集每个 act 的输入，并解包成张量（mux 布局用 unpack_mux，flat 布局用 unpack_flat）；
   - lower 抛 LoweringError 时记 `{"stage": "lower", "error": str(e)}`；
   - 依次比较：act 的个数是否与 `act_tensors` 一致；第 k 个 act 输入与 `probe_act_inputs[k]` 的最大差；最终 logits 与参考网络的最大差；
   - 记为 `{"ok": bool, "logit_max_diff": …, "first_mismatch": {"act_index", "graph_tensor", "program_line", "max_diff"} 或 null, "act_count": [程序中的个数, 网络中的个数]}`；
   - ok 的条件是：个数相等、每个 act 输入的差都 ≤1e-6、logit 差 ≤1e-9。
4. **cost**：调用 `cost.estimate(program, weights)`，取 `latency_pred_us`、`boots_pred`、`max_depth`、`ops_total`、`rot_keys`，记入 cost。
5. **validation**：在 500 张验证图上，用 `MockBackend(batch=50)` 跑近似模式。
   - observer 对每个 act 记录 `max_u`（只统计有效槽）。
   - 某一批抛 BackendError 时，记 `diverged=True` 和 `images_done`。
   - 记为 `{"agree": …, "total": 500, "diverged": bool, "worst_acts": [ {"program_line", "graph_tensor", "max_u", "deg", "domain"} …前 5 个 ], "max_u": …}`。
6. `feasible = exact.ok and not diverged and max_u <= 0.95 and agree >= 0.99*total`。
7. 返回字典的形状：`{"stage": "complete" 或失败阶段, "feasible": bool, "exact": …, "cost": …, "validation": …, "error(s)": …}`。

**program_line 的取法**：第 k 个 act 对应 `program.ops` 中第 k 个 kind 为 act 的 op 的 lineno。宏展开后的行号是宏体内的行号，这一点在 feedback 里照实给出。

## 5. planners/agent.py

```python
def run_agent(ctx: AgentContext, client, *, max_iters: int = 6, trace_path: Path) -> dict:
```

- 消息列表的开头是 `[system=SYSTEM_PROMPT, user=initial_message(ctx)]`。
- **每一轮**：
  - 调用 `client.chat(messages, purpose=f"agent-iter-{i}")`。
  - 把返回的 content 用 `json.loads` 解析，要求包含字符串字段 "program"。解析失败时，fb 记为 `{"stage": "response", "error": "response_not_json_with_program"}`。
  - 解析成功则执行 `fb = evaluate(program, ctx)`。
  - 往 messages 中追加 `assistant=content` 和 `user=feedback_message(...)`。
  - 往 trace_path 中写一行 JSON：iteration、完整的 messages 中本轮新增的两条、fb、usage 和 seconds。
- **维护 best**：所有可行轮次中 `latency_pred_us` 最小的那一轮。
- **停止条件**：达到 max_iters；或者已有可行版本，且最近两轮都没能把 best 的延迟降低 ≥5%；或者调用抛出 LLMError（记录后停止）。
- **返回**：`dict(status="feasible"|"infeasible"|"error", best_program, best_iteration, first_feasible_iteration, iterations, llm_calls, prompt_tokens, completion_tokens, seconds, history=[每轮的 stage、feasible、latency、boots、agree、max_u])`。
- 没有可行版本时，`best_program` 为 None。

## 6. run.py 的改动

- 新增参数：`--planner agent`、`--agent-iters 6`、`--llm-model deepseek-flash`、`--llm-reasoning none`。
- 外层（系统 python3）要求 `$PT_RESULTS` 存在，并把 `--budget-path $PT_RESULTS/llm_calls.jsonl` 传给内层。
- **plan 阶段**（planner=agent 时）：
  - `ctx = build_context(graph, calibration)`；
  - `result = run_agent(ctx, LLMClient(budget_path, …), trace_path=out/"agent_trace.jsonl")`；
  - 状态为 feasible 时，把 best_program 写为 `program.ptdsl`（这就是冻结版本），后续阶段与 rule 规划器完全相同；
  - 不可行时，REPORT 中写明，后续阶段不再执行。
  - REPORT 中新增 `agent` 字段，内容是 run_agent 的返回值（去掉 history 中的长文本）。
- registry 的 mode 为 "agent"，metrics 增加 llm_calls、prompt_tokens、completion_tokens、best_iteration 和 latency_pred_us。

## 7. 测试（离线，test_r7_agent.py，nixrun）

用 `FakeClient(responses: list[str])` 代替 LLMClient：按顺序返回预设的 content，并像真实客户端一样往一个临时的 budget 文件里写行。

1. **build_context**：对 R3 的 graph 和 calibration 构造上下文，n_val 取 50（为了快）。act_tensors 共 19 个；`needed_margin` 的每个值都 >0。
2. **evaluate**：
   - 对 `rule.plan(graph, cal, deg=127, margin=1.5)` 的文本：stage 为 complete，exact.ok 为真，cost.boots_pred=461，validation.diverged 为假，feasible 为真；
   - 对一份把第 3 个 act 的定义域缩成 (−0.5, 0.5) 的程序：feasible 为假，worst_acts[0] 指向这个 act；
   - 对一份把某个 conv 的 stride 改错的程序：exact.ok 为假，或者 typecheck 报出形状错误；
   - 对一份有语法错误的文本：stage 为 parse。
3. **循环逻辑**，响应依次为：
   1. 不是 JSON；
   2. 程序有语法错误；
   3. rule(127, 1.5) 的程序；
   4. rule(63, 1.4) 的程序；
   5. 再次返回 rule(63, 1.4) 的程序；
   6. 再次返回 rule(63, 1.4) 的程序；

   在 max_iters=8 下，期望：
   - status 为 feasible；
   - first_feasible_iteration=3；
   - best_iteration=4（因为 63 阶的延迟更低）；
   - 第 5、6 两轮都没能把 best 的延迟降低 ≥5%，所以第 6 轮后停止，llm_calls=6；
   - trace 文件共 6 行；
   - 每一轮发给 LLM 的反馈消息都是合法的 JSON。

4. **预算**：budget 文件中已有 300 行时，`LLMClient.chat` 抛 LLMError("budget_exhausted")，且不启动 worker。用 monkeypatch 把 subprocess.run 换成一旦被调用就失败的函数来验证。
5. SYSTEM_PROMPT 中的示例程序能通过 parse；R1–R6b 的测试依然全部通过。

**测试中不允许发起真实的 API 调用。**

## 8. 真实运行（完成标准）

1. 先做一次真实连通性检查，消耗 1 次调用：`python3 -m pt_agent.llm --smoke`，发送 `[system: "Reply with JSON {\"ok\": true}", user: "ping"]`，确认状态码为 200 且能解析出 JSON，结果写入 registry，round="R7"、mode="llm-smoke"。如果失败（TLS、网络或 key），照实记录并停下，不做别的尝试。
2. 然后运行：
   ```bash
   python3 -m pt_agent.run --model resnet20 --planner agent --stages plan,check,mock,count,emit,compile,test --final
   ```
   这一步最多消耗 6 次调用。
3. 汇报内容：
   - 每轮的 stage、feasible、latency_pred、boots_pred、agree 和 max_u；
   - best 和 first feasible 分别是第几轮；
   - 冻结程序的行数；
   - 与规则基线（R6c：127/1.5，boots 383，top-1 91.74%）的对照：每个 act 选的 deg 和 domain、cost、校准集 agree、编译耗时和 boots、top1_test、m1_item3_pass；
   - token 用量和总耗时。
4. STATUS.md 新增 R7 一行，M1 第 1 条按结果更新（冻结程序原样通过第 2–4 条时，记为通过）。

## 9. 不要做

- 不把编译结果、CKKS 结果或测试集结果放进反馈；不修改 SYSTEM_PROMPT 的文字。如果发现问题，记进"阻塞"栏，等待新 spec。
- 失败的轮次不手工修改程序。不在单次运行中超过 6 次调用；R7 总共不超过 1 次 smoke 加 6 次调用。
- 不修改 kernel、lowering、规则规划器、代价模型和 baseline，也不打印凭证。
