# R7c：让反馈能定位到问题、补充提示词，并对比思考模式开与关

## 0. 诊断（Claude 于 2026-10-21 阅读 agent_trace.jsonl 后得出）

- **第 1 轮的程序几乎完全正确**：宏、BN 融合、下采样链、mean → relayout → linear 都对。唯一的错误是 `down(h, "layer2.0", 16, …)` 和 `down(h, "layer3.0", 32, …)`：cpad 应该是 8 和 16（与图中 pad 节点的 `pads=[0,0,0,0,8,8]` 一致），LLM 却填了输入通道数，导致通道变成 48 和 96，与主路径不相等。
- **后 5 轮修不好，原因有三**：
  1. 反馈只有 `add requires equal tensor shapes`，没有给出两侧的形状；
  2. 行号 37 在 `down` 宏体内，而这个宏被调用了两次，LLM 不知道是哪一次调用出的错；
  3. 提示词没有说明 pads 采用 `F.pad` 的顺序（从最后一维往前），LLM 误以为 `(0,0,0,0,p,p)` 补的是 H 和 W，于是断定"无法扩展通道"，到第 6 轮甚至删掉了下采样。
- **结论**：主要问题出在反馈和提示词。思考模式是否有帮助，由本轮的对照实验回答。

## 1. 修改

### 1.1 形状报错带上输入形状：`pt_agent/dsl/typecheck.py`

在把 ShapeError 转成 E_SHAPE diag 的地方，消息统一改为 `f"{原消息}; input shapes: {[各输入形状]}"`，其中形状写成 tuple 形式，例如 `(1, 32, 16, 16)`。WeightError 的消息保持不变，因为它已经包含了期望形状和实际形状。

### 1.2 宏调用处的行号：`pt_agent/dsl/ir.py`、`parser.py`、`typecheck.py`、`errors.py`

- `Op` 新增字段 `call_lines: tuple[int, ...] = ()`。parser 展开宏时维护一个调用栈：每进入一次宏调用，就把这条调用语句的 lineno 压栈；在宏体内生成的 op，`call_lines` 记录当时调用栈的完整 tuple（最外层在前）。
- `canonical()` 不包含 call_lines 和 lineno。R1 的往返测试必须仍然通过。
- `Diag` 新增字段 `call_lines: tuple[int, ...] = ()`，typecheck 从对应的 op 中取值。`str(diag)` 在 call_lines 不为空时，把 `line N` 写成 `line N (in macro <最内层宏名> called at line <最内层调用行>)`。R1 测试中对 `str(diag)` 的断言如果因此受影响，按新格式同步更新。

### 1.3 结构化错误：`pt_agent/planners/agent_tools.py`

typecheck 阶段的 `errors` 改为对象列表：

```json
{"code": "E_SHAPE", "line": 37, "line_text": "<该行源码，去掉首尾空白>",
 "call_lines": [10], "call_texts": ["<第 10 行源码>"], "message": "<diag.msg>"}
```

lower 阶段的错误也采用同样的结构：通过 LoweringError 的 op_id 找到 `program.ops[op_id]`，取它的 lineno 和 call_lines。另外保留原来的字符串字段 `error`。

### 1.4 提示词升级为 `PROMPT_VERSION = "r7-v2"`：`pt_agent/planners/prompts.py`

SYSTEM_PROMPT 在 r7-v1 的基础上做**两处插入**，其余文字一字不改：

（a）在 "Downsampling shortcut" 那一条的末尾（"…must be consecutive in data flow." 之后）插入：

```text
  pads follows torch.nn.functional.pad: pairs are listed from the LAST dimension backwards, so
  (0, 0, 0, 0, p, p) leaves W and H unchanged and pads the CHANNEL axis by p zeros on each side
  (C -> C + 2p). Copy slice and pad attributes exactly from the corresponding nodes in the input.
```

（b）在 "Answer with one JSON object" 那一段之前插入一段：

```text
When feedback reports an error, it gives the line, the source text, the macro call site and the shapes
involved. Change only what the feedback points at and keep the rest of the program unchanged; do not
restructure the network or drop layers to make an error disappear.
```

## 2. 测试

1. `test_r1_*` 全部通过（如果 `str(diag)` 的格式断言受影响，就同步更新）。
2. 新增 `tests/test_r7c_feedback.py`：
   - 用 R7 真实运行的第 1 轮程序：从 `/home/lhohy/pt_agent_results/run-resnet20-agent-20261021T031807/agent_trace.jsonl` 第 1 行的 assistant content 中解析出 program。在 context 中 n_val 取 50，执行 evaluate，断言：
     - typecheck 的 errors[0] 中 `call_lines` 不为空；
     - `call_texts[0]` 包含 `down(h, "layer2.0"`；
     - message 中同时出现 `(1, 32, 16, 16)` 和 `(1, 48, 16, 16)`。
   - 把这份程序中的两个 cpad 改成 8 和 16 后再执行 evaluate，断言 stage 越过 typecheck，即不等于 "typecheck"。
3. SYSTEM_PROMPT 中包含 "C -> C + 2p"，PROMPT_VERSION 为 "r7-v2"；示例程序仍能通过 parse。
4. R2–R7b 的测试全部通过。

## 3. 真实运行（共 2 次 agent，每次 ≤6 次调用，合计 ≤12 次）

都使用 `--stages plan,check,mock,count,emit,compile,test --final`：

1. **A**：`--llm-reasoning none`（max_tokens 保持 16384）。
2. **B**：`--llm-reasoning high --llm-max-tokens 32768`。需要在 run.py 新增 `--llm-max-tokens` 参数，并传给 LLMClient。这个组合会开启思考模式。

A、B 都要跑，A 的结果不影响 B 是否执行。任何一次不可行都照实记录，不追加调用。

## 4. 汇报

- A、B 分别给出：每轮的 stage、feasible、latency_pred、boots_pred、agree 和 max_u；first_feasible 和 best 分别是第几轮；冻结程序的行数；每个 act 的 deg 和 domain；编译耗时和 boots；top1_test；token 用量（prompt、completion）和总耗时。
- A 与 B、规则基线（R6c）三者的对照表。
- STATUS.md：R7 按结果改为完成或保持阻塞，新增 R7c 一行。M1 第 1 条：只要 A、B 中有任意一个冻结程序原样通过 M1 第 2–4 条，就记为通过，并注明是哪一次运行。
