# R8b：通用化（下）——能力注册表、模式标注、示例库、通用提示词，在 4 个开发模型上运行 agent

前置：R8a 已完成（4 个开发模型可用，规则基线已有结果）。目标日期：2026-10-26。

> **2026-10-21 调整**：本轮只实现 §1–§5 并完成 §7 的离线测试。§6 的真实运行推迟到 R8c，使用 r8-v2 提示词一并执行，以免重复消耗 API。所以 §9 的完成标准改为"§7 全部通过"，冻结决定也移到 R8c 之后。

## 0. 背景与已定的决策

- **用户在 R7 后指出**：r7 的提示词太针对 ResNet。它的算子列表是按 ResNet 的写法组织的，里面写死了 ResNet 专用的模式，示例也是一个缩小版的 ResNet。本轮把提示词拆成三部分：
  1. 通用指令：和模型无关；
  2. 能力文档：由 kernel/lowering 的注册表自动生成；
  3. 示例：从示例库中按结构相似度选取，不包含目标模型。
  
  另外，模式提示写进输入数据，不写进提示词。
- **R7c 的结论**：关闭思考模式时 6 轮都不可行，开启后第 2 轮就可行。所以本轮默认 `reasoning_effort="high"`、`max_tokens=32768`，并沿用 R7c 的结构化反馈（行号、源码、宏调用处、形状）。
- **安全条件收紧**：R7c 冻结的程序在验证集上 max_u=0.93，测试集上有 2/10000 张图发散。本轮把可行性条件改为 `max_u ≤ 0.90`，验证集扩大到 1000 张。
- **冻结**：本轮结束后，由用户决定是否冻结。冻结后，提示词、注册表、示例库和反馈格式都不再修改，然后才能评测未见模型（R10）。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/capabilities.py               新：§2 注册表与 render_capabilities()
pt_agent/lower.py                      改：新增常量 HANDLED_OPS（与各处理分支一致）
pt_agent/intake/patterns.py            新：§3 annotate(graph)
pt_agent/planners/examples/            新：3 个 .ptdsl，以及 index.json（§4）
pt_agent/planners/prompts.py           改：§5 提示词 r8-v1 的模板与构造函数，initial_message 带上模式标注
pt_agent/planners/agent_tools.py       改：FEASIBLE_MAX_U=0.90，build_context 的 n_val 默认为 1000，记录 model_name
pt_agent/planners/agent.py             改：用 build_system_prompt(ctx) 代替固定的 SYSTEM_PROMPT
pt_agent/run.py                        改：--llm-reasoning 默认 high，--llm-max-tokens 默认 32768
pt_agent/experiments/r8b_dev.py        新：§6
pt_agent/tests/test_r8b_*.py           新：§7
../docs/pt_agent/STATUS.md             改
```

## 2. capabilities.py（文字逐字照抄）

```python
@dataclass(frozen=True)
class Capability:
    op: str
    signature: str
    semantics: str
    constraints: tuple[str, ...]
    cost: str
    in_prompt: bool = True

@dataclass(frozen=True)
class Pattern:
    name: str
    graph_ops: tuple[str, ...]
    description: str
    dsl_form: str

CAPABILITIES = (...)    # 内容见下
PATTERNS = (...)
def render_capabilities() -> str
```

**CAPABILITIES**：

| op | signature | semantics | constraints | cost | in_prompt |
|---|---|---|---|---|---|
| pack | `pack(x, layout="mux", gap=1)` / `pack(x, layout="flat")` | Encrypts a network input into one ciphertext. Use layout="mux" for an image input of shape [1,C,H,W] and layout="flat" for a vector input of shape [1,F]. | Every input is packed exactly once, before any other use. | No homomorphic operations. | 是 |
| conv2d | `conv2d(x, w=W[...], b=W[...] (optional), bn="<prefix>" (optional), stride=S, pad=P, algo="mpcb")` | 2-D convolution on a mux-packed tensor; with bn="<prefix>" the following batch normalization (<prefix>.weight/.bias/.running_mean/.running_var) is folded into it. | Odd square kernel; pad equals (kernel-1)//2; stride 1 or 2; dilation 1; groups 1. Stand-alone batch normalization is not supported, so a batch normalization must be folded into the convolution that feeds it. | 2 multiplicative levels; rotations grow with kernel size and channel count. | 是 |
| act | `act(x, fn=F, method="cheb", deg=D, domain=(lo, hi))` | Element-wise activation F (relu, silu, gelu, sigmoid, tanh or exp) approximated by a Chebyshev polynomial of degree D on [lo, hi]. | 1 <= D <= 255. Every input value must lie inside [lo, hi]; outside the domain the polynomial grows without bound and the whole network diverges. D and domain may differ for every act. | ceil(log2(D+1)) + 2 multiplicative levels; higher D is more accurate and more expensive; a wider domain is safer and less accurate at the same D. | 是 |
| add | `add(a, b)` | Element-wise sum. | Same shape and same layout. | No multiplicative level. | 是 |
| slice | `slice(x, axis=A, start=S, end=E, step=K)` | Strided slice along one axis. | Only allowed inside the downsample_pad pattern. | Part of the pattern. | 是 |
| pad | `pad(x, pads=(...))` | Zero padding. pads follows torch.nn.functional.pad: pairs are listed from the last dimension backwards, so (0, 0, 0, 0, p, p) keeps W and H and pads the channel axis by p zeros on each side (C -> C + 2p). | Only allowed inside the downsample_pad pattern. | Part of the pattern. | 是 |
| mean | `mean(x, axis=(2, 3))` | Average over the spatial axes of a mux-packed tensor; the result is a flat vector. | Must be followed by relayout(..., layout="flat"). | 1 multiplicative level. | 是 |
| relayout | `relayout(x, layout="flat")` | Marks a flat vector produced by mean. | Only directly after mean. | No homomorphic operations. | 是 |
| linear | `linear(x, w=W[...], b=W[...] (optional), algo="bsgs" or "diag")` | Matrix-vector product on a flat vector. | Input is flat (a packed vector input, the result of relayout, or the result of another flat operation). | 2 multiplicative levels; "bsgs" uses fewer rotations than "diag". | 是 |
| refresh | `refresh(x)` | Explicit bootstrapping hint. | With refresh="compiler" it is ignored. | — | **否** |

**PATTERNS**：

| name | graph_ops | description | dsl_form |
|---|---|---|---|
| conv_bn | `("conv2d", "batchnorm")` | A convolution whose only consumer is a batch normalization. | one statement: `conv2d(x, w=W["<conv weight>"], bn="<bn prefix>", stride=S, pad=P, algo="mpcb")` |
| downsample_pad | `("slice", "slice", "pad")` | Spatial subsampling by 2 followed by channel zero padding (a residual shortcut). | three consecutive statements: `s = slice(x, axis=2, start=0, end=None, step=2)`; `s = slice(s, axis=3, start=0, end=None, step=2)`; `s = pad(s, pads=<pads copied from the node>)` |
| mean_head | `("mean", "linear")` | Global spatial average feeding a linear layer. | `z = mean(x, axis=(2, 3))`; `z = relayout(z, layout="flat")`; `y = linear(z, ...)` |

**render_capabilities() 的输出格式**（只渲染 in_prompt=True 的条目，按表格顺序排列）：

```
Operators supported by the lowering (generated from its capability registry; anything else is rejected):
- <signature>
  <semantics> Constraints: <各条 constraints 用一个空格连接>. Cost: <cost>
...
Lowering patterns. Groups of input nodes carrying "pattern": "<name>" must be written in this form:
- <name>: <description> Write: <dsl_form>
...
```

## 3. intake/patterns.py

```python
def annotate(graph: Graph) -> list[dict]:
    """返回 [{"pattern": name, "group": k, "nodes": [node_id…]}]，按第一个节点在 graph.nodes 中的顺序编号，k 从 0 开始。"""
```

匹配规则与 lowering 和规则规划器一致：

- **conv_bn**：conv2d 节点唯一的消费者是 batchnorm。
- **downsample_pad**：一个 slice，axis=2、start=0、step=2，end 等于输入的 H；它唯一的消费者是 axis=3、start=0、step=2、end 等于 W 的 slice；后者唯一的消费者是 pads 形如 `(0,0,0,0,p,q)` 的 pad。
- **mean_head**：mean，axes=(2,3)、keepdim=False，且至少有一个消费者是 linear。组中包含这个 mean 和对应的 linear。

## 4. 示例库 planners/examples/

三个文件，内容逐字照抄。`index.json` 的格式为 `[{"name", "file", "source_model": "synthetic", "features": [...], "weight_shapes": {...}}]`。

**mlp_flat.ptdsl**，features 为 `["linear", "act", "flat_input"]`，weight_shapes 为 `{"l1.weight": [64, 512], "l1.bias": [64], "l2.weight": [10, 64], "l2.bias": [10]}`：

```python
@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def smallmlp(v: Cipher[1, 512]):
    x = pack(v, layout="flat")
    h = linear(x, w=W["l1.weight"], b=W["l1.bias"], algo="bsgs")
    h = act(h, fn="relu", method="cheb", deg=31, domain=(-4.0, 6.0))
    out = linear(h, w=W["l2.weight"], b=W["l2.bias"], algo="diag")
    return out
```

**cnn_plain_small.ptdsl**，features 为 `["conv2d", "batchnorm", "act", "mean", "linear", "conv_bn", "mean_head"]`，weight_shapes 中 c1.weight 为 [8,3,3,3]，c2.weight 为 [16,8,3,3]，n1.* 均为 [8]，n2.* 均为 [16]，fc.weight 为 [10,16]，fc.bias 为 [10]：

```python
@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def plaincnn(img: Cipher[1, 3, 16, 16]):
    x = pack(img, layout="mux", gap=1)
    h = conv2d(x, w=W["c1.weight"], bn="n1", stride=1, pad=1, algo="mpcb")
    h = act(h, fn="silu", method="cheb", deg=63, domain=(-5.0, 7.5))
    h = conv2d(h, w=W["c2.weight"], bn="n2", stride=2, pad=1, algo="mpcb")
    h = act(h, fn="silu", method="cheb", deg=31, domain=(-6.0, 8.0))
    z = mean(h, axis=(2, 3))
    z = relayout(z, layout="flat")
    out = linear(z, w=W["fc.weight"], b=W["fc.bias"], algo="bsgs")
    return out
```

**residual_small.ptdsl**，features 为 `["conv2d", "batchnorm", "act", "add", "slice", "pad", "mean", "linear", "conv_bn", "downsample_pad", "mean_head"]`。内容就是 r7-v2 中的 tinynet 示例，原样搬过来；weight_shapes 与 Claude 验证该示例时所用的形状一致：stem.weight 为 [8,3,3,3]；stem_bn.* 均为 [8]；block1.conv1.weight 为 [16,8,3,3]；block1.conv2.weight 为 [16,16,3,3]；block1.bn1.* 和 block1.bn2.* 均为 [16]；head.weight 为 [10,16]；head.bias 为 [10]。

**选择函数**（写在 prompts.py 中）：

```python
def target_features(graph, patterns) -> set:
    """graph 中出现的节点 op 集合 ∪ 模式名集合 ∪（输入 rank 为 2 时）{"flat_input"}。"""

def select_examples(features: set, model_name: str, k: int = 2) -> list[str]:
    """排除 source_model == model_name 的示例；按 Jaccard(features, ex.features) 从高到低排序，
    分数相同时按 name 的字母顺序；取前 k 个分数 > 0 的示例，返回它们的 name。"""
```

## 5. 提示词 r8-v1（prompts.py）

```python
PROMPT_VERSION = "r8-v1"
TEMPLATE = """…（见下方，逐字照抄）…"""
def build_system_prompt(ctx) -> str:
    """TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U, n_val=len(ctx.val_images),
                       examples=<所选示例，每个前面写一行 "# example: <name>"，示例之间空一行>)"""
```

**TEMPLATE**：

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
  Macro parameters may receive tensors, numbers, strings and tuples of numbers. Weight names can be built as W[p + ".suffix"].
- No for/if/while, no arithmetic expressions, no nested calls. Names must not contain "__".
- Weights are referenced as W["<name>"] using the weight names given in the input.

{capabilities}

The input lists every network node in order with its shapes, weights and attributes. Every act node also lists
calib = [min, max] observed on calibration images and needed_margin = the factor by which the exact network
exceeds calib on validation images; a domain must cover calib widened by needed_margin, plus room for the error
of the approximations before it.

Cost drivers: activations dominate the multiplicative depth and therefore the number of bootstraps;
rotations come from convolutions, pooling and linear layers. The feedback reports predicted latency,
bootstraps and depth.

Feasibility (all required): the program parses and type-checks; the exact-mode check matches the reference
network; on the {n_val} validation images no activation input leaves its domain (max |u| <= {max_u}, where u is
the input mapped to [-1, 1]) and the predicted class agrees with the exact network for at least 99% of the images.
Among feasible programs, lower predicted latency is better.

When feedback reports an error, it gives the line, the source text, the macro call site and the shapes
involved. Change only what the feedback points at and keep the rest of the program unchanged; do not
restructure the network or drop layers to make an error disappear.

Answer with one JSON object: {{"program": "<complete PT-DSL source>", "rationale": "<at most 80 words>"}}.
Always return the complete program, never a diff.

Examples (other, small networks; they show the syntax and the patterns):
{examples}
````

**initial_message 的改动**：nodes 中属于某个模式组的节点，增加 `"pattern": name` 和 `"group": k` 两个字段；顶层增加 `"patterns": annotate(graph)` 和 `"model": ctx.model_name`。其余字段与 R7c 相同。

## 6. experiments/r8b_dev.py

运行模式同前：外层和内层分开，registry 写 round="R8b"。

依次对 4 个开发模型 resnet20、resnet20_silu、cnn_plain、mlp 运行：

```
pt_agent.run --model <m> --planner agent --stages plan,check,mock,count,emit,compile,test --final
```

每个模型最多 6 次 LLM 调用，4 个模型合计不超过 24 次。

- 每个模型记录：status、first_feasible 和 best 分别是第几轮、每轮的 stage/feasible/latency/boots/agree/max_u、冻结程序的行数、编译的耗时和 boots、top1_test、n_diverged、token 用量和耗时。另外，把所选示例的名字也记录下来。
- 从 R8a 的 REPORT 中读取同一模型的规则基线结果（路径由外层作为参数传入），列一张对照表：

  | 模型 | 规划器 | 行数 | boots | 预测延迟 | top1_test | n_diverged |
  |---|---|---|---|---|---|---|

- 某个模型不可行时，照实记录，然后继续下一个模型，不追加调用。

## 7. 测试（离线；用 nixrun；不允许真实调用 API）

1. **注册表与 lowering 一致**：`{c.op for c in CAPABILITIES} == lower.HANDLED_OPS`。
2. **提示词中没有模型专用的写法**（2026-10-21 修订：原条款对整份提示词做禁词检查，与 §4 原样复制的 residual 示例、§7.4 的示例选择相冲突。`conv1.weight`、`fc.weight` 是 PyTorch 的通用命名，示例里出现属于正常情况。示例不泄露目标模型这一点，由 §7.4 排除 source_model 来保证）。分两层检查，都在小写化之后进行：
   - **(a) 与模型无关的部分**：`TEMPLATE.format(capabilities=render_capabilities(), max_u=FEASIBLE_MAX_U, n_val=50, examples="")` 中，以下字符串都不能出现：`"resnet"`、`"layer1."`、`"conv1.weight"`、`"linear.weight"`、`"fc1.weight"`、`"fc2.weight"`、`"cifar"`。
   - **(b) 整份提示词**：对 4 个开发模型分别构造 ctx（n_val 取 50），调用 build_system_prompt，结果中不能出现 `"resnet"`、`"cifar"`、`"layer1."`。不检查权重名。
   - **(c)** build_system_prompt 的结果中要包含 "C -> C + 2p" 和 "max |u| <= 0.9"。

   §4 的示例内容（包括 residual_small 中的 `.conv1.weight`）和 §7.4 的期望不变。
3. **模式标注**：
   - resnet20 和 resnet20_silu：conv_bn 19 组、downsample_pad 2 组、mean_head 1 组；
   - cnn_plain：conv_bn 3 组、mean_head 1 组；
   - mlp：0 组。
4. **示例选择**：
   - resnet20 → `["residual_small", "cnn_plain_small"]`；
   - cnn_plain → `["cnn_plain_small", "residual_small"]`；
   - mlp → `["mlp_flat", "cnn_plain_small"]`；
   - 当 model_name 等于某个示例的 source_model 时，该示例被排除（用一份临时 index 测试）。
5. **示例本身可用**：3 个示例都能通过 parse 和 typecheck（用 index 中的 weight_shapes），并且用 `rng(20261026)` 生成的随机权重在 Mock 精确模式下 lowering 不报错。
6. **R7 的 FakeClient 循环测试**改用 build_system_prompt 后仍然通过；`FEASIBLE_MAX_U == 0.90`。
7. R1–R8a 的测试全部通过。

## 8. 不要做

- 不修改 kernel、lowering 的算法、代价模型和规则规划器。lower.py 只新增 HANDLED_OPS 这个常量。
- 不在提示词中写入任何模型的名字或权重名；模型相关的信息只通过 initial_message 传递。
- 冻结程序不手工修改；不把编译结果、CKKS 结果或测试集结果放进反馈；不重试 LLM 调用。
- 不打冻结标签，也不评测未见模型（等用户决定之后，在 R10 进行）。

## 9. 完成标准

- 测试全部通过；r8b_dev 跑完 4 个模型。
- 汇报：每个模型的运行摘要、对照表、API 用量（本轮用量和累计用量）。
- STATUS.md 新增 R8b 一行。M1 第 1 条补充 r8-v1 在 resnet20 上的结果。
