# R3：PyTorch 前端、规范化图（PT-Graph）、numpy 参考前向、ResNet20 导入与精度

目标日期：2026-10-24。不依赖 R2b，可以和 R2b 并行。

## 0. 已核实的事实（照用）

- **环境**：Nix shell 内 venv 的 torch 版本是 2.0.1，没有 `torch.export`，所以只用 `torch.fx.symbolic_trace`。形状用 `torch.fx.passes.shape_prop.ShapeProp` 推断。numpy 版本 1.25.2。所有需要 torch 的代码都通过 R2 的 `pt_agent.tools.nixrun` 运行。
- **ResNet20**：
  - 模型定义：`DSL/DSL_agent/workloads/resnet20/model_reference.py`（B 线文件，只读）；
  - 加载函数：`load_original_weights(path) -> (model, info)`；
  - 权重：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th`；
  - fx 能完整 trace，节点包括 19 个 conv、19 个 bn、19 次 `F.relu`、9 次 `operator.add`、2 次 `getitem`（`x[:,:,::2,::2]`）、2 次 `F.pad`、`mean(dim=(2,3))`、`linear`。
- **CIFAR-10 记录**：
  - 目录：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-cifar-offline-records-r1804/cifar10-records/`；
  - 文件：`train-00000.records` … `train-49000.records` 和 `test-00000.records` … `test-09000.records`，每个文件 1000 行；
  - 每行 3073 字节：第 0 字节是标签，其后 3072 字节是 uint8 像素，按 RGB 平面排列为 3×32×32（C 顺序）。
- **冻结的样本选取**：`/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-resnet20-original-accuracy-r1805/frozen-selections.json`。
  - `calibration`：1000 项，训练集每类前 100 张；
  - `encrypted_test`：40 项，测试集每类前 4 张；
  - 每项含 `shard`、`row`、`label`。
- **归一化**：`(uint8/255 - mean) / std`，其中 mean=(0.485,0.456,0.406)，std=(0.229,0.224,0.225)。
- **基准精度**：B 线 r1811 的记录是 float32 模型、batch 1，在 10000 张测试图上 top-1 为 **91.73%**（9173 张）。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/intake/__init__.py          新：导出 Graph, Node, TensorInfo, IntakeError, import_module
pt_agent/intake/graph.py             新：§2
pt_agent/intake/core_ops.py          新：§3
pt_agent/intake/torch_mapping.py     新：§4
pt_agent/intake/torch_frontend.py    新：§5
pt_agent/intake/decompose.py         新：§6
pt_agent/intake/coverage.py          新：§7
pt_agent/analyzer.py                 新：§8
pt_agent/data/__init__.py            新：空
pt_agent/data/cifar.py               新：§9
pt_agent/models/__init__.py          新：空
pt_agent/models/resnet20.py          新：§9
pt_agent/experiments/r3_resnet20.py  新：§10
pt_agent/tests/test_r3_graph.py      新：§11
pt_agent/tests/test_r3_ops.py        新：§11
pt_agent/tests/test_r3_resnet20.py   新：§11
pt_agent/tests/test_r3_intake_errors.py 新：§11
../docs/pt_agent/STATUS.md           改
```

不修改 `workloads/`、`baseline/`、R1/R2/R2b 的文件。

## 2. intake/graph.py：PT-Graph

```python
@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    kind: str                 # "input" | "act" | "weight"

@dataclass
class Node:
    id: str                   # "n0", "n1", ... 按拓扑顺序编号
    op: str                   # core_ops.CORE_OPS 的键
    inputs: list[str]         # 激活张量名（不包括权重）
    outputs: list[str]        # 激活张量名；split 有多个，其余 op 只有 1 个
    attrs: dict               # 按 core_ops 的 schema；权重以名字（str）引用
    src: str                  # 来源，写成 "<fx 节点名>:<fx target 的 repr>"，供报错使用

@dataclass
class Graph:
    name: str
    nodes: list[Node]
    tensors: dict[str, TensorInfo]      # 所有输入张量、激活张量和权重
    weights: dict[str, np.ndarray]      # 权重名 -> float64 数组（只读）
    inputs: list[str]
    outputs: list[str]

    def producer(self, tensor: str) -> Node | None
    def consumers(self, tensor: str) -> list[Node]
    def save(self, directory) -> None   # 写 graph.json（不含权重数值）和 weights.npz
    @staticmethod
    def load(directory) -> "Graph"
```

- 张量命名规则：
  - 激活张量用 fx 节点名（`node.name`）；
  - `call_module` 的参数用 `<module 路径>.<参数名>`，与 `state_dict` 的键一致，例如 `layer1.0.conv1.weight`；
  - `get_attr` 用它的 target；
  - 标量和常量张量参数用 `const.<fx 节点名>.<序号>`，作为 weight 存储。
- `graph.json` 的字段：`name`、`inputs`、`outputs`、`tensors`（每项 `[name, shape, kind]`）、`nodes`（每项 `[id, op, inputs, outputs, attrs, src]`）。attrs 中的 tuple 存成 list，load 时按 schema 转回 tuple。

## 3. intake/core_ops.py：核心算子表

```python
CORE_OPS: dict[str, dict]   # op -> {"inputs": int|"2+", "attrs": {name: type}}，type 取 int/float/str/pair/tuple/weight/weight?/bool
KERNEL_OPS: frozenset       # 计划有 kernel 的算子（R4/R5），coverage 会用到
```

| op | inputs | attrs |
|---|---|---|
| conv2d | 1 | weight: weight, bias: weight?, stride: pair, padding: pair, dilation: pair, groups: int |
| linear | 1 | weight: weight, bias: weight? |
| matmul | 2 | （无） |
| add | 1 或 2 | 只有 1 个输入时，attrs 恰好有一项：scalar: float 或 const: weight |
| sub | 1 或 2 | 同 add；另有 reverse: bool（为 True 表示"标量或常量 − 输入"） |
| mul | 1 或 2 | 同 add |
| batchnorm | 1 | weight: weight, bias: weight, mean: weight, var: weight, eps: float |
| layernorm | 1 | weight: weight?, bias: weight?, normalized_shape: tuple, eps: float |
| act | 1 | fn: str，取值 relu/gelu/gelu_tanh/silu/sigmoid/tanh/hardswish/exp |
| softmax | 1 | axis: int（已归一化为非负） |
| avgpool | 1 | kernel: pair, stride: pair, padding: pair |
| adaptive_avgpool | 1 | output_size: pair |
| flatten | 1 | start_dim: int, end_dim: int（都已归一化为非负） |
| reshape | 1 | shape: tuple（等于输出的静态形状） |
| permute | 1 | dims: tuple |
| concat | 2+ | axis: int |
| slice | 1 | axis: int, start: int, end: int, step: int（都已按 Python 规则在静态形状上算好，end 不会是 None） |
| split | 1 | axis: int, sizes: tuple |
| pad | 1 | pads: tuple（`F.pad` 的顺序，从最后一维开始） |
| mean | 1 | axes: tuple, keepdim: bool |
| sum | 1 | axes: tuple, keepdim: bool |
| identity | 1 | （无；decompose 时删除） |

`KERNEL_OPS = {conv2d, linear, add, sub, mul, batchnorm, act, avgpool, adaptive_avgpool, flatten, reshape, slice, pad, mean, sum, concat}`。

## 4. intake/torch_mapping.py：PyTorch 到核心算子的映射（逐条实现，表外一律不支持）

**call_module**（按模块类型判断，必须是 `type(m) is ...`）：

| 模块 | → op | 条件（不满足即不支持，并给出原因） |
|---|---|---|
| nn.Conv2d | conv2d | `padding_mode=='zeros'`，padding 为 int 或 tuple（字符串不支持） |
| nn.Linear | linear | — |
| nn.BatchNorm2d、nn.BatchNorm1d | batchnorm | `track_running_stats` 为真，模型处于 eval 模式；weight/bias 为 None（affine=False）时，生成常量 1 和 0，命名为 `<路径>.weight` 和 `<路径>.bias` |
| nn.ReLU、nn.GELU、nn.SiLU、nn.Sigmoid、nn.Tanh、nn.Hardswish | act | GELU 按 approximate 取 gelu 或 gelu_tanh |
| nn.AvgPool2d | avgpool | `ceil_mode=False`、`count_include_pad=True`、`divisor_override=None` |
| nn.AdaptiveAvgPool2d | adaptive_avgpool | output_size 不能含 None |
| nn.Flatten | flatten | — |
| nn.Dropout、nn.Identity | identity | — |
| nn.Softmax | softmax | dim 不为 None |
| nn.LayerNorm | layernorm | — |

**call_function**：

| target | → op |
|---|---|
| operator.add、torch.add | add（`alpha` 必须为 1） |
| operator.sub、torch.sub | sub |
| operator.mul、torch.mul | mul |
| operator.truediv、torch.div | 除数必须是标量或常量张量，转成 mul，scalar 取 1/c，const 取常量的倒数；除数是激活张量时不支持 |
| operator.neg、torch.neg | mul，scalar=-1.0 |
| operator.getitem | 若第一个参数是 split 节点，取对应的那个输出；若是张量，下标必须是 slice 或 Ellipsis 组成的 tuple（不允许整数下标、None、张量下标），每个非平凡维度生成一个 slice 节点 |
| F.relu、torch.relu、F.gelu、F.silu、torch.sigmoid、F.sigmoid、torch.tanh、F.hardswish、torch.exp | act |
| F.pad | pad（mode 为 constant 且 value 为 0） |
| torch.flatten | flatten |
| torch.cat、torch.concat | concat |
| torch.matmul、operator.matmul、torch.bmm | matmul |
| torch.mean、torch.sum | mean、sum（dim 必须给出） |
| F.softmax、torch.softmax | softmax |
| F.avg_pool2d、F.adaptive_avg_pool2d | avgpool、adaptive_avgpool |
| F.linear、F.conv2d | linear、conv2d（权重必须来自 get_attr） |
| torch.permute、torch.transpose | permute（transpose 转成完整的 dims） |
| torch.reshape | reshape |
| torch.split、torch.chunk | split（按静态形状算出 sizes） |
| F.layer_norm | layernorm |

**call_method**：`mean`、`sum`、`view`、`reshape`、`flatten`、`permute`、`transpose`、`relu`、`sigmoid`、`tanh`、`exp`、`softmax`、`split`、`chunk`、`add`、`sub`、`mul`、`div` 分别对应上表；`contiguous` 对应 identity。其余方法（包括 `size`、`shape` 相关的方法）都不支持。

**get_attr**：生成 weight 张量，不生成节点。

## 5. intake/torch_frontend.py

```python
class IntakeError(Exception):
    def __init__(self, entries: list[dict]): ...   # 每项包含 fx_node, op, target, reason；str() 每行列出一项

def import_module(module: torch.nn.Module, example_inputs: tuple, name: str) -> Graph:
```

步骤：

1. `module = copy.deepcopy(module).eval().double()`，example_inputs 也转成 float64。
2. 调用 `gm = torch.fx.symbolic_trace(module)`。抛出 `torch.fx.proxy.TraceError` 或 `TypeError` 时，转成 `IntakeError([dict(fx_node="<trace>", op="trace", target=type(module).__name__, reason="dynamic control flow or untraceable Python: " + str(e))])`。
3. 调用 `ShapeProp(gm).propagate(*example_inputs)`，每个节点的形状取 `node.meta["tensor_meta"].shape`。
4. 按 §4 逐个节点映射。不支持的节点**收集起来，不要中途停下**；全部扫描完后，只要有不支持项，就抛一个 `IntakeError`，包含全部条目。
5. 权重统一转成 float64 的 `np.ndarray`，并设为只读。
6. 输入张量名用 placeholder 的名字，输出张量名取自 output 节点的参数（支持单个张量，或由张量组成的 tuple/list）。
7. 调用 `decompose.run(graph)` 后返回。

## 6. intake/decompose.py

`run(graph) -> Graph`，按顺序应用以下规则，每条规则一个函数：

1. **identity 删除**：后续节点中引用该节点输出的地方，改为引用它的输入。如果它的输出正好是图输出，就把图输出改成它的输入。
2. **adaptive_avgpool 转为 avgpool**：要求 H % oh == 0 且 W % ow == 0。kernel=stride=(H/oh, W/ow)，padding=(0,0)。不满足时保留原样，coverage 会把它算作 needs_composition。
3. 节点按拓扑顺序重新编号为 n0、n1……；`tensors` 里删掉不再被引用的激活张量。

每条规则在 test_r3_ops 中都要有一个用例。

## 7. intake/coverage.py

`report(graph) -> dict`：

```python
dict(total=节点数, supported=op 属于 KERNEL_OPS 的节点数, needs_composition=其余节点数,
     by_op={op: count}, needs_composition_ops=sorted(set(...)))
```

不支持的节点在导入阶段就已经报错，所以不会出现在这里。

## 8. analyzer.py：numpy float64 参考前向

```python
def reference_forward(graph: Graph, inputs: dict[str, np.ndarray], *, keep: bool = False) -> dict[str, np.ndarray]
def calibrate(graph: Graph, inputs: dict[str, np.ndarray], batch: int = 100) -> dict[str, dict]
```

- **批量**：`inputs[name]` 的形状是 `(N,) + shape[1:]`，图里静态形状的第 0 维是 1，N 可以任意。所有 op 都把第 0 维当作批量维；attrs 里的轴编号按完整形状计算。
- **reshape**：目标形状取 `(N,) + attrs["shape"][1:]`，要求 `attrs["shape"][0] == 1`。
- **返回值**：`keep=False` 时只返回图输出 `{name: array}`；`keep=True` 时返回所有张量。
- **各 op 的实现**：
  - conv2d：im2col 加矩阵乘，支持 groups、stride、padding、dilation。先用 `np.pad` 补零，再用 `np.lib.stride_tricks.sliding_window_view` 取窗口，`reshape` 成 `(N*Ho*Wo, Cin/g*kh*kw)` 后与权重相乘，按组拼接。
  - batchnorm：`(x - mean)/sqrt(var + eps)*weight + bias`，在通道轴（第 1 轴）上广播。
  - linear：`x @ W.T + b`。
  - act：
    - relu：`np.maximum(x, 0)`；
    - gelu：`0.5*x*(1+erf(x/√2))`，erf 用 `math.erf` 经 `np.vectorize` 实现，或者用本模块内自己写的、误差 ≤1e-15 的有理逼近，二选一，写在 docstring 里；
    - gelu_tanh：PyTorch 的 tanh 公式；
    - silu：`x*sigmoid(x)`；
    - sigmoid：`1/(1+exp(-x))`；
    - hardswish：`x*clip(x+3,0,6)/6`；
    - exp、tanh：直接调用 numpy。
  - softmax：先减去最大值再求和归一。
  - layernorm：在最后 `len(normalized_shape)` 个维度上算均值和方差（`ddof=0`）。
  - avgpool：补零后取窗口均值，与 `count_include_pad=True` 一致。
  - adaptive_avgpool：按 PyTorch 的分箱规则，第 i 个输出窗口为 `[floor(i*H/oh), ceil((i+1)*H/oh))`，W 方向同理，取窗口均值。
  - 其余 op 直接用 numpy 对应函数实现（slice 用 Python 切片，pad 用 `np.pad`）。
- **`calibrate`**：按 batch 分批前向（`keep=True`）。对每个激活张量累计 `min`、`max`、`absmax`，以及 |x| 的 99.9 百分位 `p999`（用每批的 `np.percentile` 值取最大，作为近似），再加上 `count`（元素总数）。返回 `{tensor: {min, max, absmax, p999, count}}`，值都是 Python float 或 int。

## 9. data/cifar.py 与 models/resnet20.py

```python
# data/cifar.py
RECORDS = Path(".../agent-cifar-offline-records-r1804/cifar10-records")
SELECTIONS = Path(".../agent-resnet20-original-accuracy-r1805/frozen-selections.json")
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)

def read_rows(split: str, indices) -> tuple[np.ndarray, np.ndarray]:
    """split 为 "train" 或 "test"；index i 位于文件 f"{split}-{i//1000*1000:05d}.records" 的第 i%1000 行。
    返回 (uint8 数组 [N,3,32,32]，int64 标签 [N])。"""
def normalize(pixels_uint8) -> np.ndarray      # 结果为 float64 [N,3,32,32]
def calibration_set() -> tuple[np.ndarray, np.ndarray]     # 按 SELECTIONS["calibration"] 的 shard 和 row 读取，共 1000 张
def encrypted_test_set() -> tuple[np.ndarray, np.ndarray]  # SELECTIONS["encrypted_test"]，共 40 张
def test_set(limit: int | None = None) -> tuple[np.ndarray, np.ndarray]   # 测试集前 limit 张，None 表示全部 10000 张
```

读取 selections 时，要求每项的 shard 与 index 推出的文件名一致，label 与记录中的标签字节一致，不一致就抛 ValueError。不做哈希校验。

```python
# models/resnet20.py
WEIGHTS = Path(".../agent-cifar-offline-records-r1804/public/resnet20-12fca82f.th")
def load_torch() -> torch.nn.Module     # 把 workloads/resnet20 加入 sys.path，调用 model_reference.load_original_weights(WEIGHTS)[0]，返回 float32 eval 模型
def load_graph() -> Graph               # import_module(load_torch(), (torch.zeros(1,3,32,32),), "resnet20")
```

## 10. experiments/r3_resnet20.py

用法：`cd DSL/DSL_agent/scripts && python3 -m pt_agent.experiments.r3_resnet20`

- 外层与 R2b 的 r2b_smoke 采用同一种模式：用系统 python3 以 subprocess 调用 nixrun 运行 `--inside --out <目录>`，打印子进程输出，最后由外层写 registry。目录取 `$PT_RESULTS/r3-resnet20-<UTC 时间戳>`，外层创建；没有 `PT_RESULTS` 就报错退出。
- inside 依次执行：
  1. `load_graph()`，`graph.save(out/"graph")`，coverage 写入 `out/coverage.json`。
  2. `calibrate(graph, {"x": calibration_set()[0]})`，写入 `out/calibration.json`。
  3. 在全部 10000 张测试图上，按每批 250 张统计 top-1：
     - torch float32 模型；
     - torch float64 模型（`copy.deepcopy(model).double()`）；
     - `reference_forward`。
  4. 在 1000 张校准图上，比较 `reference_forward` 与 torch float64 的 logits，取最大绝对误差。
  5. 写入 `out/REPORT.json`：`dict(nodes, coverage, top1_torch32, top1_torch64, top1_reference, agree_reference_vs_torch64=两者 argmax 一致的张数, max_abs_logit_diff_calibration, seconds_reference_10000, relu_input_ranges={每个 act 节点的输入张量: [min, max]})`。
- 外层写 registry：round="R3"，model="resnet20"，mode="reference"，metrics 取 REPORT 中除 `relu_input_ranges` 以外的字段。
- 资源：单进程不超过 4 GiB，批大小不要超过 250。预计 30 分钟以内跑完；超过 90 分钟就中止，并记录已经算出的部分。

## 11. 测试（全部通过 nixrun 运行）

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun --seconds 1800 -- -m unittest discover -s pt_agent/tests -p 'test_r3_*.py' -v
```

**test_r3_graph.py**

1. 用一个小 CNN（conv→bn→relu→avgpool→flatten→linear）导入；`save` 再 `load` 后，节点、张量和权重完全一致。
2. `producer` 和 `consumers` 的结果正确。

**test_r3_ops.py**：表中每个模块都执行 `import_module(m, (x,), name)`。输入 `x = torch.randn(shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64)`；判定为 `reference_forward(g, {"x": x.numpy()})` 与 `m.double().eval()(x)` 的最大绝对误差 ≤1e-9，同时检查生成的 op 序列符合期望。

| # | 模块的 forward（输入形状） | 期望的 op 序列 |
|---|---|---|
| 1 | `nn.Conv2d(3,8,3,padding=1)`，(1,3,8,8) | conv2d |
| 2 | `nn.Conv2d(8,8,3,stride=2,padding=1,groups=8,bias=False)`，(1,8,8,8) | conv2d |
| 3 | `nn.Conv2d(4,6,3,dilation=2)`，(1,4,9,9) | conv2d |
| 4 | `nn.Linear(16,5)`，(1,16) | linear |
| 5 | `torch.matmul(x, x.transpose(1,2))`，(1,4,6) | permute, matmul |
| 6 | `x + x*2.0 - 1.0`，(1,8) | mul, add, sub（mul 和 sub 带 scalar） |
| 7 | `3.0 - x`，(1,8) | sub（reverse=True） |
| 8 | `x / 4.0`，(1,8) | mul（scalar=0.25） |
| 9 | `-x`，(1,8) | mul（scalar=-1.0） |
| 10 | `x + self.b`，b 为 `nn.Parameter(randn(8))`，(1,8) | add（const 为 b） |
| 11 | `nn.BatchNorm2d(4)`，先设置随机的 running_mean/var/weight/bias 再 eval，(1,4,5,5) | batchnorm |
| 12 | `nn.LayerNorm(8)`，(1,3,8) | layernorm |
| 13–19 | 依次为 `F.relu`、`nn.GELU()`、`nn.GELU(approximate='tanh')`、`nn.SiLU()`、`torch.sigmoid`、`torch.tanh`、`nn.Hardswish()`，(1,16) | act（fn 依次对应） |
| 20 | `torch.exp`，(1,16) | act(exp) |
| 21 | `F.softmax(x, dim=-1)`，(1,3,8) | softmax(axis=2) |
| 22 | `nn.AvgPool2d(2)`，(1,4,8,8) | avgpool |
| 23 | `nn.AdaptiveAvgPool2d(1)`，(1,4,8,8) | avgpool（经 decompose 转换） |
| 24 | `nn.AdaptiveAvgPool2d(3)`，(1,4,8,8) | adaptive_avgpool（不能整除，保留原样） |
| 25 | `torch.flatten(x, 1)`，(1,2,3,4) | flatten |
| 26 | `x.view(1, -1)`，(1,2,3,4) | reshape(shape=(1,24)) |
| 27 | `x.permute(0,2,1)`，(1,3,5) | permute |
| 28 | `torch.cat([x, x*2.0], dim=1)`，(1,4) | mul, concat |
| 29 | `x[:, :, ::2, 1:]`，(1,2,6,6) | slice, slice |
| 30 | `a, b = torch.split(x, 3, dim=1); a + b`，(1,6) | split, add |
| 31 | `F.pad(x, (0,0,0,0,2,2))`，(1,4,3,3) | pad |
| 32 | `x.mean(dim=(2,3))`，(1,4,5,5) | mean |
| 33 | `torch.sum(x, dim=1, keepdim=True)`，(1,4,5) | sum |
| 34 | `nn.Dropout(0.5)` 后接 `nn.Identity()`，再接 `F.relu`，(1,8) | act（identity 已被删除） |

另外：`reference_forward` 输入批量 N=3 时，结果与 torch 逐样本计算一致（用 #1 和 #11 验证）。

**test_r3_resnet20.py**

1. `load_graph()` 共 74 个节点：`Counter(op)` 等于 `{conv2d:19, batchnorm:19, act:19, add:9, slice:4, pad:2, mean:1, linear:1}`；所有 act 的 fn 都是 relu；输出张量形状为 (1,10)；`coverage.needs_composition == 0`。
2. pad 节点的 `pads` 为 `(0,0,0,0,8,8)` 和 `(0,0,0,0,16,16)`；slice 节点的 step 为 2，axis 为 2 或 3。
3. 在 8 张校准图上，`reference_forward` 与 torch float64 的 logits 最大绝对误差 ≤1e-9。
4. 前 500 张测试图：reference 的 argmax 与 torch float64 完全一致。
5. `calibration_set()` 形状为 (1000,3,32,32)，每类各 100 张；`encrypted_test_set()` 形状为 (40,3,32,32)，每类各 4 张；第 0 张测试图（label 3）归一化后的 `[0,0,0,0]` 等于 `(像素字节/255-0.485)/0.229`。

**test_r3_intake_errors.py**

1. 一个依次含 `nn.MaxPool2d(2)`、`torch.sort(x)[0]` 和 `F.relu` 的模块，抛出 IntakeError，并且同时列出 MaxPool2d 和 sort 两项。
2. forward 中含 `if x.sum() > 0:` 的模块，抛出 IntakeError，op 为 "trace"。
3. 除数是激活张量（`x / x`），抛 IntakeError，原因里写明不支持激活张量作除数。
4. `x[:, 0]`（整数下标）抛 IntakeError。

## 12. 不要做

- 不安装或升级 torch，不用 `torch._dynamo`。
- 不修改 `workloads/resnet20/`、`baseline/`、数据和权重文件；不加哈希校验。
- 不做 BN 折叠、近似、布局，也不生成 PT-DSL（这些在 R4–R6）。
- 不支持表外的算子；不支持的情况一律报 IntakeError，不要自行补充映射。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 13. 完成标准

- `test_r3_*.py` 全部通过；R1、R2 的测试依然通过。
- `r3_resnet20` 跑完，在汇报里贴出 REPORT.json（`relu_input_ranges` 只贴前 3 项）。期望结果如下，不符合的项照实报告：
  - `top1_torch32` 为 91.73；
  - `top1_reference` 与 `top1_torch64` 相差不超过 0.1 个百分点；
  - `max_abs_logit_diff_calibration` ≤1e-9。
- registry 中新增一行；STATUS.md 的 R3 一行写明结论。
