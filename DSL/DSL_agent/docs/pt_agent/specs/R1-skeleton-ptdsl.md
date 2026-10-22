# R1：骨架收尾 + PT-DSL v1 的 IR、解析器、打印器、类型检查

目标日期：2026-10-21。本轮只用 Python 3.10 标准库。

## 0. 本轮开始前

- 如果已经按旧版 PLAN 的 T2 写了 `ir.py`、`parser.py`、`printer.py`、`typecheck.py`、`ops_spec.py` 等文件，全部删除，按本 spec 重写。
- 已有的 `pt_agent/__init__.py`、`config.py`、`registry.py`、`tests/test_t0_registry.py` 保留，只按 §2 修改。
- 正式结果目录定为 `PT_RESULTS=/home/lhohy/pt_agent_results`。单测一律用临时目录。

## 1. 文件清单

路径相对 `DSL/DSL_agent/`：

```
scripts/pt_agent/registry.py                 改：字段 card → round
scripts/pt_agent/tests/test_t0_registry.py   重命名为 test_r1_registry.py，同步改字段
scripts/pt_agent/dsl/__init__.py             新：只有一行 docstring
scripts/pt_agent/dsl/errors.py               新
scripts/pt_agent/dsl/ir.py                   新
scripts/pt_agent/dsl/ops_spec.py             新
scripts/pt_agent/dsl/shapes.py               新
scripts/pt_agent/dsl/parser.py               新
scripts/pt_agent/dsl/printer.py              新
scripts/pt_agent/dsl/typecheck.py            新
scripts/pt_agent/dsl/check.py                新：命令行入口
scripts/pt_agent/examples/__init__.py        新：空
scripts/pt_agent/examples/shapes.py          新
scripts/pt_agent/examples/mlp.ptdsl          新：内容见 §9.1，逐字照抄
scripts/pt_agent/examples/resnet20.ptdsl     新：内容见 §9.2，逐字照抄
scripts/pt_agent/examples/invalid/*.ptdsl    新：29 个，内容见 §9.4，逐字照抄
scripts/pt_agent/tests/test_r1_parser.py     新
scripts/pt_agent/tests/test_r1_printer.py    新
scripts/pt_agent/tests/test_r1_shapes.py     新
scripts/pt_agent/tests/test_r1_typecheck.py  新
scripts/pt_agent/tests/test_r1_cli.py        新
docs/pt_agent/STATUS.md                      改：换成轮次模板（§14）
```

## 2. registry 修改

- `REQUIRED_FIELDS` 里的 `"card"` 改成 `"round"`，其余逻辑不变。
- 新增两个函数：

```python
def git_rev() -> str:
    """`git rev-parse --short HEAD`（cwd 取本文件所在目录）；`git status --porcelain` 非空时追加 "-dirty"。
    git 调用失败时返回 "unknown"。"""

def new_run_id(round_name: str, model: str, mode: str) -> str:
    """返回 f"{round_name}-{model}-{mode}-{UTC %Y%m%dT%H%M%S}-{secrets.token_hex(2)}"。"""
```

- `test_r1_registry.py`：保留原来 3 个测试（字段改成 round），再加 2 个：
  - `new_run_id("R1","mlp","unit")` 匹配正则 `^R1-mlp-unit-\d{8}T\d{6}-[0-9a-f]{4}$`；
  - `git_rev()` 返回非空 str。

## 3. errors.py

```python
PARSE_CODES = frozenset({
    "E_SYNTAX", "E_NO_ENTRY", "E_MULTI_ENTRY", "E_HEADER", "E_PARAM", "E_STMT", "E_EXPR",
    "E_UNKNOWN_OP", "E_UNDEFINED", "E_ARITY", "E_ATTR_UNKNOWN", "E_ATTR_MISSING",
    "E_ATTR_TYPE", "E_ATTR_VALUE", "E_TARGET", "E_RETURN", "E_MACRO", "E_REGION",
    "E_NAME", "E_WEIGHT_REF",
})
CHECK_CODES = frozenset({
    "E_SHAPE", "E_WEIGHT_MISSING", "E_WEIGHT_SHAPE", "E_LAYOUT_UNPACKED",
    "E_LAYOUT_REPACK", "E_LAYOUT_MISMATCH", "E_KIND", "W_UNUSED", "W_REFRESH_COMPILER",
})

class PTDSLError(Exception):
    def __init__(self, code: str, lineno: int, msg: str):
        assert code in PARSE_CODES
        self.code, self.lineno, self.msg = code, lineno, msg
        super().__init__(f"{code} line {lineno}: {msg}")

@dataclass(frozen=True)
class Diag:
    code: str          # ∈ CHECK_CODES
    severity: str      # "error"（E_ 开头）或 "warning"（W_ 开头）
    lineno: int
    op_id: int | None
    msg: str
    def __str__(self): return f"{self.severity.upper()} {self.code} line {self.lineno}: {self.msg}"
```

解析遇到第一个错误就抛 `PTDSLError`。类型检查收集全部问题，返回 `Diag` 列表。和具体行无关的错误（目前只有 E_NO_ENTRY）用 lineno=0。

## 4. ir.py

```python
@dataclass(frozen=True)
class WeightRef:
    name: str

@dataclass(frozen=True)
class Header:
    slots: int
    refresh: str = "compiler"          # "compiler" | "agent"
    placement: str = "depth-dp"        # "depth-dp" | "optimized" | "greedy"
    l_max: int | None = None
    boot_out_level: int | None = None
    lineno: int = 0

@dataclass(frozen=True)
class Param:
    name: str
    kind: str                          # "cipher" | "plain"
    shape: tuple[int, ...]
    lineno: int

@dataclass(frozen=True)
class Region:
    name: str
    level_budget: int
    lineno: int

@dataclass(frozen=True)
class Op:
    id: int                            # 在 Program.ops 中的下标
    kind: str                          # ops_spec.OPS 的键
    inputs: tuple[str, ...]            # 值 id
    outputs: tuple[str, ...]           # 值 id
    attrs: dict                        # 已补全默认值，键顺序与 ops_spec 声明顺序一致
    lineno: int
    region: str | None
    macro_stack: tuple[str, ...]       # 展开路径，例如 ("basic_block",)

@dataclass(frozen=True)
class Program:
    name: str
    header: Header
    params: tuple[Param, ...]
    ops: tuple[Op, ...]
    outputs: tuple[str, ...]
    regions: dict[str, Region]         # 按出现顺序

    def producer(self, vid: str) -> "Op | None": ...   # 产生该值的 op；参数返回 None
    def param(self, name: str) -> "Param | None": ...

def canonical(program: Program) -> dict:
    """与名字和行号无关的结构表示，用于往返比较；结果可直接 json.dumps。"""
```

`canonical` 的规则：

- 参数值 id 映射成 `"p:<参数名>"`；op 输出按定义顺序映射成 `"%0"`、`"%1"`……
- `header` 转成 dict，去掉 lineno；`params` 是 `[name, kind, list(shape)]` 的列表。
- `ops` 是 `[kind, inputs, outputs, attrs, region]` 的列表。attrs 里 `WeightRef` 写成 `{"weight": name}`，tuple 写成 list，其余原样。
- `outputs` 是映射后的 id 列表。
- `regions` 是 `[name, level_budget]` 的列表，只包含至少有一个 op 的 region，按第一个 op 出现的顺序排列。

## 5. ops_spec.py

```python
ATTR_TYPES = ("int", "float", "str", "bool", "weight", "int_or_pair", "int_tuple",
              "float_tuple", "float_pair", "axis")

@dataclass(frozen=True)
class AttrSpec:
    type: str
    required: bool = False
    default: object = None             # 默认值是 None 的非必填属性允许写 None
    choices: tuple | None = None

@dataclass(frozen=True)
class OpSpec:
    name: str
    inputs: int | str                  # 固定个数，或 "2+"
    outputs: str                       # "one" | "split"（个数 = len(attrs["sizes"])）
    attrs: dict[str, AttrSpec]         # 声明顺序即打印顺序
    validate: Callable[[dict], str | None]   # 属性之间的约束；返回错误信息表示违反（→ E_ATTR_VALUE）
    layout: str                        # "pack" | "relayout" | "same" | "binary"

OPS: dict[str, OpSpec]
RESERVED_NAMES = frozenset({"W", "fhe", "region", "Cipher", "Plain"}) | frozenset(OPS)
def describe_markdown() -> str:
    """一个 markdown 表，列为：op | inputs | attributes。attributes 里每项写成
    `name: type = default {choices}`，必填项写 `name: type (required)`。R7 的 prompt 会直接用它。"""
```

### 5.1 属性类型的转换与检查（在解析阶段完成）

| type | 接受的字面量 | 存进 IR 的形式 |
|---|---|---|
| int | int（bool 不算） | int |
| float | int 或 float（bool 不算） | float |
| str | str | str |
| bool | True/False | bool |
| weight | `W[...]` | WeightRef |
| int_or_pair | int，或 2 个 int 的 tuple | 2 元 tuple |
| int_tuple | 1 个以上 int 的 tuple | tuple[int] |
| float_tuple | 1 个以上数的 tuple | tuple[float] |
| float_pair | 2 个数的 tuple | tuple[float, float]；要求 lo < hi，否则 E_ATTR_VALUE |
| axis | int，或 1 个以上 int 的 tuple | tuple[int] |

类型不符报 E_ATTR_TYPE。取值不在 choices 里报 E_ATTR_VALUE。缺少必填属性报 E_ATTR_MISSING。给了未声明的属性名报 E_ATTR_UNKNOWN。显式写 None 只在默认值是 None 的非必填属性上允许，其他情况报 E_ATTR_TYPE。

### 5.2 算子表（v1，共 26 个）

"pow2" 表示 2 的幂。所有约束违反都报 E_ATTR_VALUE。

| op | inputs | layout | 属性（按声明顺序） | validate 约束 |
|---|---|---|---|---|
| pack | 1 | pack | layout: str (required) {flat, mux, rep}; gap: int = 1; period: int = None; copies: int = None | gap 是 pow2 且 ≥1；layout≠mux 时 gap 必须为 1；period 为 None 或 pow2≥2，layout=mux 时必须为 None；layout=rep 时 copies 必填且为 pow2≥2，否则必须为 None |
| relayout | 1 | relayout | 同 pack | 同 pack |
| conv2d | 1 | same | w: weight (required); b: weight = None; bn: str = None; bn_eps: float = 1e-5; stride: int_or_pair = 1; pad: int_or_pair = 0; dilation: int_or_pair = 1; groups: int = 1; algo: str = "mpcb" {mpcb, tap} | stride、dilation 各分量 ≥1；pad ≥0；groups ≥1；bn_eps > 0 |
| linear | 1 | same | w: weight (required); b: weight = None; algo: str = "bsgs" {bsgs, diag, rowsum}; n1: int = None | n1 为 None 或 pow2≥1；n1 非 None 时 algo 必须是 bsgs |
| matmul | 2 | binary | algo: str = "diag" {diag, rowsum} | 无 |
| avgpool | 1 | same | k: int_or_pair (required); stride: int_or_pair = None; pad: int_or_pair = 0 | k ≥1；stride 为 None 时在解析阶段替换为 k；stride ≥1；pad ≥0 |
| mean | 1 | same | axis: axis (required); keepdim: bool = False | axis 内不重复 |
| sum | 1 | same | 同 mean | 同 mean |
| add | 2 | binary | 无 | 无 |
| sub | 2 | binary | 无 | 无 |
| mul | 2 | binary | 无 | 无 |
| scale | 1 | same | c: float (required) | c 有限 |
| bias | 1 | same | b: weight (required); axis: int = 1 | 无 |
| batchnorm | 1 | same | bn: str (required); eps: float = 1e-5 | eps > 0 |
| act | 1 | same | fn: str (required) {relu, silu, gelu, sigmoid, tanh, exp, inv, rsqrt, hardswish, square}; method: str (required) {minimax_comp, cheb, exact}; degrees: int_tuple = None; deg: int = None; domain: float_pair = None | 见 §5.3 |
| poly | 1 | same | coeffs: float_tuple (required); basis: str = "power" {power, cheb}; domain: float_pair = None | coeffs 长度 1..256 且有限；basis=cheb 时 domain 必填；basis=power 时 domain 必须为 None |
| softmax | 1 | same | axis: int = -1; exp_deg: int (required); exp_domain: float_pair (required); inv_deg: int (required); inv_domain: float_pair (required) | exp_deg、inv_deg ∈ [1,255]；inv_domain 的 lo > 0 |
| layernorm | 1 | same | w: weight = None; b: weight = None; eps: float = 1e-5; rsqrt_deg: int (required); rsqrt_domain: float_pair (required) | eps > 0；rsqrt_deg ∈ [1,255]；rsqrt_domain 的 lo > 0 |
| flatten | 1 | same | start_dim: int = 1 | 无 |
| reshape | 1 | same | shape: int_tuple (required) | 每项 ≥1，或者恰好一项为 -1 |
| permute | 1 | same | dims: int_tuple (required) | 无（在 shape 阶段检查） |
| concat | 2+ | binary | axis: int (required) | 无 |
| slice | 1 | same | axis: int (required); start: int = 0; end: int = None; step: int = 1 | step ≥1 |
| split | 1 | same | axis: int (required); sizes: int_tuple (required) | 每项 ≥1 |
| pad | 1 | same | pads: int_tuple (required) | 长度为偶数且在 2..8 之间；每项 ≥0 |
| refresh | 1 | same | 无 | 无 |

除 split 外，`outputs` 都是 `"one"`；split 是 `"split"`。

### 5.3 act 的约束

- method=minimax_comp：fn 必须是 relu；degrees 必填，长度 1..4，每项是 [3,63] 内的奇数；deg 必须为 None；domain 必填。
- method=cheb：fn 不能是 square；deg 必填，在 [1,255] 内；degrees 必须为 None；domain 必填。
- method=exact：fn 必须是 square；degrees、deg、domain 都必须为 None。
- fn 是 inv 或 rsqrt 时，domain 的 lo 必须 > 0。

## 6. parser.py

```python
def parse(text: str) -> Program
```

只用 `ast.parse`。**禁止 exec、eval、compile 以及 import 用户内容。**

### 6.1 顶层结构

1. `ast.parse` 抛 SyntaxError → E_SYNTAX，lineno 取 `e.lineno`（为 None 时取 0）。
2. 模块体只允许 `FunctionDef`，其他语句（import、表达式语句、赋值、class 等）一律报 E_STMT。
3. 带装饰器的函数是入口：没有入口报 E_NO_ENTRY（lineno 0）；有多个入口报 E_MULTI_ENTRY，lineno 取第二个入口的 `def` 行。
4. 不带装饰器的函数是宏。函数重名，或宏名落在 `RESERVED_NAMES` 里，报 E_MACRO（lineno 取后定义的那个 def 行）。

### 6.2 入口头部

- `decorator_list` 必须恰好是一个 `Call`，func 为 `Attribute(Name("fhe"), "program")`，只能有关键字参数。不满足就报 E_HEADER，lineno 取装饰器所在行。
- 关键字规则：
  - `slots`：int，必填，pow2，1024 ≤ slots ≤ 65536；
  - `refresh`：取值 {"compiler", "agent"}，默认 compiler；
  - `placement`：取值 {"depth-dp", "optimized", "greedy"}，默认 depth-dp；
  - `l_max`、`boot_out_level`：int，范围 1..64，可省略；
  - refresh=agent 时，l_max 和 boot_out_level 都必填，且 boot_out_level ≤ l_max。
  - 出现未知关键字，或任何一条不满足，都报 E_HEADER。
- 参数：
  - 每个参数必须带注解 `Cipher[d0, d1, ...]` 或 `Plain[...]`；单维写法如 `Cipher[64]` 也允许；每维是 int 常量且 ≥1。
  - 不允许默认值、`*args`、`**kwargs`、仅关键字参数、仅位置参数，也不允许返回注解。
  - 至少要有一个 Cipher 参数。
  - 以上不满足报 E_PARAM，lineno 取该参数所在行；"至少一个 Cipher" 不满足时取 def 行。

### 6.3 名字规则（E_NAME）

被赋值的名字、入口参数名、宏参数名必须：

- 匹配 `^[A-Za-z][A-Za-z0-9_]*$`；
- 不含 `__`；
- 不在 `RESERVED_NAMES` 里，也不是任何一个宏的名字；
- 入口参数名还不能匹配 `^v\d+$`，这类名字留给打印器使用。

### 6.4 语句（入口和宏的函数体）

- `Assign`：
  - targets 只能有 1 个，可以是 `Name` 或由 `Name` 组成的 `Tuple`，其他形式报 E_STMT。
  - 右边必须是 `Call` 且 func 为 `Name`，否则报 E_EXPR。
  - func 名在 `OPS` 里按 §6.5 处理，是宏名按 §6.6 处理，两者都不是报 E_UNKNOWN_OP。
- `With`：
  - 在宏里出现报 E_REGION；已经在某个 region 里再嵌套也报 E_REGION。
  - 必须恰好一个 item，形如 `region("<字符串字面量>", level_budget=<int 1..64>)`，没有 `as`；region 名不能重复。不满足报 E_REGION。
  - 函数体里的 op 都带上这个 region 名。
- `Return`：只能是函数体顶层的最后一条语句，否则报 E_RETURN。入口可以返回 `Name` 或由 `Name` 组成的 `Tuple`；宏只能返回单个 `Name`；返回值必须是已定义的张量。函数体最后一条不是 Return 时报 E_RETURN，lineno 取最后一条语句的行。
- 其他语句（for、if、while、表达式语句、pass、嵌套 def……）报 E_STMT。

### 6.5 op 调用

- 位置参数是张量输入，每个都必须是 `Name`，否则报 E_EXPR。
  - 名字未定义报 E_UNDEFINED；
  - 名字是绑定到字面量的宏参数，报 E_MACRO。
- 输入个数与 `OpSpec.inputs` 不符报 E_ARITY。
- 关键字参数就是属性；出现 `**kw` 报 E_EXPR。属性值的求值规则：
  - `Constant`（int/float/str/bool/None）取原值。
  - `-<int|float 常量>`，即 `UnaryOp(USub)`，取负数。
  - `Tuple`：元素只能是数字常量、负数常量，或绑定到数字的宏参数 Name。
  - `W[<字符串表达式>]` 求值为 `WeightRef`。下标不是字符串表达式，或单独出现 `W`，报 E_WEIGHT_REF。
  - `Name`：如果绑定到宏字面量就取该字面量；如果是张量报 E_EXPR；未定义报 E_UNDEFINED。
  - 字符串表达式指：str 常量、绑定到 str 的宏参数 Name，以及用 `+` 连接这两种得到的 `BinOp(Add)`。连接的任一侧不是 str 时报 E_EXPR。
  - 其他节点（IfExp、Compare、非字符串的 BinOp、Call、Attribute、Lambda……）一律报 E_EXPR。
- 按 §5.1 转换类型 → 检查 choices → 补默认值（avgpool 的 stride=None 替换为 k）→ 调用 `validate`。
- 输出：
  - 单输出 op 的目标必须是 `Name`，写成 Tuple 报 E_TARGET；
  - split 的目标必须是 Tuple，且元素个数等于 `len(sizes)`，否则报 E_TARGET。
- 生成 `Op(id=当前长度, ...)`，lineno 取赋值语句所在行。

### 6.6 宏展开

- 维护一个全局计数器 k，从 1 开始，每展开一次宏调用加 1。
- 绑定实参：
  - 位置参数按顺序，关键字参数按名字；
  - 个数不符、未知参数名、缺少参数都报 E_MACRO；
  - 实参允许 `Name`（张量绑定为张量；字面量绑定原样传入）、字面量（§6.5 中 Constant、负数、Tuple 的规则）和字符串表达式；
  - 实参写 `W[...]` 报 E_MACRO。
- 递归（宏出现在当前展开栈里）或展开深度超过 16，报 E_MACRO，lineno 取那次调用所在行。
- 宏体继承调用处的 region；宏体内部不允许写 region。
- 宏的返回值：调用处的目标名直接绑定到宏返回的那个值 id，不新建 id。

### 6.7 SSA 值 id

- 每个作用域（入口，或一次宏展开）带一个前缀：入口的前缀是 `""`，第 k 次展开宏 m 的前缀是 `f"{m}__c{k}__"`。
- 某个名字 n 第 i 次被赋值（从 0 开始计）得到的 id 是：i=0 时为 `前缀 + n`，否则为 `前缀 + n + f"__{i}"`。
- 入口参数先绑定（i=0）。之后对参数名重新赋值，会得到 `x__1` 这样的 id。

## 7. shapes.py

```python
class ShapeError(Exception): ...                       # → E_SHAPE
class WeightError(Exception):                          # → E_WEIGHT_MISSING / E_WEIGHT_SHAPE
    def __init__(self, code, msg): ...
def infer(kind: str, in_shapes: list[tuple], attrs: dict,
          wshape: Callable[[str], tuple]) -> list[tuple]
```

- `wshape(name)`：权重不存在时抛 `WeightError("E_WEIGHT_MISSING", name)`。
- 下文写"期望 w 为 S"时，实际形状不等于 S 就抛 `WeightError("E_WEIGHT_SHAPE", f"{name}: expected {S}, got {actual}")`。
- 所有负数轴先加上 rank 归一化，越界抛 ShapeError。

各算子的规则：

- **pack, relayout, act, poly, scale, refresh**：形状不变。
- **softmax**：axis 必须在范围内；形状不变。
- **conv2d**：
  - x 必须 rank 4 且 x[0]==1。
  - w 必须 rank 4。设 Co=w[0]，g=groups，Ci=x[1]；要求 Ci%g==0、Co%g==0、w[1]==Ci//g。不满足抛 E_WEIGHT_SHAPE。
  - `Ho = (H + 2*ph - dh*(kh-1) - 1)//sh + 1`，Wo 同理；Ho、Wo 都要 ≥1。
  - b 期望为 (Co,)。bn 前缀 P 时，期望 `P.weight`、`P.bias`、`P.running_mean`、`P.running_var` 都为 (Co,)。
  - 输出 (1, Co, Ho, Wo)。
- **linear**：x 的 rank ≥1；w 必须 rank 2 且 w[1]==x[-1]（否则 E_WEIGHT_SHAPE）；b 期望为 (w[0],)；输出 x[:-1]+(w[0],)。
- **matmul**：两个输入的 rank 都 ≥2；a[:-2]==b[:-2]；a[-1]==b[-2]；输出 a[:-2]+(a[-2], b[-1])。
- **avgpool**：rank 4；`Ho=(H+2p-k)//s+1`，W 同理，都要 ≥1。
- **mean, sum**：轴归一化后不能越界、不能重复；keepdim 时把这些轴设为 1，否则删掉；结果为空 tuple 时抛 ShapeError（v1 不支持归约成标量）。
- **add, sub, mul**：两个输入形状必须相等。
- **bias**：axis 必须在范围内；b 期望为 (x[axis],)。
- **batchnorm**：rank ≥2；4 个 BN 权重期望为 (x[1],)。
- **layernorm**：w、b 如果给了，期望为 (x[-1],)。
- **flatten**：s=start_dim 归一化后满足 0≤s<rank；输出 x[:s]+(prod(x[s:]),)。
- **reshape**：用 -1 推出那一维（要求能整除），乘积必须相等。
- **permute**：dims 必须是 range(rank) 的一个排列；输出按 dims 重排。
- **concat**：所有输入 rank 相同；除 axis 外各维相等；输出在 axis 上求和。
- **slice**：n=x[axis]；`L=len(range(*builtins.slice(start, end, step).indices(n)))`，要求 L≥1；输出把 axis 那一维替换为 L。
- **split**：sum(sizes)==x[axis]；输出 len(sizes) 个形状。
- **pad**：k=len(pads)//2，要求 k≤rank。第 i 对（i 从 0 开始）作用在维 rank-1-i 上：新长度 = 原长度 + pads[2i] + pads[2i+1]。与 `torch.nn.functional.pad` 的顺序一致。

## 8. typecheck.py

```python
@dataclass
class CheckResult:
    diags: list[Diag]
    shapes: dict[str, tuple | None]    # 每个值 id 的形状；无法推出时为 None
    layouts: dict[str, str]            # 每个值 id 的布局标签
    @property
    def ok(self) -> bool: ...          # 没有 severity=="error" 的 diag

def check(program: Program, weights: Mapping[str, Sequence[int]]) -> CheckResult
def load_weight_shapes(path) -> dict[str, tuple[int, ...]]   # JSON {名字: [维度...]}
```

按 op 顺序逐个检查：

1. **种类（E_KIND）**：Plain 参数只能作为 add、sub、mul 的第 2 个输入，出现在其他任何位置都报错。
2. **布局**（v1 只比较标签；R4 会细化）：
   - Cipher 参数的标签是 `"raw"`，Plain 参数是 `"plain"`。
   - pack 的输入必须是 raw，否则报 E_LAYOUT_REPACK；输出标签取 attrs.layout。
   - 其他 op 的任一张量输入是 raw 时报 E_LAYOUT_UNPACKED（relayout 也一样）。
   - binary 类 op 的所有非 plain 输入标签必须相同，否则报 E_LAYOUT_MISMATCH。
   - relayout 的输出取 attrs.layout，其他 op 的输出取第一个输入的标签。
3. **形状**：任一输入形状为 None 时跳过本 op，输出形状全部记为 None，这样同一个问题不会连锁报错。否则调用 `infer`：ShapeError 报 E_SHAPE，WeightError 按它携带的 code 报；出错时本 op 的输出形状记为 None。
4. 全部 op 检查完后：
   - 既没被任何 op 使用、也不是程序输出的值（参数也算），每个报一条 W_UNUSED；
   - header.refresh=="compiler" 时，每个 refresh op 报一条 W_REFRESH_COMPILER（提示它会被 lower 成恒等）。

diag 的 lineno 取对应 op 的 lineno。W_UNUSED 对参数取参数所在行，对 op 输出取产生它的 op 的行。

## 9. 样例文件（逐字照抄）

### 9.1 examples/mlp.ptdsl

```python
# PT-DSL v1 sample: two-layer MLP with square activation.
@fhe.program(slots=8192)
def mlp(x: Cipher[1, 64]):
    x = pack(x, layout="flat")
    h = linear(x, w=W["fc1.weight"], b=W["fc1.bias"], algo="bsgs")
    h = act(h, fn="square", method="exact")
    out = linear(h, w=W["fc2.weight"], b=W["fc2.bias"], algo="diag")
    return out
```

### 9.2 examples/resnet20.ptdsl

```python
# PT-DSL v1 sample: ResNet20 (CIFAR-10).
# Weight names follow workloads/resnet20/model_reference.py (B line).
@fhe.program(slots=16384, refresh="compiler", placement="depth-dp")
def resnet20(x: Cipher[1, 3, 32, 32]):
    x = pack(x, layout="mux", gap=1)
    with region("stem", level_budget=16):
        y = conv2d(x, w=W["conv1.weight"], bn="bn1", stride=1, pad=1, algo="mpcb")
        y = act(y, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-16.0, 16.0))
    y = basic_block(y, "layer1.0")
    y = basic_block(y, "layer1.1")
    y = basic_block(y, "layer1.2")
    y = down_block(y, "layer2.0", 8)
    y = basic_block(y, "layer2.1")
    y = basic_block(y, "layer2.2")
    y = down_block(y, "layer3.0", 16)
    y = basic_block(y, "layer3.1")
    y = basic_block(y, "layer3.2")
    z = mean(y, axis=(2, 3))
    z = relayout(z, layout="flat")
    out = linear(z, w=W["linear.weight"], b=W["linear.bias"], algo="bsgs")
    return out


def basic_block(x, p):
    t = conv2d(x, w=W[p + ".conv1.weight"], bn=p + ".bn1", stride=1, pad=1, algo="mpcb")
    t = act(t, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-16.0, 16.0))
    t = conv2d(t, w=W[p + ".conv2.weight"], bn=p + ".bn2", stride=1, pad=1, algo="mpcb")
    t = add(t, x)
    t = act(t, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-16.0, 16.0))
    return t


def down_block(x, p, cpad):
    t = conv2d(x, w=W[p + ".conv1.weight"], bn=p + ".bn1", stride=2, pad=1, algo="mpcb")
    t = act(t, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-16.0, 16.0))
    t = conv2d(t, w=W[p + ".conv2.weight"], bn=p + ".bn2", stride=1, pad=1, algo="mpcb")
    s = slice(x, axis=2, start=0, end=None, step=2)
    s = slice(s, axis=3, start=0, end=None, step=2)
    s = pad(s, pads=(0, 0, 0, 0, cpad, cpad))
    t = add(t, s)
    t = act(t, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-16.0, 16.0))
    return t
```

### 9.3 examples/shapes.py（只用标准库）

```python
def mlp_weight_shapes() -> dict:
    return {"fc1.weight": (32, 64), "fc1.bias": (32,), "fc2.weight": (10, 32), "fc2.bias": (10,)}

def resnet20_weight_shapes() -> dict:
    """97 个条目。conv1.weight (16,3,3,3)；bn1.{weight,bias,running_mean,running_var} (16,)；
    stage s=1,2,3 的通道 c=16,32,64，block b=0,1,2：
      输入通道 in_c = c；但 (s,b)=(2,0) 时为 16，(3,0) 时为 32
      layer{s}.{b}.conv1.weight (c,in_c,3,3)；layer{s}.{b}.conv2.weight (c,c,3,3)
      layer{s}.{b}.bn1.* 和 bn2.* 各 4 个，都是 (c,)
    linear.weight (10,64)；linear.bias (10,)。"""
```

### 9.4 examples/invalid/

约定：

- 第 1 行写 `# expect: <CODE>`（解析错误），或 `# expect-check: <CODE>`（解析能通过、类型检查报错）。
- expect-check 文件的第 2 行写 `# weights: mlp`。
- 出错那一行的行尾加 `# ERR`；没有 `# ERR` 的文件期望 lineno=0。

用 `H` 代表下面两行：

```python
@fhe.program(slots=8192)
def mlp(x: Cipher[1, 64]):
```

| 文件 | 内容（从第 2 行起；正文 4 空格缩进） |
|---|---|
| syntax.ptdsl | H；`    x = pack(x, layout="flat"   # ERR`；`    return x` |
| no_entry.ptdsl | `def mlp(x):`；`    return x` |
| two_entries.ptdsl | H；`    y = pack(x, layout="flat")`；`    return y`；空行；`@fhe.program(slots=8192)`；`def b(x: Cipher[1, 64]):   # ERR`；`    y = pack(x, layout="flat")`；`    return y` |
| bad_header.ptdsl | `@fhe.program(slots=1000)   # ERR`；`def mlp(x: Cipher[1, 64]):`；`    y = pack(x, layout="flat")`；`    return y` |
| agent_header.ptdsl | `@fhe.program(slots=8192, refresh="agent")   # ERR`；`def mlp(x: Cipher[1, 64]):`；`    y = pack(x, layout="flat")`；`    return y` |
| bad_param.ptdsl | `@fhe.program(slots=8192)`；`def mlp(x: Tensor[1, 64]):   # ERR`；`    y = pack(x, layout="flat")`；`    return y`（期望 E_PARAM） |
| for_loop.ptdsl | H；`    y = pack(x, layout="flat")`；`    for i in range(2):   # ERR`；`        y = act(y, fn="square", method="exact")`；`    return y` |
| if_expr.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = act(y, fn="square", method="exact") if True else y   # ERR`；`    return z` |
| arith.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = y * 2   # ERR`；`    return z` |
| nested_call.ptdsl | H；`    z = act(pack(x, layout="flat"), fn="square", method="exact")   # ERR`；`    return z` |
| unknown_op.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = relu(y)   # ERR`；`    return z` |
| undefined.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = act(q, fn="square", method="exact")   # ERR`；`    return z` |
| arity.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = add(y)   # ERR`；`    return z` |
| attr_unknown.ptdsl | H；`    y = pack(x, layout="flat", stride=2)   # ERR`；`    return y` |
| attr_missing.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = linear(y, b=W["fc1.bias"])   # ERR`；`    return z` |
| attr_type.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = act(y, fn="square", method=3)   # ERR`；`    return z` |
| attr_value.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = act(y, fn="relu", method="minimax_comp", deg=7, domain=(-4.0, 4.0))   # ERR`；`    return z` |
| weight_ref.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = linear(y, w=W)   # ERR`；`    return z` |
| tuple_target.ptdsl | H；`    y = pack(x, layout="flat")`；`    a, b = act(y, fn="square", method="exact")   # ERR`；`    return a` |
| return_not_last.ptdsl | H；`    y = pack(x, layout="flat")`；`    return y   # ERR`；`    z = act(y, fn="square", method="exact")` |
| macro_recursion.ptdsl | H；`    y = pack(x, layout="flat")`；`    z = f(y)`；`    return z`；空行；`def f(a):`；`    b = f(a)   # ERR`；`    return b` |
| nested_region.ptdsl | H；`    y = pack(x, layout="flat")`；`    with region("a", level_budget=4):`；`        with region("b", level_budget=2):   # ERR`；`            z = act(y, fn="square", method="exact")`；`    return z` |
| dunder_name.ptdsl | H；`    y__1 = pack(x, layout="flat")   # ERR`；`    return y__1` |
| tc_unpacked.ptdsl | 第 2 行 `# weights: mlp`；H；`    h = linear(x, w=W["fc1.weight"], b=W["fc1.bias"])   # ERR`；`    return h` |
| tc_repack.ptdsl | 第 2 行 `# weights: mlp`；H；`    a = pack(x, layout="flat")`；`    b = pack(a, layout="flat")   # ERR`；`    return b` |
| tc_layout_mismatch.ptdsl | 第 2 行 `# weights: mlp`；H；`    a = pack(x, layout="flat")`；`    b = relayout(a, layout="rep", copies=2)`；`    c = add(a, b)   # ERR`；`    return c` |
| tc_weight_missing.ptdsl | 第 2 行 `# weights: mlp`；H；`    a = pack(x, layout="flat")`；`    h = linear(a, w=W["fc9.weight"])   # ERR`；`    return h` |
| tc_shape.ptdsl | 第 2 行 `# weights: mlp`；H；`    a = pack(x, layout="flat")`；`    h = linear(a, w=W["fc1.weight"], b=W["fc1.bias"])`；`    c = add(a, h)   # ERR`；`    return c` |
| tc_kind.ptdsl | 第 2 行 `# weights: mlp`；`@fhe.program(slots=8192)`；`def mlp(x: Cipher[1, 64], p: Plain[1, 64]):`；`    a = pack(x, layout="flat")`；`    c = add(p, a)   # ERR`；`    return c` |

期望 code 就是文件名对应的那一类：

- syntax→E_SYNTAX，no_entry→E_NO_ENTRY，two_entries→E_MULTI_ENTRY；
- bad_header 和 agent_header→E_HEADER，bad_param→E_PARAM；
- for_loop→E_STMT；if_expr、arith、nested_call→E_EXPR；
- unknown_op→E_UNKNOWN_OP，undefined→E_UNDEFINED，arity→E_ARITY；
- attr_unknown→E_ATTR_UNKNOWN，attr_missing→E_ATTR_MISSING，attr_type→E_ATTR_TYPE，attr_value→E_ATTR_VALUE；
- weight_ref→E_WEIGHT_REF，tuple_target→E_TARGET，return_not_last→E_RETURN；
- macro_recursion→E_MACRO，nested_region→E_REGION，dunder_name→E_NAME；
- tc_unpacked→E_LAYOUT_UNPACKED，tc_repack→E_LAYOUT_REPACK，tc_layout_mismatch→E_LAYOUT_MISMATCH；
- tc_weight_missing→E_WEIGHT_MISSING，tc_shape→E_SHAPE，tc_kind→E_KIND。

上表 29 行，对应 29 个文件。

## 10. printer.py

```python
def print_program(program: Program) -> str
```

输出展开后的规范文本，末尾带一个 `\n`：

- 第 1 行是 `@fhe.program(...)`，按 slots、refresh、placement、l_max、boot_out_level 的顺序列出，值为 None 的省略，字符串用双引号。
- 第 2 行是 `def <name>(<参数>):`，参数写成 `x: Cipher[1, 64]`，参数之间用 `", "` 分隔。
- 函数体每行一个 op，缩进 4 空格，格式为 `<输出> = <kind>(<输入>, <属性>)`：
  - 参数保留原名；op 的输出按定义顺序依次命名为 `v0`、`v1`……（split 的多个输出依次各占一个编号，写成 `v4, v5 = split(...)`）。
  - 属性按 spec 声明顺序列出，跳过值为 None 的项（默认值也照常写出），格式为 `name=value`。
  - 值的写法：WeightRef 写成 `W["name"]`；str 用 `json.dumps`；bool 写 True/False；int 写 `str(v)`；float 写 `repr(v)`；tuple 写 `(a, b)`，单元素写 `(a,)`，元素按同样规则递归。
- 连续属于同一 region 的 op 包在 `    with region("<name>", level_budget=<n>):` 里，内部缩进 8 空格。
- 最后一行是 `    return v3`，多个输出时用 `", "` 连接。

**MLP 的期望输出（测试逐字比较）：**

```python
@fhe.program(slots=8192, refresh="compiler", placement="depth-dp")
def mlp(x: Cipher[1, 64]):
    v0 = pack(x, layout="flat", gap=1)
    v1 = linear(v0, w=W["fc1.weight"], b=W["fc1.bias"], algo="bsgs")
    v2 = act(v1, fn="square", method="exact")
    v3 = linear(v2, w=W["fc2.weight"], b=W["fc2.bias"], algo="diag")
    return v3
```

## 11. check.py（命令行入口）

```
python3 -m pt_agent.dsl.check FILE [--weights mlp|resnet20|PATH.json] [--print]
```

- 解析失败：打印 `ERROR <code> line <n>: <msg>`，退出码 2。
- 给了 `--weights` 时做类型检查，每条 diag 用 `str(diag)` 打印一行；有 error 时退出码 1，否则 0。
- `--print` 时，在最后把 `print_program` 的结果打印到 stdout。
- `--weights` 的值是 mlp 或 resnet20 时，使用 `examples/shapes.py` 里对应的函数。

## 12. 测试（文件名以 test_r1_ 开头，unittest）

用 `Path(__file__).parents[1] / "examples"` 定位样例文件。

**test_r1_parser.py**

1. mlp：op kind 序列为 `["pack","linear","act","linear"]`；`header == Header(slots=8192, refresh="compiler", placement="depth-dp", l_max=None, boot_out_level=None, lineno=2)`；params 为 `(Param("x","cipher",(1,64),3),)`；outputs 长度为 1。
2. resnet20：
   - 共 57 个 op；`Counter(op.kind)` 等于 `{conv2d:19, act:19, add:9, slice:4, pad:2, pack:1, mean:1, relayout:1, linear:1}`；
   - `regions` 只有 "stem"，level_budget=16；region=="stem" 的 op 恰好 2 个；
   - 第一个 macro_stack 为 `("basic_block",)` 的 conv2d，其 `attrs["w"] == WeightRef("layer1.0.conv1.weight")`、`attrs["bn"] == "layer1.0.bn1"`、`stride == (1,1)`、`dilation == (1,1)`、`groups == 1`、`bn_eps == 1e-5`；
   - 第一个 pad 的 `attrs["pads"] == (0,0,0,0,8,8)`，第二个是 `(0,0,0,0,16,16)`；
   - 第一个 act 的 `attrs["domain"] == (-16.0, 16.0)`，`degrees == (15,15,27)`。
3. SSA：mlp 中第二个 linear 的输入等于 act 的输出；所有 op 输出 id 互不相同。
4. 遍历 invalid/ 里所有 `# expect:` 文件：`parse` 抛出 `PTDSLError`，code 与第 1 行写的一致，lineno 等于带 `# ERR` 那一行的行号（没有标记时为 0）。
5. 每个抛出的 code 都在 `PARSE_CODES` 里；每个 diag 的 code 都在 `CHECK_CODES` 里。
6. 源码里没有 `exec(`、`eval(`、`compile(`：读入 `dsl/` 下全部 .py 做字符串检查。

**test_r1_printer.py**

1. mlp 的打印结果与 §10 的期望文本逐字相等。
2. 对 mlp 和 resnet20：s1=print(parse(src))，s2=print(parse(s1))，要求 s1==s2，且 `canonical(parse(src)) == canonical(parse(s1))`。
3. resnet20 打印出的文本里包含 `with region("stem", level_budget=16):`、`domain=(-16.0, 16.0)`、`bn_eps=1e-05`，并且不包含 `def basic_block`。
4. 内联一个含 split 的小程序 `a = pack(x, layout="flat")`；`b, c = split(a, axis=1, sizes=(32, 32))`；`d = add(b, c)`，确认打印出 `v1, v2 = split(v0, axis=1, sizes=(32, 32))`，并且往返稳定。

**test_r1_shapes.py**（直接调用 `infer`）

| 用例 | 期望 |
|---|---|
| conv2d，x (1,16,32,32)，w (32,16,3,3)，stride (2,2)，pad (1,1) | (1,32,16,16) |
| conv2d，groups=16，x (1,16,8,8)，w (16,1,3,3)，pad (1,1) | (1,16,8,8) |
| conv2d，w (32,8,3,3) 配 x (1,16,8,8) | WeightError E_WEIGHT_SHAPE |
| avgpool，k (2,2)，stride None→(2,2)，x (1,16,32,32) | (1,16,16,16) |
| slice，axis=2，step=2，end=None，x (1,16,32,32) | (1,16,16,32) |
| pad，pads (0,0,0,0,8,8)，x (1,16,16,16) | (1,32,16,16) |
| mean，axis (2,3)，x (1,64,8,8) | (1,64)；keepdim=True 时为 (1,64,1,1) |
| mean，对 (64,) 取 axis (0,) | ShapeError |
| linear，w (10,64)，x (1,64) | (1,10)；w (10,63) 时为 E_WEIGHT_SHAPE |
| reshape，shape (1,-1)，x (1,4,4) | (1,16)；shape (1,5) 时为 ShapeError |
| permute，dims (0,2,1)，x (1,3,4) | (1,4,3)；dims (0,0,1) 时为 ShapeError |
| concat，axis 1，(1,3)+(1,5) | (1,8)；(1,3)+(2,5) 时为 ShapeError |
| split，sizes (2,2)，x (1,4) | [(1,2),(1,2)]；sizes (2,3) 时为 ShapeError |
| matmul，(2,3,4)×(2,4,5) | (2,3,5) |

**test_r1_typecheck.py**

1. mlp 配 mlp 权重：`ok` 为真，输出形状 (1,10)，没有 W_UNUSED。
2. resnet20 配 resnet20 权重：`ok` 为真，输出形状 (1,10)；第一个 stride=2 的 conv2d 输出形状为 (1,32,16,16)；第一个 pad 输出 (1,32,16,16)；mean 输出 (1,64)。`len(resnet20_weight_shapes()) == 97`。
3. 遍历 invalid/ 里所有 `# expect-check:` 文件：`parse` 成功；`check` 的结果里有 code 匹配、lineno 等于 `# ERR` 行的 error diag。
4. mlp 权重里把 fc1.weight 改成 (32,63)：报 E_WEIGHT_SHAPE，lineno=5。
5. 一个多出未使用值的小程序：`a = pack(x, layout="flat")`；`b = act(a, fn="square", method="exact")`；`return a`。结果里有 W_UNUSED（lineno 为 b 那一行），而且 ok 仍然为真。
6. 在 refresh=compiler 的程序里用 `refresh(a)`：出现 W_REFRESH_COMPILER。
7. 带 tc_shape 那种错误时，后续用到 None 形状的 op 不再报 E_SHAPE：在 tc_shape 程序的 `c = add(a, h)` 之后加 `d = act(c, fn="square", method="exact")` 并返回 d，E_SHAPE 恰好只有 1 条。

**test_r1_cli.py**（subprocess，cwd=scripts/，用 `sys.executable -m pt_agent.dsl.check`）

1. `examples/mlp.ptdsl --weights mlp`：退出码 0。
2. `examples/invalid/if_expr.ptdsl`：退出码 2，stdout 里包含 `E_EXPR`。
3. `examples/invalid/tc_shape.ptdsl --weights mlp`：退出码 1，stdout 里包含 `E_SHAPE`。
4. `examples/resnet20.ptdsl --weights resnet20 --print`：退出码 0，stdout 里包含 `= linear(`。

运行：

```bash
cd DSL/DSL_agent/scripts && python3 -m unittest discover -s pt_agent/tests -p 'test_r1_*.py' -v
```

## 13. 不要做

- 不 import numpy、torch、hecate，也不 import `scripts/baseline/` 里的任何模块（本轮是纯标准库）。
- 不对 DSL 文本做 exec/eval/compile；不调用 LLM；不编译；不跑 CKKS。
- 不支持 for、if、region 嵌套，也不支持宏里写 region；不实现 kernel，不做布局细节（这些在后面的轮次）。
- 不新增 §1 以外的文件；不修改 AGENTS.md、PLAN.md 和本 spec；不 commit。
- spec 有矛盾或者写不通的地方，不要自己选，写进 STATUS.md 的"阻塞"一栏，然后停下。

## 14. STATUS.md

把"任务卡"表换成 PLAN.md §3.3 的轮次表，列出 R1–R7（R1 写本轮结果，其余为"未开始"），M1 表保留。"阻塞"栏删掉 PT_RESULTS 那一条（已定为 `/home/lhohy/pt_agent_results`），其余照旧。

## 15. 完成标准

- `test_r1_*.py` 全部通过，在汇报里贴出 `-v` 的输出。
- 汇报里列出：改动的文件、与 spec 不一致之处（没有就写"无"）、未完成项。
