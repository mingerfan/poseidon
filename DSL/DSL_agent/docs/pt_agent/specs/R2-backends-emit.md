# R2：三个后端（Mock / Count / Emit）+ Nix 运行入口（离线）

目标日期：2026-10-22。本轮不调用真实的 Hecate 前端、编译器和 CKKS 运行时，那部分放在 R2b。

## 0. 背景（只读，不需要实现）

- 后续所有 kernel 都只写一份，用同一套接口在三个后端上运行：
  - **MockExpr**：numpy 明文仿真；
  - **CountExpr**：统计 op 次数、乘法深度和 level；
  - **EmitBackend**：生成一份独立的 Hecate 前端程序 `candidate_trace.py`，外加常量文件 `constants.npz`。
- R2b 会把 Emit 生成的程序放进现有的 `baseline/poseidon_candidate_sandbox.py` 沙箱，运行后得到 `/out/candidate_trace.mlir`，再交给 `hecate-opt` 编译。所以生成程序的文件名、路径和函数名必须严格按 §5 来写。
- 槽位旋转的约定与现有流水线保持一致：`rotate(k)` 等价于 `np.roll(v, -k)`。参照 `baseline/first_output_dsl_math.py` 的 `_Value.rotate`。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/tools/__init__.py                  新：空
pt_agent/tools/nixrun.py                    新：§2
pt_agent/backends/__init__.py               新：导出 MockBackend, CountBackend, EmitBackend, BackendError, LevelError, run_program
pt_agent/backends/base.py                   新：§3
pt_agent/backends/mock.py                   新：§4.1
pt_agent/backends/count.py                  新：§4.2
pt_agent/backends/emit.py                   新：§5
pt_agent/tests/fixtures/__init__.py         新：空
pt_agent/tests/fixtures/fake_hecate_expr.py 新：§6
pt_agent/tests/test_r2_backends.py          新：§7
pt_agent/tests/test_r2_emit.py              新：§7
../docs/pt_agent/STATUS.md                  改：R2 一行
```

不修改 R1 的任何文件。

## 2. tools/nixrun.py：在项目的 Nix shell 里运行 venv 的 Python

系统 `python3` 里没有 numpy，venv 的 Python 只能在 Nix shell 里运行。本轮起，凡是需要 numpy 的测试都走这个入口。

```
cd DSL/DSL_agent/scripts
python3 -m pt_agent.tools.nixrun [--seconds 600] -- <传给 venv python 的参数...>
```

实现步骤：

1. 若环境变量未设置，就设 `POSEIDON_PLATFORM=aarch64-linux`、`POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work`。这一步必须在 import `hecate_python_env` 之前完成。
2. 把 `scripts/baseline` 加入 `sys.path`，然后 `from hecate_python_env import enter_nix, ROOT, VENV`。
3. `scripts_dir = Path(__file__).resolve().parents[2]`，即 `DSL/DSL_agent/scripts`。
4. 拼命令字符串：

   ```python
   command = ("cd " + shlex.quote(str(scripts_dir)) + " && env LD_LIBRARY_PATH=\"$HECATE_PYTHON_LIBRARY_PATH\" "
              "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
              + shlex.quote(str(VENV / "bin/python")) + " -B " + shlex.join(args))
   ```

5. `os.chdir(ROOT)`。`enter_nix` 内部使用的是相对仓库根的路径，所以必须先切换目录。
6. `sys.exit(enter_nix(command, seconds=seconds))`。

不要往 Nix shell 里传任何别的环境变量；它是 `--pure` 模式，测试也不能依赖 `PT_RESULTS`。

本轮测试命令：

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun -- -m unittest discover -s pt_agent/tests -p 'test_r2_*.py' -v
```

R1 的测试照旧用系统 `python3` 运行，不受影响。

## 3. backends/base.py：统一接口

```python
class BackendError(Exception): ...           # 接口误用：两个 Plain 相加、跨后端混用、步长不是 int 等
class LevelError(BackendError):              # 只有 CountBackend 会抛
    def __init__(self, op_index: int, msg: str): ...

class Backend:
    slots: int                               # 2 的幂，≥2
    def input(self, index: int): ...         # 第 index 个密文输入，index 从 0 开始且必须连续使用
    def const(self, vec): ...                # 返回 Plain；vec 长度为 1（广播到 slots）或等于 slots；元素必须有限
    def output(self, values: list) -> None   # 声明输出，只能调用一次，至少 1 个；每项必须是本后端的密文值，且不能直接是某个 input

def run_program(backend: Backend, fn, n_inputs: int):
    """xs = [backend.input(i) for i in range(n_inputs)]；outs = fn(backend, *xs)；
    outs 可以是单个值或 list/tuple，统一转成 list 后调用 backend.output(outs)；返回这个 list。"""
```

### 3.1 密文值（三个后端都要支持）

- 二元运算 `a + b`、`a - b`、`a * b`，以及反向运算 `b + a`、`b - a`、`b * a`。另一个操作数可以是：
  - 同一后端的密文值；
  - 同一后端的 Plain；
  - Python 的 int 或 float（bool 不行），相当于 `backend.const([v])`。
- 一元运算 `-a`、`a.rotate(k)`、`a.bootstrap()`。
- `rotate(k)`：k 必须是 int（bool 不行）。先算 `k % slots`，为 0 时直接返回 a 本身，不产生 op。
- Plain 和 Plain 之间做运算，或者对 Plain 调用 rotate/bootstrap，都抛 `BackendError`。Plain 本身只支持出现在 `+`、`-`、`*` 的一侧。
- 不同后端实例的值混用，抛 `BackendError`。

### 3.2 op 分类（Count 和 Emit 共用这套口径）

| op | 触发方式 |
|---|---|
| add | 密文 + 密文 / Plain / 标量 |
| sub | 密文 − 任意，或 Plain/标量 − 密文 |
| neg | `-a` |
| mul_cc | 密文 × 密文 |
| mul_cp | 密文 × Plain / 标量 |
| rot | 步长取模后不为 0 的 rotate |
| boot | `bootstrap()` |

## 4. Mock 与 Count

### 4.1 mock.py：`MockBackend(slots)`

- 值是长度为 slots 的 `np.float64` 数组，创建后设为只读（`flags.writeable = False`）。
- `input(i)` 从 `MockBackend.bind(inputs)` 设定的输入里取值。`bind` 接收 `list[np.ndarray]`，每个长度都等于 slots；必须先 bind 再 input，否则抛 `BackendError`。
- 运算就是 numpy 的逐元素运算；rotate 是 `np.roll(v, -k)`；bootstrap 原样复制。
- `output` 保存结果，之后通过属性 `outputs: list[np.ndarray]` 读取。
- 计算中出现非有限值时抛 `BackendError`：用 `np.errstate(over='raise', invalid='raise')` 包住运算。

### 4.2 count.py：`CountBackend(slots, input_level=None, boot_out_level=None)`

每个值记录 `depth`（自上次 bootstrap 以来的乘法深度）、`total_depth`（不因 bootstrap 清零），以及 `level`（`input_level` 为 None 时 level 也为 None，即不追踪）。规则：

| op | depth / total_depth | level |
|---|---|---|
| add, sub | 两个密文操作数取 max；Plain 视为 0 | 两个密文取 min；有 Plain 时取密文的值 |
| neg, rot | 不变 | 不变 |
| mul_cc | max + 1 | min − 1 |
| mul_cp | 密文的 depth + 1 | 密文的 level − 1 |
| boot | depth 置 0；total_depth 不变 | 置为 boot_out_level；为 None 时抛 BackendError |

- level 算出来 < 0 时抛 `LevelError(op_index, msg)`。op_index 是这个 op 在全部 op 中的序号，从 0 开始。
- `stats()` 返回一个 dict：

  ```python
  dict(counts={"add":..,"sub":..,"neg":..,"mul_cc":..,"mul_cp":..,"rot":..,"boot":..},
       ops_total=上面各项之和, rot_steps=sorted(list(取模后的步长集合)),
       distinct_consts=按内容去重后的 Plain 个数（标量先扩展成 slots 长度再去重）,
       max_depth=所有值的 depth 最大值, max_total_depth=所有值的 total_depth 最大值,
       min_level=所有值 level 的最小值或 None, outputs=输出个数)
  ```

## 5. emit.py：`EmitBackend(slots, *, frontend="/frontend/hecate/expr.py", constants="/out/constants.npz", save_dir="/out")`

- **命名**：第 i 个输入叫 `x{i}`；op 结果按产生顺序依次叫 `t0`、`t1`……；Plain 按内容去重后依次叫 `k0`、`k1`……。标量先扩展成长度为 slots 的向量再去重。
- **Plain 的定义行**：在第一次被使用的那个 op 行之前，插入 `    kN = hc.Plain(C["kN"])`。
- **op 行的写法**：

  | 情况 | 写法 |
  |---|---|
  | add、mul_cc、mul_cp | 密文写在左边，例如 `t0 = x0 * k0`、`t3 = t1 + t2` |
  | 两个密文 | 保持调用时的左右顺序 |
  | sub | 保持调用时的左右顺序，例如 `t4 = k1 - t3` |
  | neg | `t5 = -t4` |
  | rot | `t6 = t5.rotate(3)`（写取模后的步长） |
  | boot | `t7 = hc.bootstrap(t6)` |

- `stats()` 与 CountBackend 的 `counts`、`ops_total`、`distinct_consts` 三项完全一致，可以直接复用同一个计数器类。
- `render() -> tuple[str, dict[str, np.ndarray]]`：返回程序文本和常量表 `{"k0": array, ...}`，必须在 `output()` 之后调用。
- `write(out_dir)`：写出 `out_dir/candidate_trace.py`，以及用 `np.savez`（不压缩）写出 `out_dir/constants.npz`。没有常量时也要写，内容为空。

### 5.1 程序文本格式（逐字遵守）

```
# Generated by pt_agent.backends.emit. Trusted lowering output; do not edit.
# ops: add=A sub=S neg=N mul_cc=M mul_cp=P rot=R boot=B consts=K
import importlib.util
import sys
import numpy as np
_spec = importlib.util.spec_from_file_location("hecate_expr", "<frontend>")
hc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hc
_spec.loader.exec_module(hc)
C = np.load("<constants>")


@hc.func("<sig>")
def golden(<x0, x1, ...>):
<函数体：每行缩进 4 空格>
    return <单个输出写 tN；多个输出写 [tA, tB]>


hc.save("<save_dir>", "<save_dir>")
```

- `<sig>` 是 n 个 `c` 用逗号连接，例如两个输入写 `c,c`。
- 路径用 `json.dumps` 加双引号。

**期望输出示例**（测试逐字比较）：slots=4，程序为 `out = x0 * 2.0 + x0.rotate(1)`，参数取默认值：

```
# Generated by pt_agent.backends.emit. Trusted lowering output; do not edit.
# ops: add=1 sub=0 neg=0 mul_cc=0 mul_cp=1 rot=1 boot=0 consts=1
import importlib.util
import sys
import numpy as np
_spec = importlib.util.spec_from_file_location("hecate_expr", "/frontend/hecate/expr.py")
hc = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hc
_spec.loader.exec_module(hc)
C = np.load("/out/constants.npz")


@hc.func("c")
def golden(x0):
    k0 = hc.Plain(C["k0"])
    t0 = x0 * k0
    t1 = x0.rotate(1)
    t2 = t0 + t1
    return t2


hc.save("/out", "/out")
```

末尾带一个换行。常量表中 `C["k0"]` 为 `np.full(4, 2.0)`。

## 6. tests/fixtures/fake_hecate_expr.py（只供测试使用）

用 numpy 模拟 Hecate 前端 API 中被生成程序用到的那一部分，这样不需要真实工具链也能验证生成的程序是否正确：

- `class Expr`：内部是一个 numpy 数组。支持 `+ - *` 及反向运算、取负，以及 `rotate(k)`（`np.roll(v, -k)`）。另一个操作数只能是 Expr。
- `class Plain(Expr)`：`Plain(data)` 把 data 转成 `np.float64` 数组。
- `def bootstrap(x)`：原样返回一个新的 Expr。
- `def func(sig)`：装饰器，把被装饰的函数存进模块级变量 `_REGISTERED`，并原样返回该函数。
- `def save(dirs, cst_dirs)`：
  - 从环境变量 `FAKE_HECATE_INPUTS` 指定的 npz 里读出 `x0`、`x1`……（个数等于 sig 中 c 的个数）；
  - 包装成 Expr，调用 `_REGISTERED`；
  - 把结果统一转成 list，写入 `dirs/fake_outputs.npz`，键名为 `o0`、`o1`……。

## 7. 测试

所有测试都用 `slots=64`，另有说明的除外。随机输入取 `np.random.default_rng(20261022).uniform(-1, 1, 64)`。

**test_r2_backends.py**

1. **P1**：`out = (x0 * c1 + x0.rotate(3)) * x1 - 0.5`，其中 `c1 = np.arange(64)/64`。
   - Mock 结果与 numpy 直接计算的结果完全相等（`np.array_equal`）。
   - Count 的 counts 为 add=1、sub=1、mul_cp=1、mul_cc=1、rot=1，其余为 0；`max_depth=2`；`distinct_consts=2`；`rot_steps=[3]`。
2. **P2 Horner**：系数 a0..a5 = 0.1..0.6；`y = x0*a5 + a4`，然后依次执行 `y = y*x0 + a3`、`a2`、`a1`、`a0`。
   - Mock 与 `np.polyval([0.6,0.5,0.4,0.3,0.2,0.1], x)` 的差 ≤1e-12。
   - Count 为 mul_cp=1、mul_cc=4、add=5，`max_depth=5`，`distinct_consts=6`。
3. **P3 level**：`y = x0*x0; y = y.bootstrap(); out = y*y`，取 `input_level=2, boot_out_level=3`，不报错，`min_level=1`；去掉 bootstrap、取 `input_level=1` 时抛 `LevelError`，且 `op_index=1`。
   - 带 bootstrap 那组：`max_depth=1`，`max_total_depth=2`，counts.boot=1。
4. **P4 求和**：依次对 s ∈ [1,2,4,8,16,32] 执行 `x = x + x.rotate(s)`。
   - Mock 输出的每个槽都等于 `x.sum()`，差 ≤1e-12。
   - Count 为 rot=6、add=6，`rot_steps=[1,2,4,8,16,32]`，`max_depth=0`。
5. **P5 多输出**：`[x0 + x1, x0 * x1]`，Mock 两个输出都正确，Count 的 `outputs=2`。
6. **rotate 约定**：Mock 的 `rotate(3)`、`rotate(-5)` 与 `first_output_dsl_math._Value(v, _Work(10**6)).rotate(k).value` 完全相等。导入时把 `scripts/baseline` 加入 `sys.path`。
7. **边界**：
   - `rotate(64)` 和 `rotate(0)` 返回原对象，Count 不计数；
   - `rotate(True)` 抛 BackendError；
   - 两个 Plain 相加抛 BackendError；
   - 混用两个 MockBackend 实例的值抛 BackendError；
   - `const` 长度既不是 1 也不是 slots 时抛 BackendError；
   - `output` 调用两次抛 BackendError；
   - 输出直接是 input 时抛 BackendError；
   - Mock 计算溢出（`x * 1e308 * 1e308`）抛 BackendError。

**test_r2_emit.py**

1. 用 §5.1 的例子生成文本，与期望文本逐字相等；常量表等于 `{"k0": np.full(4, 2.0)}`。
2. 对 P1–P5（P3 用带 bootstrap 的版本）：
   - Emit 的 counts、ops_total、distinct_consts 与 Count 一致；
   - 把生成程序写到临时目录，其中 `frontend` 指向 fixtures 里 `fake_hecate_expr.py` 的绝对路径，`constants` 和 `save_dir` 指向临时目录；
   - 把输入写成 npz，用 `subprocess.run([sys.executable, "-B", str(tmp/"candidate_trace.py")], env={**os.environ, "FAKE_HECATE_INPUTS": ...})` 运行，退出码为 0；
   - `fake_outputs.npz` 中的每个输出与 Mock 的结果完全相等（`np.array_equal`）。
3. 生成文本中不包含 `exec(`、`eval(`、`__import__`；`import` 只出现在前 5 行。
4. 用 `ast.parse` 能解析生成的文本；`golden` 函数体里每条语句都是 `Assign` 或最后一条 `Return`。

## 8. 不要做

- 不调用真实的 `hecate/expr.py`、`libHecateFrontend.so`、`hecate-opt`、`poseidon_agent_cpu` 或沙箱（这些属于 R2b）。
- 不修改 `scripts/baseline/`、R1 文件、AGENTS.md、PLAN.md 和本 spec；不新增 §1 以外的文件；不 commit。
- 不实现任何 kernel 或 PT-DSL lowering（R4、R5 的内容）。
- 不在 Nix shell 之外安装 numpy，也不修改 venv。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 9. 完成标准

- `test_r2_*.py` 通过 §2 的命令全部通过；`test_r1_*.py` 用系统 python3 依然全部通过。
- 汇报：改动的文件、两组测试的 `-v` 输出、与 spec 不一致之处、未完成项。
