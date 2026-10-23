# P2：导出后端——从 PT-DSL 生成结构化 Hecate DSL 包

用户要求（2026-10-23）：给 Agent 加一个导出后端，使用相对路径，从 PT-DSL 生成结构化的 Hecate 包，不再是展开的直线代码。流程变为：

```
PT-DSL（Agent 的决策） → 结构化 Hecate DSL 包（交付物，与 DSL_benchmark 同格式） → 编译器
```

本轮不调用 LLM，不读取报告集，不改变任何 Agent 决策。预计 1 天实现，运行约 1 小时。

## 0. 现状与问题

- `backends/emit.py` 把整个程序展开成一条直线：`candidate_trace.py` 中每个密文运算一行（ResNet20 有数万行），常量放在 `constants.npz` 里按 `k0, k1, …` 编号。
- 生成的文件写死了沙箱路径 `/frontend/hecate/expr.py`、`/out/constants.npz`、`hc.save("/out", "/out")`，要靠 `toolchain.retarget` 改写后才能在本机运行。
- 这份文件无法阅读，也无法交给别人：看不出每层是什么运算、用了哪个近似；离开 pt_agent 的目录约定就无法运行。

团队交付的 DSL 包（`DSL/DSL_benchmark/llm_dsl/Qwen25_24Layer_DSL`）的格式是：

- 结构化的 Python 源码：模型代码调用 kernel 函数，循环保留在 kernel 内；
- `config/` 下以 JSON 显式给出模型参数、近似参数和权重清单；
- `interfaces/` 下给出密文输入输出的接口快照；
- 入口脚本：`trace_*.py` 支持 `--plan`、`--weights`、`--output-dir`、`--bytecode`，并写出 `signature.json`；另有 `prepare_inputs.py`、`unpack_outputs.py`、`check_outputs.py`、`verify_manifest.py`；
- `manifest.json` 记录每个文件的字节数和 sha256，`provenance.json` 记录来源；`README.md` 用中文说明；
- 不含权重，不含大型中间产物；运行时 `import hecate`，使用对方环境中已有的前端。

本轮让 pt_agent 产出同样格式的包，并让编译阶段直接使用这个包。

## 1. 包的目录结构

包目录名为 `<ModelName>_PTAgent_DSL`，其中 ModelName 由程序名转写而来，例如 `resnet20` 对应 `ResNet20_PTAgent_DSL`，转写规则写在代码里，与模型无关。Python 包名使用 PT-DSL 的函数名（例如 `resnet20`），必须是合法标识符，否则报错。

```
ResNet20_PTAgent_DSL/
  README.md                    生成的中文说明（§5）
  manifest.json                与 Qwen 包同字段：package、date、config、scope、validation 标志、files{bytes,sha256}
  provenance.json              来源：PT-DSL 文件、pt_agent 提交号及是否有未提交改动、freeze tag、Agent 运行目录、复制源码的对应关系
  program.ptdsl                Agent 的决策，原样复制
  config/
    model.json                 name、slots、refresh、placement、compile_mode、boot_profile、输入输出形状；refresh="agent" 时另含 input_level/boot_in_level/boot_out_level
    approximations.json        每个 act 运算一项（§2.3），显式给出系数
    weight_schema.json         程序引用的全部权重名和形状（含 BN 前缀展开后的四个张量）
    compiler-config.json       复制自 compiler_profiles/qwen48/compiler-config.json
    operator-spec.json         复制自 compiler_profiles/qwen48/operator-spec.json
    compile_argv.json          编译命令模板（§3.2）
  interfaces/
    default.json               format、function、slots；inputs/outputs 每项给出 name、shape、ciphertext 下标和布局（mux: n,C,H,W,k；flat: n,size）
  resnet20/                    Python 包
    __init__.py
    model.py                   由 PT-DSL 生成的结构化模型（§2.1）
    ops.py                     每种 PT-DSL 运算一个函数（§2.2，固定模板）
    hecate_backend.py          把运算映射到 hecate Expr（§2.4，固定模板）
    numpy_backend.py           明文槽模拟，用于自检（§2.4，固定模板）
    runtime.py                 读取 config、构造并注册 hc.func（固定模板）
    weights.py                 读取 NPZ 并按 weight_schema 检查（固定模板）
    core/                      从 pt_agent 原样复制的算术源码（§2.5）
  trace_model.py               导出 MLIR；--plan 只打印接口，不导入 hecate
  compile_model.py             用 config/ 下的文件调用 hecate-opt（§3.2）
  prepare_inputs.py            明文输入张量 (N,…) → 打包后的槽向量 .npy
  run_numpy.py                 用 numpy_backend 运行整个程序，生成参考输出
  unpack_outputs.py            解密后的槽向量 → 输出张量
  check_outputs.py             按 CKKS 容差 1e-5+1e-4|ref| 比较实际输出与参考
  verify_manifest.py           按 manifest 检查文件
```

所有路径都相对包根目录（`Path(__file__).resolve().parent`）解析。包内不得出现绝对路径，也不得导入 pt_agent。包里不放权重；权重 NPZ 由导出器写在包外（§4）。

## 2. 生成规则

### 2.1 model.py：每个 PT-DSL 运算对应一行函数调用

`pt_agent/export/codegen.py` 把 `Program` 翻译成 `model.py`，例如：

```python
"""resnet20: generated from program.ptdsl by pt_agent.export. Edit program.ptdsl and re-export instead."""
from . import ops


def forward(b, W, A, img):
    v0 = ops.pack(b, img, layout="mux", gap=1)
    v1 = ops.conv2d(b, v0, W, w="conv1.weight", bn="bn1", bn_eps=1e-05, stride=1)
    v2 = ops.act(b, v1, A["v2"])
    ...
    v23 = ops.downsample(b, v17, pads=(8, 8))   # slice v21, slice v22, pad v23
    v24 = ops.add(b, v20, v23)
    ...
    return [v59]
```

- 值的命名与 PT-DSL 相同，便于对照。
- 近似参数一律通过 `A["<值名>"]` 从 config 读取，不内联在代码里。
- 权重只按名称引用，不内联。
- 下采样链 slice→slice→pad 的识别条件与 `lower.py` 完全相同（同一套 consumers 判断），合并为一次 `ops.downsample` 调用，并在注释中写出被合并的三个值名。
- `refresh` 运算生成 `ops.refresh(b, v, enabled=<header.refresh == "agent">)`。
- 每种运算都由通用规则生成，与具体模型无关。遇到 `HANDLED_OPS` 之外的运算，或 lower 不接受的属性组合，导出时报错，错误信息与 lowering 的 `LoweringError` 一致（op id 和行号）。
- 不做跨层的循环折叠，例如把重复的残差块合成 for 循环：这需要识别特定模型的模式，违反通用性原则。结构化体现在 kernel 是函数、kernel 内部保留循环，模型代码每个运算一行。

### 2.2 ops.py：每种 PT-DSL 运算一个函数

这是固定模板（`pt_agent/export/templates/<pkg>/ops.py`），逐条复刻 `lower.py` 中各分支的检查和计算，参数改为显式传入：`pack`、`conv2d`（含 BN 折叠）、`act`、`add`、`downsample`、`mean`、`relayout`、`linear`、`refresh`。

值用 `Tensor(value, layout)` 二元组在函数之间传递。各分支的条件、错误文本和调用的 core 函数都必须与 `lower.py` 相同。`act` 的仿射变换必须与 `act_cheb` 逐个表达式相同（`a=2/(hi-lo)`、`c=-(hi+lo)/(hi-lo)`、`u=x*a+c`）；minimax 必须与 `act_minimax` 相同（`s=x*(1.0/hi)`，逐段 `cheb_eval`，最后 `x*(s+0.5)`）。

**不修改 `lower.py`**（冻结模块）。两份实现的一致性由 §6 的测试 A 和 B 保证。

### 2.3 approximations.json

每个 act 运算一项，键为输出值名，内容为：`op_id`、`lineno`、`fn`、`method`，以及：

- cheb：`deg`、`domain`、`coefficients`（由 `cheb_fit` 计算的 float64 列表）；
- minimax_comp：`degrees`、`domain`、`coefficients`（`minimax_coeffs` 返回的各段列表）；
- exact square：只记录 method。

JSON 写入使用 Python 默认的 float repr，保证 float64 读回后与原值完全相同。包运行时只用这些系数，不再读取 DaCapo 的 `sgn151527.txt`，也不重新拟合。

### 2.4 两个后端

- `hecate_backend.py`：`HecateBackend(Backend)`。输入是 `hc.func` 传入的 Expr。`_apply` 的映射为：add/sub/mul_cc/mul_cp 对应 `+ - *`，neg 对应取负，rot 对应 `.rotate(step)`，boot 对应 `hc.bootstrap`。常量对应 `hc.Plain(data)`，每个 `_Plain` 只创建一次并缓存（与 Emit 中"同一常量只定义一次变量"等价）。Plain 的 scale 使用前端默认值，与 Emit 相同。`hc` 作为参数注入，模块本身不在导入时 `import hecate`。
- `numpy_backend.py`：最小的 float64 槽模拟，运算语义与 `MockBackend._apply` 相同（rot 为 `np.roll(x, -step, axis=-1)`，boot 为复制），支持批量。不需要 watch、noise 或 apply_exact。

两者都继承 `core/base.py` 中复制来的 `Backend`，因此标量广播、常量去重和运算分派与 pt_agent 完全相同。refresh="agent" 时，runtime 用 `core/refresh.py` 的 `RefreshBackend` 包一层，levels 取自 `config/model.json`，与 `run.py` 的 emit 分支相同。

### 2.5 core/：原样复制的源码

原样复制以下文件，provenance.json 记录每个文件的来源路径：

| 包内文件 | 来源 |
|---|---|
| `core/base.py` | `backends/base.py` |
| `core/refresh.py` | `backends/refresh.py` |
| `core/layout.py`、`slots.py`、`conv.py`、`pool.py`、`linear.py` | `kernels/` 同名文件 |
| `core/cheb.py` | 由 `inspect.getsource(kernels.poly.cheb_eval)` 生成，只加上它需要的 import（numpy、`KernelError`、`accumulate`），函数体不改 |
| `core/__init__.py` | 生成，只导出包内用到的名字；不复制 `kernels/__init__.py`，因为它会导入依赖 `paths` 的 poly.py |

复制前检查每个文件的导入只指向 core 内部的模块、numpy 或标准库；不满足就报错，不允许静默改写。

## 3. 入口脚本

### 3.1 trace_model.py

参数：`--plan`、`--weights <npz>`、`--output-dir <新目录>`、`--bytecode`。

- `--plan` 只打印 `interfaces/default.json` 加上 approximations 摘要，不导入 hecate，也不读权重。
- 正常运行时：`import hecate as hc`（由调用方设置 `HECATE`、`HECATE_BUILD`、`PYTHONPATH`），加载权重并检查 schema，注册 `@hc.func("c,…")` 函数，函数名为程序名，然后 `hc.save(out, out[, bytecode=True])`。输出目录必须不存在（与 Qwen 包相同），不覆盖已有产物。
- 输出目录中写 `signature.json`：interfaces 内容，加上 `function`（`_hecate_<name>`）、`mlir` 文件名（hecate 按脚本名命名为 `trace_model.mlir` 或 `.mlirbc`）、`native_export_format`、`weights_binding`（NPZ 文件名和字节数）。

### 3.2 compile_model.py

参数：`--hecate-opt <路径>`、`--input <trace 输出的 mlir>`、`--output-dir <新目录>`、`--dry-run`。

`config/compile_argv.json` 由导出器根据 `toolchain.compile_argv(compile_mode, "qwen48", …)` 生成，把沙箱路径换成占位符：

| 沙箱路径 | 占位符 |
|---|---|
| `/hecate-opt` | `{hecate_opt}` |
| `/out/candidate_trace.mlir` | `{input}` |
| `/out/boot-compiler-config.json` | `{package}/config/compiler-config.json` |
| `/out/lowered.mlir` | `{output_dir}/lowered.mlir` |

出现其他绝对路径时导出报错。compile_model.py 只做占位符替换并执行，不包含任何编译参数逻辑，因此编译参数只有 toolchain 这一个来源。`--dry-run` 只打印 argv。compile_mode 为：refresh="compiler" 用 `dacapo-depth-dp`，refresh="agent" 用 `pars`，与 run.py 相同。

### 3.3 其余脚本

- `prepare_inputs.py --inputs <(N,…) .npy> --output <(N,slots) .npy>`：用 core 的 pack_mux/pack_flat 打包。输入是模型接收的张量，归一化不在包内，README 写明。
- `run_numpy.py --weights --inputs --output`：用 numpy_backend 运行 forward，按输出布局解包，保存参考输出。
- `unpack_outputs.py --signature --slots --output`：按 interfaces 中的布局解包。
- `check_outputs.py --actual --reference`：默认容差 `1e-5+1e-4|ref|`，并给出 argmax 一致率。
- `verify_manifest.py`：与 Qwen 包行为相同。manifest 中的 sha256 是交付格式的一部分（用户要求同格式）；本轮验收不依赖比较编译产物的哈希。

## 4. pt_agent 侧改动

只允许修改或新增以下内容：

1. **新增 `pt_agent/export/`**：`__init__.py`、`codegen.py`、`package.py`、`__main__.py`、`templates/`。
   - `package.py` 提供 `export_package(program_text, program, weights, out_dir, *, run_info=None) -> Path`：写出整个包；在包外写 `<out_dir>/weights/<name>.npz`，只包含 weight_schema 中的名字；最后写 manifest。
   - `__main__.py` 提供 CLI：`python -m pt_agent.export --program <ptdsl> --model <zoo name> --models-dir <dir> --out <dir>`。
   - 模板是普通 .py 文件，pt_agent 不导入它们，导出时复制，只有 model.py 和 README 是生成的。
2. **`run.py`**：
   - 新增 `export` 阶段，位于 count 之后、compile 之前，结果写入 `report["export"]`（包路径、权重路径、model.py 行数、文件数）。
   - `compile` 改为要求 `export`：用 `toolchain.trusted_run` 运行包内的 `trace_model.py`（env 增加 `PYTHONPATH=<DACAPO_ROOT>/python/hecate`，其余与现在 trusted trace 相同），再运行包内的 `compile_model.py`，读取 `plan_stats` 和 `plan_latency`。
   - `emit` 阶段保留，作为旧路径，不再被 compile 依赖。
   - `_server_job` 改为传输包目录和权重 NPZ，脚本只调用包内的 trace_model.py 和 compile_model.py，用 `toolchain.server_env_exports()` 设置环境，不再依赖 pt_agent 源码。
3. **`toolchain.py`**：`plan_stats(run_dir, function="golden")` 增加函数名参数，默认值保持原行为；需要时增加一个计算包 trace 环境变量的小函数。`prune` 对包路径的产物同样适用（只删 R6b §4 批准的中间文件类型）。
4. **`pt_agent/README.md`**：更新流程图和命令，说明交付物是包。
5. **测试**：新增 `tests/test_p2_export.py`。

**不修改**：`lower.py`、`cost.py`、`placement.py`、`kernels/*`、`backends/*`、`planners/*`、`dsl/*`、`intake/*` 以及所有冻结的 Agent 模块。不往 `DSL/DSL_benchmark/` 写入任何内容：放进仓库需要用户另行批准提交。不把权重加入包或 git。

## 5. README.md（生成）

中文，章节与 Qwen 包对应：

1. 模型内容和接口：程序名、slots、输入输出、布局。
2. 近似参数表：每个 act 一行，列出值名、函数、方法、阶数、定义域。
3. 文件入口。
4. 依赖：Python ≥3.10、NumPy；trace 时另需 hecate 前端及其依赖（torch）。
5. 权重 NPZ 的键名约定，以及从 PyTorch state_dict 导出的示例片段。示例只是 README 中的代码，包本身不加载 pickle。
6. 命令：verify、plan、prepare_inputs、run_numpy、trace、compile、unpack、check。
7. 验证状态：从 manifest 的 validation 标志生成，例如"已原生编译，自举 N 次；CKKS 和 GPU 尚未验证"。
8. 说明：本包由 pt_agent 从 program.ptdsl 自动生成，请勿手改。

## 6. 测试（tests/test_p2_export.py）

依赖本机产物的测试使用 `require(...)` 跳过。

被测程序集合 P：
- R8e7 冻结的 4 个模型程序：`pt_result("r8e7-reconfirm-20261022T122558")/models/*/program.ptdsl`；
- 两个基线 `baselines/R6c`、`baselines/R7c-B`；
- 规则规划器以 `--refresh agent` 为 ResNet20 生成的程序（即 R6b 的 pars 路径）。

测试项：

1. **codegen**：对 P 中每个程序，生成的 model.py 能通过 `ast.parse`；forward 中的调用数等于运算数减去被合并的 slice/slice/pad 数；不含数组字面量；所有近似参数都通过 `A[...]` 读取。
2. **测试 A：与 Emit 一致**。对 P 中每个程序，用 pt_agent 的 `EmitBackend` 分别运行 `lower()` 和包的 `forward`（refresh=agent 时两边都用 `RefreshBackend` 包一层），`render()` 得到的文本必须完全相同，常量逐个 `np.array_equal`。这一项证明两条路径发出的运算序列相同。
3. **测试 B：与 Mock 一致**。对 mlp、cnn_plain、resnet20，在留出集 test[9000:9020] 上，包的 `numpy_backend` 输出与 `lower()` 加 `MockBackend` 的输出 `np.array_equal`。
4. **自包含**：
   - 用 ast 扫描包内所有 .py：import 只允许标准库、numpy、包内模块和 `hecate`；
   - 包内所有文本文件中不出现以 `/` 开头的路径字面量，不出现 `WORKSPACE` 或 `REPO` 的字符串；
   - 把包复制到另一个临时目录（改名）并在另一个 cwd 下运行：`trace_model.py --plan` 成功，`run_numpy.py` 在 2 张图上与测试 B 的结果相同。
5. **编译命令**：`compile_model.py --dry-run` 打印的 argv，与 `toolchain.to_host_argv(toolchain.compile_argv(mode, "qwen48", d), d)` 在相同路径替换后逐项相同，两种 mode 都测。
6. **manifest**：verify_manifest 通过；改动包内任意一个文件的一个字节后报错。
7. **run.py**：`--stages plan,check,export` 生成包；只给 `compile` 而没有 `export` 时报错；原有 emit 路径的测试仍然通过。
8. **未支持的运算**：构造一个含 standalone pad 的程序，导出时报与 lowering 相同的错误（op id 和行号）。
9. 既有 315 项测试全部通过。

## 7. 运行

结果目录：`pt_result("p2-export-<时间戳>")`。

1. 导出 P 中全部程序的包。
2. 对每个包：trusted 运行 trace_model.py 和 compile_model.py，读取 plan_stats。
3. **验收**：编译得到的自举数和各类运算计数，必须与原路径记录的值相同：
   - 4 个冻结模型：与 `r8e7-reconfirm` 中各模型 REPORT 的 plan 记录比较（ResNet20 自举 226、mlp 24，其余以记录为准）；
   - 规则 ResNet20 与 agent 刷新：分别为 383 和 461。

   任何不一致都停下来汇报，不改代码去迁就。
4. 每个包运行 `run_numpy.py`（test[9000:9020]）和 `check_outputs.py`，以 Mock 输出为参考，必须完全相同。
5. 记录每个包的 model.py 行数、包文件数，以及对比旧 `candidate_trace.py` 的行数。

## 8. 汇报

- 每个程序：测试 A、B 的结果；编译自举数和运算计数与记录值的对比；model.py 行数与旧直线代码行数。
- 一个包的目录树和 model.py 前 20 行。
- 测试总数和通过数。
- STATUS.md 新增 P2 一行。

## 9. 不要做

- 不修改 §4 未列出的模块，不修改任何冻结的 Agent 模块，不调用 LLM，不读取报告集。
- 不在包中放权重，不向 `DSL/DSL_benchmark/` 写入，不提交。
- 不做针对特定模型的结构折叠。
- 不修改编译器参数；编译参数只来自 toolchain。
