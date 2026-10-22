# R2b：接入真实工具链（trace、编译、CPU CKKS）+ 自举编译试验

前置：R2 已完成（`pt_agent/backends/` 可用）。目标日期：R2 通过后一天。

## 0. 已核实的事实（照用，不需要重新调查）

2026-10-20 已在 `results/pt-agent-toolchain-smoke-20261020/` 实际跑通。

- **工具链来源**：
  - 环境变量 `POSEIDON_SDK_MANIFEST=/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-normal-cpu-approved-overlay-staging-r2680/sdk/ordinary-cpu-selector.json`；
  - 再加 `POSEIDON_PLATFORM=aarch64-linux`、`POSEIDON_WORK_ROOT=/home/lhohy/poseidon-work`；
  - 这三个变量必须在 import 任何 baseline 模块之前设置。
- **沙箱**：`baseline/poseidon_candidate_sandbox.run(payload, output, argv, log, seconds)`。
  - payload 和 output 必须位于 `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/` 之下；
  - 沙箱内 `output` 挂载为 `/out`，Hecate 前端位于 `/frontend/hecate/expr.py`；
  - 编译器是 `/hecate-opt`，编译配置是 `/profile.json`，OperatorSpec 是 `/operator-spec.json`，执行器是 `/poseidon-agent-cpu`；
  - 必须在 Nix shell 内调用，因为它要用到 `$HECATE_PYTHON_LIBRARY_PATH`。
- **trace**：
  - 在 output 目录放好 R2 的 `EmitBackend.write()` 生成的 `candidate_trace.py` 和 `constants.npz`；
  - argv 为 `[str(VENV/"bin/python"), "-B", "/out/candidate_trace.py"]`；
  - 运行后产生 `/out/candidate_trace.mlir`。
- **编译**：基础参数由 `baseline/poseidon_runtime_compile.command(45, serialization_policy=RUNTIME_PLAN_SERIALIZATION_POLICY)` 生成（eva 管线），两个常量从 `compiler_configuration` 导入。
  - 产物：`lowered._hecate_golden.runtime-plan.json` 和 `lowered._hecate_golden.bundle`。
  - eva、dacapo、pars 三种管线在不含自举的程序上都已编译通过。
- **显式写 `hc.bootstrap` 时**：`--pars` 把自举保留在源码写的位置，`--dacapo` 会删掉后重新放置。两者导出 RuntimePlan 时都报 `RuntimePlan Boot requires --boot-profile`，因为普通 CPU 的配置没有自举。
- **CPU 执行器的输入输出**（源码 `src/poseidon/tools/agent_runtime/cpu_runner.cpp`）：
  - 普通协议下，`inputs.json` 是一个 JSON 数组，每项是一个样本，样本数为 1–16；
  - 每个样本是一个数组，每个外部输入对应其中一个向量；
  - 向量长度为 P，P 必须属于 `packed_input_abi.PERIODS = (4,8,…,256)`；执行器把向量平铺到 16384 个槽后加密；
  - 输出 `execution.json` 中的 `outputs` 形状为 `[样本][输出][256]`，即解码前 256 个槽。
- **常量绑定**：执行前要先运行 `poseidon_artifact_gate.bind_periodic_constants(plan, bundle, output/"bound", P)`，把常量平铺到 16384 个槽，生成 `bound/plan.json` 和 `bound/bundle`。
- **执行**：用 `baseline/normal_cpu_overlay_sandbox.run(payload, output, argv, log, 150, selected_library=POSEIDON_BUILD/"libposeidon_shared.so")`，其中 `POSEIDON_BUILD` 来自 `poseidon_runtime_paths`。
  - argv 为 `["/poseidon-agent-cpu","/out/bound/plan.json","/operator-spec.json","/out/bound/bundle","/out/inputs.json","/out/execution.json"]`；
  - 每个 output 目录只能执行一次，因为凭证文件只写一次。
- **Mock 与真实执行的对应关系**：输入和常量都是 P 周期的，所以长度为 P 的 Mock 向量做 `np.roll(-k)`，与 16384 槽上的真实旋转结果一致。比较时取解码结果的前 P 个槽。
- **自举编译配置**：只读复用 `DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config/`，即 `compiler-config.json` 和 `operator-spec.json`（N=65536，自举配置名 `cpu-bootstrap48-level1-to5`，waterline 43）。这套参数只用于编译期统计，本机执行器跑不了。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/toolchain.py                     新：§2
pt_agent/experiments/__init__.py          新：空
pt_agent/experiments/r2b_smoke.py         新：§3
pt_agent/tests/test_r2b_toolchain.py      新：§4
../docs/pt_agent/STATUS.md                改
```

不修改 baseline、R1、R2 的文件，也不修改 `DSL_benchmark/` 下的文件。

## 2. pt_agent/toolchain.py

只能在 Nix shell 内调用的函数，在 docstring 里注明 "inside nix"。

```python
RESULTS_ROOT = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
SDK_SELECTOR = RESULTS_ROOT / "agent-normal-cpu-approved-overlay-staging-r2680/sdk/ordinary-cpu-selector.json"
REPO = Path(__file__).resolve().parents[4]          # /home/lhohy/Code/Poseidon
BOOT_CONFIG_DIR = REPO / "DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config"
COMPILE_MODES = ("eva", "dacapo-optimized", "dacapo-depth-dp", "pars")
BOOT_PROFILES = (None, "qwen48")

def setup_env() -> None:
    """若尚未设置，就设好三个 POSEIDON_* 环境变量，并把 scripts/baseline 加入 sys.path。
    可以重复调用。必须在 import 任何 baseline 模块之前调用。"""

def new_run_dir(tag: str) -> Path:
    """RESULTS_ROOT / f"pt-agent-{tag}-{UTC %Y%m%dT%H%M%S}-{secrets.token_hex(2)}"；
    新建这个目录，并写入 payload.json，内容为 "{}\\n"。tag 只能包含 [a-z0-9-]。"""

def compile_argv(mode: str, boot: str | None, run_dir: Path) -> list[str]:
    """纯函数，不调用沙箱（需先 setup_env）。规则见 §2.1。"""

def trace(run_dir: Path, seconds: int = 600) -> dict:
    """inside nix。要求 run_dir 下已有 candidate_trace.py 和 constants.npz。
    返回 dict(exit=int, seconds=float, mlir=bool)。日志写到 run_dir/'trace.log'。"""

def compile(run_dir: Path, mode: str, boot: str | None = None, seconds: int = 900) -> dict:
    """inside nix。boot=="qwen48" 时，先把 BOOT_CONFIG_DIR 里的两个文件复制为
    run_dir/'boot-compiler-config.json' 和 run_dir/'boot-operator-spec.json'。
    再用 compile_argv 生成的命令调用沙箱，日志写到 run_dir/'compile.log'。
    返回 dict(exit, seconds, plan=bool, log_tail=日志最后 800 个字符)。"""

def plan_stats(run_dir: Path) -> dict:
    """读取 lowered._hecate_golden.runtime-plan.json。统计 initialization+execution+finalization 中
    每一步的 step.get("op") or step["kind"]，得到 ops 字典；
    返回 dict(ops=..., steps=总步数, boots=ops.get("boot",0), values=len(plan["values"]),
    inputs=len(plan["external_inputs"]), outputs=len(plan["final_outputs"]))。"""

def bind(run_dir: Path, period: int) -> dict:
    """inside nix。调用 bind_periodic_constants(plan, bundle, run_dir/'bound', period)，返回它的 record。"""

def write_inputs(run_dir: Path, samples: list[list[np.ndarray]], period: int) -> None:
    """按普通协议写 run_dir/'inputs.json'：
    检查 1 ≤ 样本数 ≤ 16，每个向量长度等于 period 且元素有限，period 属于 PERIODS；
    写 json.dumps([[v.tolist() for v in s] for s in samples])。"""

def execute_cpu(run_dir: Path, seconds: int = 150) -> np.ndarray:
    """inside nix。用 §0 的 argv 和 normal_cpu_overlay_sandbox.run 执行，日志写到 run_dir/'execute.log'。
    退出码不为 0 时抛 RuntimeError，消息里带上日志最后 800 个字符。
    读取 execution.json：要求 encrypted_execution is True、bootstrap is False、executor_secret_key is False；
    返回 np.asarray(report["outputs"], dtype=float64)，形状为 (样本, 输出, 256)。"""
```

### 2.1 compile_argv 的规则（逐条实现）

1. `base = command(45, serialization_policy=RUNTIME_PLAN_SERIALIZATION_POLICY)`，其中包含 `--eva`。
2. 按 mode 处理：

   | mode | 写法 |
   |---|---|
   | `eva` | `argv = base` |
   | `dacapo-optimized` | `argv = command(45, serialization_policy=RUNTIME_PLAN_SERIALIZATION_POLICY, state_management_policy=COMPILER_STATE_POLICY)` |
   | `dacapo-depth-dp` | 同上，再把 `--boot-placement=optimized` 换成 `--boot-placement=depth-dp`（必须恰好替换一处） |
   | `pars` | 把 base 中的 `--eva` 换成 `--pars`（必须恰好一处） |

3. `boot == "qwen48"` 时，在上一步结果上继续替换，每项都必须恰好命中一处，否则抛 `ValueError`：
   - `--ckks-config=/profile.json` → `--ckks-config=/out/boot-compiler-config.json`
   - `--waterline=45` → `--waterline=43`
   - 以 `--runtime-plan-operator-spec-id=` 开头的参数 → `--runtime-plan-operator-spec-id=cpu-qwen-bootstrap48-compiler-contract`
   - 以 `--runtime-plan-operator-spec-sha256=` 开头的参数 → `--runtime-plan-operator-spec-sha256=sha256:` 加上 `BOOT_CONFIG_DIR/operator-spec.json` 文件字节的 sha256。这是编译器要求的参数值，写进 plan 里，不是 pt_agent 做的校验。
   - 以 `--runtime-plan-context-id=` 开头的参数 → `--runtime-plan-context-id=qwen-bootstrap48`
   - 在 `-o` 之前插入 `--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5`

   `boot` 为 None 时不做这一步。boot 只允许与 `dacapo-depth-dp`、`dacapo-optimized`、`pars` 组合，与 `eva` 组合时抛 `ValueError`。
4. 返回新的 list，不修改 base。

## 3. pt_agent/experiments/r2b_smoke.py

用法：

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.experiments.r2b_smoke
```

### 3.1 运行方式

- 外层用系统 `python3` 运行：`subprocess.run([sys.executable, "-m", "pt_agent.tools.nixrun", "--seconds", "2700", "--", "-m", "pt_agent.experiments.r2b_smoke", "--inside"], cwd=scripts 目录, capture_output=True, text=True)`。把子进程的 stdout 和 stderr 原样打印出来，再从 stdout 中找 `REPORT=` 开头的最后一行。
- `--inside` 跑完全部用例，把结果写到 `<summary_dir>/REPORT.json`，并在 stdout 最后一行打印 `REPORT=<路径>`。`summary_dir` 由 `new_run_dir("r2b-summary")` 创建。
- 外层读取 REPORT.json，按用例各写一行 registry（需要外层环境里有 `PT_RESULTS`，没有就报错并退出）：
  - 字段 round="R2b"、model=用例名、mode=编译模式，metrics=该用例的结果，artifacts_dir=用例目录；
  - run_id 用 `registry.new_run_id`。

### 3.2 用例

每个"用例 × 模式"组合使用一个独立的 `new_run_dir`。程序用 R2 的后端构造：同一个函数分别跑一遍 `EmitBackend(P)`（写入 run_dir）和 `MockBackend(P)`。

| 用例 | P | 程序 | 模式 | 步骤 |
|---|---|---|---|---|
| `c1-rotmul` | 256 | `(x0*c1 + x0.rotate(3))*x1 - 0.5`，其中 `c1 = np.arange(256)/256` | `eva`、`dacapo-optimized` | trace → compile → bind → 4 个样本执行 → 与 Mock 比较 |
| `c2-horner` | 256 | R2 的 P2 Horner 多项式（系数 0.1..0.6） | `eva`、`dacapo-optimized` | 同上 |
| `c3-boot-auto` | 256 | 8 次平方：`y=x0`，循环 8 次 `y=y*y` | `dacapo-depth-dp` + `qwen48` | trace → compile → `plan_stats`，不执行 |
| `c3-boot-agent` | 256 | 先做 4 次平方，然后 `y=y.bootstrap()`，再做 4 次平方 | `pars` + `qwen48` | trace → compile → `plan_stats`，不执行 |

- **输入**：`rng = np.random.default_rng(20261023)`，每个样本的每个输入取 `rng.uniform(-1, 1, 256)`。c3 只构造程序和编译，不需要输入。
- **比较**：`ckks = execute_cpu(...)[s, o, :P]`，`mock = Mock 的第 o 个输出`。
  - 判定标准为 `|ckks - mock| <= 1e-5 + 1e-4*|mock|`，逐元素检查；
  - 记录 `max_abs_err`、`max_rel_err`（只在 |mock|>1e-12 的位置计算）、`passed`。
- **c3 记录**：trace 和 compile 的结果、`plan_stats`（重点是 boots）、编译耗时。如果编译失败，就照实记下 `log_tail`，不重试，也不换参数。

### 3.3 REPORT.json 格式

```json
{"round": "R2b", "cases": [
  {"case": "c1-rotmul", "mode": "eva", "run_dir": "...", "trace": {...}, "compile": {...},
   "plan": {...}, "execute": {"samples": 4, "max_abs_err": 0.0, "max_rel_err": 0.0, "passed": true}},
  ...]}
```

不执行的用例，`execute` 写 `null`。任何一步失败时，后续步骤不再运行，`error` 字段记下异常信息，然后继续跑下一个用例。

### 3.4 资源

- 全部步骤串行执行，同一时间只编译一个。
- 执行器的 CPU 时间受沙箱自带限制，不要修改。
- 整个脚本预计 15 分钟以内跑完；超过 40 分钟就中止，并记录已经得到的结果。

## 4. tests/test_r2b_toolchain.py

用 R2 的 nixrun 运行（Nix shell 内）。命令：

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun -- -m unittest discover -s pt_agent/tests -p 'test_r2b_*.py' -v
```

单元测试，不调用沙箱：

1. `compile_argv("eva", None, d)` 中包含 `--eva`，不包含 `--pars` 和 `--dacapo`。
2. `compile_argv("pars", None, d)` 中恰好有一个 `--pars`，没有 `--eva`。
3. `compile_argv("dacapo-depth-dp", None, d)` 中包含 `--dacapo` 和 `--boot-placement=depth-dp`，不包含 `--boot-placement=optimized`。
4. `compile_argv("pars", "qwen48", d)`：
   - 包含 `--ckks-config=/out/boot-compiler-config.json`、`--waterline=43`、`--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5`、`--runtime-plan-context-id=qwen-bootstrap48`；
   - 不包含 `/profile.json` 和 `--waterline=45`；
   - `-o` 紧跟在 boot-profile 参数之后。
5. `compile_argv("eva", "qwen48", d)` 抛 `ValueError`；未知 mode 抛 `ValueError`。
6. 两次调用 `compile_argv` 返回的不是同一个 list 对象，修改其中一个不影响另一个。
7. `write_inputs`：
   - 2 个样本 × 2 个输入、P=8 时，写出的 JSON 读回后形状为 [2][2][8]；
   - P=12（不属于 PERIODS）、17 个样本、向量中含 NaN，这三种情况都抛 `ValueError`。
8. `new_run_dir("t-x")` 创建在 RESULTS_ROOT 之下且 payload.json 存在，测试结束后删除；`new_run_dir("Bad_Tag")` 抛 `ValueError`。

## 5. 不要做

- 不修改 baseline、`DSL_benchmark/`、`third_party/`、SDK 的 JSON 文件和 Qwen 的配置文件。自举配置复制到 run_dir 后只读使用。
- 不加任何新的哈希校验（编译器参数里需要的 sha256 值除外）。不写 handoff 文件或 rNNNN 文档。
- 不尝试在 CPU 上执行带自举的 plan；不改执行器的 CPU 时间限制；失败不重试。
- 不实现 kernel 或 PT-DSL lowering。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 6. 完成标准

- `test_r2b_*.py` 全部通过；R1、R2 的测试依然通过。
- `r2b_smoke` 跑完。在汇报里贴出 REPORT.json 的内容，至少说明三件事：
  - c1、c2 共 4 个组合是否全部通过 CKKS 容差，以及 max_abs_err；
  - c3-boot-auto 和 c3-boot-agent 是否编译成功、boots 各是多少、编译各用了几秒；
  - 失败的组合给出 log_tail。
- registry 中新增对应行；STATUS.md 的 R2b 一行写明结论。
