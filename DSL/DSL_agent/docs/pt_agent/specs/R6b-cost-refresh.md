# R6b：代价模型、由 Agent 惰性放置自举（--pars）、中间文件清理

前置：R6、R6c 已完成。目标日期：2026-10-23。本 spec 的算法和期望值，都已在 2026-10-21 用原型在整网上实测过。

## 0. 背景与已核实的事实

- **Qwen 自举配置（qwen48）**：输入在 level 5；自举的输入必须在 level 1，输出为 level 5；每次乘法后编译器都会做一次 rescale，即每次乘法消耗 1 层（R6c 的 plan 中 mul_cp 6016 次加 mul_cc 456 次，正好等于 rescale 6472 次）。
- **延迟表**：`DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config/compiler-config.json` 中的 `latencyTable`，按 level 索引（下标 1–16，单位微秒），键包括 `earth.add_double`、`earth.add_single`、`earth.mul_double`、`earth.mul_single`、`earth.rotate_single`、`earth.rescale_single`、`earth.modswitch_single`、`earth.negate_single`、`earth.bootstrap_single`。
- **RuntimePlan**：`execution` 中每一步有 `op`、`inputs`（值 id 列表）和 `output`；`values` 中每个值有 `level`。
- **原型实测（ResNet20，惰性策略）**：

  | 配置 | 惰性策略预测的自举数 | `--pars` 编译后 plan 中的自举数 | dacapo depth-dp 的自举数 |
  |---|---|---|---|
  | (127, 1.5) | 461 | — | 383 |
  | (63, 1.4) | 382 | 382，编译 6.2 秒、峰值 1.45 GB | 288，编译 5.5 秒 |

  结论：`--pars` 原样保留 Agent 放置的自举；惰性策略比 depth-dp 多用约 30% 的自举；把阶数从 127 降到 63，depth-dp 的自举数减少 25%。
- **磁盘**：一次整网编译的运行目录约 5.2 GB，本机剩余约 21 GB。所以每次编译完成后都要清理中间文件（用户已批准，见 §4）。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/backends/refresh.py          新：§2
pt_agent/backends/__init__.py         改：导出 RefreshBackend
pt_agent/cost.py                      新：§3
pt_agent/toolchain.py                 改：新增 prune（§4）
pt_agent/planners/rule.py             改：新增参数 refresh
pt_agent/run.py                       改：新增 --refresh，并在编译后调用 prune
pt_agent/experiments/r6b_refresh.py   新：§5
pt_agent/tests/test_r6b_refresh.py    新：§6
pt_agent/tests/test_r6b_cost.py       新：§6
../docs/pt_agent/STATUS.md            改
```

## 2. backends/refresh.py：RefreshBackend

它是一个包装层，可以套在任意后端（Mock、Count、Emit）外面，对外提供与 Backend 相同的接口。

```python
class RefreshBackend:
    def __init__(self, inner, *, input_level: int, boot_in_level: int, boot_out_level: int,
                 latency_table: dict | None = None):
        self.inner = inner; self.slots = inner.slots
    def input(self, i) -> RValue:  return RValue(self, self.inner.input(i), input_level)
    def const(self, vec):          return self.inner.const(vec)      # 直接返回 inner 的 Plain
    def output(self, values):      self.inner.output([v.value for v in values])
    def apply_exact(self, x, fn):  # 仅当 inner 有这个方法时可用
        return RValue(self, self.inner.apply_exact(x.value, fn), x.level)
    def stats(self) -> dict:       # boots、min_level、latency_us（未提供延迟表时为 None）、level_hist={level: 在该 level 执行的 op 数}
```

`RValue` 包住一个 inner 的密文值 `value` 和它的 `level`。规则如下（直接照原型实现）：

- **二元运算**（`+ - *` 及其反向）：另一个操作数是 RValue 时取其 `.value`；是 Plain 或 Python 数时原样传给 inner。
  - **乘法之前**：对每个 RValue 操作数 x，若 `x.level <= boot_in_level`，就用 `self._refreshed(x)` 替换它。
  - 结果的 level：乘法为参与运算的 RValue 中最小 level 减 1；加减法为最小 level。
  - 若结果 level < boot_in_level，抛 BackendError。按这套规则不应该出现这种情况。
- **`-x`、`x.rotate(k)`**：level 不变。rotate 时如果 inner 返回的是原对象（步长取模为 0），就返回原 RValue。
- **`x.bootstrap()`**（程序里显式写的 refresh）：要求 `x.level >= boot_in_level`，计 1 次 boot，调用 `inner` 的 bootstrap，结果 level 为 boot_out_level。
- **`_refreshed(x)`**：用 `id(x)` 作为缓存键，同时把 x 本身也存进缓存，防止它被回收后 id 被复用。缓存中没有时执行 `x.bootstrap()`。同一个值无论被多少次乘法使用，最多只自举一次。
- **延迟累计**（提供了 latency_table 时）：L 取本次运算中 RValue 操作数在刷新后的最小 level。

  | 运算 | 累加的延迟 |
  |---|---|
  | 两个密文相加或相减 | `add_double[L]` |
  | 密文与 Plain 或数相加减 | `add_single[L]` |
  | 取负 | `negate_single[L]` |
  | 旋转（步长不为 0） | `rotate_single[L]` |
  | 两个密文相乘 | `mul_double[L] + rescale_single[L]` |
  | 密文乘 Plain 或数 | `mul_single[L] + rescale_single[L]` |
  | 每次自举 | `bootstrap_single[boot_in_level]` |

  表中键名都省略了前缀 `earth.`。modswitch 和 relinearize 不计入。

## 3. cost.py

```python
PROFILES = {"qwen48": dict(input_level=5, boot_in_level=1, boot_out_level=5,
                           latency=BOOT_CONFIG_DIR / "compiler-config.json")}   # BOOT_CONFIG_DIR 取自 toolchain

def load_latency(profile: str) -> dict          # 读取 latencyTable，返回 {键: list}，键名不带前缀 "earth."

def estimate(program, weights, *, profile: str = "qwen48") -> dict:
    """在 RefreshBackend(CountBackend(slots, boot_out_level=...), ...) 上执行 lower。返回：
    counts、ops_total、max_depth（取 inner Count 的 total depth）、rot_keys=len(rot_steps)、distinct_consts、
    const_bytes = distinct_consts * slots * 8、boots_pred、latency_pred_us、level_hist。
    在本机上应在 10 秒内完成。"""

def plan_latency(plan_path, profile: str = "qwen48") -> dict:
    """对 RuntimePlan 中 execution 的每一步，按下表映射到延迟表的键，level 取 inputs[0] 所在值的 level。
    add_cc 和 sub_cc → add_double；add_cp 和 sub_cp → add_single；negate → negate_single；rotate → rotate_single；
    mul_cc → mul_double；mul_cp → mul_single；rescale → rescale_single；mod_switch → modswitch_single；boot → bootstrap_single。
    其他 op（relinearize、encode 等）不计时，统计次数后记入 untimed。
    返回 dict(latency_us, by_op={op: us}, untimed={op: count}, boots, steps)。"""
```

## 4. toolchain.prune、规划器与运行入口的改动

- `toolchain.prune(run_dir)`：删除 `candidate_trace.mlir`、`constants.npz`、`lowered.mlir`、`lowered.earth.mlir`、`lowered.ckks.mlir` 和 `lowered._hecate_golden.bundle/`，然后写入 `PRUNED.txt`，说明哪些文件被删除、何时删除、用户已批准。保留 RuntimePlan、各类日志、`candidate_trace.py` 和配置文件。`run.py` 的 compile 阶段和 r6b 实验在取得 `plan_stats` 和 `plan_latency` 之后**必须**调用 prune。
- `rule.plan(..., refresh="compiler")`：refresh="agent" 时，头部写成 `@fhe.program(slots=..., refresh="agent", placement="depth-dp", l_max=5, boot_out_level=5)`，程序正文不变。
- `run.py --refresh compiler|agent`，默认 compiler。设为 agent 时：
  - 规划器的头部带上 refresh="agent"；
  - count 阶段改用 `cost.estimate`；
  - emit 阶段用 `RefreshBackend(EmitBackend(slots), …qwen48 的三个 level)`；
  - compile 阶段用 `mode="pars", boot="qwen48"`。
- mock 阶段始终使用不加包装的 MockBackend。RefreshBackend 对 Mock 只会插入恒等的 bootstrap，不改变数值，测试 §6.1-5 会验证这一点。

## 5. experiments/r6b_refresh.py

运行模式同前：外层和内层分开，registry 写 round="R6b"。结果目录为 `$PT_RESULTS/r6b-refresh-<时间戳>`。

对每种配置 (deg, margin) ∈ {(127, 1.5), (63, 1.4)}，图、校准和权重同 R6c，各跑下面两种模式：

| 模式 | 程序 | 编译 |
|---|---|---|
| compiler | 用 `rule.plan(refresh="compiler")` 生成，`EmitBackend` 写出 | `trace(trusted)` → `compile("dacapo-depth-dp", "qwen48", trusted)` |
| agent | 用 `rule.plan(refresh="agent")` 生成，`RefreshBackend(EmitBackend)` 写出 | `trace(trusted)` → `compile("pars", "qwen48", trusted)` |

每个组合记录：`cost.estimate` 的 boots_pred 和 latency_pred_us；编译的 exit、seconds 和 peak_rss_mib；`plan_stats` 中的 boots 和 steps；以及 `plan_latency` 的 latency_us、by_op（只记前 5 项）和 untimed。记录完成后调用 prune。

另外，在 4 张校准图上分别用 `RefreshBackend(MockBackend(batch=4))` 和 `MockBackend(batch=4)` 跑 (63, 1.4)，两者输出必须逐位相等；结果记为 `mock_equal`。

REPORT.json 中再加一张汇总表，4 行，列为 deg、margin、mode、boots、plan_latency_s、compile_s、peak_rss_mib。

## 6. 测试（nixrun）

**test_r6b_refresh.py**，统一使用 slots=64，level 参数取 (input=5, boot_in=1, boot_out=5)：

1. 连续平方 8 次：counts.boot=1，`stats().boots=1`；Mock 结果与不加包装时逐位相等；Emit 的文本中恰好有 1 处 `hc.bootstrap(`。
2. 同一个 level 为 1 的值被两次乘法使用时，只自举 1 次。
3. 两个 level 为 1 的值相乘，自举 2 次。
4. 显式 `x.bootstrap()` 之后的 level 为 5；`rotate(0)` 返回原对象。
5. 提供延迟表时，对"一次 mul_cc 加一次 rotate"的小程序，`latency_us` 等于按 §2 规则由表中数值算出的结果（期望值在测试里用表计算，不要写死数字）。

**test_r6b_cost.py**

1. 对 R3 的 graph 和 calibration 运行 `cost.estimate(parse(rule.plan(..., deg=127, margin=1.5)))`：boots_pred=461，counts 中 mul_cp=6016、rot=2624；deg=63、margin=1.4 时 boots_pred=382。
2. `plan_latency` 作用在一个手写的 3 步小 plan JSON 上，结果与按表手算的值一致，relinearize 被计入 untimed。
3. `toolchain.prune` 在临时目录中删除了指定文件，保留了 plan JSON 和日志，并生成了 PRUNED.txt。

R1–R6c 的测试依然全部通过。

## 7. 不要做

- 不实现除惰性策略之外的自举放置策略；不修改编译器、kernel 和近似算法。
- 不把编译结果或 plan 统计反馈给规划器；代价模型只使用 Count 加延迟表。
- 编译失败不重试。不删除本 spec 和 prune 规定以外的任何文件。

## 8. 完成标准

- 测试全部通过；`r6b_refresh` 跑完。汇报中贴出 4 行汇总表、boots_pred 与 plan 中实际 boots 的对照，以及 mock_equal。
- STATUS.md 新增 R6b 一行。
