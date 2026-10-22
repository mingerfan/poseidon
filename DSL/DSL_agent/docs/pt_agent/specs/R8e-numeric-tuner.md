# R8e：精调工具（数值旋钮的贪心坐标下降）+ 冻结前的真实运行

前置条件：R8b、R8d、R8c 的实现和离线测试完成。目标日期：2026-10-26～27。结束后由用户决定是否冻结（10/27）。

## 0. 背景与分工（2026-10-21 定）

- **LLM 负责离散的结构决策**：近似方法（cheb / minimax_comp）、布局和算法、宏结构、刷新位置、长尾算子的组合、修错。
- **精调工具负责数值旋钮**：每个激活的阶数和定义域。在 LLM 给出的可行程序上逐个激活尝试，每一步都用代价模型和验证集检查：可行就保留，不可行就回退。整个过程不调用 LLM，结果交回 LLM，进入下一轮。
- 两者都只依赖与编译器一致的代价模型（R8c）和验证集。循环中不编译、不加密，也不使用测试集。
- 原型结果见 §7（Claude 于 2026-10-21 实测）。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/planners/agent_tools.py   改：§2.1 把验证阶段抽成 validate()
pt_agent/planners/tuner.py         新：§2.2
pt_agent/planners/agent.py         改：§2.3 每个可行轮次之后调用精调工具
pt_agent/planners/prompts.py       改：§2.4 r8-v3（feedback 中的 tuned 一节）
pt_agent/run.py                    改：--tuner {on,off}（默认 on）、--tuner-budget-s（默认 2400）
pt_agent/experiments/r8b_dev.py    改：§4 增加 --tuner 参数，REPORT 记录精调结果
pt_agent/tests/test_r8e_tuner.py   新：§3
../docs/pt_agent/STATUS.md         改
```

## 2. 实现

### 2.1 agent_tools.validate

```python
def validate(program, ctx, n: int | None = None, *, boot_nodes=None) -> dict:
    """在 ctx.val_images[:n] 上（n=None 表示全部）执行现有 evaluate 验证阶段的同一段逻辑：
    逐批 Mock、observer 统计每个 act 的 max_u 和观测到的 [min, max]、与精确网络对比的 agree、
    以及 R8c 的 boot_input_max_abs。
    返回 dict(agree, total, diverged, max_u, rows, boot_input_max_abs, feasible)。
    rows[i] 在 R8c 的字段之外，增加 obs_min、obs_max（第 i 个 act 输入的观测极值）。
    feasible 的判定与 evaluate 完全相同：未发散；max_u <= FEASIBLE_MAX_U；agree >= 0.99*total；
    boot_input_bound 为 None 或未超出。"""
```

evaluate 的验证阶段改为调用 validate(program, ctx)，输出保持逐字段不变，用 R7/R8b/R8c 的测试回归。

### 2.2 planners/tuner.py

```python
LADDER = (255, 127, 63, 31, 15, 7)     # cheb 的阶数阶梯；每降一档，深度减 1
TARGET_U = 0.80                        # 让观测范围映射到 |u| <= 0.80（可行条件是 0.90，留出余量）
N_COARSE = 200                         # 粗筛用的验证图数量
TUNER_AGREE_MIN = 0.992                # 精调结果的全量确认门槛，比可行条件 0.99 更严，留出余量，避免在验证集上贴边过拟合

@dataclass
class TuneResult:
    program: Program            # 精调后的程序（与输入相同或更优，并且在全部验证集上可行）
    source: str                 # print_program(program)，即展开后的程序
    changes: list[dict]         # 每项为 dict(line, call_lines, kind="deg"|"domain", old, new)
    cost_before: dict; cost_after: dict
    coarse_evals: int; full_evals: int; seconds: float
    history: list[dict]         # 每次尝试的记录：act、旧值、新值、预测延迟、agree、max_u、是否接受

def tune(program, ctx, *, budget_s: float = 2400, max_sweeps: int = 1) -> TuneResult
```

算法：精调只修改展开后 IR 中各 act op 的 attrs，不改变 kind、method 和程序结构。

0. 用 `copy.deepcopy(program)` 得到 cur，对全部验证集执行 `validate(cur, ctx)`，得到每个 act 的 obs_min 和 obs_max。
   - 发散（diverged）或 exact 不通过：原样返回，changes 为空。
   - 只有 max_u 或 agree 不满足：继续执行阶段 A，由它尝试修复。
1. **定义域对准安全目标（阶段 A）**：根据第 0 步的 rows，对每个 act 计算：
   ```
   c = (obs_min + obs_max) / 2
   h = (obs_max - obs_min) / 2 / TARGET_U
   ```
   - cheb：新定义域为 `(c-h, c+h)`。这一步**既可能收紧，也可能放宽**：过宽的定义域收窄以提高精度，不安全的定义域（max_u > 0.90）放宽以恢复安全；
   - minimax_comp：要求对称，`B' = max(|obs_min|, |obs_max|)/TARGET_U`，新定义域为 `(-B', B')`。

   数值保留 3 位小数。对全部验证集执行 validate：
   - 可行就采用；
   - 不可行时，如果第 0 步的程序本身可行，就整体回退到第 0 步；如果第 0 步也不可行，就返回 `TuneResult(program=原程序, feasible=False, ...)`，agent 不把它加入候选池。
2. **降阶（阶段 B）**：
   - 激活的处理顺序：按 cost.estimate(cur) 的 hotspots 中所在行的延迟占比降序；同一行在不同宏调用处的 act 视为不同项；不在 hotspots 中的 act 排在后面，按程序顺序。
   - 对每个 method=="cheb" 的 act，取 LADDER 中比当前 deg 小的最大值，生成候选程序：
     - 候选的 latency_pred_us 不低于 cur 的，跳过；
     - 否则执行 `validate(trial, ctx, N_COARSE)`，可行就令 cur = trial，并把这一项记入 accepted 列表。
   - 每完成一遍（sweep），对全部验证集执行 validate(cur, ctx)，并且要求 `agree >= TUNER_AGREE_MIN*total`（阶段 A 的结果也用这个门槛）。不满足时，**对本遍 accepted 列表的前缀长度做二分**：找出最大的 k，使得"本遍开始时的程序 + accepted 前 k 项"在全部验证集上可行，令 cur 等于它。逐项撤销需要的全量验证次数太多，二分只需约 log2(len) 次。原型中，200 张图的粗筛每一步都通过，但 18 项累积后全量一致率降到 984/1000，见 §7。
   - 某一遍没有任何接受，或者已满 max_sweeps 遍，或者超过 budget_s，就停止。超时之前必须先完成当前这一遍的全量确认（如果全量确认失败，还要完成回退）。
3. TuneResult 增加字段 `feasible: bool`。为 True 时，返回的 program 必须在全部验证集上可行，并且（输入可行时）latency_pred_us ≤ 输入程序的值。

### 2.3 planners/agent.py

- `run_agent(..., tuner: bool = True, tuner_budget_s: float = 2400)`。默认每次调用只做 1 遍（max_sweeps=1）；ResNet20 上一遍实测约 29 分钟（见 §7）。
- **只在 LLM 给出了新的"LLM 侧最优"时才调用精调**：本轮 LLM 程序的 latency_pred_us 低于此前所有 LLM 程序（不含精调结果）的最小值，或者这是第一个可精调的程序。这样每个模型通常只精调 2–3 次。
- 某一轮 LLM 程序通过 exact 检查且未发散时（fb 为 feasible，或者只有 max_u、agree 不满足），如果 tuner 为 True，就调用 `tune(program, ctx, budget_s=...)`；`result.feasible` 为 True 时：
  - 候选池同时加入 LLM 的原程序和精调后的程序，best 取 latency_pred_us 最小的那个。best 记录中增加 `source="llm"` 或 `source="tuned"`，以及 iteration；
  - 冻结输出的是 best 对应的源码。如果是精调后的程序，冻结的就是展开后的程序文本。REPORT 同时记录 LLM 原程序的行数和冻结程序的行数。
- 停止条件仍然用 best 的 latency_pred_us：连续 2 轮改进不足 5% 就停。
- trace.jsonl 的每轮记录中增加 `tuned` 一项：changes、cost_before、cost_after、coarse_evals、full_evals、seconds。

### 2.4 提示词 r8-v3（prompts.py）

`PROMPT_VERSION = "r8-v3"`。TEMPLATE 在 r8-v2 的基础上只做一处插入：在 "Spend changes where the share is large." 之后接着写

```text
After each feasible program, a deterministic tuner lowers activation degrees and tightens domains one at a time and reports the settings it kept ("tuned"). Concentrate on structural decisions (approximation method, layout and algorithms, macro structure); you may adopt the tuned settings.
```

feedback_message 在 tuner 运行过时增加字段：

```python
tuned = dict(latency_pred_us=..., boots_pred=..., changes=[{line, line_text, call_texts, kind, old, new}, ...])
```

最多列 30 项，按 line 排序。feasible 分支的 instruction 改为：

```text
The program is feasible. The tuner's result is in "tuned". Reduce predicted latency further by structural changes (for example a different approximation method on the hotspot lines) while keeping all feasibility conditions. Return the complete program.
```

## 3. 测试（test_r8e_tuner.py，离线，不调用 API，用 nixrun）

1. **validate 与 evaluate 一致**：对 R6c 的 program.ptdsl，在 n_val=50 的 ctx 上，`validate(p, ctx)` 的 agree、max_u、feasible 与 `evaluate(src, ctx)["validation"]` 和 `["feasible"]` 相等。
2. **单调与可行**：resnet20 使用 R7c B 的程序和 n_val=200 的 ctx，以 `N_COARSE=100`、`budget_s=900`、`max_sweeps=1` 运行 tune：
   - 返回的 program 在全部 200 张图上可行；
   - latency_pred_us 不高于输入；
   - program 与输入的 `canonical()` 只在 act 的 deg 和 domain 上不同（逐项比较 kind、inputs 和其他 attrs）；
   - changes 中每一项都能在两者的差异里对上。
3. **回退**：构造一个 FakeValidate，粗筛时总是可行，全量时只有当所有 deg ≥ 63 才可行。运行 tune 后，结果中所有 deg 都 ≥ 63，并且 history 中有撤销记录。
4. **minimax_comp 的 act 不降阶**，只收紧对称定义域：用 R8d 的 rule 程序（act_method=minimax_comp）在 cnn_plain 上验证。
5. **agent 循环**：用 FakeClient 返回 R7c B 程序的 2 轮测试中，tuner=True 时 trace 中有 tuned 记录，best.source 为 "tuned" 或 "llm"，冻结源码可以 parse；tuner=False 时行为与 R8c 相同。
6. 提示词版本为 r8-v3；R8b 的禁词检查仍然通过。
7. R1–R8c 的测试全部通过。

## 4. 真实运行（即原 R8b §6 / R8c §6；使用 r8-v3）

`python3 -m pt_agent.experiments.r8b_dev --round R8e --tuner on`：4 个开发模型，每个最多 6 次 LLM 调用，合计不超过 24 次。

每个模型记录：
- R8b §6 和 R8c §6 要求的全部内容，以及每一轮的 LLM 程序与精调后程序的 boots、latency 和 agree；
- 精调的评估次数和耗时；
- 冻结程序的来源（llm 或 tuned）；
- 编译后 plan 的 boots 和延迟（与预测对照）；
- 测试集 top-1 和 n_diverged。

## 5. 汇报

- 测试输出和 4 个模型的对照表。
- resnet20 与 R7c B（288 次自举）的对比。
- API 用量。
- STATUS.md：新增 R8e 一行，R8c 标为完成；M1 第 1–4 条以本轮 resnet20 的结果为准。

## 6. 不要做

- 精调工具不修改方法、结构和宏，不调用 LLM，不使用测试集，不编译。
- 不放宽可行性条件（max_u ≤ 0.90，agree ≥ 99%）。
- 不打冻结标签。

## 7. 原型结果（Claude 于 2026-10-21 实测，ResNet20，起点是 R7c B 的冻结程序）

| 步骤 | 自举（预测，depth-dp） | 预测延迟 | 验证集（1000 张） | 累计用时 |
|---|---|---|---|---|
| 起点：R7c B（19 个激活全为 cheb 63） | 288 | 77.4 s | agree 996，max_u **0.929**（新标准 0.90 下不可行） | 2.3 min |
| 阶段 A：定义域对准 \|u\|≤0.80 | 288 | 77.4 s | agree **998**，max_u **0.820**，可行 | 5 min |
| 阶段 B：按程序顺序 19 项 63→31，200 张图粗筛 | 18 项通过，合计 198 | 54.5 s | 全量 agree 984，不可行 | 17 min |
| 前缀二分（5 次全量验证） | 保留前 8 项：**248（−14%）** | **67.2 s（−13%）** | agree **990**（恰好在门槛上），max_u 0.842 | **29 min** |

结论：
1. **阶段 A 必不可少**：它把新标准下不可行的程序修成可行，并留出余量。
2. **精调在 LLM 的结果上还能再降约 14% 的自举和 13% 的延迟**，但这比只看粗筛时（−31%）少得多：200 张图的粗筛对累积误差不敏感，后面的激活即使降阶，在前缀二分时也会被整体舍弃。
3. **贴边风险**：最终一致率正好等于 990。所以加了 `TUNER_AGREE_MIN = 0.992`。按这个门槛，原型会保留前 7 项：254 次自举（−12%），agree 992。
4. **改进方向**（已写入 §2.2，R8e 照做）：按 hotspots 顺序处理（先处理延迟占比高的激活）；只在 LLM 侧出现新最优时才精调。下一个版本可以再考虑"按激活的误差贡献排序"，本轮不做。
5. **耗时**：一遍约 29 分钟（粗筛每项约 31 s，全量验证约 2.5 min）。所以默认 budget 为 2400 s，并且只在 LLM 侧出现新最优时调用。
6. **测试集检查未完成**：脚本在测试集那一步被系统杀掉（退出码 137，疑似内存超限）。精调后程序的测试集结果，由 R8e §4 的真实运行统一报告（测试集只用于报告）。
