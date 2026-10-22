# R8c：让代价模型与编译器一致，并把目标平台的约束和热点反馈给 Agent

前置条件：R8a2、R8a 完成；R8b、R8d 的实现和离线测试完成。目标日期：2026-10-25。

> **2026-10-21 调整**：
> - 顺序为 R8b → R8d → R8c → R8e。
> - 本轮只实现 §1–§5，并完成 §5 的离线测试。**§6 的真实运行推迟到 R8e**，与精调工具一并执行，以免重复消耗 API。
> - §7 中与真实运行有关的汇报项（4 个模型的 agent 摘要、hotspots、API 用量）也一并移到 R8e。本轮的完成标准为"§5 全部通过"。

## 0. 背景（Claude 于 2026-10-21 核对 R6b 的数据，并用原型验证）

**原则：Agent 只靠内部反馈，产出能一次编译通过、运行正确的高性能程序。** 真实编译和 CKKS 结果不回传，所以内部预测必须与目标编译器、目标平台一致。目前有 4 个缺口：

1. **编译器模式下的预测有系统偏差。** Agent 写的是 `refresh="compiler", placement="depth-dp"`，但 cost.estimate 用的是惰性插入自举的模型。R6b 中惰性模型与真实 depth-dp 的对比：

   | 程序 | 预测自举 | 实际自举 | 预测延迟 | 实际延迟 |
   |---|---|---|---|---|
   | deg127/m1.5 | 461 | 383 | 122.0 s | 101.3 s |
   | deg63/m1.4 | 382 | 288 | 101.4 s | 76.8 s |

   偏差为 +20~33%，而且偏差大小随程序变化。Agent 因此在优化一个与实际不一致的目标。

   depth-dp 的放置算法在 `third_party/ckks-runtime/third_party/dacapo/lib/Dialect/Earth/Transforms/GreedyBootstrapPlacement.cpp`（第 54–265 行）。Claude 已把它移植成原型，在同一份 Earth op DAG 上运行，结果是：边界数 76/76，自举 383/383、288/288，延迟误差 1.0% 和 0.7%，两个程序都完全一致。
   - 唯一需要核实的语义是：**自举之后，密文回到满 level**（plan 中 288 次自举的输出都是 level 5），而不是消耗一层。原型最初按"消耗一层"实现，自举次数多出 5~10%。
2. **资源约束没有进入可行性判断**：常量字节数、旋转 key 数。
3. **自举输入的取值范围没有检查。** CKKS 自举要求输入的幅度在一个范围内，但 Agent 看不到哪些值会被自举、它们有多大。目前目标平台的自举输入上界和精度还未知（`qwen48` 配置里没有写），本轮只实现这个机制：值为 None 时只报告，不作为约束。
4. **反馈只有总延迟。** Agent 不知道延迟主要花在哪几行，只能整体降阶数。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/backends/base.py            改：Backend.__init__ 增加 self.op_context = None
pt_agent/backends/dag.py             新：§2.1 DagBackend
pt_agent/backends/__init__.py        改：导出 DagBackend
pt_agent/backends/mock.py            改：§2.4 节点编号、watch、可选噪声
pt_agent/lower.py                    改：每处理一个 program op，先执行 b.op_context = op.id
pt_agent/placement.py                新：§2.2 depth_dp_cuts、simulate
pt_agent/cost.py                     改：§2.3 profile 扩展，estimate 按模式分派，hotspots
pt_agent/planners/agent_tools.py     改：§3 resources 阶段、自举输入检查、hotspot 文本
pt_agent/planners/prompts.py         改：§4 r8-v2
pt_agent/experiments/r8b_dev.py      改：§6 增加 --round 参数和预测与实际的对照
pt_agent/tests/test_r8c_placement.py 新：§5
pt_agent/tests/test_r8c_feedback.py  新：§5
pt_agent/tests/test_r6b_cost.py      改：§5.6 断言值
pt_agent/tests/test_r7_agent.py      改：§5.6 断言值
../docs/pt_agent/STATUS.md           改
```

## 2. 实现

### 2.1 backends/dag.py

```python
class DagExpr(_Cipher):            # 字段 nid
class DagBackend(Backend):
    _keep_const_data = False
    kind: list[str]                # "input" | "add" | "sub" | "neg" | "mul_cc" | "mul_cp" | "rot" | "boot"
    args: list[tuple[int, ...]]    # 密文操作数的 nid，按出现顺序，可重复（x*x 记两次）
    prog_op: list[int | None]      # 节点创建时的 self.op_context
    counts: _Counters              # 与 CountBackend 相同的计数
```

- `_make_input` 新建一个 "input" 节点。
- `_apply(kind, left, right, step)` 新建一个节点，只记录密文操作数，明文不记录。rot 的 step 写入 counts。
- 节点编号按创建顺序，从 0 开始。
- `stats()` 返回与 CountBackend.stats() 相同的键，外加 `nodes=len(kind)`。

### 2.2 placement.py（移植 depth-dp，算法已由 Claude 的原型验证）

```python
@dataclass
class Placement:
    cuts: list[int]; barriers: int; boots: int; latency_us: float
    boot_nodes: list[int]                 # 被自举的节点 nid（同一个节点可能重复出现）
    node_latency: dict[int, float]        # 节点 nid → 执行它的延迟；自举延迟记到被自举的节点上
    max_depth: int

class PlacementError(Exception): ...

def depth_dp(dag: DagBackend, *, last_boot_level: int, boot_in_level: int, boot_out_level: int,
             latency_table: dict) -> Placement
```

记号：`L = last_boot_level`，即 `boot_out_level - boot_in_level`，qwen48 中为 4。level 用"已消耗层数"c 表示：输入为 0，自举后为 0；查延迟表时用 `R(c) = boot_out_level - c`。

1. 如果 DAG 中有 "boot" 节点，抛 PlacementError("depth-dp model requires a program without explicit refresh")。
2. **users**：对每个节点，按 args 列出使用它的节点（重复使用就记多次）。输出值额外加一个虚拟的 RET 用户，`RET = len(nodes)`。
3. **depth**：input 为 0。其他节点为 max(各密文操作数的 depth) + (1 if kind ∈ {mul_cc, mul_cp} else 0)；操作数全是明文时，前一项取 0。`maxd` = 所有真实节点 depth 的最大值，`depth[RET] = maxd + 1`。
4. **延迟一元算子**：按节点编号倒序遍历，kind ∈ {rot, neg} 且有 users 的节点，`depth[i] = min(depth[u] for u in users[i])`。
5. **crossing**：长度为 maxd+2 的数组。对每个有 users 的节点：start 为 0（input）或 depth[i]，`end = max(start, max(depth[u]))`；`crossing[start] += 1`，`crossing[end] -= 1`。然后对 i=1..maxd 做前缀和。
6. **DP**：
   - `cost[0] = 0`，其余为 ∞；`prev` 全为 -1。
   - 对 e=1..maxd：若 `e ≤ L`，`cost[e] = 0`。然后 s 从 e-1 倒序到 max(0, e-(L-1))：`cand = cost[s] + crossing[s]`，若 `cand < cost[e]`（严格小于），则 `cost[e] = cand`、`prev[e] = s`。
   - 回溯：`e = maxd`；当 `prev[e] ≥ 0` 时，`e = prev[e]`，把 e 加入 cuts。最后把 cuts 反转成升序。
7. **调度**：所有节点加上 RET，按 depth 稳定排序（depth 相同时保持编号顺序）。
8. **模拟**：`level = [0]*n`；`remaining[i] = len(users[i])`；live 是保持插入顺序的集合，初始为有 users 的 input 节点；`nxt = 0`。按调度顺序遍历每个节点 i（input 节点跳过）：
   - **边界**：若 `nxt < len(cuts)` 且 `depth[i] > cuts[nxt]`：
     - barriers += 1；
     - `end` 取 `cuts[nxt+1]`，没有下一个就取 maxd；`safe = L - (end - cuts[nxt])`；
     - 对 live 中的每个 v（按插入顺序）：若 `level[v] > L`，抛 PlacementError；若 `level[v] > safe`，则 `level[v] = 0`、boots += 1、`boot_nodes.append(v)`，延迟加上 `table["bootstrap_single"][boot_in_level]`，记到 v；
     - nxt += 1。每个节点最多处理一个边界。
   - 若 i 是 RET，结束本次处理。
   - `lv = max(level[j] for j in args[i])`。
   - **mul_cc / mul_cp**：若 lv ≥ L，抛 PlacementError(f"multiplication exceeds the bootstrap budget at node {i}")。否则延迟为 `table["mul_double" 或 "mul_single"][R(lv)] + table["rescale_single"][R(lv)]`，`level[i] = lv + 1`。
   - **其他算子**：rot 用 rotate_single，neg 用 negate_single；add/sub 有两个密文操作数时用 add_double，否则用 add_single，取 `[R(lv)]`；`level[i] = lv`。
   - 对每个操作数 j：`remaining[j] -= 1`，降到 0 时把 j 从 live 中移除。若 `remaining[i] > 0`，把 i 加入 live。

### 2.3 cost.py

- `PROFILES["qwen48"]` 新增以下字段：
  - `limits=dict(const_bytes=4*2**30, rot_keys=None)`：const_bytes 的依据是可信运行器单文件 8 GiB，而 MLIR 体积约为常量体积的 1.2 倍；rot_keys 等后端给出 key 内存后再定；
  - `boot_input_bound=None`、`boot_abs_error=None`：等待后端提供。
- `estimate(program, weights, *, profile="qwen48", placement_model=None)`：
  - placement_model 为 None 时自动选择：`refresh == "agent"` 用 "lazy"（现有 RefreshBackend 路径，不改）；`refresh == "compiler"` 且 `placement == "depth-dp"` 用 "depth-dp"；其他情况用 "lazy"，并在返回值中标注 `placement_exact=False`。
  - **"depth-dp"**：用 DagBackend 执行 lower，然后调用 placement.depth_dp。
    - `counts` 为 DagBackend 的计数，并把 `counts["boot"]` 设为 placement.boots；
    - `boots_pred = placement.boots`，`latency_pred_us = placement.latency_us`；
    - 新增键：`barriers`、`boot_nodes`，以及 `hotspots`（见下）、`boot_share`（自举延迟 / 总延迟）。
  - 两种模型都返回 `placement_model` 和 `placement_exact`（depth-dp 和 agent 模式下为 True）。lazy 模型的 hotspots 与 boot_nodes 为空列表。
  - **hotspots**：按 `program.ops[prog_op]` 的 `(lineno, call_lines)` 汇总 node_latency。prog_op 为 None 的节点记到 `(None, ())`。按延迟降序取前 5 项，每项为 `dict(line=lineno, call_lines=list(call_lines), latency_share=round(值/总延迟, 3), boots=该键下的自举次数)`。
- `program.ops` 按 op.id 查找：如果 ops 列表的下标不等于 id，先建立 `{op.id: op}`。

### 2.4 backends/mock.py

- `MockBackend(slots, batch=None, *, watch=None, noise=None)`：
  - 新增节点计数 `self._nid`，在 `_make_input` 和每次 `_apply` 时各加 1，编号规则与 DagBackend 相同；
  - `watch` 是 nid 的集合：这些节点产生值时，在 `self.watch_max[nid]` 中记录 `float(np.max(np.abs(值)))`，范围是全部槽位（真实自举作用于整个密文）；
  - `noise=(sigma, seed)`：对 watch 中的节点，输出加上 `np.random.default_rng(seed + nid).normal(0, sigma, 形状)`。
- 不传 watch 和 noise 时，行为与现在逐位相同。

## 3. planners/agent_tools.py

1. **cost 阶段**之后新增 **resources 阶段**：对 profile.limits 中不为 None 的项，`cost["const_bytes"]`、`cost["rot_keys"]` 超过上限就返回：
   ```python
   stage="resources", feasible=False,
   errors=[dict(code="E_RESOURCE", limit=name, value=v, max=limit)]
   ```
2. `fb["cost"]` 在现有字段之外，增加 placement_model、placement_exact、barriers、boot_share 和 hotspots。hotspots 的每项补充 `line_text` 和 `call_texts`，用 evaluate 中已有的 line_text 函数生成。
3. **validation 阶段**：若 `cost["boot_nodes"]` 不为空，构造 `MockBackend(..., watch=set(boot_nodes), noise=(profile.boot_abs_error, 20261022) if profile.boot_abs_error else None)`，每个批次后取 `max(watch_max.values())`，汇总为 `validation["boot_input_max_abs"]`，并记录 `validation["boot_input_bound"] = profile.boot_input_bound`。
4. **feasible** 在原有条件之外再加两条：resources 阶段通过；`boot_input_bound` 为 None，或 `boot_input_max_abs ≤ boot_input_bound`。
5. `AgentContext` 增加字段 `profile: str = "qwen48"`。`initial_message` 顶层增加：
   ```python
   "target": dict(profile=..., placement="depth-dp", limits={非 None 的项}, boot_input_bound=...)
   ```

## 4. 提示词 r8-v2（prompts.py）

`PROMPT_VERSION = "r8-v2"`。在 R8b 的 TEMPLATE 上只做 2 处替换，其余一字不改。

（a）把

```text
rotations come from convolutions, pooling and linear layers. The feedback reports predicted latency,
bootstraps and depth.
```

替换为

```text
rotations come from convolutions, pooling and linear layers. The feedback reports predicted latency and
bootstraps, computed with the same bootstrap placement as the compiler, and "hotspots": the program lines
(with their macro call sites) that carry the largest share of predicted latency. Spend changes where the
share is large.
```

（b）在 `the predicted class agrees with the exact network for at least 99% of the images.` 之后，接着插入一句（在同一段内，前面加一个空格）：

```text
The program must also stay within the target limits given in the input ("target"); a violated limit is reported at stage "resources". When the target gives boot_input_bound, every value the compiler bootstraps must stay within it on the validation images.
```

`feedback_message` 中 feasible 分支的 instruction 改为：

```text
The program is feasible. Reduce predicted latency where the hotspots show the largest share (for example a lower activation degree or a tighter but still safe domain on those lines) while keeping all feasibility conditions. Return the complete program.
```

## 5. 测试（离线；用 nixrun；不允许真实调用 API）

1. **链**：DagBackend(16)，一个输入连续乘 8 次标量 0.5 后输出。在 qwen48 参数下，depth_dp 结果为：`cuts == [4, 7]`、`barriers == 2`、`boots == 1`、`max_depth == 8`。
2. **分叉**：
   ```
   y = x*0.5; p = y; q = y.rotate(1)
   重复 4 次：p = p*0.5; q = q*0.25
   z = p + q
   重复 2 次：z = z*0.5
   输出 z
   ```
   结果为：`cuts == [4]`、`barriers == 1`、`boots == 2`、`max_depth == 7`，并且 boot_nodes 是 p 和 q 第 3 次乘法的结果节点。
3. **与真实编译一致（核心验收）**：用 `Graph.load(R3/"graph")` 的权重，读取
   - `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/pt-agent-r6b-127-1p5-compiler-20261021T025533-1843/program.ptdsl`
   - `/home/lhohy/poseidon-work/platforms/aarch64-linux/results/pt-agent-r6b-63-1p4-compiler-20261021T025558-09cd/program.ptdsl`

   分别执行 estimate，要求：
   - boots_pred 分别**等于** 383 和 288，barriers 都等于 76；
   - latency_pred_us 与 101340690 和 76809672 的相对误差 ≤2%；
   - `placement_model == "depth-dp"`，counts["rot"] == 2624。

   这两个文件缺失，或者数值不符时，停下来写进阻塞，不要调整算法去凑数。
4. **节点编号一致**：deg63 的程序分别在 DagBackend 和 MockBackend(batch=2) 上 lower，`len(dag.kind) == mock._nid`。在 Mock 上设置 watch 为 boot_nodes 时，watch_max 的键集合等于 set(boot_nodes)。不传 watch 和 noise 时，输出与原来 `np.array_equal`。
5. **hotspots**：deg63 程序的 hotspots 有 5 项，latency_share 降序且总和 ≤1，每项的 line 都是程序中的有效行号；boot_share 在 (0.8, 1.0) 之间（实际约为 0.95）。
6. **旧测试的断言更新**（这是行为变化，不是放宽）：
   - test_r6b_cost.py：`(127, 1.5, 461), (63, 1.4, 382)` 改为 `(127, 1.5, 383), (63, 1.4, 288)`；`counts['boot']` 同步更新。
   - test_r7_agent.py：`boots_pred` 由 461 改为 383。
   - 另外在 test_r6b_cost.py 中增加：`estimate(..., placement_model="lazy")` 仍然得到 461 和 382。
7. **反馈**：
   - 把 const_bytes 的上限临时设为 1，evaluate 返回 `stage == "resources"`，errors[0] 的 limit 为 "const_bytes"；
   - 把 boot_input_bound 临时设为 1e-6，返回 `feasible == False`，并且 validation 中有 boot_input_max_abs；
   - 用 `noise=(1e-3, 0)` 时，agree 仍然能正常计算，不报错。
8. **提示词**：PROMPT_VERSION 为 "r8-v2"；TEMPLATE 中包含 "hotspots" 和 `stage "resources"`；R8b 测试 2 中禁止出现的字符串都不出现。
9. R1–R8b 的测试全部通过。

## 6. 真实运行（执行 R8b §6 推迟的部分，使用 r8-v2）

`python3 -m pt_agent.experiments.r8b_dev --round R8c`：4 个开发模型，每个最多 6 次调用，合计不超过 24 次。r8b_dev 的改动：

- registry 中的 round 取 --round 参数的值；
- 每个模型在 REPORT 中增加 `pred_vs_actual = dict(boots_pred, boots_plan, latency_pred_us, latency_plan_us, rel_err)`，plan 一侧用现有的 plan_latency 计算；
- 同时对 R8a 的规则基线程序也计算一次新的 estimate（不调用 LLM，也不重新编译），写进同一张对照表。

## 7. 汇报

- 测试输出。
- 4 个模型的 agent 运行摘要：内容同 R8b §6 的对照表，增加 boots_pred、boots_plan 和延迟误差这 3 列。
- 每个模型的 hotspots 前 3 项，以及 boot_input_max_abs。
- API 用量（本轮和累计）。
- STATUS.md：新增 R8c 一行，R8b 标为完成；M1 第 1–4 条补充 resnet20 在 r8-v2 下的结果。

## 8. 不要做

- 不修改 kernel、lowering 的算法和规则规划器，也不修改 RefreshBackend 的 lazy 路径。
- 不为了对齐真实值调整 depth-dp 算法的步骤。§5.3 不通过就报告阻塞。
- 不往 profile 中填写自举输入上界和精度的猜测值（保持 None）。
- 不把编译、CKKS、测试集的结果放进反馈；不重试 LLM 调用；不打冻结标签。
