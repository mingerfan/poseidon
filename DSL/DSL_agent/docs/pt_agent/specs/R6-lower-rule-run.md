# R6：PT-DSL lowering、规则规划器、端到端运行入口（ResNet20）

前置：R5b、R5c、R5d 已完成。目标日期：2026-10-31。本轮只做"编译器负责刷新"（refresh="compiler"）这一条路线；代价模型和由 Agent 放置自举留到 R6b。

## 0. 背景（只读）

- **组件链路**：规划器生成 PT-DSL 文本 → R1 的 `parse` 和 `typecheck` → **本轮新增的 lowering** → R4/R5 的 kernel 在三个后端上运行 → Mock 测精度 / Count 统计 / Emit 生成 Hecate 程序 → R2b 的工具链编译。
- R5 的 `models/resnet20_fhe.build` 是"直接遍历 PT-Graph"的组装方式；本轮的 lowering 遍历 PT-DSL 的 IR。两者调用同一组 kernel、遵循同样的融合规则，所以同一配置下 Count 的统计必须完全相等，测试会检查这一点。
- 规则规划器的默认参数取 R5 网格中**未发散**且 agree 最高的那组（R5 汇报中的 `emit` 配置）。若没有可用结果，就用 deg=63、margin=1.25。参数都可以通过命令行覆盖。
- 注意：margin=1.0 时，近似误差会把个别 act 输入推出定义域，高阶多项式随即发散（见 R5c 的诊断）。所以 `plan` 的 margin 默认值是 1.25，规则规划器不得使用小于 1.1 的 margin（传入时抛 ValueError）。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/lower.py                    新：§2
pt_agent/planners/__init__.py        新：空
pt_agent/planners/rule.py            新：§3
pt_agent/run.py                      新：§4（端到端运行入口）
pt_agent/tests/test_r6_lower.py      新：§5
pt_agent/tests/test_r6_rule.py       新：§5
../docs/pt_agent/STATUS.md           改
```

不修改 R1–R5 的文件。

## 2. pt_agent/lower.py

```python
class LoweringError(Exception):
    def __init__(self, op_id: int | None, lineno: int | None, msg: str): ...   # str() 为 f"op {op_id} line {lineno}: {msg}"

@dataclass(frozen=True)
class RawLayout:
    shape: tuple[int, ...]          # pack 之前的密文参数，只记录形状

def lower(program: Program, weights: Mapping[str, np.ndarray], b: Backend, *,
          act_mode: str = "program", observer=None) -> tuple[list, list]:
    """按 program.ops 的顺序发出 kernel 调用，返回 (输出值列表, 输出布局列表)，并调用 b.output(输出值列表)。
    act_mode 取 "program"（按程序里的 act 属性）或 "exact"（只在 Mock 下允许，所有 act 改用 b.apply_exact）。
    observer 的签名与 R5 的 build 相同，在每个 act 之前调用。"""

def pack_inputs(program: Program, arrays: list[np.ndarray]) -> list[np.ndarray]:
    """客户端打包：arrays[i] 的形状为 (N,) + params[i].shape[1:]。
    按 params[i] 对应的那个 pack op 的布局，对每个样本调用 pack_mux 或 pack_flat，返回形状为 (N, slots) 的数组列表。"""

def decode_outputs(values_np: list[np.ndarray], layouts: list) -> list[np.ndarray]:
    """对每个输出（形状为 (N, slots) 或 (slots,)），在 FlatLayout 上取前 size 个槽，在 MuxLayout 上调用 unpack_mux。"""
```

- **环境**：维护 `env: vid -> (value, layout)`。第 i 个参数为 `(b.input(i), RawLayout(param.shape))`，n = `program.header.slots`。
- 先预计算每个 vid 的消费者列表，供模式匹配使用。
- **各 op 的处理**（attrs 都已由 R1 补全默认值；`W` 引用通过 `weights[ref.name]` 取出，取不到时抛 LoweringError）：

| op | 处理 |
|---|---|
| pack | 输入必须是 RawLayout。layout="mux" 时，要求 shape 为 (1,C,H,W)，新布局为 `mux_layout(n, C, H, W, gap)`；layout="flat" 时，要求 shape 为 (1,F)，新布局为 `FlatLayout(n, F)`；其他 layout 抛 LoweringError。值不变，不发出任何 op。 |
| conv2d | 输入必须是 MuxLayout。要求 algo="mpcb"、dilation=(1,1)、groups=1、stride[0]==stride[1]、pad[0]==pad[1]==(fh-1)//2。bn 不为 None 时，用 `fold_bn(W[bn+".weight"], W[bn+".bias"], W[bn+".running_mean"], W[bn+".running_var"], bn_eps, conv_bias)` 得到 scale 和 shift；否则 scale=None，shift=conv_bias（b 为 None 时 shift 也为 None）。调用 `conv2d_mux`。 |
| act | 按 act_mode 处理：exact 模式调用 `b.apply_exact(x, fn)`；program 模式下，method="cheb" 调用 `act_cheb(b, x, fn, deg, domain)`，method="exact"（只能是 fn="square"）计算 `x * x`，method="minimax_comp" 抛 LoweringError("minimax_comp not implemented in v1")。布局不变。调用 kernel 前先调用 observer。 |
| add | 两个输入都是 MuxLayout 时用 `add_mux`；都是相同的 FlatLayout 时直接用 `+`；其他情况抛 LoweringError。 |
| slice | 必须满足 R5 §7 规则 5 的 slice→slice→pad 链（按消费者关系判断，第二个 slice 和 pad 必须是各自上游唯一的消费者）。满足时，在处理第一个 slice 时一次性调用 `downsample_pad_mux`，并把后面的 slice 和 pad 标记为已处理。不满足时抛 LoweringError。 |
| pad | 只能作为上面这条链的一部分出现，单独出现时抛 LoweringError。 |
| mean | 要求 axis=(2,3)、keepdim=False，输入为 MuxLayout，调用 `mean_mux`。 |
| relayout | 目标是 flat、输入已经是 FlatLayout 时为恒等，其他情况抛 LoweringError。 |
| linear | 输入为 FlatLayout，调用 `linear_flat(algo=algo, n1=n1)`。 |
| refresh | `header.refresh=="agent"` 时调用 `x.bootstrap()`，否则为恒等。 |
| 其他 op | 抛 LoweringError，提示"v1 lowering 不支持"。 |

kernel 抛出的 KernelError 要包装成 LoweringError，并带上 op_id 和 lineno。

## 3. pt_agent/planners/rule.py

```python
def plan(graph: Graph, calibration: dict, *, deg: int = 63, margin: float = 1.25,
         slots: int = 16384, placement: str = "depth-dp") -> str:
    """返回一份 PT-DSL 源码文本（不含宏和 region），按 graph.nodes 的顺序一条节点生成一行。"""
```

- **头部**：

  ```
  @fhe.program(slots=<slots>, refresh="compiler", placement="<placement>")
  def <graph.name>(x: Cipher[1, 3, 32, 32]):
  ```

  参数名和形状取自 `graph.inputs[0]` 和 `graph.tensors`。
- **变量名**：图张量名 t 转成 `"v_" + re.sub(r"[^A-Za-z0-9]", "_", t)`，再把连续的下划线合并成一个，并去掉结尾的下划线。如果与已有名字重复，依次加后缀 `_2`、`_3`……。
- **第一行**：`<input 变量> = pack(<param>, layout="mux", gap=1)`。
- **各节点**：

| 图节点 | 生成的语句 |
|---|---|
| conv2d，且唯一消费者是 batchnorm | 一行 `conv2d(prev, w=W["…weight"], b=…, bn="<prefix>", bn_eps=<eps>, stride=<s>, pad=<p>, algo="mpcb")`，跳过那个 batchnorm。prefix 由 BN 的 weight 名去掉 ".weight" 得到；要求 BN 另外三个名字分别是 prefix+".bias"、".running_mean"、".running_var"，否则抛 ValueError。b 为 None 时省略。 |
| conv2d，没有接 BN | 同上，但不带 bn。 |
| act | `act(prev, fn="<fn>", method="cheb", deg=<deg>, domain=(<lo>, <hi>))`，其中 lo=calibration[输入]["min"]*margin，hi=calibration[输入]["max"]*margin，用 `repr(float)` 写出。 |
| add | `add(a, b)` |
| slice | `slice(prev, axis=…, start=…, end=…, step=…)` |
| pad | `pad(prev, pads=(…))` |
| mean | `mean(prev, axis=(…))` |
| mean 之后紧跟的 linear | 先输出 `<名> = relayout(<mean 结果>, layout="flat")`，再输出 `linear(…, w=W[…], b=W[…], algo="bsgs")` |
| 其他节点 | 抛 ValueError，提示信息中写明节点 id |

- **最后一行**：`return <输出变量>`。
- 生成的文本必须能通过 `parse` 和 `typecheck`（权重形状取自 graph.weights）。

## 4. pt_agent/run.py：端到端运行入口

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.run --model resnet20 --planner rule [--deg D] [--margin M] \
    --stages plan,check,mock,count,emit[,compile][,test] [--final]
```

### 4.1 运行方式

- 与 R3/R5 的实验相同：外层用系统 python3，创建 `$PT_RESULTS/run-resnet20-rule-<UTC 时间戳>/`，然后用 nixrun 在 Nix 内执行 `--inside --out <目录>`。最后由外层把 REPORT.json 的摘要写进 registry（round="R6"、model="resnet20"、mode="rule"）。
- `test` 这个 stage 必须同时带 `--final` 才执行，否则直接报错退出。这是"测试集只用于最终验收"的约定。

### 4.2 inside 依次执行（只跑 `--stages` 中列出的步骤；后面的步骤依赖前面的产物）

1. **plan**：`graph = models.resnet20.load_graph()`；`calibration = analyzer.calibrate(graph, {"x": calibration_set()[0]})`；`text = rule.plan(graph, calibration, deg, margin)`。写出 `program.ptdsl` 和 `calibration.json`。
2. **check**：`parse` 加 `typecheck`，结果写进 `check.json`（字段 ok、diags、`n_statements=len(program.ops)`、`source_lines` 为非空行数）。
3. **mock**：
   - 精确模式：取 8 张校准图，在 `MockBackend(slots, batch=8)` 上执行 `lower(act_mode="exact")`，与 `reference_forward` 的最大绝对误差记为 `exact_max_abs_diff`，要求 ≤1e-9。
   - 近似模式：1000 张校准图按 batch=50 计算，记录 agree（argmax 与精确参考一致的张数）、top1、logit_rmse 和 seconds。
4. **count**：`CountBackend(slots)` 加 `lower`，记录 `stats()`。
5. **emit**：
   - `EmitBackend(slots)` 加 `lower`，写进 `toolchain.new_run_dir("r6-resnet20-emit")`，记录 constants.npz 的字节数和 counts。
   - 同时生成 `server_jobs/r6-resnet20/run.sh`。它从仓库根目录调用 `pt_agent.toolchain` 的 trace 和 compile，只需要设置路径和环境变量，在服务器上执行；把 run_dir 也打包进去的事由人工处理。
6. **compile**：只在 constants.npz ≤1.5 GiB 时执行，依次运行 `toolchain.trace(trusted=True)` 和 `toolchain.compile(mode="dacapo-depth-dp", boot="qwen48", seconds=3600, trusted=True)`（R5d 的可信运行器），记录退出码、耗时、峰值内存、plan_stats 和 log_tail。失败不重试。
7. **test**（需要 `--final`）：10000 张测试图按 batch=50 跑近似模式的 Mock，记录 top1_test，并给出 `m1_item3_pass = top1_test >= 91.73 - 1.0`。

REPORT.json 包含以上所有字段，再加上 `deg`、`margin` 和 `program_path`。

## 5. 测试（nixrun）

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun --seconds 2400 -- -m unittest discover -s pt_agent/tests -p 'test_r6_*.py' -v
```

**test_r6_rule.py**

1. 用 R3 的 graph 和 calibration.json（路径同 R5b）运行 `plan(..., deg=31, margin=1.25)`：
   - 文本能通过 parse 和 typecheck，typecheck 的 ok 为真；
   - op 的种类计数为 {pack:1, conv2d:19, act:19, add:9, slice:4, pad:2, mean:1, relayout:1, linear:1}，共 57 个；
   - 第一个 act 的 domain 等于 `(min*1.25, max*1.25)`；
   - 所有 conv2d 都带 bn；
   - 变量名中没有 `__`，也没有落在 RESERVED_NAMES 里的名字。
2. 对一个含不支持节点的图（把某个 act 的 op 改成 "softmax"）调用 plan，抛 ValueError；`plan(..., margin=1.0)` 也抛 ValueError。

**test_r6_lower.py**

1. **与 R5 一致**：对同一份 graph 和 calibration，所有 act 取 (deg=31, margin=1.25)：
   - `lower(parse(plan(...)))` 在 CountBackend 上的 `stats()` 与 `resnet20_fhe.build` 在 CountBackend 上的 `stats()` 完全相等，包括 counts、ops_total、distinct_consts、rot_steps 和 max_depth。
2. **精确模式**：取 2 张校准图，Mock 精确模式 lowering 的结果与 `reference_forward` 的误差 ≤1e-9。
3. **flat 路径**：用 R1 的 `examples/mlp.ptdsl`，权重按 `examples/shapes.mlp_weight_shapes()` 的形状由 `rng(20261031)` 随机生成；在 Mock（slots=8192、batch=3）上 lowering 后，与 numpy 计算 `fc2(square(fc1(x)))` 的误差 ≤1e-9。
4. **错误**：
   - R1 的 `examples/resnet20.ptdsl` 使用了 minimax_comp，lowering 时抛 LoweringError；
   - mux 布局上的 `relayout(layout="flat")` 抛 LoweringError；
   - 单独出现的 `pad` 抛 LoweringError；
   - 权重名不存在时抛 LoweringError。以上每种情况的提示里都要带 op_id。
5. **refresh**：一个 refresh="agent"、l_max=10、boot_out_level=8 的小 flat 程序 `pack → act(square, exact) → refresh → act(square, exact)`，在 `CountBackend(8192, input_level=2, boot_out_level=8)` 上 lowering 后，counts.boot=1，不报 LevelError；同一程序在 refresh="compiler" 的头部下，counts.boot=0。
6. **pack_inputs 与 decode_outputs**：
   - ResNet20 程序：pack_inputs 的结果与 `resnet20_fhe.pack_input` 逐元素相等；
   - MLP 程序：`decode_outputs` 得到的形状为 (N, 10)。

## 6. 不要做

- 不实现代价模型、延迟估计、自举放置和 LLM 规划器（R6b、R7 的内容）。
- 没有 `--final` 时不读取测试集；不在测试中读取测试集。
- 不修改 R1–R5 的文件、baseline、workloads 和 DSL_benchmark。
- 编译失败不重试，不换参数。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 7. 完成标准

- `test_r6_*.py` 全部通过；R1–R5 的测试依然通过。
- 依次运行：
  1. `python3 -m pt_agent.run --model resnet20 --planner rule --stages plan,check,mock,count,emit,compile`
  2. 确认第 1 步的 mock 结果中 agree ≥ 990 之后，再运行 `python3 -m pt_agent.run --model resnet20 --planner rule --stages plan,check,mock,test --final`
- 汇报中贴出两次 REPORT 的摘要：source_lines、exact_max_abs_diff、agree、top1（校准）、counts 和 max_depth、constants 大小、编译结果或服务器作业路径、top1_test 和 m1_item3_pass。
- STATUS.md 的 R6 一行写明结论，M1 表中第 2、3 条按实际结果更新。
