# R8d：组合 minimax ReLU 近似（`method="minimax_comp"`）

前置条件：R8b 完成（本轮会修改 R8b 新建的 capabilities.py）。在 R8c 之前完成。目标日期：2026-10-24。本轮不调用 LLM API。

## 0. 背景（Claude 于 2026-10-21 用原型验证）

- 目前只实现了 Chebyshev 近似。两个小开发模型的准确率不高，很多预测的 logits 很接近；而 Chebyshev ReLU 在宽定义域上误差较大。结果是：**cnn_plain 即使把阶数开到上限 255，与精确网络的一致率也只有 97.4~98.0%**，达不到 Agent 的可行性要求（≥99%）。这是 kernel 能力不够，不是 Agent 的问题。
- **组合 minimax 符号函数**（Lee 等，Dacapo `poly/Func.py` 中 `HE_ReLU` 的做法）：
  ```
  relu(x) = x · (0.5 + s(x/B))
  s = P3 ∘ P2 ∘ P1 ≈ sign/2
  ```
  P1、P2、P3 是 Chebyshev 基下的奇多项式，次数分别为 15、15、27。系数来自 Dacapo 的 `data/sgn151527.txt`：
  - 第 0–15 行除以 2.0 → P1；
  - 第 16–31 行除以 1.7 → P2；
  - 第 32–59 行除以 2.0 → P3。

  这与 Dacapo `Poly.py` 中的 poly1、poly2、poly3 一致。
- **原型结果**（1000 张验证图，与精确网络的一致率，余量 1.5）：

  | 模型 | cheb deg255 | minimax_comp |
  |---|---|---|
  | mlp | 994 | **1000** |
  | cnn_plain | 974 | **998** |
  | resnet20 | 999 | **1000** |

  另外测了 Cheon 等的闭式组合 fₙ，一致率只有 689–991，不采用。
- **代价**：在 CountBackend 上对单个激活计数：
  - minimax_comp：深度 17，mul_cc 26、mul_cp 46；
  - cheb deg255：深度 10，mul_cc 33、mul_cp 241；
  - cheb deg63：深度 8。

  minimax_comp 更准，乘法更少，但更深（自举更多）。这给了 Agent 一个真正的"方法选择"旋钮：逐层在精度和深度之间权衡。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/kernels/poly.py          改：§2.1
pt_agent/lower.py                 改：§2.2
pt_agent/dsl/ops_spec.py          改：§2.3
pt_agent/capabilities.py          改：§2.4（act 一行的文字）
pt_agent/planners/agent_tools.py  改：§2.5
pt_agent/planners/rule.py         改：§2.6
pt_agent/run.py                   改：--act-method
pt_agent/experiments/r8d_methods.py 新：§4
pt_agent/tests/test_r8d_minimax.py  新：§3
../docs/pt_agent/STATUS.md        改
```

## 2. 实现

### 2.1 kernels/poly.py

```python
_REPO = Path(__file__).resolve().parents[5]          # /home/lhohy/Code/Poseidon
MINIMAX_DATA = _REPO / "third_party/dacapo/python/poly/poly/data/sgn151527.txt"
MINIMAX_SETS = {(15, 15, 27): ((0, 16, 2.0), (16, 32, 1.7), (32, 60, 2.0))}   # (起始行, 结束行, 除数)

def minimax_coeffs(degrees: tuple) -> list[np.ndarray]:
    """按 MINIMAX_SETS 从 MINIMAX_DATA 读取，每行一个浮点数（strip 后 float），切片后除以对应的除数。
    degrees 不在表中，抛 KernelError(f"minimax_comp supports degrees {sorted(MINIMAX_SETS)}")。
    结果缓存（functools.lru_cache）。只读第三方文件，不复制、不修改。"""

def act_minimax(b, x, degrees: tuple, domain: tuple[float, float]):
    lo, hi = domain
    if not (hi > 0 and lo == -hi):
        raise KernelError("minimax_comp requires a symmetric domain (-B, B)")
    s = x*(1.0/hi)
    for c in minimax_coeffs(degrees):
        s = cheb_eval(b, s, c)
    return x*(s+0.5)
```

运算顺序必须与上面完全一致，否则 §3.2 的计数对不上。

### 2.2 lower.py

在 act 分支中：
- `method == "minimax_comp"` 时，调用 `act_minimax(b, value, tuple(a["degrees"]), a["domain"])`，删除原来的 `raise KernelError("minimax_comp not implemented in v1")`；
- 传给 observer 的 cfg 写为 `("minimax_comp", tuple(a["degrees"]), lo, hi)`，保持 4 个元素，与 cheb 的 `("cheb", deg, lo, hi)` 形式相同；
- exact 模式不变。

### 2.3 dsl/ops_spec.py 中 `_act` 的 minimax_comp 分支

条件改为：
- fn 为 "relu"；
- degrees 是 `kernels.poly.MINIMAX_SETS` 的一个键（比较时转成 tuple）；
- deg 为 None；
- domain 不为 None，且 `lo == -hi`、`hi > 0`。

不满足时，错误消息为：

```text
minimax_comp requires fn="relu", degrees in [(15, 15, 27)], deg=None and a symmetric domain (-B, B)
```

这里的列表由 MINIMAX_SETS 生成。如果 ops_spec 无法 import kernels（循环依赖），就把 `MINIMAX_DEGREES = ((15, 15, 27),)` 定义在 ops_spec 里，由 poly.py 引用；只保留一个定义处。

### 2.4 capabilities.py 中 act 一行（逐字替换 R8b 写入的 signature、semantics、constraints、cost 四个字段）

- **signature**：
  ```text
  act(x, fn=F, method="cheb", deg=D, domain=(lo, hi)) | act(x, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-B, B))
  ```
- **semantics**：
  ```text
  Element-wise activation F (relu, silu, gelu, sigmoid, tanh or exp). method="cheb" uses a Chebyshev polynomial of degree D fitted on [lo, hi]. method="minimax_comp" (relu only) computes x * (0.5 + s(x / B)), where s is a composition of three minimax polynomials (degrees 15, 15, 27) approximating sign(u) / 2; it is much more accurate than cheb on wide domains.
  ```
- **constraints**：
  ```text
  For cheb, 1 <= D <= 255. For minimax_comp, fn must be relu, degrees must be (15, 15, 27) and the domain must be symmetric (lo = -hi). Every input value must lie inside the domain; outside it the polynomial grows without bound and the whole network diverges. Method, degree and domain may differ for every act.
  ```
- **cost**：
  ```text
  cheb: ceil(log2(D+1)) + 2 multiplicative levels; higher D is more accurate and more expensive; a wider domain is safer and less accurate at the same D. minimax_comp: 17 multiplicative levels (deeper than cheb with D = 255, which needs 10, so it usually costs more bootstraps) but fewer multiplications.
  ```

in_prompt 仍为"是"。PROMPT_VERSION 不变，R8c 会统一升为 r8-v2。

### 2.5 planners/agent_tools.py

- validation 的 rows 中，`deg=op.attrs["deg"]` 旁边增加 `method=op.attrs["method"]` 和 `degrees=op.attrs.get("degrees")`。
- max_u 的计算不变：对称定义域下，它正好等于 |x/B|。

### 2.6 planners/rule.py 和 run.py

- `plan(..., act_method="cheb")` 新增参数。取 "minimax_comp" 时，每个 relu act 输出：
  ```
  method="minimax_comp", degrees=(15, 15, 27), domain=(-B, B)
  B = margin * max(|calib.min|, |calib.max|)
  ```
  非 relu 的 act 仍用 cheb。默认值保持原有输出逐字不变。
- run.py 新增 `--act-method {cheb,minimax_comp}`，默认 cheb，传给 rule.plan。

## 3. 测试（test_r8d_minimax.py，用 nixrun）

1. **系数**：`MINIMAX_DATA` 存在且共 60 行；`minimax_coeffs((15,15,27))` 返回的 3 个数组长度为 16、16、28，第一个数组的 `[1]` 等于 `1.27424444144022223/2.0`。
2. **数值和计数**，B=10：
   - 在 `[-10, 10]` 上取 16 个等距点，MockBackend 的 `act_minimax` 与 numpy 组合计算的结果之差 ≤1e-12。numpy 一侧写作：
     ```python
     x*(0.5+Chebyshev(c3)(Chebyshev(c2)(Chebyshev(c1)(x/10))))
     ```
   - 在 `[-10, 10]` 上取 2001 个点，numpy 组合与 relu 的差 ≤2e-3（原型实测为 4.7e-4）。
   - CountBackend 在一个新输入上执行 act_minimax：`max_depth == 17`、`counts["mul_cc"] == 26`、`counts["mul_cp"] == 46`。
3. **typecheck**：
   - `degrees=(15,15,27)` 加 `domain=(-5.0, 5.0)` 通过；
   - `(7,7,7)`、`domain=(-4.0, 5.0)`、`fn="silu"`、同时给出 deg，这 4 种情况各自报 E_ATTR 或现有的对应错误码，消息中含 "minimax_comp requires"。
4. **lowering**：用 cnn_plain（models-v2）做测试：
   - `rule.plan(graph, calib, deg=127, margin=1.5, act_method="minimax_comp")` 能通过 parse 和 typecheck；
   - 精确模式结构检查通过（差 ≤1e-9）；
   - 在 Mock 上用 4 张校准图运行，不报错。
5. **rule 默认输出不变**：`act_method` 取默认值时，resnet20 的程序与 R6c 的 program.ptdsl 逐字相同（复用现有的回归断言）。
6. **能力文档**：R8b 测试 1（注册表与 HANDLED_OPS 一致）、测试 2（没有模型专用字符串）仍然通过；`render_capabilities()` 中包含 "minimax_comp" 和 "17 multiplicative levels"。
7. R1–R8b 的测试全部通过。

## 4. 实验：`python3 -m pt_agent.experiments.r8d_methods`（不调用 API）

- 对 4 个开发模型（models-v2），用 rule 规划器生成 `--act-method minimax_comp --deg 127 --margin 1.5` 的程序，运行 `--stages plan,check,mock,count,emit,compile`。
- 外层把 R8a v2 的 REPORT 路径作为参数传入：`/home/lhohy/pt_agent_results/r8a-dev-20261021T074003-v2/REPORT.json`，从中读取 cheb 一行的数据，列出对照表：

  | 模型 | 方法 | agree/1000 | max_depth | boots（plan） | 预测延迟 | plan 延迟 | 编译耗时 / 峰值内存 |
  |---|---|---|---|---|---|---|---|

  其中 plan 延迟用 cost.plan_latency 计算。
- registry 中 round 记为 "R8d"。某个模型失败时，照实记录，然后继续下一个模型。

## 5. 汇报

测试输出、§4 的对照表、各模型的编译结果。STATUS.md 新增 R8d 一行。

## 6. 不要做

- 不修改 Dacapo 的任何文件，也不把系数复制进仓库。
- 不修改 cheb 的实现和默认的 rule 输出；不修改代价模型（R8c 负责）。
- 不实现其他组合多项式族（Cheon 的 fₙ 原型精度不足），也不自己求解 minimax 系数。
- 不调用 LLM API，不打冻结标签。
