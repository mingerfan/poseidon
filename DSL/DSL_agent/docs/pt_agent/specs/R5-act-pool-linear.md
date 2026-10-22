# R5：Chebyshev 激活近似、全局平均池化、全连接、批量 Mock、ResNet20 整网组装

前置：R3、R4 已完成。目标日期：2026-10-29。本 spec 的全部算法和期望值，都已在 2026-10-20 用 numpy 原型实测过。

## 0. 已有的数据（照用）

- R3 的 ResNet20 结果：`/home/lhohy/pt_agent_results/r3-resnet20-20261020T131658/`，其中有 `graph/` 和 `calibration.json`。19 个 ReLU 输入张量的范围从 [-5.0, 7.3] 到 [-13.7, 27.2] 不等。
- 统一约定：测试集只用于最终验收，本轮只使用 1000 张校准图，不读取测试集。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/backends/mock.py          改：§2 批量与 apply_exact（R2 的测试必须仍然通过）
pt_agent/kernels/layout.py         改：新增 index_grid、FlatLayout、pack_flat/unpack_flat（§3）
pt_agent/kernels/conv.py           改：常量构造改为向量化（§3），算法不变（R4 的测试必须仍然通过）
pt_agent/kernels/poly.py           新：§4
pt_agent/kernels/pool.py           新：§5
pt_agent/kernels/linear.py         新：§6
pt_agent/kernels/__init__.py       改：导出新函数
pt_agent/models/resnet20_fhe.py    新：§7
pt_agent/experiments/r5_resnet20.py 新：§8
pt_agent/tests/test_r5_poly.py     新
pt_agent/tests/test_r5_pool_linear.py 新
pt_agent/tests/test_r5_mock_batch.py  新
pt_agent/tests/test_r5_resnet20.py    新
../docs/pt_agent/STATUS.md         改
```

## 2. backends/mock.py：批量计算与精确激活

- 构造函数改为 `MockBackend(slots, batch=None)`。
- `batch` 为 None 时行为与 R2 完全一样。
- `batch=N` 时：
  - `bind` 的每个输入必须是形状为 (N, slots) 的有限 float64 数组；
  - 密文值的形状为 (N, slots)；Plain 仍然是长度为 slots 的一维数组，运算时靠 numpy 广播；
  - rotate 改为 `np.roll(v, -k, axis=-1)`；
  - `outputs` 返回形状为 (N, slots) 的数组列表。
- 新增 `MockBackend.apply_exact(value, fn: str)`：对密文值逐元素调用 `poly.EXACT[fn]`，返回新的密文值。它**只存在于 Mock 中**，不计入任何 op，仅用于"精确模式"的结构检查。另外两个后端没有这个方法。

## 3. layout.py 的补充，以及卷积常量的向量化

```python
MuxLayout.index_grid(self) -> np.ndarray   # 形状 (p, C, H, W) 的 int64，元素为 slot(c,y,x,copy)，用 numpy 广播一次算出
@dataclass(frozen=True)
class FlatLayout:
    n: int
    size: int          # 有效元素放在槽 0..size-1，其余槽为 0；只有一份拷贝
def pack_flat(v: np.ndarray, lay: FlatLayout) -> np.ndarray
def unpack_flat(x: np.ndarray, lay: FlatLayout) -> np.ndarray   # 取前 size 个槽
```

- `pack_mux`、`unpack_mux`、`channel_slots`，以及 `conv.py` 第 2、4、6 步里构造 u、m、h 这三类常量，全部改用 `index_grid` 加 numpy 花式索引，不再写逐元素的 Python 循环。算法和 op 序列保持不变：R4 的 T1 计数必须仍然是 15/13/15。
- 性能要求：在 n=16384 下构造 ResNet20 全部 19 个卷积的常量，Mock 单次建图（batch=1、精确模式）不超过 60 秒。

## 4. kernels/poly.py：Chebyshev 近似

```python
EXACT = {"relu": lambda x: np.maximum(x, 0), "silu": ..., "gelu": ..., "sigmoid": ..., "tanh": np.tanh, "exp": np.exp}
# silu = x/(1+exp(-x))；gelu = 0.5*x*(1+erf(x/sqrt(2)))，erf 用 R3 analyzer 里同一套实现

def cheb_fit(fn: str, deg: int, lo: float, hi: float) -> np.ndarray:
    """系数 c[0..deg]：np.polynomial.chebyshev.chebinterpolate(lambda u: EXACT[fn](((hi-lo)*u + (hi+lo))/2), deg)。
    要求 1 ≤ deg ≤ 255 且 lo < hi。"""

def cheb_eval(b, u, coeffs) -> value:
    """在 u（取值应在 [-1,1] 内）上计算 Σ c_j T_j(u)，算法见 §4.1。"""

def act_cheb(b, x, fn: str, deg: int, domain: tuple[float, float]):
    """lo, hi = domain；a = 2/(hi-lo)；c = -(hi+lo)/(hi-lo)
    u = x * a + c       # 1 次 mul_cp + 1 次 add，消耗 1 层深度
    return cheb_eval(b, u, cheb_fit(fn, deg, lo, hi))"""
```

### 4.1 cheb_eval 的算法（逐行实现）

```python
d = len(coeffs) - 1                 # 要求 d ≥ 1
l = max(1, (d + 1).bit_length() // 2)
g = 2 ** l                          # 小步规模
# 小步：T[1..g]
T = {1: u}
for j in range(2, g + 1):
    if j % 2 == 0:
        s = T[j//2] * T[j//2]; T[j] = s + s - 1.0
    else:
        h = j // 2; q = T[h] * T[h+1]; T[j] = q + q - T[1]
# 大步：G[g], G[2g], ..., G[g*2^(m-1)]，m 为满足 g*2^m > d 的最小值
m = 0
while g * 2**m <= d: m += 1
G = {g: T[g]}
for i in range(1, m):
    prev = G[g * 2**(i-1)]; s = prev * prev; G[g * 2**i] = s + s - 1.0

def base(c):                        # len(c) ≤ g
    out = None
    for j in range(1, len(c)):
        if c[j] != 0: out = accumulate(out, T[j] * float(c[j]))
    c0 = float(c[0]) if len(c) else 0.0
    if out is None: return c0       # 返回 Python float，表示常数
    return out + c0 if c0 != 0.0 else out

def rec(c, level):
    c = np.trim_zeros(np.asarray(c, float), 'b')
    if len(c) <= g: return base(c)
    k = g * 2**level
    while k >= len(c): level -= 1; k = g * 2**level
    q, r = np.polynomial.chebyshev.chebdiv(c, [0]*k + [1])
    hq = rec(q, level - 1); hr = rec(r, level - 1)
    t = G[k] * hq if isinstance(hq, float) else hq * G[k]
    return t + hr if (not isinstance(hr, float) or hr != 0.0) else t

result = rec(coeffs, m - 1)
# result 是 float 时（coeffs 只剩常数项），抛 KernelError
```

- 说明：`s + s - 1.0` 用加法代替乘以 2，避免多消耗一层。`T[j] * float(c)` 是标量乘法，在后端里算作 mul_cp。
- 深度：除去仿射那一层，d = 15/31/63/127 时分别为 5/6/7/8，一般情况满足 ≤ ceil(log2(d+1)) + 1。

## 5. kernels/pool.py：mean_mux

```python
def mean_mux(b, x, lin: MuxLayout) -> tuple[value, FlatLayout]:
    """对应 PyTorch 的 mean(dim=(2,3))。要求 lin.W 和 lin.H 都是 2 的幂。
    B = sum_slots(b, x, lin.W, lin.k)
    B = sum_slots(b, B, lin.H, lin.k*lin.k*lin.W)
    out = None
    for t in range(lin.T):
        for a in range(lin.k):
            src = t*k*k*H*W + a*k*W;  dst = t*k*k + a*k
            m = 长度为 n 的零向量；对 bb in range(k)：若 c = dst+bb < C，则 m[c] = 1/(H*W)
            if m 全为 0: continue
            out = accumulate(out, B.rotate(src - dst) * b.const(m))
    return out, FlatLayout(lin.n, lin.C)"""
```

## 6. kernels/linear.py：linear_flat

```python
def linear_flat(b, x, lin: FlatLayout, weight: np.ndarray, bias: np.ndarray | None, *, algo="bsgs", n1=None):
    """weight 的形状为 (O, I)，要求 I == lin.size。
    Ip = 不小于 I 的最小 2 的幂；Op = 不小于 O 的最小 2 的幂
    要求 Op ≤ Ip 且 2*Ip ≤ n，否则抛 KernelError
    Wp = 形状为 (Op, Ip) 的零矩阵，Wp[:O, :I] = weight
    xr = x + x.rotate(-Ip)          # 把 x 复制一份到 Ip..2Ip-1，形成循环窗口
    u(d)：长度为 n 的向量，对 s<Ip 令 u[s] = Wp[s % Op, (s + d) % Ip]，其余为 0
    algo == "diag"：
        z = Σ_{d<Op，u(d) 不全为 0} xr.rotate(d) * u(d)
    algo == "bsgs"：
        g = n1 若给出，否则 2**ceil(log2(Op)/2)；要求 g 是 2 的幂且 g ≤ Op
        R[i] = xr.rotate(i)，i < g
        对 j in range(ceil(Op/g))：
            inner = Σ_{i<g，d=g*j+i<Op，np.roll(u(d), g*j) 不全为 0} R[i] * np.roll(u(d), g*j)
            z = accumulate(z, inner.rotate(g*j))
    y = sum_slots(b, z, Ip // Op, Op)
    m = 长度为 n 的向量，m[:O] = 1
    y = y * b.const(m)
    bias 不为 None 时：y = y + b.const(槽 0..O-1 为 bias、其余为 0 的向量)
    return y, FlatLayout(n, O)"""
```

`algo` 不是 "diag" 或 "bsgs" 时，抛 KernelError。

## 7. models/resnet20_fhe.py：按图组装整网

```python
def build(b, graph: Graph, act_cfg: dict, *, n: int = 16384, x=None, observer=None):
    """把 R3 的 PT-Graph 按规则映射到 R4、R5 的 kernel，返回 (logits_value, FlatLayout)。
    x 为 None 时使用 b.input(0)。act_cfg 将每个 act 节点的输入张量名映射为
    ("exact",)（只在 Mock 下允许）或 ("cheb", deg, lo, hi)。
    observer 不为 None 时，每个 act 计算前调用 observer(张量名, value, layout, cfg)。
    §8 的 out_of_domain 统计就通过它实现：对 Mock 取 value.value，只统计 layout 中有效槽的元素。"""
def pack_input(img: np.ndarray, n=16384) -> np.ndarray    # img 形状为 (3,32,32)，结果为 pack_mux(img, mux_layout(n,3,32,32,1))
def decode_logits(vec, lay: FlatLayout) -> np.ndarray     # unpack_flat
```

按 `graph.nodes` 的顺序处理，同时维护 `张量名 → (value, layout)` 的映射。规则如下：

1. **conv2d**：
   - 唯一的消费者是 batchnorm 时，两者合并：`conv2d_mux(..., stride, scale, shift = fold_bn(...))`，并跳过那个 batchnorm 节点；
   - 否则直接调用 `conv2d_mux`，`shift` 取 conv 的 bias。
   - 要求 padding=(fh-1)//2、dilation=(1,1)、groups=1、stride 为 (1,1) 或 (2,2)。
2. **单独出现的 batchnorm**（前面不是 conv）：抛 KernelError，提示"v1 不支持"。
3. **act**：按 act_cfg 选 `act_cheb` 或 `b.apply_exact`。fn 必须是 relu、silu、gelu、sigmoid、tanh、exp 之一。
4. **add**：两个输入都是激活张量时用 `add_mux`；其他情况抛 KernelError。
5. **slice → slice → pad 链**：
   - 识别条件：第一个 slice 为 axis=2、start=0、step=2、end=H；它唯一的消费者是 axis=3、start=0、step=2、end=W 的 slice；后者唯一的消费者是 pads=(0,0,0,0,pb,pa) 的 pad。
   - 识别成功后，三个节点合成一次 `downsample_pad_mux(pb, pa)`。
   - 不满足这个模式的 slice 和 pad 都抛 KernelError。
6. **mean**：要求 axes=(2,3) 且 keepdim=False，调用 `mean_mux`。
7. **linear**：调用 `linear_flat(algo="bsgs")`。
8. **其他 op**：抛 KernelError，提示信息中写明节点 id 和 op。

图输出必须是 linear 产生的 flat 值。

## 8. experiments/r5_resnet20.py

运行模式同 R3：外层用系统 python3，内层在 Nix 中执行，registry 写 round="R5"。结果目录为 `$PT_RESULTS/r5-resnet20-<时间戳>`。

1. **结构检查（精确模式）**：
   - 取 8 张校准图，用 `MockBackend(16384, batch=8)` 和全部 ("exact",)；
   - logits 与 R3 的 `reference_forward` 相比，最大绝对误差 ≤1e-9；
   - 记录 `exact_max_abs_diff` 和 `build_seconds`。
2. **计数**：用 `CountBackend(16384)`，全部 act 取 ("cheb", 63, lo, hi)，lo、hi 取自 calibration.json 中该张量的 min 和 max。记录 `stats()`，其中最重要的是 counts、rot_steps 的长度和 max_depth。
3. **近似网格（只用校准集）**：
   - deg 取 {31, 63, 127}，margin 取 {1.0, 1.25}，每个 act 的定义域为 `(min*margin, max*margin)`；
   - 1000 张校准图按 batch=200 分 5 批，用 `MockBackend(16384, batch=200)` 计算；
   - 每种组合记录：`agree`（argmax 与精确参考一致的张数）、`top1`（对标签的准确率，单位 %）、`logit_rmse`（相对精确参考）、`max_abs_logit_diff`、`seconds`。
   - margin=1.0 时，校准输入按定义不会越界；但多项式在每层的误差会累积，所以还要额外记录每批中有多少个 act 输入元素超出了 [lo, hi]，记为 `out_of_domain`。实现方式：在 build 里对 Mock 的 u 值做统计，只在 Mock 下进行，用一个可选的 `observer` 回调实现。
4. **整网 Emit 与编译尝试**：
   - 取网格中 agree 最高的组合（同分时取 deg 较小的），用 `EmitBackend(16384)` 生成整网程序，写进 `toolchain.new_run_dir("r5-resnet20-emit")`；
   - 记录 `constants.npz` 的字节数和 Emit 的 counts；
   - constants.npz 超过 1.5 GiB 时不编译，只生成 `server_jobs/r5-resnet20/run.sh`，内容是在服务器上执行 trace 和 compile 的命令；
   - 否则依次执行 trace 和 `compile(mode="dacapo-depth-dp", boot="qwen48", seconds=3600)`，记录退出码、耗时、`plan_stats` 中的 boots 和 steps，失败时记录 log_tail。失败不重试，也不换参数。
5. **REPORT.json** 包含以上所有字段。

预计总耗时在 2 小时以内。超过 3 小时就中止，并保存已完成的部分。

## 9. 测试（nixrun）

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun --seconds 2400 -- -m unittest discover -s pt_agent/tests -p 'test_r5_*.py' -v
```

**test_r5_poly.py**

1. 用 `rng(20261029)` 生成随机系数，阶数 d ∈ {1,2,3,7,8,15,16,31,63,127}。在 u ∈ uniform(-1,1, 64) 上，Mock 的 `cheb_eval` 与 `chebval` 的误差 ≤1e-12；Count 的 `max_depth ≤ ceil(log2(d+1)) + 1`。
2. d ∈ {15,31,63,127} 时，`cheb_eval` 的 Count `max_depth` 分别等于 5、6、7、8。
3. `act_cheb(relu, deg, (-13.7, 27.2))` 在 slots=4096 上：
   - deg=15：counts 为 mul_cp=13、add=20、mul_cc=7、sub=4，`max_depth=6`；
   - deg=127：counts 为 mul_cp=121、add=145、mul_cc=24、sub=17，`max_depth=9`；与 relu 的最大误差 ≤0.05（输入在该定义域内均匀取样）。
4. 只有常数项的系数，或 d=0，抛 KernelError。

**test_r5_pool_linear.py**

1. `mean_mux` 在 (16384,64,8,8,4)、(256,4,4,4,1)、(256,8,2,2,2) 三种布局上，与 `x.mean(axis=(1,2))` 的误差 ≤1e-12，尾部槽全为 0。(16384,64,8,8,4) 的 counts 为 rot=21、add=21、mul_cp=16，`max_depth=1`。
2. `linear_flat`，64→10，n=16384：
   - diag：counts 为 rot=18、add=19、mul_cp=17；
   - bsgs：counts 为 rot=9、add=19、mul_cp=17；
   - 两者误差都 ≤1e-12、深度都为 2。
   - 另外 16→16、10→3 两种形状数值也通过；32→64 抛 KernelError。
3. EmitBackend 运行 64→10 bsgs 的结果与 Count 计数一致，经 fake_hecate 运行后与 Mock 逐位相等。

**test_r5_mock_batch.py**

1. batch=5 时，R2 的 P1 程序结果与逐样本运行 Mock 完全相等。
2. `apply_exact` 在 batch 模式和非 batch 模式下都正确；对 Count 和 Emit 调用会抛出 AttributeError（因为它们没有这个方法）。

**test_r5_resnet20.py**

1. `build` 精确模式，batch=2，取两张校准图，与 `reference_forward` 的误差 ≤1e-9。
2. 用 CountBackend、全部 ("cheb", 31, -10, 10) 跑一遍，不报错；`counts["mul_cc"] > 0`，`max_depth ≥ 100`。
3. 往图里人为加入一个不支持的节点（例如把某个 act 的 fn 改成 "hardswish"），抛 KernelError，且提示里写明节点 id。

## 10. 不要做

- 不读取测试集；不在本轮选定"最终"的近似配置（那是 R6 或 Agent 的工作），只记录网格结果。
- 不实现 minimax_comp、多密文、PT-DSL lowering，也不实现代价模型。
- 不修改 R1–R3 的文件（mock.py、layout.py、conv.py 按 §1 修改除外）、baseline 和 workloads。
- 编译失败不重试；不在 CPU 上执行整网。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 11. 完成标准

- `test_r5_*.py` 全部通过；R1–R4 的测试依然通过。
- `r5_resnet20` 跑完。汇报中贴出：
  - 精确模式的误差；
  - Count 的 counts 和 max_depth；
  - 网格的 6 行结果（agree、top1、rmse、out_of_domain）；
  - Emit 常量文件大小；
  - 编译结果（或者服务器作业路径）。
- STATUS.md 的 R5 一行写明结论。
