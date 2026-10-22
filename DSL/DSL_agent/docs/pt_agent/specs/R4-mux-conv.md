# R4：多路复用布局（mux）+ 卷积 kernel（含 BN 折叠、步长 2）+ 下采样捷径 + 残差加

前置：R2 和 R2b 已完成。R3 可以还在进行中，本轮不依赖它。目标日期：2026-10-27。

## 0. 背景（只读）

- **布局方法**：沿用 Lee 等人（ICML'22）提出的多路复用并行卷积（MPCB）。参考实现是 `third_party/ckks-runtime/third_party/dacapo/python/poly/poly/MPCB.py` 中的 `MultParPack`、`MultParConvBN`、`SumSlots` 和 `InferShapes`，只读，不 import。本 spec 已经把需要的公式全部写出，照 spec 实现即可，**不要**去移植 MPCB.py 里的其他函数。
- **只做单密文**：一个张量的所有数据放在一个长度为 n 的槽向量里。ResNet20 每层都放得下：n=16384；输入用 3072 个槽；三个阶段分别用 16384、8192、4096 个槽。
- **后端**：全部 kernel 都只通过 R2 的 Backend 接口（`const`、`+`、`*`、`rotate`）发出 op，三种后端可以通用。`rotate(k)` 等价于 `np.roll(v, -k)`，即新值的槽 s 取自原来的槽 s+k。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/kernels/__init__.py        新：导出 MuxLayout, mux_layout, pack_mux, unpack_mux, conv2d_mux, downsample_pad_mux, add_mux, KernelError
pt_agent/kernels/layout.py          新：§2
pt_agent/kernels/slots.py           新：§3（sum_slots、accumulate）
pt_agent/kernels/conv.py            新：§4、§5、§6
pt_agent/experiments/r4_conv.py     新：§8
pt_agent/tests/test_r4_layout.py    新：§7
pt_agent/tests/test_r4_conv.py      新：§7
../docs/pt_agent/STATUS.md          改
```

## 2. kernels/layout.py：mux 布局

```python
class KernelError(Exception): ...

@dataclass(frozen=True)
class MuxLayout:
    n: int      # 槽数，2 的幂
    C: int      # 通道数
    H: int
    W: int
    k: int      # gap，2 的幂，≥1
    # 以下为派生量，用 property 实现：
    #   T     = ceil(C / k²)                         通道组数
    #   used  = T * k² * H * W                       一份拷贝占用的槽数
    #   p     = 2 ** floor(log2(n / used))           拷贝数（复制份数）
    #   block = n // p                               每份拷贝占用的槽区间长度

    def slot(self, c: int, y: int, x: int, copy: int = 0) -> int:
        t, r = divmod(c, k*k); a, b = divmod(r, k)
        return copy*block + t*(k*k*H*W) + (k*y + a)*(k*W) + (k*x + b)

def mux_layout(n, C, H, W, k) -> MuxLayout:
    """检查：n 和 k 是 2 的幂；C、H、W ≥1；used ≤ n。不满足时抛 KernelError。"""

def pack_mux(x: np.ndarray, lay: MuxLayout) -> np.ndarray:
    """x 的形状为 (C,H,W)。返回长度为 n 的 float64 向量：
    对所有 c<C、y<H、x<W、copy<p，令 v[slot(c,y,x,copy)] = x[c,y,x]；其余槽为 0。"""

def unpack_mux(v: np.ndarray, lay: MuxLayout, copy: int = 0) -> np.ndarray:
    """pack_mux 的逆操作，读取第 copy 份拷贝，返回 (C,H,W)。"""

def channel_slots(lay: MuxLayout, c: int) -> np.ndarray:
    """第 0 份拷贝中通道 c 的全部 H*W 个槽下标，按 (y,x) 的 C 顺序排列，返回 int64 数组。"""
```

## 3. kernels/slots.py

```python
def accumulate(acc, term):
    """acc 为 None 时返回 term，否则返回 acc + term。所有求和都用它，不要先建零常量再累加。"""

def sum_slots(b, A, M: int, stride: int):
    """完全照搬 MPCB.SumSlots 的算法，结果中槽 s 等于 Σ_{m<M} A[s + m*stride]。
    L = floor(log2(M))
    B = [A]
    for j in 1..L:   B.append(B[j-1] + B[j-1].rotate(2**(j-1) * stride))
    C = B[L]
    for j in 0..L-1:
        if (M // 2**j) % 2 == 1:
            C = C + B[j].rotate((M // 2**(j+1)) * 2**(j+1) * stride)
    return C
    M == 1 时直接返回 A，不产生 op。"""
```

`rotate` 的参数为 0 时，R2 的后端会直接返回原对象，不产生 op。所以这里不需要特殊处理。

## 4. conv2d_mux

```python
def conv2d_mux(b, x, lin: MuxLayout, weight: np.ndarray, *, stride: int,
               scale: np.ndarray | None = None, shift: np.ndarray | None = None):
    """返回 (value, lout)。weight 的形状为 (Co, Ci, fh, fw)。"""
```

### 4.1 前置检查（不满足时抛 KernelError）

- `Ci == lin.C`；`fh == fw` 且为奇数；`stride ∈ {1, 2}`；`lin.H % stride == 0` 且 `lin.W % stride == 0`。
- 隐含的卷积参数是 padding=(fh-1)//2、dilation=1、groups=1。这些检查交给调用方，本函数不负责。
- `lout = mux_layout(lin.n, Co, lin.H//stride, lin.W//stride, lin.k*stride)`，lout 必须构造成功。
- scale 和 shift 若给出，长度都必须为 Co。

### 4.2 算法

符号约定：n、k、H、W、T、p、block 取自 lin；ko、Ho、Wo、To、po、blocko 取自 lout；c0 = (fh-1)//2。

**第 1 步：输入平移。** 对每个卷积核位置 (i1,i2)，i1、i2 取 0..fh-1，令 dy = i1-c0、dx = i2-c0，计算 `tmp[i1][i2] = x.rotate(k*k*W*dy + k*dx)`。位移为 0 时，后端不产生 op。

**第 2 步：分组乘权重。** 共 `q = ceil(Co / p)` 组，对每组 i3 = 0..q-1：

- 对每个 (i1,i2) 构造权重向量 `u`，长度为 n、初值全 0。对每份拷贝 j<p，令 o = p*i3 + j；若 o < Co，则对所有 c<Ci、y<H、x<W：
  - 若 `0 ≤ y+dy < H` 且 `0 ≤ x+dx < W`，令 `u[lin.slot(c,y,x,copy=j)] = weight[o, c, i1, i2]`；
  - 否则令该槽为 0（这就是补零）。
- 若 u 全为 0，就跳过这个 (i1,i2)；否则执行 `B = accumulate(B, tmp[i1][i2] * b.const(u))`。B 在每组开始时为 None。整组的 u 都是全 0 时，跳过这一组。

**第 3 步：通道求和。** 依次执行：

```
Cs = sum_slots(b, B, k, 1)
Cs = sum_slots(b, Cs, k, k*W)
Cs = sum_slots(b, Cs, T, k*k*H*W)
```

执行后，第 j 份拷贝中槽 `j*block + (k*y)*(k*W) + k*x` 的值，就是输出通道 o=p*i3+j 在输入网格 (y,x) 处的卷积值。

**第 4 步：搬到输出布局。** 对本组中每个 j（满足 o = p*i3 + j < Co）：

- 令 `to, r = divmod(o, ko*ko)`、`ao, bo = divmod(r, ko)`；
- `delta = to*(ko*ko*Ho*Wo) + ao*(ko*Wo) + bo - j*block`；
- 掩码 `m`：长度 n，在 `channel_slots(lout, o)` 处的值为 `scale[o]`（scale 为 None 时取 1.0），其余为 0；
- `D = accumulate(D, Cs.rotate(-delta) * b.const(m))`。D 在第一组开始前为 None，所有组共用。

步长 2 时，第 3 步中 y=2*yo、x=2*xo 处的值，正好落在 lout 中 (yo,xo) 对应的源槽位置上：因为 ko*Wo = k*W，且 ko = 2k。所以不需要额外的抽样步骤。

**第 5 步：复制。** 对 j = 0 .. log2(po)-1，执行 `D = D + D.rotate(-(2**j) * blocko)`。

**第 6 步：偏移。** shift 不为 None 且不全为 0 时，构造长度为 n 的向量 `h`：对所有拷贝 j<po、通道 o<Co、yo<Ho、xo<Wo，令 `h[lout.slot(o,yo,xo,j)] = shift[o]`。然后 `D = D + b.const(h)`。

**返回** `(D, lout)`。

### 4.3 乘法深度

第 2 步一次 mul_cp，第 4 步一次 mul_cp，所以每次卷积的乘法深度恰好是 2。

### 4.4 BN 折叠（工具函数，同在 conv.py 中）

```python
def fold_bn(gamma, beta, mean, var, eps, conv_bias=None) -> tuple[np.ndarray, np.ndarray]:
    """G = gamma / sqrt(var + eps)
    H = beta - G*mean + (G*conv_bias if conv_bias is not None else 0)
    返回 (scale=G, shift=H)。"""
```

没有 BN、但有卷积 bias 时，调用方传 `scale=None, shift=bias`。

## 5. downsample_pad_mux：ResNet 的 option A 捷径

对应 PyTorch 的 `F.pad(x[:, :, ::2, ::2], (0,0,0,0,pb,pa))`。

```python
def downsample_pad_mux(b, x, lin: MuxLayout, pad_before: int, pad_after: int):
    """构造 1×1 的恒等选择权重 w：形状为 (Co, Ci, 1, 1)，其中 Co = Ci + pad_before + pad_after，Ci = lin.C；
    当 o == c + pad_before 时 w[o, c, 0, 0] = 1，否则为 0。
    返回 conv2d_mux(b, x, lin, w, stride=2)。"""
```

第 2 步会跳过全 0 的权重向量，所以这个函数实际产生的 op 远少于一次普通卷积。

## 6. add_mux

```python
def add_mux(b, x, lx: MuxLayout, y, ly: MuxLayout):
    """要求 lx == ly，否则抛 KernelError。返回 (x + y, lx)。"""
```

## 7. 测试

用 R2 的 nixrun 运行：

```bash
cd DSL/DSL_agent/scripts && python3 -m pt_agent.tools.nixrun --seconds 1800 -- -m unittest discover -s pt_agent/tests -p 'test_r4_*.py' -v
```

测试文件里自己写一个 numpy 参考卷积 `conv_ref(x[C,H,W], w, stride, pad)`：先补零，再用 `sliding_window_view` 加 einsum 计算。不要 import torch 和 R3。随机数统一用 `np.random.default_rng(20261027)`，输入取 uniform(-1,1)，权重取 normal(0, 0.3)。

**判定标准**：`unpack_mux(Mock 输出, lout, copy=j)` 对**每一份**拷贝 j<lout.p 都与参考结果的最大绝对误差 ≤1e-9。另外，lout 中所有不属于任何通道和拷贝的槽，值必须为 0（误差 ≤1e-12）。

**test_r4_layout.py**

1. `mux_layout(256,4,4,4,1)`：T=4，used=64，p=4，block=64。`mux_layout(256,4,4,4,2)`：T=1，used=64，p=4。`mux_layout(16384,3,32,32,1)`：T=3，used=3072，p=4，block=4096。
2. 对 (n,C,H,W,k) 取 (256,4,4,4,1)、(256,6,2,2,2)、(16384,32,16,16,2)、(16384,64,8,8,4) 四组：`unpack_mux(pack_mux(x))` 与 x 完全相等；所有拷贝都相等；`slot()` 对不同的 (c,y,x,copy) 两两不同。
3. used > n 时抛 KernelError；k=3 时抛 KernelError。

**test_r4_conv.py**

| # | 输入布局 (n,C,H,W,k) | 卷积 | 期望 |
|---|---|---|---|
| T1 | (256,4,4,4,1) | Co=4，3×3，stride=1 | 数值通过；Count 为 rot=15、mul_cp=13、add=15、mul_cc=0、`max_depth=2` |
| T2 | (256,4,4,4,1) | Co=4，3×3，stride=2 | 数值通过（参考用 stride=2、pad=1）；lout 为 (256,4,2,2,2) |
| T3 | (128,2,4,4,1) | Co=6，3×3，stride=1 | 数值通过（p=4，所以 q=2） |
| T4 | (256,4,4,4,1) | 依次为 Co=4 s1 → Co=8 s2 → Co=8 s1 三层串联 | 数值通过；Count 的 `max_depth=6` |
| T5 | (256,4,4,4,1) | Co=4 s1，带 bias 和 BN（随机 gamma、beta、mean，var 取 uniform(0.5,2)，eps=1e-5），用 fold_bn | 与"conv + bias → BN"的参考一致 |
| T6 | (256,4,4,4,1) | `downsample_pad_mux(pb=2, pa=2)` | 与 `np.pad(x[:, ::2, ::2], ((2,2),(0,0),(0,0)))` 一致；lout 为 (256,8,2,2,2)；Count 的 mul_cp 数小于 T2 |
| T7 | (256,4,4,4,1) | 主路径 conv s2 (Co=8) 加上 T6 的捷径，用 add_mux 相加 | 数值通过 |
| T8 | (16384,3,32,32,1) | Co=16 s1 → Co=32 s2 → Co=64 s2，即 ResNet20 的三种形状 | 三层的 lout 分别为 (16384,16,32,32,1)、(16384,32,16,16,2)、(16384,64,8,8,4)；数值通过；把 Mock 用时写进测试输出（print） |
| T9 | (256,4,4,4,1) | 与 T1 相同的程序，用 EmitBackend 生成 | counts 与 Count 一致；用 R2 的 fake_hecate_expr 运行后与 Mock 逐位相等 |
| T10 | — | 错误：fh=2、stride=3、Ci 不等于 lin.C、H 为奇数且 stride=2 | 都抛 KernelError |

T1 的计数推导如下，写进测试的注释里：

- 第 1 步：8 次 rot（9 个核位置中有 1 个位移为 0）；
- 第 2 步：9 次 mul_cp、8 次 add；
- 第 3 步：只有 M=T=4 那一次 sum_slots 产生 op，2 次 rot、2 次 add；
- 第 4 步：4 次 mul_cp；o=1..3 各 1 次 rot，共 3 次；3 次 add；
- 第 5 步：po=4，2 次 rot、2 次 add。

合计 rot=15、mul_cp=13、add=15。

## 8. experiments/r4_conv.py：真实编译与本机 CKKS

外层和内层的运行模式与 r2b_smoke 相同：外层写 registry，round 为 "R4"。每个用例用 `toolchain.new_run_dir`，用例之间串行。

| 用例 | 内容 | 步骤 |
|---|---|---|
| `r4-t1-ckks` | T1 的程序（P=n=256），4 个随机样本 | 用 EmitBackend 写出 → trace → compile("eva") → bind(256) → write_inputs（每个样本的输入为 `pack_mux(x_s, lin)`）→ execute_cpu → 取 `[s,0,:256]`，同时与 Mock 输出和 `pack_mux(conv_ref(...), lout)` 比较。判定标准：CKKS 容差 `1e-5+1e-4|ref|` |
| `r4-t7-ckks` | T7 的程序 | 同上 |
| `r4-resnet-conv1-compile` | T8 的第一层（n=16384，Co=16 s1） | 只做 trace → compile("eva") → plan_stats，记录编译耗时、steps、values，以及 Count 的 counts。不执行，因为本机执行器的周期上限是 256 |

REPORT.json 的格式沿用 r2b_smoke 的 cases 结构。另外，每个用例都要写进 Count 的 `counts` 和 `max_depth`。

## 9. 不要做

- 不实现多密文、depthwise/groups>1、dilation>1、偶数核或非"same"补零。这些情况一律抛 KernelError。
- 不实现 tap 算法、avgpool、mean、linear 和激活（这些属于 R5）。
- 不 import torch、MPCB.py 或 R3 的模块。不修改 R1/R2/R2b 的文件和 baseline。
- 第 2 步的跳过条件只有"权重向量全为 0"这一种，不要加其他优化。
- spec 写不通的地方，记进 STATUS.md 的"阻塞"一栏后停下。

## 10. 完成标准

- `test_r4_*.py` 全部通过，T1 的计数与 §7 完全一致；R1、R2、R2b 的测试依然通过。
- `r4_conv` 跑完，在汇报里贴出 REPORT.json 的摘要：两个 CKKS 用例的 max_abs_err 及是否通过；conv1 的编译耗时和 counts。
- STATUS.md 的 R4 一行写明结论。
