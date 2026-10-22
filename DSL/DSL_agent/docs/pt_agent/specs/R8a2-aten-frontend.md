# R8a2：通用的 ATen 层前端（make_fx），跑完 R8a

> 本文件取代先前的 `R8a2-proxy-trace.md`。那一版只包装了 3 个池化函数，属于点状补丁，**不要实现它**。

## 0. 为什么换前端（Claude 于 2026-10-21 实测）

- fx 符号跟踪下，`x.size()`、`x.shape` 返回的是 Proxy，C 实现的函数（如 `F.avg_pool2d`、`torch.reshape`、`torch.flatten`）拿到 Proxy 参数会直接报错。
  - 即使把 `torch.nn.functional` 的 137 个公开函数全部包装起来，R3 的 47 个测试虽然都能通过，但 `torch.reshape(…, (x.shape[0], -1))`、`torch.flatten(start_dim=x.dim()-3)` 这类写法仍然失败。
  - 在 fx 上逐个打补丁，覆盖不完，没有通用性。
- **PyTorch 2.0.1 已经带有 `torch.fx.experimental.proxy_tensor.make_fx`。** 它用真实张量执行模型，把调用记录到 ATen 算子这一层，`torch.export` 内部也是这个原理。形状运算和依赖形状的 Python 控制流都会被直接算成常数；module、functional、Tensor 方法、lambda 等各种写法，最终都归一成一套 ATen 算子。实测结果：
  - `torch.reshape(F.avg_pool2d(x, x.size()[3]), (x.shape[0], -1))`、`torch.flatten(…, start_dim=x.dim()-3)`、`n,c,h,w=x.shape; x.view(n,c,h*w).mean(-1)`、依赖形状的 if 加通道重排加 hardswish，以及 Dacapo 的 ResNet20-SiLU，跟踪后与直接运行的误差都是 **0.0**；
  - 19 类常见模块和写法（conv/bn/relu、depthwise、linear、各种池化、flatten、dropout、6 种激活、layernorm、softmax、maxpool、算术、split、矩阵乘、归约、attention、slice/pad）总共只产生 **37 种 ATen 算子**；
  - 参数会变成 `_param_constant*` 和 `_tensor_constant*` 常量，但它们与原模块的参数共享存储，按 `data_ptr` 可以对回 state_dict 中的名字（例如 `0.weight`、`1.running_mean`）。
- **通用性原则**：前端只维护一张 ATen 核心算子映射表，不针对任何模型或写法打补丁。

## 1. 文件清单

路径相对 `DSL/DSL_agent/scripts/`：

```
pt_agent/intake/aten_frontend.py   新：§2
pt_agent/intake/__init__.py        改：import_module(..., frontend="fx"|"aten")，默认仍为 "fx"（R3 的测试不变）
pt_agent/models/zoo.py             改：zoo 的导入一律用 frontend="aten"
pt_agent/run.py                    改：zoo 模型用 aten 前端
pt_agent/tests/test_r8a2_aten.py   新：§3
../docs/pt_agent/STATUS.md         改
```

R8a 中已经实现的 fx 静态解析（torch_mapping.py）保留，不删除。

## 2. intake/aten_frontend.py

```python
def import_module_aten(module, example_inputs: tuple, name: str) -> Graph:
```

1. 执行 `m = copy.deepcopy(module).eval().double()`，输入转成 float64，然后在 `torch.no_grad()` 下调用 `gm = make_fx(m, tracing_mode="real")(*inputs)`，再执行 `ShapeProp(gm).propagate(*inputs)`。make_fx 抛出的异常转成 `IntakeError`，op 记为 "trace"，原因里写上异常文本。
2. **权重命名**：建立 `{t.data_ptr(): name}`，来源是 `m.named_parameters()` 和 `m.named_buffers()`，同一地址取先出现的名字。每个 get_attr 节点对应的张量，如果在表里，就用 state_dict 中的名字；不在表里，就命名为 `const.<attr 名>`。权重一律转成 float64 的只读 numpy 数组。
3. **激活张量名**用 aten 节点名。placeholder 是图的输入，output 节点的参数是图的输出。
4. **映射表**：逐个节点处理；不支持的节点全部收集起来，最后一起抛 IntakeError，不要遇到第一个就停。"无消费者"指该节点没有 users。

| aten target | → 核心 op / 处理 |
|---|---|
| `convolution` | conv2d。要求 `transposed=False`，否则报不支持，原因写 "transposed convolution"。weight、bias 取 get_attr 的名字（bias 为 None 则不填），stride、padding、dilation、groups 取自参数 |
| `native_batch_norm`、`_native_batch_norm_legit_no_training` | batchnorm。要求 training 为 False；weight、bias、running_mean、running_var 取名字；eps 取自参数。只有 `getitem[0]` 的输出被使用，`getitem[1]` 和 `[2]` 必须无消费者 |
| `empty.memory_format` 等无消费者的节点 | 忽略 |
| `relu`、`silu`、`sigmoid`、`tanh`、`hardswish`、`exp` | act，fn 取对应名字 |
| `gelu` | act，approximate 为 "tanh" 时 fn 为 gelu_tanh，否则为 gelu |
| `add.Tensor`、`add_.Tensor` | add（alpha 必须为 1）；另一个操作数为标量时用 scalar，为 get_attr 张量时用 const |
| `sub.Tensor` | sub（规则同上） |
| `rsub.Scalar(x, c)` | sub，reverse=True，scalar=c |
| `mul.Tensor` | mul（规则同上） |
| `div.Tensor` | 除数必须是标量或常量，转成 mul，取倒数 |
| `neg` | mul，scalar=-1.0 |
| `addmm(b, x, t(W))` | linear，weight 为 W 的名字，bias 为 b 的名字。第三个参数必须是作用在权重 get_attr 上的 `t`。被吸收的 `t` 节点不再单独生成节点 |
| `mm(x, t(W))` | linear，无 bias，规则同上 |
| `t`（不是作用在权重上，或者有其他消费者） | permute，dims=(1,0) |
| `bmm`、`mm`（两个操作数都是激活） | matmul |
| `expand` | 输出形状等于输入形状时视为 identity，否则报不支持，原因写 "broadcast expand" |
| `view`、`_unsafe_view`、`reshape` | reshape，shape 取输出的静态形状 |
| `permute` | permute |
| `transpose.int(x, a, b)` | permute，dims 由 a、b 交换得到 |
| `slice.Tensor(x, dim, start, end, step)` | 按静态形状归一化。start=0、end≥该维长度且 step=1 时为 identity；否则为 slice，end 取 min(end, 该维长度) |
| `constant_pad_nd(x, pads, value)` | pad。value 必须为 0 |
| `avg_pool2d(x, k, stride, padding, ceil_mode, count_include_pad, divisor_override)` | avgpool。stride 为空列表时取 k；要求 ceil_mode=False、count_include_pad=True、divisor_override 为 None |
| `_adaptive_avg_pool2d` | adaptive_avgpool |
| `mean.dim` | mean |
| `sum.dim_IntList` | sum |
| `cat` | concat |
| `split.Tensor`、`split_with_sizes` | split，配合 getitem 使用 |
| `_softmax(x, dim, half_to_float)` | softmax。要求 half_to_float=False |
| `native_layer_norm(x, shape, w, b, eps)` | layernorm。只使用 `getitem[0]` |
| `clone`、`alias`、`detach`、`contiguous`、`lift_fresh_copy` | identity |
| `max_pool2d_with_indices` | 不支持，原因写 "max pooling has no polynomial lowering in v1" |
| 其他 | 不支持，原因写 "aten op not in mapping table" |

5. 最后调用 `decompose.run(graph)`。其中已有 identity 删除、adaptive 转普通池化、global_avgpool_flatten 合并这几条规则；ResNet20-SiLU 的 `avg_pool2d + view` 会由最后一条合并成一个 mean。

## 3. 测试（test_r8a2_aten.py，nixrun）

1. **R3 的 34 个单算子模块**：直接复用 test_r3_ops.py 中的模块定义。如果现在是写在测试函数内部，把它们抽成模块级函数 `r3_cases()`，fx 和 aten 两边共用。每个模块都用 aten 前端导入，`reference_forward` 与 torch float64 的差 ≤1e-9。这里不断言 op 序列，因为 ATen 层的分解方式不同。例外：maxpool 和 adaptive(3) 这两个用例要按"不支持"或 needs_composition 来判断，判定结果写在测试里。
2. **resnet20（B 线）和 resnet20_silu**（aten 前端）：
   - 都是 74 个节点，op 计数为 {conv2d:19, batchnorm:19, act:19, add:9, slice:4, pad:2, mean:1, linear:1}；
   - 权重名与 state_dict 一致，例如存在 `layer1.0.conv1.weight`；
   - 与 torch float64 的差 ≤1e-9（取 2 张校准图）；
   - 用 aten 图生成的 `rule.plan(…, deg=127, margin=1.5)`，与 R6c 的 program.ptdsl 解析后的 `canonical()` 相等。变量名可以不同，但结构、属性和权重名必须一致。
3. **原先失败的写法**：S1、S2、S3（来自 R8a 的 size/shape 用例）、"依赖形状的 if 加通道重排加 hardswish"，以及 `conv → bn → relu → flatten → linear`，用 aten 前端导入后，reference 都与 torch 一致。
4. **参数命名**：`nn.Sequential(Conv2d, BatchNorm2d, ReLU, Flatten, Linear)` 导入后，权重名中包含 `0.weight`、`1.running_mean`、`4.weight`、`4.bias`。
5. **不支持的情况一次列全**：一个包含 MaxPool2d 和 ConvTranspose2d 的模块，抛出的 IntakeError 中同时列出两项。
6. R1–R8a 的测试全部通过（fx 前端行为不变）。

## 4. 然后

按 R8a spec §7 完整运行 `r8a_dev`，zoo 模型全部改用 aten 前端。SiLU 的归一化已经选定为 ImageNet（训练集验证子集准确率 100%）。之后训练两个小模型，并对 4 个模型跑规则基线。汇报内容同 R8a §7/§9，并在 REPORT 中注明 `frontend: "aten"`。STATUS.md 中 R8a 改为完成，新增 R8a2 一行，删除"阻塞"栏中的对应条目。

## 5. 不要做

- 不针对某个模型或某种写法加特判；遇到映射表里没有的 aten 算子，按 §2 报不支持。
- 不把 fx 前端的默认值改成 aten，也不删除 fx 前端。
- 不安装或升级 torch，不用 `torch._dynamo`。

## 6. 口径修订（2026-10-21，回应 codex 的核对报告）

### 6.1 §3.2 的范围
- 与 R6c program.ptdsl 做 canonical 对照，**只针对 resnet20**（ReLU）。
- resnet20_silu 单独验收（aten 前端），不与 ReLU 程序比较属性：
  1. 共 74 个节点；op 计数与 resnet20 的 aten 图相同；所有 act 节点的 fn == "silu"；
  2. 图中每个权重名都在该模块的 state_dict 中，形状一致；
  3. 用 zm.normalization 取 2 张校准图，reference 与 torch float64 的差 ≤1e-9；
  4. 用这 2 张图做 calibrate，`rule.plan(graph, calib, deg=127, margin=1.5)` 能通过 parse 和 typecheck，程序中的 act 全部为 fn="silu"。

### 6.2 新增 §2.1b：检查依赖数据的控制流（通用，不针对模型）
make_fx 得到 gm 后，用固定种子 20261022 生成 3 个形状与示例输入相同、取值服从 N(0,1) 的 float64 输入，比较 gm 与模型 m 的输出。只要有一处 |gm−m| > 1e-9·(1+|m|)，就抛出 IntakeError，op 记为 "trace"，原因写 "data-dependent control flow: traced graph differs from module on other inputs"。
test_r8a2_aten.py 新增一项测试：`if x.sum() > 0: return x; return -x` 这个模块用 aten 前端导入时，必须抛出上述错误。

### 6.3 批量语义（适用于所有前端）
导入得到的图表示的是模型在**示例输入形状（batch=1）**下的语义；用 batch>1 计算 reference，只是我们的批量扩展。因此，torch 侧的对照值一律逐样本计算：
`expected = np.concatenate([model(x[i:i+1]) for i in range(n)])`。
对于不依赖 batch 大小的模型，结果与整批计算完全相同。

### 6.4 允许修改 tests/test_r8a_models.py，只做以下两处
1. `compare_reference` 改为 §6.3 的逐样本计算。
2. `test_size_shape_and_flatten` 改用 `frontend="aten"` 导入，其余断言不变（只有一个 `mean` 节点，axes=(2,3)，keepdim=False，输出即图输出）。4 个 style 的期望输出形状都是 **(2, 4)**。原因：style 2 和 3 用 `x.size(0)` 作为 flatten 的 start_dim；在 batch=2 时直接调用 torch，start_dim 会变成 2，得到 (2,4,1)，这是另一种形状下的语义，不是导入时的模型。
其余测试（SiLU 导入、小模型导入、规则精确、resnet20 原文回归、函数恢复、size 当张量用被拒绝）**继续用默认的 FX 前端，保持原样**，作为 FX 行为的回归测试。aten 前端对应的验收都在 test_r8a2_aten.py 里。

### 6.5 §3.6 改为
R1–R7c 的测试全部通过；test_r8a_models.py 按 §6.4 修改后全部通过；FX 前端源码（包括旧轮已有的 3 个函数包装）不改动、不扩展。这个包装属于遗留的点状修复，只是保留下来，不再在它上面继续打补丁。
