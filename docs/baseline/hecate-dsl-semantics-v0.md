# Hecate 语义契约 v0：已验证子集与尚未验证的边界

## 结论与证据范围

团队所说的 DSL，在当前路线中指 **Hecate Python frontend**，不是本仓库
另有一个叫“FHE DSL”的独立语言。下面的 `hecate-function-v0` 是本项目为
后续生成任务定义的**受限契约**，不是宣称上游已有这份 grammar。

Confirmed fact：固定 Dacapo commit
`4616402710f39df3e5f5bd7930a6c036025aaac3` 的真实 frontend/compiler/SEAL CPU
已完成 8 个人工案例、每例 4 组输入，共 32 次密态执行、112 个输出值比较。
`Linear(4,4) -> square -> Linear(4,2)` 已通过；尚无 Agent 生成成功率结果。

证据目录：`/home/lhy/poseidon-work/results/seal-cpu-golden-l7iu78ak`。
本轮不安装依赖、不修改 submodule，不表示 Poseidon GPU adapter 已兼容。

### 代码证据索引（路径相对于源码根）

| 层 | 入口与用途 |
|---|---|
| Python frontend | `third_party/dacapo/python/hecate/hecate/expr.py`：`func`、`Func.eval`、`resolveType`、`Plain`、`rotate`、`save` |
| Native frontend | `third_party/dacapo/tools/frontend.cpp`：构造 Earth function/op、导出常量 |
| Scale/level 管理 | `third_party/dacapo/lib/Dialect/Earth/Transforms/WaterlineRescaling.cpp`、`Common.cpp`、`IR/EarthDialect.cpp` |
| Earth→CKKS | `third_party/dacapo/lib/Conversion/EarthToCKKS/EarthToCKKS.cpp` |
| HEVM emission | `third_party/dacapo/lib/Dialect/CKKS/Transforms/RemoveLevel.cpp`、`EmitHEVM.cpp`；`include/hecate/Support/HEVMHeader.h` |
| SEAL 执行 | `third_party/dacapo/lib/Runtime/SEAL_HEVM.cpp` |
| 高层模型辅助函数 | `third_party/dacapo/python/poly/poly/Func.py`：`HE_Linear`、`HE_ConvBN` 等，不是本轮采用的实现 |
| 独立 reference / 人工 DSL | `scripts/baseline/golden_cases/*/model.py`、`fixtures.json`、`trace_golden.py` |
| 项目静态契约检查 | `scripts/baseline/hecate_contract.py`；**不执行代码，不代替密态验证，不是沙箱** |

## 1. 函数结构与 grammar

上游：`@hc.func("c")` 注册函数，`hc.save()` 调用 `Func.eval()`，将函数实际
执行于符号 `Expr` 上以构图。Python 的循环在 tracing 时执行，并不成为
密态控制流。`hc.save(output_dir, constant_dir)` 写 MLIR/CST。

项目 v0：模型只生成下面的**函数片段**，不是带任意 import/I/O 的独立脚本。
可信 harness 提供 `hc` 和只读公开常量，负责保存、编译与执行。
grammar 的空白、缩进、标识符由 Python parser 处理；类型约束见下一节。

```ebnf
program      = '@hc.func("c")', NEWLINE,
               'def golden(x):', NEWLINE, INDENT,
               { assignment, NEWLINE }, return_stmt, NEWLINE, DEDENT ;
assignment   = fresh_name, '=', expr ;
expr         = atom | '(', expr, ')'
             | expr, '+', expr | expr, '*', expr
             | expr, '.rotate(', step, ')' ;
atom         = 'x' | previous_cipher_name | approved_public_constant_name ;
step         = '1' | '2' ;
return_stmt  = 'return', expr
             | 'return [', expr, { ',', expr }, ']' ;
```

单双引号等价。`+`/`*` 使用 Python 正常优先级；需要改变顺序时加括号。
静态检查器还要求一个函数、一个参数、一次最终 return、SSA 风格的新变量名。
不允许变量重绑定、`+=`、函数注解/default 参数、任意方法/函数调用、循环、
条件、下标读取、列表推导式或新常量字面量。权重通过已批准的常量名引用。
人工 golden 中的固定循环由可信作者编写；Agent v0 必须输出展开的直线代码。

例如，公开 `w=[w0,w1,w2,w3]`、`bias=[b]` 由 harness 注入：

```python
@hc.func("c")
def golden(x):
    p = x * w
    pairs = p + p.rotate(1)
    total = pairs + pairs.rotate(2)
    return total + bias
```

这返回一个承载点积的 ciphertext；选 slot 0 由输出 manifest 规定。
返回 `[a,b]` 表示**两个 ciphertext**，不是一个 ciphertext 的两个 slot。

## 2. 类型、shape 与 broadcast

v0 的逻辑模型：CPU float64、eval、静态 shape `[4]`、公开固定权重、加密输入。
四组测试输入是四次独立加密执行，不是已经支持 batch shape `[4,4]`。

| 值 | 项目契约 | 责任边界 |
|---|---|---|
| 输入 `x` | 一个 ciphertext，四个实数按周期 4 重复 | harness 负责输入布局 |
| ciphertext 临时值 | 同一 CKKS 参数上下文，静态推断周期不超过 4 | 不能按 Python tensor 进行切片/reshape |
| 公开常量 | 有限实数 scalar、长度 1 或 4 的向量；v0 检查绝对值≤1024 | 来自固定模型/受信 manifest，不能由生成器修改 reference |
| ciphertext `+` / `*` | 左侧 ciphertext；右侧 ciphertext 或公开常量 | 只接受当前已验证的组合；公开-only 计算在 harness 中完成 |
| scalar broadcast | 长度 1 的常量重复到全部 slot | 不是任意 NumPy/PyTorch broadcasting |
| 点积/隐藏神经元 | 人工归约后所有 slot 相等，每个神经元一个 ciphertext | 静态检查器不尝试证明这种相等，只保守记为周期 4 |

Confirmed fact：`frontend.cpp::createFunc` 当前构造 `tensor<1x!earth.ci<0*0>>`，
并未把 PyTorch 的 shape `[4]` 放进函数类型，也没有依据 `inputTys` 构造真实
plaintext 输入。因此不要把 `@hc.func("p")` 当成已验证的明文参数接口。
原始 tensor shape、batch、layout、隐私边界必须另存 manifest。

Confirmed fact：`resolveType` 接受若干 Python/NumPy/Torch 类型，但这不是
“支持任意 tensor”的保证。`Plain` 传的是 `double*` 和 `len(data)`；本契约
只允许一维 float64 常量，拒绝空、ragged、多维、NaN/Inf 和隐式广播。
`Plain(data, scale=40)` 的 `scale` 参数在该实现中未传给 native frontend；
不能用它覆盖 compiler 的精度规划。

## 3. 算子矩阵

“通过”只表示下表测试配置通过，不代表所有 shape/参数/组合都正确。

| 语义 | Hecate / lowering | 本轮状态 | Agent v0 |
|---|---|---|---|
| ciphertext 加法 | `a+b` → Earth add → AddCC | add、Linear、MLP 通过 | 允许 |
| 加公开 bias | `a+bias` → AddCP | Linear、MLP 通过 | 允许 |
| 乘公开 scalar/vector | `a*w` → MulCP | mul_plain、Linear、MLP 通过 | 允许 |
| ciphertext 乘法/平方 | `a*b` / `a*a` → MulCC | square、quartic、MLP 通过 | 允许；不能自动保证深度可行 |
| rotation | `a.rotate(1/2)` → RotateC | 非对称输入独立验证 | 仅 1、2 |
| 四次方 | `s=a*a; s*s` | 编译出 2 次 MulCC、2 次 rescale，执行通过 | 用乘法表达，不允许 `**` |
| Linear(4,2) | 显式点积、rotation、加法、bias | 真正跨元素归约通过 | 组合表达，不是新 runtime opcode |
| Linear(4,4)-square-Linear(4,2) | 4 个隐藏 ciphertext → 平方 → 2 个输出 | 通过 | 组合表达，固定权重/布局 |
| Negate/Sub | `-a`、`a-b` 的 frontend 构造已由代码确认 | 未做独立密态语义覆盖 | 暂不允许 |
| 负 rotation/其他步数 | frontend/native 存在整数 offset 路径 | 本轮未验证、未配相应 key | 暂不允许 |
| reshape/flatten/batch/Conv/pooling | 需 layout-aware 转换或高层辅助函数 | 未验证 | 拒绝，不自动改写 |
| ReLU/SiLU | 上游辅助函数可能采用多项式/调用 bootstrap | 非原模型精确等价，未纳入 | 不自动近似 |
| bootstrap / upscale | 上游有接口或 opcode | 本 SEAL gate 硬拒绝 | 禁止 |
| 数据相关分支、训练、随机 forward | Python 可能执行不等于密态语义成立 | 不支持 | 禁止 |

`HE_Linear` 在 `python/poly/poly/Func.py` 调用 `MPCB.Linear(..., 2**16)` 并
带有额外缩放假设。本机 SEAL 是 16384 slots；不能因为辅助函数名称匹配
就直接宣称兼容。当前 golden 采用显式低层组合，不调用该辅助函数。

## 4. 编码、packing、rotation、权重

SEAL runtime 的输入和 CST 都执行 `slot[i] = src[i % len(src)]`，长度 1
为标量广播，长度 4 为四元素周期。N=32768 提供 16384 complex CKKS slots；
本轮仅验证实数编码/解码，没有宣称 complex 输入支持。

Confirmed fact（执行）：正 step=1 对 `[0.5,-1,0.25,-0.75]` 的前四 slot
产生近似 `[-1,0.25,-0.75,0.5]`；正 step=2 对应 `[0.25,-0.75,0.5,-1]`。
因此本布局下 `rotate(s)[i] = x[(i+s) mod 16384]` 的左旋方向与测试一致。
四元素循环是输入周期性造成的，不能把整个旋转域误写成永远只有 4 slots。
非周期全 slot 输入、其他步数仍需测试；完整归约的最终值本身不能鉴别方向。

Linear 行：先乘四元素权重，再 rotate-and-add 1、2，得到四项 dot product，
加 bias。MLP 的第一层输出四个 broadcast ciphertext，各自在密文中平方；
第二层用 scalar weights 加权四个 ciphertext，不做中间解密或 plaintext 回填。
最终两输出均读取各自 slot 0，并与原始 PyTorch reference 逐项比较。

CST 是常量向量表；HEVM 的 Encode 引用表索引。检查器拒绝非法索引、空值、
非有限值和不安全的 eager-encode 目的寄存器复用。常量的原模型名称/shape
不由 CST 保留，manifest 必须记录其来源、数组 hash 与布局。

## 5. Scale、level、modswitch、rescale 与密钥

不改变安全参数：SEAL CKKS degree=32768、14 个 60-bit 模数；完整 key context
包含 14 个模数，首个 data context 包含 13 个。创建时显式通过 tc128 检查。
编译使用固定 `profiled_SEAL_CPU.json`、EVA、`--waterline=40`。

这里 `s=log2(scale)` 是编码精度缩放的指数，不是安全强度；`q` 是 rescale
丢弃的那个实际素数。乘法使 scale 相乘，即 `s_out=s_a+s_b`；rescale 将
scale 除以 q，近似 `s_out=s_in-60`，同时消耗一个 data 模数。
modswitch 丢弃指定数量的模数但不改变 scale。不能混用这两个操作。

| 表示 | level 含义/观测 |
|---|---|
| 管理后的 Earth | 消耗量递增；quartic 中 0→10→11→12 |
| CKKS/HEVM | 剩余 data 模数数量；同一 quartic 为 13→3→2→1 |
| SEAL | chain_index 是剩余 data 模数数减 1；首个 data index=12 |
| Poseidon adapter | 另有上下文/level 约定；不得直接套用 SEAL 的整数 |

quartic 的真实 artifact 为：modswitch(10)，square，square，rescale，rescale。
Earth scale 指数走 40→80→160→100→40；运行时最终观测
`log2(scale)=40.000000000060275`、剩余模数=1、ciphertext 多项式数=2。
微小指数偏差来自实际 q 不等于 2^60；代码检查误差≤1e-6，不修改实际 scale。
该元数据检查门限不是明文输出差分门限，输出门限始终不变。

Confirmed fact：SEAL_HEVM 的 MulCC 调用 multiply 后立即 relinearize；本轮
平方、quartic 和 MLP 输出观测为 2-polynomial ciphertext。relin key 必须存在。
rotation 1、2 使用实际 Galois keys；public/secret key 负责输入加密/最终解密。
key setup 只减少未使用 rotation key 的生成，没有缩短 Q/P 链或降低安全检查。

风险：stock AddCC/AddCP 会将 lhs 的 scale 元数据改为 rhs/plain 的 scale；
不能据此认为任意不匹配的 scale 都能相加而不改变数值。当前只执行 compiler
输出并检查最终元数据/数值，复杂分支仍需新测试。

frontend 没有可依赖的显式 `x.rescale()` API；由 compiler 插入 rescale、
modswitch、常量 encoding level/scale、buffer reuse。Agent 不能自行改变
profile、waterline、容差或安全参数以使样例通过。

## 6. Bootstrap 与控制流的硬边界

上游 `hc.bootstrap` 可产生 Earth bootstrap；这只证明语法/IR 存在。
SEAL_HEVM 的 bootstrap 实现含 assert 后的 decrypt/decode/reencode/encrypt。
release 构建可能去掉 assert，因此**opcode 10 在加载 native runtime 前被
整段扫描拒绝**，不允许用此路径冒充真实 FHE bootstrap。opcode 5 和未知
opcode 同样拒绝；本轮 8 个程序没有执行这些 opcode。

同理，Python `if Expr` 可能只测试对象真值，不能解释为加密条件分支。
禁止数据相关控制流、in-place 操作、任意访问环境、网络、文件或进程。
受限 AST 检查不等于权限隔离；执行未来 Agent 代码仍需 OS 级能力隔离。

## 7. 每层保留/丢失的信息，以及谁负责

| 转换 | 保留 | 丢失/需另存 |
|---|---|---|
| PyTorch→Hecate | 显式算术依赖、权重数值、人工表达的 packing | 原 shape/batch、模块名、隐私边界、算子意图不是自动保留 |
| Hecate→Earth | 符号 dataflow、cipher/plain 运算、rotation、constant 引用 | Python 循环/辅助函数结构被展开；原 tensor shape 不能从 tensor<1xci> 恢复 |
| Earth→CKKS | 管理后的依赖、剩余 level、encoding、CC/CP 区分 | 高层 Linear/MLP 意图、部分 scale 信息只保留在函数 metadata |
| CKKS→HEVM/CST | opcode、buffer 索引、I/O level/scale、常量值 | SSA 名称、逻辑 shape/layout/常量来源；RemoveLevel 擦去内部类型 level |
| HEVM→SEAL runtime | 实际密文、modulus、scale、evaluation key 操作 | 无原 PyTorch 模型信息，必须靠外部 reference/输出选择器解释结果 |
| HEVM→Poseidon schedule | 已支持 opcode 的导入/静态调度已有代码 | Modswitch 等兼容性仍未打通；本轮没有执行此后端 |

Agent/前端必须表达：运算图、公开权重引用、支持的改写、逻辑 I/O、packing 与
输出 slot。compiler 负责 FHE 参数规划/必要的低层操作；harness 固定安全配置、
生成密钥、执行、解密和比较；独立 validator 拒绝非法结构与产物。
任何一层“声明正确”都不能取代后续实际测试。

## 8. 当前验证结果与下一步

新增案例最大绝对误差：rotation 1 为 2.9948464e-6，rotation 2 为 4.1102908e-6，
square 为 1.2376522e-8，quartic 为 2.2704281e-8，MLP 为 5.4681472e-8。
全部满足 `|actual-reference| <= 1e-5 + 1e-4*|reference|`。rotation 的误差
比算术样例大，应保留观察，不能用 MLP 的较小误差代表所有 operator。

完整回归 72/72 通过，无跳过：包括此前的编译/环境/产物测试、本轮实际
密态结果与元数据检查，以及 9 项静态 grammar/type/layout 拒绝测试。

Confirmed：当前具体案例的编译/执行/差分与 metadata 检查通过。
Evidence-based inference：该小子集可作为规则转换器与 Agent 的共同目标。
Unconfirmed：新 shape/model family、跨 key 随机性稳定性、完整 48 例泛化、
Agent repair 收益、任意 Hecate 程序正确性，以及 Poseidon GPU 功能/性能。

下一步：把受限 PyTorch FX 检查、固定 I/O/layout manifest、此静态契约、
编译/执行驱动连接成可批量诊断接口；保留相同输入/后端的 deterministic
translator 对照。模型服务、凭据、版本与费用上限确认前不发外部 LLM 请求。
首批 48 例应逐语义扩展，而不是将这里的 8 个程序误称为 8 类模型全覆盖。

## 本轮本地文件与 Git 边界

- `scripts/baseline/golden_cases/`：新增 `mlp4x4x2/model.py`、`rotate1/model.py`、
  `rotate2/model.py`、`square/model.py`、`quartic/model.py`；扩展 `fixtures.json`
  和 `trace_golden.py`，保留已有输入、权重与容差数值。
- `scripts/baseline/python_compiler_smoke.py`：仅扩展共享 reference 准备函数；
  原 Poseidon diagnostic 三例选择保持不变。
- `scripts/baseline/seal_cpu_golden.py`、`test_seal_cpu_golden.py`：扩展案例、
  必须实际出现的 opcode coverage、运行时只读元数据与证据回归。
- `scripts/baseline/seal_keys/CMakeLists.txt`、新增 `metadata.cpp`：只读 SEAL
  ciphertext metadata observer；不改 key 参数或原 evaluator。
- 新增 `scripts/baseline/hecate_contract.py`、`test_hecate_contract.py`：函数片段
  静态白名单检查及其测试，不执行生成代码。
- 文档：新增本文件，更新 `compiler-chain-status.md`、`seal-cpu-golden.md`。

以上 17 个本轮文件位于现有未跟踪的 scripts/docs 下，尚未 stage/commit。
原有 5 项 tracked 文件修改未动，分支/HEAD、sparse checkout、Zone.Identifier
的 skip-worktree 状态保持；Dacapo working tree 干净。tracked diff check 通过
（仅原有 CMake CRLF 提示）。未安装、未 sudo、未下载、未 commit/push/PR。
以后经批准可拆为“MLP/算子 golden 与 runtime 观测”和“DSL 契约/检查器/文档”
两组本地提交；不从测试通过推断获得提交授权。
