# Poseidon / Dacapo / Agent：当前系统的分层说明

日期：2026-09-07。本文依据当前 checkout 的源码与已保存实验，不把计划当成实现。
源码路径保留空格：`D:\Code Space\Poseidon`，WSL 为 `/mnt/d/Code Space/Poseidon`。

## 1. 先建立正确的整体认识

目标是让程序生成 Agent 根据公开的模型计算结构，编写 Hecate 同态程序，
经过现有编译器，在加密数据上计算，最后由持有密钥的本地验证端解密并比较。

必须区分两条执行路径：

- **当前已验证路径**：受限模型描述 → 本地 PyTorch/FX → LLM → Hecate →
  Earth/CKKS 编译 → HEVM/CST → Dacapo 现有 SEAL CPU runtime → 解密差分。
- **原定 Poseidon GPU 目标路径**：相同编译产物 → Poseidon Dacapo adapter →
  静态执行计划 → Poseidon GPU evaluator。它还存在指令和参数语义对齐门禁，
  不能把第一条路径的成功算成第二条成功。

本轮重新运行的 48 个案例属于第一条，最终48/48通过，首次45/48通过，证据审计通过。
详见 [完整Flash批次报告](deepseek-flash-48-case-results-2026-09-07.md)。它使用真实 CKKS 密文，不是明文模拟；
但仍不是 Poseidon GPU 端到端验证，也不包含 bootstrap。

## 2. FHE 是什么，解决什么问题

FHE（Fully Homomorphic Encryption，全同态加密）解决的是：计算方不必先拿到
输入明文，也能对密文实施特定计算，数据拥有者解密后得到对应结果。

令 `x` 是输入，`f` 是模型，`Enc` 是加密，`Eval` 是密态求值，`Dec` 是解密。
本项目使用近似数值方案 CKKS，因此追求的是：

`Dec(Eval(f, Enc(x))) ≈ f(x)`。

左边是“加密后计算再解密”，右边是普通 PyTorch 计算。约等号非常重要：
CKKS 的编码、加密和缩放操作带来数值误差，不要求逐 bit 相同。
这不是把明文运算包一层加密接口；底层要在模多项式环中运算，维护密文、
模数链和误差预算。[CKKS 原始论文](https://eprint.iacr.org/2016/421)

数据通路如下：

1. **Encode**：把实数向量映射为 CKKS 明文多项式，包含 scale 与槽位布局。
2. **Encrypt**：用密钥生成带随机性的密文。公开权重通常只编码，不必加密。
3. **Evaluate**：执行密文加法、乘法、槽位旋转等，不读取输入明文。
4. **Decrypt**：只有持有 secret key 的一端恢复 CKKS 明文。
5. **Decode**：把多项式转换回近似实数向量。

API 服务商在这里负责“生成程序”，不负责对用户密文执行神经网络，也没有收到
解密密钥。当前隐私边界是公开模型/权重、加密输入；模型权重保密不是当前目标。
本地测试把客户端和求值端部署在同一开发机，属于验证设施，不是已完成的生产
客户端/服务端隔离部署。

### 同态操作为什么比普通张量计算复杂

| 概念 | 解决的问题 | 在本项目中的含义 |
|---|---|---|
| Packing / slots | 一个密文里并行装多个数 | CKKS 为多个槽位执行 SIMD 式运算；不是一个密文只能放一个标量 |
| Rotation | 普通逐槽运算不能直接跨槽求和 | 循环移动槽位，需要对应 Galois/rotation key |
| Scale | 用近似整数表示小数 | 本轮输入 scale 为 `2^40`；两数相乘后 scale 也相乘 |
| Level / 模数链 | 随深度消耗的运算资源 | 必须根据具体 runtime 的层编号和模数定义解释，不能只比较一个整数 |
| Rescale | 降低乘法后过大的 scale | 同时降低模数，带舍入误差；不是纯元数据改写 |
| Modswitch | 让分支处于兼容模数层 | 与 rescale 目的不同；具体实现必须按 opcode 语义对齐 |
| Relinearization | 密文乘法会增大密文表示 | 用 relin key 将结果降回较小表示；不是解密 |
| Bootstrap | 可用深度耗尽后刷新密文 | 必须是真正同态刷新，不能解密再加密冒充 |

例如两个 scale 约为 `Δ` 的密文相乘，结果 scale 约为 `Δ²`。
若 rescale 丢弃的模数因子为 `q`，新 scale 约为 `Δ²/q`。
`Δ` 是小数放大倍数，`q` 来自实际参数链，不应猜它一定等于 `Δ`。
编译器据此安排缩放/层对齐，而 runtime 执行实际多项式操作。

ReLU、比较、数据相关分支不是当前 CKKS 算术路径的原生低成本操作。
如果未来用多项式代替 ReLU，那首先改变了明文模型，必须把“近似模型误差”与
“CKKS 执行误差”分开。当前测试使用显式多项式，不偷偷替换激活函数。

## 3. 原有项目由哪些层构成

### Poseidon：密态执行库和设备后端

`src/poseidon` 是 C++ 实现主体。典型应用是人工编写 C++，直接构造参数、密钥、
encoder/encryptor/evaluator/decryptor 并调用同态操作。当前开发分支在此基础上
还有 compiler adapter、GPU 与 multi-GPU 静态调度模块，不等于所有分支都有这些内容。

| 位置 | 输入 / 输出及职责 |
|---|---|
| `parameters_literal.*`, `poseidon_context.*`, `crt_context.*` | 参数 → 模数链、环及运算上下文 |
| `plaintext.*`, `ciphertext.*`, `rns_poly.*` | 明文/密文与 RNS 多项式表示 |
| `ckks_encoder.*` | 实数/复数槽向量 ↔ CKKS 明文 |
| `keygenerator.*`, `key/` | 创建公钥、私钥及评估所需密钥材料 |
| `encryptor.*`, `decryptor.*` | 明密文转换；不属于 Agent 生成任务 |
| `factory/`, `evaluator/` | 上下文和 evaluator 构造、方案运算接口及软件实现 |
| `gpu/` | GPU 密文/明文、内存、参数上传、NTT、key switching、rescale 等 |
| `frontends/dacapo/` | 读取/检查 HEVM/CST，将编译结果接入 Poseidon 执行体系 |
| `mgpu/` | ciphertext 级静态图、计划、调度与执行；不是完整神经网络 frontend |
| `tests/`, `examples/`, `tools/` | 单元测试、直接 API 应用示例、工具入口 |

GPU 算术相关文件包括 `gpu_evaluator`、`gpu_parameter`、`gpu_ntt`、
`gpu_keyswitch`、`gpu_rescale`、`gpu_memory`。这些层承担模运算、NTT 变换、
密钥切换和设备数据移动，不理解“这是一个 PyTorch Linear 层”。

multi-GPU 的静态计划还要显式描述设备位置、通信与依赖；本机只有一张 GPU，
可以检查部分 CPU-side 计划，却不能得到真实多卡通信或性能结论。

### Dacapo / Hecate：把程序编译成同态操作

固定 submodule 为 `third_party/dacapo`，commit
`4616402710f39df3e5f5bd7930a6c036025aaac3`。
当前隔离依赖使用 LLVM/MLIR 18.1.2 与 SEAL 4.0.0。

Dacapo 仓库中承载 Hecate frontend、MLIR dialect/pass、HEVM emitter 和 runtime。
不要把“用了 Dacapo 仓库”理解成“本轮执行了所有 DaCapo 优化算法”。
当前候选编译命令使用 `--eva --waterline=40`，不是 bootstrap-placement 实验。

| 位置 / 表示 | 解决什么问题 | 下游 |
|---|---|---|
| `python/hecate/hecate/expr.py` | Python 操作符重载与符号表达式 | native frontend |
| `tools/frontend.cpp` | Python/ctypes 创建操作与常量 | Earth MLIR |
| `lib/Dialect/Earth` | 较高层同态计算表示及变换 | Earth → CKKS lowering |
| `lib/Conversion` | dialect 之间的转换 | 更具体的 CKKS IR |
| `lib/Dialect/CKKS` | scale/level 等更接近 runtime 的操作表示 | HEVM emission |
| `lib/Dialect/CKKS/Transforms/EmitHEVM.cpp` | 输出寄存器式指令 | HEVM + CST |
| `lib/Runtime/SEAL_HEVM.cpp` | 解释 HEVM 并调用 SEAL | 密态执行结果 |

源码中存在 rescale、modswitch、bootstrap placement、成本估计等多种 pass。
“存在某 pass”与“当前 pipeline 调用了它”不同；实际执行应以编译命令和 IR/log 为准。

### Poseidon adapter：桥接两套软件语义

位于 `src/poseidon/frontends/dacapo/`：

- `dacapo_artifacts`、`dacapo_constants` 读程序和常量；
- `hevm_artifact_report` 提供指令、寄存器和兼容性检查；
- `hevm_plaintext_encoding` 处理编译常量的槽位表示；
- `hevm_io_binding` 绑定输入输出；
- `dacapo_adapter` 翻译到 Poseidon 静态图/操作；
- `hevm_static_execution_plan` 建立执行计划；
- `poseidon_gpu_hevm_executor` 连接 GPU evaluator。

桥接不能只做 opcode 名字替换。已有检查发现的关键门禁包括 ModswitchC、
UpscaleC 的支持情况，MulCC 是否自带 relinearization，以及 compiler/SEAL
与 GPU 的 Q/P、RNS 位宽、逻辑 level 和物理 level 是否一致。
历史兼容性修复已将 HEVM 常量按周期重复编码，而保留直接 CKKS encoder 的
原有行为。详见 `hevm-compatibility-repair.md`；这些记录不是 GPU 端到端通过证据。

## 4. DSL 到底在哪里，为什么需要它

DSL 是面向某个领域的程序表达方式。这里的生成目标是 **Hecate Python frontend**，
不是凭空新增一种语言。开发者用 Python 外观描述同态表达式，符号执行生成 IR。
例如 `@hc.func("c")` 描述一个密文输入函数，`x * w` 建立符号乘法，
`x.rotate(1)` 建立旋转，`hc.save()` 触发构图和产物保存。

上游 Python 函数在 tracing 时运行，操作的是符号 Expr，而不是输入明文。
Python 层可以有比当前 Agent 合同更丰富的辅助逻辑，但这不意味着密文 runtime
支持任意 Python 控制流。上游高层 HE_Linear/HE_ConvBN 等 helper 也不能仅凭
名称就认定支持任意模型 shape。

FHE 与 DSL 的作用不同：**FHE 使密文计算成为可能，DSL 使该计算能被表达、检查
和编译**。DSL 自己不提供密码学安全；它是应用意图与密码学操作之间的接口。

### 当前 Agent 允许的是一个更小的 Hecate 子集

规则在 `hecate_contract.py` 与 `candidate_contract.py`，不是只写在 prompt 中：

- 固定 `@hc.func("c") def golden(x)`；
- 顺序局部赋值和最终密文返回；允许返回 1–4 个密文组成的列表；
- 密文加法、乘法以及 `.rotate(1)` / `.rotate(2)`；
- 只能引用登记的公开常量，不自己读文件或任意创建权重；
- 禁止 imports、任意函数调用、循环、分支、修改对象和任意索引；
- 源码、AST、操作数量都有界；scale、level、安全参数由受信 harness 控制。

返回 `[y0, y1]` 指的是两个密文，不是“一个密文中的两个槽”。
这种区分是模型语义映射里非常容易出错的地方。

### 一个真实跨元素 Linear 如何表示

设输入四个数为 `x0..x3`，某输出行权重为 `w0..w3`，偏置为 `b`。
所需结果是 `y = w0*x0 + w1*x1 + w2*x2 + w3*x3 + b`。
本轮将四元素向量在所有 CKKS 槽中周期重复；常量也用同样周期编码。
概念性 Hecate 片段如下，`w`、`b` 必须是 harness 注册的常量名：

```python
@hc.func("c")
def golden(x):
    p = x * w
    pairs = p + p.rotate(1)
    total = pairs + pairs.rotate(2)
    y = total + b
    return y
```

第一步只是逐槽乘法。第二步合并相邻两项，第三步合并两组二项和，因输入周期为4，
各槽最后都得到完整四项和。正旋转按当前验证规则为向左循环移动。
`Linear(4,2)` 为两行权重分别完成这个计算，返回两个密文并从约定槽位读出。
它没有用逐元素乘法冒充矩阵乘法。后续 MLP 的隐藏单元采用多个密文分别表示，
也是当前固定布局，不是优化后的最省内存 packing。

## 5. Agent 现在的输入、处理与输出

### 输入不是任意 PyTorch 文件

当前 CLI 读取 JSON 描述，例如：

```json
{"schema":1,"id":"linear-1","family":"linear","configuration":1}
```

该描述选择 `model_catalog.py` 中一个确定的 PyTorch 模型。公开权重由固定规则
生成，模型在 CPU float64/eval 模式下执行。它是可重复实验接口，不是训练模型。
当前尚未完成“任意用户 model.py / ONNX / checkpoint 导入”这一更大的接口。

catalog 有 affine、polynomial、linear、mlp2、mlp3、fanout、residual、
flatten_linear 八类，各六种配置。输入都只有四个逻辑元素；flatten 案例测试
二维到一维的 row-major 语义。深度、宽度、权重和图结构有变化，但不是大模型测试。

### 本地预处理和双路径

`fx_to_hecate.py` 接受受信的 PyTorch Module，用 Torch FX 提取静态图并检查
算子、shape、dtype 和公开状态。注意 FX tracing 会执行原 Python：它不是
不可信 model.py 的安全沙箱，目前不能以此接口直接运行陌生代码。

本地规则转换器有两个角色：

1. 生成 deterministic 对照程序，测量不用 LLM 是否也能完成转换；
2. 给 Agent 准备公开常量表、常量来源、固定 packing 和输出选择规则。

规则转换器的完整答案保留在本地，不作为 live prompt 的答案发送给模型。
但 Agent 仍获得了规则系统准备的布局和常量，因此当前属于**规则辅助的受限程序生成**，
不是 LLM 独立解决从原模型到所有低层决策的问题。

### 发给模型什么

`candidate_contract.make_request` 构造带 SHA256 request ID 的结构化请求，包含：

- 描述文件、FX 图、公开模型结构；
- 公开常量及其来源；
- 输入/输出 shape、slot period、output selectors；
- Hecate 语义规则、响应 schema、编译 profile 身份；
- 明确的隐私边界和禁止事项。

不发送隐藏测试输入、明文参考答案、加密/解密密钥或规则转换器答案。
这份“知识”通过每次请求提供，并没有微调 DeepSeek，也没有新训练一个模型。

### 模型返回什么

模型应返回严格 JSON：

```json
{
  "schema": 1,
  "request_id": "与请求一致的SHA256值",
  "hecate_source": "@hc.func(\"c\")\ndef golden(x):\n    ...\n"
}
```

模型只能生成函数片段；不能修改输入、权重、layout、reference、门限或安全参数。
输出不是密文结果，也不是编译好的 GPU 程序。后续本地工具负责验证和运行。

### 从候选到验证闭环

1. `deepseek_provider.py` / `deepseek_http_worker.py` 调用选定服务，限制响应体和超时。
2. `candidate_contract.py` 检查 schema、request ID 与源码边界。
3. `hecate_contract.py` 进行 AST 白名单、变量/类型和输出结构检查。
4. `candidate_trace.py` 解释允许 AST，调用真实 Hecate Expr 构图；**不 exec 模型源码**。
5. `candidate_sandbox.py` 用 bubblewrap 隔离 tracing/编译/执行，禁网、只读依赖，
   不给生成过程暴露整个项目目录和凭据。
6. `hecate-opt --eva --ckks-config=... --waterline=40 --verify-each` 编译。
7. `seal_artifact_gate.py` 检查二进制、指令/寄存器、常量和参数兼容性，拒绝
   bootstrap/upscale/未知指令，再进入 native runtime。
8. `candidate_worker.py` 经 `seal_cpu_golden.execute_artifact` 调用已有 SEAL HEVM
   runtime，真实加密输入、执行、解密并保存结果。
9. 受信比较端对独立 PyTorch reference 逐项比较；不把解密中间值回填 reference。
10. 候选失败可将受限诊断反馈给模型，最多修复三轮。基础设施/完整性失败停止，
    不暗中用规则答案代替生成失败。

这不是模型“自评正确”：决定通过的是本地编译、执行、数值和完整性检查。
沙箱和白名单降低风险，但不等于对内核/native依赖安全作形式化证明。

### 最终保存什么

每个实验在 `/home/lhy/poseidon-work/results/` 建独立目录，包含报告、公开请求、
每轮 `candidate.py`、tracing 证据、Earth IR、lowered CKKS IR、HEVM/CST、日志、
解密数组和差分指标。批量报告保存每个 case 的状态、调用次数与失败层。
精确文件名与哈希以该实验 report 为准。不要分享 `private-keys` 或整个结果目录。

失败不能混为一个 compile error：provider、JSON/AST、frontend、compiler、
artifact、runtime、numerical、integrity/infrastructure 分层记录，便于定位原因。

## 6. 每次 lowering 保留或失去什么

| 边界 | 保留 / 具体化 | 可能丢失或需旁路保存 |
|---|---|---|
| PyTorch → FX | 静态算子、参数引用、依赖 | 原 Python 写法、模块设计意图；数据相关控制不在支持范围 |
| FX → Hecate | 算术与明确 packing 的密态程序 | 原始 Linear/Residual 名称可能展开为很多操作 |
| Hecate → Earth | 符号运算图、常量、加密性表达 | Python 辅助逻辑已在 tracing 完成，不是 runtime 控制流 |
| Earth → CKKS | 更具体的 CKKS 操作、层和缩放处理 | 更高层应用含义减弱，需 metadata 保留来源 |
| CKKS → HEVM/CST | opcode、寄存器、常量和 I/O | tensor shape、模型节点、用户意图不能仅从指令恢复 |
| HEVM → runtime | 实际密文、评估密钥、模数与设备执行 | 必须从外部配置对齐 layout/参数；opcode 同名不足以证明同义 |

这解释了为什么 compiler feedback 和 metadata 很重要：如果未来 Agent 要选择
packing 或改算法，它需要知道哪些高层决策造成了 rotations、深度和内存消耗。
单独看最终 HEVM 很难恢复“为什么用了这种卷积布局”。

## 7. 如何理解 48 个案例的正确性结果

每个模型用零输入、有符号固定输入、固定种子随机输入、声明范围边界输入四组。
48 模型全部执行对应 192 组输入。本轮门限固定为逐项：

`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。

报告 MAE、最大绝对误差、非零 reference 的相对误差，并在适用时记录 cosine。
近零 reference 的相对误差会放大，所以通过条件使用 absolute + relative 联合门限。
这些是有限输入上的差分证据，不是对所有可能输入的语义等价证明。

当前密钥设置显式采用 N=32768、16384 slots、14 个 60-bit key-context 模数，
first data context 有13个模数，并要求 SEAL tc128 参数校验；只生成旋转1和2所需密钥。
没有为了跑通修改这些参数。测试不调用 bootstrap；上游 SEAL runtime 的 bootstrap
路径不能作为真实 FHE 刷新使用，artifact gate 会在执行前拒绝它。

参数满足 tc128 检查不代表整个旧软件栈可直接生产部署。SEAL 上游当前另有建议
升级至至少4.4.0的安全公告；这里为复现实验仍固定4.0.0，没有擅自升级依赖。
对外服务前需要单独评估升级与回归。[SEAL 官方仓库安全提示](https://github.com/microsoft/SEAL)

## 8. 接入 Agent 改变了什么，没有改变什么

**改变了**：人工编写 Hecate 函数现在可由模型 API 生成；本地闭环能自动检查、编译、
真实执行、差分、反馈修复；同一批固定案例能统计首次成功率、修复收益和失败层。
Provider 可换而不改 compiler/runtime。

**没有改变**：FHE 算法、Hecate lowering、SEAL HEVM 运算语义、Poseidon GPU 核心，
以及数值通过标准。没有用 LLM 在密文上推理，也没有“模型说正确就通过”。

**尚未完成**：任意用户模型导入、一般 shape/Conv/pooling/ReLU、多样 packing 搜索、
Poseidon GPU 全链路、完整 GPU ResNet 的功能/性能对照、bootstrap、multi-GPU 性能。
现有 fanout/residual 案例提供结构测试，但 prompt 含有其目标图，而且同一 catalog
被反复使用，不能仅以通过48例宣称广泛 unseen 泛化或已实现研究贡献。

确定性 translator 是必要对照：若任务始终在静态支持算子中一对一映射，它可能
更简单可靠。Agent 的潜在研究价值在未知结构、受约束重写、诊断修复和应用语义
决策，但这些需要额外对照实验，不是当前工程成功自动附带的结论。

## 9. 当前最合理的后续门禁

先用新的全48报告确认可复现成功率与失败来源；随后把 catalog 选择接口扩展为
受限用户模型接口，并引入真正独立的结构/权重/shape 留出集。Poseidon GPU 的
opcode与参数兼容是另一条明确工作线，必须先做最小真实执行再扩大模型。
性能与更强 DSL 表达能力仍排在正确性与范围扩展之后。

本说明取代旧 `compiler-chain-status.md` 中关于“Agent 尚未实现”的时点描述；
旧报告保留作为历史证据，不应被改写成当前实验结果。
