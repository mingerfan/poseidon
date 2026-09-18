# 完整项目 DSL 支持：新目标与第一轮代码证据

## 验收口径

新目标是完整支持项目 DSL 的所有语义，仍未完成。旧的受限 CPU 扩展阶段已完成，
不代表新目标完成。旧的 13 类图算子、29/32 语法分区和 126 个成功案例只作为回归
基线，不能重新定义为全部 Hecate 的分母。

固定源码版本：Poseidon 4995e7cadedf2bfb9104658b5638662ecf6a1d0a，Dacapo gitlink
4616402710f39df3e5f5bd7930a6c036025aaac3；已有本地修改保留。源码以当前 working tree
为准，不能将本地修改后的行为说成原始上游 commit 行为。

“全部”要覆盖值/类型/shape/layout、操作语义、函数/调用/公开控制流、编译器管理
语义和运行时要求；只收集操作名字不能证明完成。不包含授予生成程序任意文件、
网络、进程访问权限。公开 Python 循环是 tracing 时构图，不等于加密数据分支。

## 按实际代码分层的未完成项

| 层 | 源码依据 | 当前差距与下一门禁 |
|---|---|---|
| Python Expr | python/hecate/hecate/expr.py | 增强赋值绑定错误已修复；新 v5 AST / v6 请求已接入反向 +、-、* 和 =、+=、-=、*=，完成下述人工密态验证；未宣称所有 Expr 语义或线上 Agent 覆盖完成 |
| 常量/类型 | expr.py 的 resolveType、Plain | int/float/list/NumPy/Torch 路径、数组维度与长度、构造参数的实际效果需逐一规范；当前 scalar/period4 不是完整支持 |
| 容器与初始累加器 | expr.py 的 Empty、unaryFactory | Empty 的减法语义、bootstrap iterable 返回行为需要实际对照；不能猜它们等价于普通数值容器 |
| 函数与参数 | expr.py 的 func/Func.eval/Func.__call__；tools/frontend.cpp | 普通构图 helper 的作用域、组合、高阶调用和有界递归已完成第四轮人工密态验证；mixed IR 参数与多 decorated 函数调用仍待核验。Python 调用了 createCall，但当前 frontend.cpp 未找到该定义，不能宣称该 IR 调用链已支持 |
| Python 构图语法 | examples/benchmarks、poly/MPCB.py | 公开 for/range、局部列表/元组、索引、公开条件已通过第三轮受控展开及人工密态验证；辅助函数、推导式、数组元素计算等仍待实现；不允许执行任意生成 Python |
| 上层模型辅助库 | poly/Func.py、MPCB.py、Poly.py | BN、ConvBN、Linear、ReshapeLinear、DwConv、Concat、Pool、ReLU/SiLU 等真实 helper 的 packing、参数和近似策略未完整接通；本地同名图算子不是这些 helper 的证据 |
| Earth/CKKS | include/hecate/Dialect/{Earth,CKKS}/IR/*Ops.td | 原清单覆盖操作名，但还需逐项记录类型、属性、合法输入、scale/level、lowering 和错误行为 |
| HEVM/SEAL | lib/Runtime/SEAL_HEVM.cpp | 真实 bootstrap 缺失，upscale 未实现；不能仅修改 Agent allowlist |
| 其他运行时 | lib/Runtime/HEAAN_HEVM.cpp、lib/Runtime/CMakeLists.txt | 有 Bootstrapper 调用，但构建项被注释且调用 CUDA 同步；不能据源码存在就宣布可用 CPU 后端，需要依赖、接口与真实执行核验 |

以上路径均相对 third_party/dacapo。后续逐项增加实现与验证，不以拒绝本来属于 DSL
的能力来降低目标；必要依赖/后端选择保持显式门禁，不使用假执行顶替。

### 已确认的运行时硬缺口

SEAL_HEVM::bootstrap 的函数体包含 assert 后的 decrypt/decode/encode/encrypt。
assert 不是生产安全门禁，release 构建可能去掉它。现有 artifact gate 会提前拒绝
bootstrap，不能解除该检查并把解密重加密称为 FHE bootstrap。

SEAL_HEVM::upscale 只有不支持的断言。HEAAN_HEVM 也有未实现 upscale 的断言。
HEAAN 源码中的 Bootstrapper 调用还不足以证明完整执行路径可用；当前没有安装
新依赖、修改构建后端或再次开展 GPU 对接。

### 声明导出不等于实际功能

__init__.py 的 __all__ 声明 31 个名称；对 expr.py/runner.py 顶层定义、赋值与
动态 unary 工厂的静态扫描，24 个名称没有解析到定义，包括 Model、PlainMat、
sigmoid、sqrt、inverse、compile 等。这是静态“未解析”，尚不等于运行时证明不存在。
这类声明需要核对实际导入与文档，不能从陈旧 __all__ 直接发明 DSL 能力。

## 本轮实际修复：增强赋值的操作数顺序

原代码将 __iadd__/__isub__/__imul__ 全部绑定到 binaryReverseMethod。
这使 x -= y 构造 y-x；对加法/乘法虽然数学上交换，但构图顺序也被改变。

修复仅把增强赋值绑定到 binaryMethod。保持原有“返回新的 Expr、Python 变量重新
绑定、不修改其他别名”的行为，没有改成未使用的 binaryInplaceMethod（该实现
会修改 obj 且没有返回值）。普通 forward/reverse 方法未改变。

验证分为两层：

1. 记录型 C ABI 替身单元测试：修复前 6 项中 3 项失败；修复后 6 项全部通过。
   它测试真实上游 metaclass 定义的派发和别名，不是密态执行。
2. 真实 Hecate frontend + 现有 Dacapo 编译器：分别 trace/compile 普通算术和
   增强赋值共 6 个手写程序。修复前 6 个程序均能编译，但三对产物不相同；修复后
   add/subtract/multiply 三对 HEVM/CST 全部逐字节相同。保存 Earth IR、HEVM/CST、
   命令、日志与文件哈希，使用 --verify-each，没有降低参数。

证据目录（位于 /home/lhy/poseidon-work/results）：

- 修复前：frontend-augmented-b_yfzp9c/report.json。
- 修复后：frontend-augmented-f_bk3hf9/report.json。

**本轮没有新增 Agent 调用或真实密态执行。** 编译产物等价是这一层的证据，不能
单独描述为 Agent 端到端支持已完成。旧 hecate-function-v0..v4 仍拒绝 AugAssign；
后续需要版本化语法、受控派发、语义规则和密态正反例一起接入，不能直接放开 exec。

复现（WSL 源码根，不调用 API、不生成密钥）：

```bash
timeout -k 3s 30s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/baseline \
  python3 -m unittest test_frontend_augmented_ops -v
timeout -k 3s 340s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/probe_frontend_augmented.py
```

设置 POSEIDON_AUGMENTED_BEFORE / POSEIDON_AUGMENTED_AFTER 为上述目录可启用
真实编译产物复核；默认不把缺少外部报告的条件测试当成通过。

本轮最终回归：启用上述真实证据的 8 项测试全部通过，无跳过；全量离线
437 项中 333 通过、104 条件跳过。Dacapo diff --check 通过，gitlink 未变。
修改留在本地，没有 commit/push、安装、密钥生成或付费 API 调用。

## 后续顺序

1. 对固定上游构图语法和值语义建立比旧操作名表更完整的规范与最小复现。
2. 版本化接入 reverse/augmented 算术、容器、公开循环、函数组合；保持旧契约。
3. 泛化 shape/packing、常量与 plaintext 参数，并逐个验证上层 helper。
4. 为 bootstrap/upscale 的真实后端缺口确定实现路线；新增依赖或安装前请求批准。
5. 各层均有 reference、静态正反例、真实产物和密态执行后，再扩大线上 Agent 验证。

这不是缩小目标：所有尚未支持的语义保留为待办，而不是从最终验收范围删除。

## 第二轮：反向算术与增强赋值已接入受控生成执行链

新增 opt-in `--extended-arithmetic`：请求版本 `hecate-function-synthesis-v6`，
AST 契约 `hecate-function-v5`。旧 v0..v4 AST、旧请求 rules/guidance 和默认
版本选择不变；不要把请求 v5（encrypted-zero ABI）与 AST v5 混淆。

实际变化：

- 允许具名公开常量位于 +、-、* 左侧，至少一个操作数必须是 ciphertext。
- 允许密文局部变量/输入变量重新绑定与 +=、-=、*=；先读取原绑定计算右侧，
  再绑定结果。`a=x; x-=w` 不改变 a。常量和 zero_ct 的绑定仍只读。
- 受控 AST 解释器直接派发 Hecate Expr 操作，不执行生成 Python。
  在活动 frontend 函数内将常量转换为 Hecate Plain，避免公开数组左侧运算
  走 NumPy object ufunc。scalar/length1/length4 的既有广播规则保持不变。
- 请求中同步给出版本化语义规则；CLI 在 prepare/manual/replay/live 路径均传递版本。
- 覆盖统计用 `name@statement_index` 区分赋值版本，追踪旧别名依赖，不能用
  最后一次变量定义覆盖历史依赖。新增反向/增强赋值分区不修改旧 32 分区清单。
- 执行报告增加真实 Python frontend 文件哈希，并在执行前后检查未变化。

这不代表 shape/packing 已泛化、容器/公开循环/函数组合已接入，亦未解决
真实 bootstrap/upscale。它们继续属于完整目标待办。

### 真实密态证据（无付费 API）

命令（WSL 源码根）：

```bash
timeout -k 3s 2200s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_extended_arithmetic_goldens.py
```

结果：`/home/lhy/poseidon-work/results/extended-arithmetic-goldens-m9oqynnt/report.json`。
5 个正确程序通过，2 个故意错误程序在 numerical_comparison 阶段被拒绝，7/7
符合预期。每个程序包含 4 组输入、16 个输出值；共 28 组、112 个比较值。

| 人工程序 | 期望 | 最大绝对误差 |
|---|---|---:|
| 公开 length1 左加 | 通过 | 1.4815202398210658e-8 |
| 公开 length1 左乘 | 通过 | 6.094638294040777e-9 |
| 公开 length1 左减（有序） | 通过 | 2.323503434098484e-8 |
| 公开 length4 左减（有序） | 通过 | 1.2576927860274978e-8 |
| 旧别名、重新绑定与三种增强赋值链 | 通过 | 1.753400524773241e-8 |
| 故意颠倒减法方向 | 数值拒绝 | 3.000000000017306 |
| 故意使用更新后的 x 代替旧别名 | 数值拒绝 | 0.5000000055049032 |

真实路径仍是 Hecate → Dacapo → HEVM/CST → SEAL CPU，不是 Poseidon GPU。
SEAL 4.0.0，N=32768，14×60-bit modulus，tc128，既有 scale/水位配置不变，
atol=1e-5、rtol=1e-4 不变，无 bootstrap，无参数降级。
这批是人工 golden 验收，不是新增线上 Agent 成功案例。

`test_extended_arithmetic` 在指定结果环境中 12/12 通过、0 跳过。其中真实证据
测试重新计算独立公式与解密误差，校验不可变请求/模型/输入、HEVM/CST 哈希、
安全参数、AST 检查、实际密态执行标志和错误程序的拒绝层。
设置 `POSEIDON_EXTENDED_ARITHMETIC_REPORT` 为上述 report.json，在现有 pinned
Nix/Python 环境运行 `python -m unittest test_extended_arithmetic -v` 可复核。

7 个临时 private-keys 目录已由现有逐例清理逻辑删除，总计 4,752,860,070 字节
（约 4.43 GiB）；保留原始模型、固定输入、reference、生成源码、IR、产物、
解密输出和诊断。随机测试密钥可重新生成，不保留或上传密钥。

本轮全量离线回归：449 项中 344 通过、105 条件跳过，无失败；跳过不计通过。
`git diff --check` 通过（既有 CMakeLists.txt 的 CRLF 提示不是本轮错误）。
没有 commit/push、安装、依赖下载或新的付费 API 调用。

## 第三轮：公开构图语法

新增 `--public-construction`，使用请求 v7 / AST v6；在不执行生成 Python 的
前提下展开公开循环、局部容器、索引、公开条件、列表共享修改与解包。
展开后再次使用已验证的直线算术契约检查，再走真实 Dacapo/SEAL CPU。
完整语义、边界、可复现命令和证据见 [公开构图验证](public-construction-cpu.md)。

3 个正例（含真正跨 slot 归约的循环式 Linear(4,2)）通过，2 个错误程序被
数值比较拒绝；共 20 组输入、64 个输出值。最大正例误差 3.383e-8。
真实证据复核 18/18 通过；全量离线 467 项中 361 通过、106 条件跳过。
此批没有 Agent API 调用，不能计入线上 Agent 生成成功率。

完整目标仍未完成：下一步接入函数组合与更多公开数据构造语义；shape/packing、
高层 helper、真实 bootstrap/upscale 等缺口仍保留在最终验收范围。

## 第四轮：构图函数的作用域、组合与返回

新增 `--function-composition`，请求 v8 / AST v7，保持旧版公开构图引擎不变。
新引擎在独立调用帧中解释普通 helper 的 AST，支持参数绑定、局部隔离、共享
列表副作用、组合/高阶调用、公开有界递归和提前返回；不执行生成的 Python，
不冒充 Dacapo 多 IR 函数调用。详见 [函数组合 CPU 验证](function-composition-cpu.md)。

3 个正例与 3 个针对性反例最终符合预期，共 24 组输入、80 个输出值；
最大正例误差 2.553e-8。首次四次多项式人工程序遗漏系数/偏置，被数值比较
正确拒绝；保留失败报告，仅修改人工程序并补跑该正反例，未更改 reference
或门限。加入独立明文公式 preflight，避免类似遗漏进入后续密态运行。

新旧真实证据复核 48/48 通过，无跳过；全量离线 485 项中 378 通过、107 条件
跳过。没有新增 Agent API 调用。这不完成全部 DSL：闭包/嵌套定义、公开数组
计算与推导式、一般 packing、完整上游 helper、IR 调用及真实 bootstrap/upscale
仍待实现或验证。

## 第五轮：嵌套函数、闭包与 nonlocal

新增 `--closures`（请求 v9 / AST v8），使用共享词法 frame，而不是定义时复制
外层值。支持返回后仍存活的闭包、不同 factory 调用隔离、兄弟闭包共享 cell、
循环变量 late binding、最近外层 nonlocal 写回、未初始化变量拒绝及嵌套递归。
旧契约不自动放开新语法；历史报告保留实际生成器哈希，并复核旧模式展开一致性。

本轮真实 Dacapo/SEAL CPU：3 个正例通过、3 个错误程序在数值比较拒绝，
24 组输入 / 80 个比较值；最大正例误差 1.538e-8 以下。新旧真实证据复核
65/65 通过，无跳过；全量离线 502 项中 394 通过、108 条件跳过，无失败。
详见 [闭包 CPU 验证与边界](closure-construction-cpu.md)。没有新增 Agent API 调用。

完整目标仍未完成。下一步补齐参数调用绑定（默认参数、关键字参数等）及更多
公开数组/构造语义，继续推进一般 packing、完整高层 helper、IR 调用及真实
bootstrap/upscale；不能把 Python 构图语法扩展当成这些同态后端缺口已经解决。

## 第六轮：默认参数、关键字与可变参数绑定

新增 `--call-binding`（请求 v10 / AST v9），支持 helper 的 positional-only、
keyword-only、默认值、关键字参数、*args/**kwargs 和有界转发。默认值在定义时
求值并保留引用，区别于调用时读取的闭包 cell；新参数种类不改变 golden 的加密
输入 ABI。公开 string-keyed dict 的基本构造/索引/修改支持不等于一般数组支持。

192 组参数组合与独立 Python 原生函数调用对照一致。人工实际密态最终选定
4 个正例通过、4 个反例数值拒绝，32 组输入、112 个输出值，最大正例误差
2.255e-8 以下。首次 wrong_snapshot 意外构造 -x+x，SEAL 以透明密文异常终止；
保留失败报告，调整人工正反例中的外层重新绑定后只补跑这一对，没有修改 reference
或关闭运行时检查。因此不是首次 8/8 通过。新旧真实证据审计 84/84 通过。
全量离线 521 项中 412 通过、109 条件跳过，无失败；跳过不计通过。

详见 [参数绑定 CPU 验证](call-binding-cpu.md)。无新增 Agent API 调用，不能
把人工 goldens 算作线上生成成功率。一般公开数组/数值表达、更多控制语法、完整
packing 与上游 helper、IR calls、真实 bootstrap/upscale 继续属于完整目标待办。

## 第七轮：推导式与惰性公开迭代

依据上游 MPCB.py 实际使用的构图语法，新增 `--public-iteration`（请求 v11 /
AST v10）：列表/字符串键字典推导、嵌套 for/filter、独立隐式作用域，以及
enumerate/zip/reversed/iter/next/list/tuple。迭代器保留共享消费状态和允许的
容器修改行为，不提前复制成列表；每次消费受既有步数门限约束。

人工真实 Dacapo/SEAL CPU 首次 6/6 符合预期：3 个正例通过，3 个错误程序
数值拒绝，24 组输入 / 80 个比较值，最大正例误差 1.965e-8 以下。
新旧六组语义连同实际产物审计 105/105 通过，无跳过。
全量离线回归 542 项，432 通过、110 条件跳过、无失败。详见
[公开迭代 CPU 验证](public-iteration-cpu.md)。未调用付费 API，不能计作新线上
Agent 成功案例，也不说明完整 GenPoly 或其他高层 helper 已经接通。

公开数值/数组表达、lambda/更多控制语法、一般 packing、完整上游 helper、
IR 调用和真实 bootstrap/upscale 仍属于完整目标，尚未完成。

## 第八轮：lambda 与公开键稳定排序

依据 Poly.py 返回 lambda、MPCB.py 使用 sorted(key=lambda...) 的实际代码，
新增 `--function-literals`（请求 v12 / AST v11）。支持匿名函数的词法捕获、
默认参数、返回/容器选择后的函数调用，以及仅基于公开键的稳定排序。排序回调
每元素调用一次，相同键在 reverse=True 时仍保持原序；回调结束后重新检查共享
键容器，禁止密文或 opaque Plain 进入比较。

人工真实 Dacapo/SEAL CPU 首次 6/6 符合预期：3 个正例通过、3 个反例数值拒绝，
24 组输入 / 80 个输出比较，最大正例误差 1.514e-8 以下。新旧七组真实证据审计
122/122 通过，无跳过。详见 [匿名函数与排序 CPU 验证](function-literals-cpu.md)。
未调用付费 API，不计作新线上 Agent 成功案例，不代表完整 GenPoly/genRelu6 已实现。

公开数值/数组表达、剩余控制语法、一般 packing、完整上游 helper、IR 调用、
真实 bootstrap/upscale 与新增语法的线上 Agent 验收继续保留在完整目标中。

## 第九轮：公开序列与切片语义

新增 `--public-sequences`（请求 v13 / AST v12），支持公开 list/tuple/string
的索引、切片、拼接、重复，以及列表切片赋值和命名列表的原地 +=/*=。正确保留
浅拷贝与嵌套共享引用，区分普通表达式新建列表和原地修改；负步长赋值使用原始
切片边界，不能将归一化的 -1 再次解释为相对索引。所有操作保留既有资源界限，
禁止密文索引、符号常量切片和循环容器。上游证据与边界见
[公开序列 CPU 验证](public-sequences-cpu.md)。

216 组切片读取和 180 组赋值边界与原生 Python 对照。实际 Dacapo/SEAL CPU
首次 6/6 符合预期：3 个正例通过、3 个错误程序数值拒绝，共 24 组输入 / 80
个输出值，最大正例误差 2.961e-8 以下。新旧八组语义及真实证据审计 136/136
通过，无跳过。临时密钥清理约 3.79 GiB，保留完整实验产物。

全量离线回归 573 项：461 通过、112 条件跳过、无失败。跳过项不算覆盖通过。

本轮没有付费 API 调用，因此不计为新增线上 Agent 生成成功。完整目标仍未完成：
一般公开数值/数组、剩余控制/容器语义、一般 shape/packing、完整上游 helpers、
IR calls、真实 bootstrap/upscale 和新语义的 Agent 生成验收继续保留。

## 第十轮：公开数值/数组表达与派生常量

新增 `--public-numbers`（请求 v14 / AST v13），将公开 int/float 运算、受控
np.array/asarray、逐元素广播、公开索引/切片、shape、C-order reshape/flatten
接到现有 Hecate tracing。原始常量、权重、request/reference 不变，派生值单独
保存为 `derived-constants.json` 并记录哈希。详见
[公开数值与真实 CPU 证据](public-numbers-cpu.md)。

实际 Dacapo/SEAL CPU 首次 6/6 符合预期：3 个正确程序通过，3 个反例完成真实
密态执行后数值拒绝，24 组输入 / 80 个输出值，最大正例误差小于 1.626e-8。
空数组 dtype 显式保存；未实现的数组原地修改明确拒绝，不以重新绑定伪装 NumPy
原地语义。175 组数组运算与固定版本 NumPy 对照。最终新旧语义/产物及批量接口
170/170 项审计通过，无跳过；全量离线 590 项，476 通过、114 条件跳过、无失败。
清理本批临时密钥约 3.79 GiB，保留实验材料；历史产物记录实际 producer 哈希。

付费入口已支持此契约，并阻止将改变契约的新实验标记为同 prompt 重试。用户已
授权目标内付费测试；12 例新契约外发尝试仍被执行环境的具体 payload 审批拒绝，
进程未创建、无 API 生成调用。等待用户明确确认该模型清单/公开权重/DSL规则与
修复诊断发往 DeepSeek，不以旁路启动。允许新语法不等于 Agent 已覆盖新语法。

完整目标仍未完成：数组修改、更多数值/数组算子、一般 shape/packing、完整上游
helpers、其余控制/容器语义、IR calls、真实 bootstrap/upscale 及 Agent 全面验收
均继续保留；当前 rank/元素数与编码长度界限不代表完整 Hecate 能力。

## 第十一轮：公开控制流与不可变字典键

新增 `--public-control`（请求 v15 / AST v14）：公开 while、for/while-else、
break/continue/pass、短路 and/or/not、公开比较与 membership，以及整数/元组
等不可变字典键。保持循环控制的词法归属、短路的操作数返回语义、迭代器消费状态
及数值键冲突行为。对应 MPCB.GenPoly 中真实存在的构图需求，不代表完整 GenPoly。

1,176 组比较和 250 组循环与原生 Python 对照通过。实际 Dacapo/SEAL CPU
首次 6/6 符合预期：3 正例通过、3 反例在真实密态执行后数值拒绝；24 组输入 /
80 个输出比较，正例最大误差 3.364e-8 以下。新旧真实证据审计 187/187 通过，
无跳过；全量离线 607 项，492 通过、115 条件跳过，无失败。清理临时密钥约
3.79 GiB，实验产物保留。详见 [控制流语义与 CPU 证据](public-control-cpu.md)。

本轮无付费 API 调用，新语义的线上 Agent 生成验证仍未完成。完整目标继续保留：
一般数组及修改、更多公开数值/字符串/容器语义、一般 shape/packing、完整
上游 helpers、IR calls、真实 bootstrap/upscale 和完整 Agent 验收。

## 第十二轮：公开系数与树描述字符串解析

根据 MPCB.GenPoly 的 strip/split/int/float 构图流程，新增
`--public-strings`（请求 v16 / AST v15）：公开字符串 strip/split 系列、
partition/rpartition、replace、join。保留字符集合删除、显式分隔符空字段、
参数绑定、接收者求值顺序与迭代器消费语义，不执行字符串中的代码。

616 组字符串组合同时与原生 Python 对照并经过 AST 解释器；另有 12 组 join
及边界测试。人工实际 Dacapo/SEAL CPU 首次 6/6 符合预期：3 个正确解析程序
通过，3 个解析错误在真实密态执行后数值拒绝，24 组输入 / 80 个输出比较，
正例最大误差 6.214e-8 以下。新旧真实证据审计 196/196 通过，无跳过；
全量离线 616 项，500 通过、116 条件跳过。清理临时密钥约 3.79 GiB。
详见 [字符串语义与真实 CPU 证据](public-strings-cpu.md)。

本轮无付费 API 调用，线上 Agent 对新增契约的生成成功率仍未测量。
公开文本解析不是完整 GenPoly；np.polynomial.Chebyshev、多项式除法/余数、
更多 NumPy 运算和 helper 验证仍缺失。一般数组/修改、其余公开语义、
shape/packing、IR calls、真实 bootstrap/upscale 与完整 Agent 验收继续保留。

## 第十三轮：公开 Chebyshev 与实际上游 GenPoly

新增 `--public-polynomial`（请求 v17 / AST v16），复用已安装固定 NumPy 的
公开 Chebyshev 构造、算术、商/余数、只读系数及 floor/ceil/log2、dtype marker。
golden 中的 fint/GenPoly 直接取自上游，测试逐项核对函数 AST，并使用独立
单项式模型作 reference。没有重写上游函数来绕过语义限制。

真实 Dacapo/SEAL CPU 初始 7 例及补充 missing-giant 分支 1 例均首次符合预期：
4 正例通过、4 反例数值拒绝，32 组输入 / 128 个输出比较，最大正例误差
9.053e-8 以下。特别验证 GenPoly 叶节点仅读取奇数阶系数：偶数项模型没有被
静默修复，而是在数值比较中被拒绝。详见
[公开 Chebyshev / GenPoly 证据](public-polynomial-cpu.md)。

固定 NumPy 1.25.2 下新旧语义及产物审计 210/210 通过，无跳过。系统 Python
全量离线 630 项：500 通过、130 条件跳过；新增 NumPy 测试在固定环境中实际
执行，不能以系统环境的跳过冒充通过。清理两批临时密钥约 5.06 GiB。

本轮没有付费 API 调用；支持这些 GenPoly 配置不代表任意 tree/length/系数或
整个 DSL 已完整支持。一般数组与修改、其余公开语义、通用 shape/packing、
其他上游 helpers、IR calls、真实 bootstrap/upscale 及 Agent 全面验收仍保留。

## 第十四轮（构图阶段）：对象数组与 Empty

新增显式实验开关 `normalize(..., object_arrays=True)`，实现对象数组构造、
切片共享、复制、索引修改、元素增强赋值、迭代与 Empty 的已验证操作数行为。
没有修改已有请求契约；新语义尚未加入 Agent prompt/正式 CLI。

实际上游 SumSlots AST 的 6 组 m/p 配置与独立明文求和对照通过。固定
NumPy 1.25.2 环境新增 15/15 测试通过；新旧语义及既有密态产物审计
225/225 通过，无跳过；系统 Python 全量离线 645 项，500 通过、145 条件
跳过。以上是新构图语义与旧证据回归，**不是新增密态执行通过**。

详见 [对象数组与 Empty 语义证据](object-array-semantics.md)。下一门禁是
正式版本化接线、golden/反例密态执行及 Agent 生成验证。完整 DSL 目标不变；
vectorized object 运算、一般数组/packing、完整上游 helpers、IR calls、
真实 bootstrap/upscale、Poseidon GPU 与整体 Agent 验收仍未完成。

## 第十五轮：对象数组正式契约与真实 CPU 验证

`--object-arrays` 已接入请求 v18 / AST v17、语义规则、sandbox、trace、单例及
批量运行和审计。保留旧规则/请求哈希，不把更换合约标记为同 prompt 重试。

实际上游 SumSlots 四槽/三槽归约、对象数组 Linear 和 Empty/别名仿射共 4 个
正例通过真实 Dacapo/SEAL CPU；错误旋转方向和错误 Empty 符号共 2 个反例在
密态执行后数值拒绝。四槽首轮最大误差 9.776e-6 靠近零 reference 的阈值，
保持程序/参数/门限补做两次独立密钥测试，最大误差 3.149e-6 / 2.698e-6，均通过；
有限重复不代表任意输入或任意随机密钥下的精度保证。

共 8 次执行、32 组输入、120 个输出比较；新旧语义/真实证据审计 229/229
通过，零跳过；系统离线 649 项：500 通过、149 条件跳过。临时密钥清理约
5.06 GiB，原始实验结果保留。详见 [对象数组 CPU 证据](object-array-cpu.md)。

本轮无付费 API 调用，新增语义的实际 Agent 生成验收未完成。完整 DSL 目标保持
开放，尚缺的对象/数值数组运算、一般 packing、高层 helpers、IR calls、真实
bootstrap/upscale 和 Poseidon GPU 等不由本批人工 CPU 结果代替。

## 第十六轮（构图阶段）：公开字典与实际上游形状计算

针对 `CascadeDS` / `CascadePool` 的实际 `shapes.copy()` 阻断，增加显式
`normalize(..., public_mappings=True)`。支持受限 dict 构造、copy/get/setdefault/
pop/popitem/clear/update，以及动态 keys/values/items 视图、迭代和公开 membership。
保持浅复制、插入顺序、接收者/参数求值顺序与 iterator 失效；拒绝通过视图形成循环。
旧 v18 及以前契约不变，新语义尚未接入正式 Agent 请求。

实际上游 InferShapes / CascadeDS / CascadePool 在 36 组配置下共 108 次对照，
同时核对原生函数、独立整数 reference 和 AST 展开器。新增 10/10 固定环境测试
通过；联合旧语义/密态证据审计 239/239 通过，零跳过；全量离线 659 项：500
通过、159 条件跳过。本轮没有新增密态执行或付费 API 调用，不把公开 shape
计算通过解释为通用 packing 或 Conv 已验证。

详见 [公开字典与形状 helper 证据](public-mapping-semantics.md)。正式版本化接线、
对应密态 golden 与 Agent 验收仍待完成；其余完整 DSL / backend 目标保持开放。

## 第十七轮：正式公开字典契约与逐项真实 Agent 构造覆盖

公开字典接入 request v19 / AST v18、CLI、sandbox、trace 和批量驱动；旧契约保持
不变。新增冻结 30 例 construction exercises，要求实际使用指定语法构造输出，
而不是让普通模型随机出现目标语法。

经主批与定向补批，共 55 次 DeepSeek deepseek-flash 真实调用，最终 30/30 例、
117/117 去重观察项具备真实 Agent 源码及 Dacapo/SEAL CPU 数值通过证据。
最终 120 组输入、480 个标量；MAE 5.8921e-9，最大误差 3.2142e-8。
主批首次仅 20/30，通过反馈修复为 28/30；补批与覆盖复核单独记录，不改写失败历史。

34 项有有限输出扰动敏感证据，83 项是执行结构证据加源码复核。清除空字典、冗余
loop-else、items 结果被覆盖等不计有效覆盖，已补测。329 项相关回归中 310 通过、
19 条件跳过；测试密钥清理约 22.13 GiB，证据保留。

30 例计算目标均为同一个四元素仿射模型，是不同构造写法的专项验证，不是 30 类
网络泛化，也不是完整上游语义或所有边界的证明。Poseidon GPU、真实 bootstrap、
一般 packing、完整 helpers 等目标仍开放。逐项矩阵、原始记录、失败解释及命令见
[最近新增构造语法的真实 Agent 验收](construction-agent-coverage.md)。

## 第十八轮：对象数组逐元素与原地算术

正式 request v20 / AST v19 增加 bounded object-array +、-、* 和 whole-array /
slice +=、-=、*=；新增公开存储维度广播、0-D 标量结果、共享引用更新和重叠读取
语义。保持 Expr 左操作数与 ndarray 左操作数的实际派发差异，不把数组形状当作
密文 slot 布局。旧请求不变，新功能通过 --object-arithmetic 显式选择。

四个正确 golden 经真实 Dacapo/SEAL CPU 通过，两个错误的别名/Empty 程序也真实
执行并在数值阶段被拒绝，6/6 符合预期。共24组密态输入、96个输出比较；正例
最大误差 1.7522e-8。344项相关回归：325通过、19条件跳过，包含本轮真实证据
再审计；旧付费30例/117项仍通过。测试密钥清理约3.79GiB，产物和数值证据保留。

本轮没有调用付费 API；新增 v20 语义仍待定向 Agent 生成和批量接线，不能沿用
v19 的付费覆盖结论。其他对象算术、Plain/array 派发、一般packing、helpers、
bootstrap/upscale及Poseidon GPU保持未完成。详见
[对象数组算术与真实CPU正反例](object-arithmetic-cpu.md)。

## 第十九轮：v20 批量接线与定向真实 Agent 验收

新增 --object-arithmetic 批量与独立 --object-arithmetic-exercises 冻结十例，
不改变旧v19的30例/117项要求。八例仿射、两例二次多项式；检查真实typed array
事件、广播shape、存储重叠及有限输出影响，不把普通标量运算当作对象数组覆盖。

DeepSeek deepseek-flash/high/API并发10：首次8/10，修复后10/10，13次调用，
231485 tokens，零传输重试；40组密态输入、160个标量比较，最大误差1.9217e-8。
352项相关回归333通过、19条件跳过；本轮真实证据审计通过，旧v19仍30/30、117/117。
测试密钥清理约6.32GiB，原始失败和数值证据保留，无额外补批。

三个失败尝试分别为不存在的常量名、公开数组直接作为对象标量单元、数组到float
转换。后两者提示当前值语义仍有未实现路径；不能以Agent避开它们后的成功宣称
完整支持。普通重跑禁止丢掉冻结构造要求。详情见
[v20对象数组算术真实Agent验收](object-arithmetic-agent.md)。完整DSL与GPU目标仍开放。
