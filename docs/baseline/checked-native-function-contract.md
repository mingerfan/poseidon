# 原生装饰函数：受限 Agent 契约与真实 sandbox 验证

## 结论与边界

2026-09-16，本地新增显式 `--native-functions` 契约，将经过类型检查的多装饰函数
AST 分派到真实 Hecate `Func`/native `createCall`；不执行候选 Python。
这不是 v22 构造语法的并集，也不代表所有上游 DSL 语义完成。

- 四个正确人工候选通过隔离 trace → Dacapo/Earth/CKKS → HEVM/CST →
  upstream SEAL HEVM CPU → decrypt → 独立 reference 差分。
- 一个减法输入顺序故意写反的候选完成真实密态执行，并在 numerical_comparison 被拒绝。
- 另有十一类原生函数 AST 程序重新 trace/compile，golden HEVM/CST 与此前真实
  密态执行的对应产物逐字节一致。这十一类检查本身不是十一例新密态执行。
- 本轮新付费 API 请求为零；尚未取得本契约的真实 Agent 生成覆盖。
- 没有 Poseidon GPU、bootstrap 或全输入形式化等价结论。

## 系统位置和输入输出

`candidate_contract.make_request(native_functions=True)` 生成独立任务
`hecate-native-function-synthesis-v1`。输入仍是受限明文模型描述、公开常量、
FX 文本、固定 layout 和规则；响应仍为 `schema/request_id/hecate_source` JSON。

`decorated_functions.validate` 检查整个调用图，再由 `register` 生成可信 Python
wrapper 来解释已校验 AST。wrapper 调用真实 Hecate 对象，native createCall 静态
复制已验证 IR；不是在 HEVM 中添加动态函数调用，也不是规则答案代替 Agent 输出。

`candidate_trace` 在既有 bubblewrap 隔离中运行这一路径，保存真实 IR、常量、
`native-function-plan.json` 和统一的 `trace-evidence.json`。候选无文件、进程、
网络和密钥访问权限。已有参数、输入、reference、容差及隐私边界不变。

本报告首次完成时仅接通单案例 CLI；后续已补批量逐项 cohort 和调用图审计，见
`docs/baseline/native-function-exercise-coverage.md`。老单函数
`dsl_grammar_coverage.analyze_source` 对此契约仍明确拒绝，不能把第一个 helper
误当作 golden。新审计使用原生调用记录；只有出现相关声明不算覆盖。

## 受限语义

支持 c/p 位置参数、零参数 helper、顶层多装饰函数、前向和嵌套调用、重复调用、
标量及平坦 list/tuple 返回、解包、固定整数索引、有限公开常量、密文算术和六种旋转。
检查 c/p 类型、输入顺序、返回数量、递归和展开资源界限。

禁止递归、任意 Python 执行、I/O、bootstrap、控制流、普通/嵌套 Python helper、
lambda、ndarray 方法、公开值之间的算术和 augmented assignment。
native frontend 中已有的 object-array 返回人工证据不等于此 Agent 契约已支持它。
`expanded_native_work` 是保守资源估计，不是运行时 opcode 数或性能指标。

## 两个实际缺陷及修复

### 1. 缺少 trace 证据文件

新分支在 `hc.save` 后提前返回，IR/CST 已生成，却没有写驱动要求的 JSON。
第一批五例均停在 dsl_trace，诊断为 `JSON file size limit`（文件不存在也触发该检查）。
统一 `save_trace` 后，两个路径都会在 save 成功后写证据；save 失败则不写成功证据。

失败批（保留）：`/home/lhy/poseidon-work/results/checked-native-goldens-mg3jct6x`。
report SHA256：`4deeeff62e0c36f366fb09763f9e69de6efd2d68bd4f5d1b611636a96f24f43f`。

### 2. compiler 的参数 scale 与函数签名不同步

第二批 tuple 和双密文程序通过；`helper(c,p,p)` 的乘法后加偏置、以及包含这一结构的
前向调用失败。EVA 将 bias block argument 从 Plain scale 40 调为 80，而函数签名
还使用 `refineInputValues` 捕获的旧类型。`--verify-each` 正确报告类型不一致。

修复位于 Dacapo `lib/Dialect/Earth/Transforms/Common.cpp`：在 refineReturnValues
建立最终函数类型前重新收集 block argument 类型，共增加七行。没有关闭 verifier，
没有剔除 helper，没有改变乘加运算、安全参数或容差。仅重建 hecate-opt，parallel=2。
独立可重放补丁：`scripts/baseline/patches/dacapo-refined-function-input-types.patch`；
已通过当前修改的 reverse apply --check 和 scoped diff --check。

失败批（保留）：`/home/lhy/poseidon-work/results/checked-native-goldens-wcefp7x_`。
report SHA256：`a5591b8eef48a4dcc3d11106aa081a70719f50258c78b327a12e0b113bf8ee77`。

## 最终证据

批次：`/home/lhy/poseidon-work/results/checked-native-goldens-9pcd1b9p`。
report SHA256：`29d3bce3ad7b736f6555ba13aa79df02e158872a5de9a83d6b72eab864b4cb19`。

| 人工候选 | 独立 reference | 预期和实际结果 |
|---|---|---|
| affine helper(c,p,p) | 0.5x+x+0.375 | 密态差分通过 |
| tuple 返回、解包和合并 | 同上 | 密态差分通过 |
| forward + nested helper | 同上 | 密态差分通过 |
| 双密文 ordered subtract | left-right | 密态差分通过 |
| 故意 right-left | reference 仍为 left-right | 密态执行完成，数值比较拒绝 |

四个正确候选共16组输入、64个输出值；最大绝对误差 `2.2924344422747822e-8`，
加权 MAE `4.535487571147559e-9`。逐项门限仍为 `1e-5 + 1e-4*abs(reference)`。
反例不计入正确程序误差统计。每例保留参数、命令、请求、源码、产物哈希、解密值和诊断。

十一类当前 AST 编译复核：`/home/lhy/poseidon-work/results/checked-native-functions-6w0qazd6`。
report SHA256：`4ab49f4889524c3a2842b0e4880e43a84e14d4a24496abf87581d485e2471f15`。
编译器 SHA256：`1a0a5d24b9dcce727e7d0b8a46c82098b2a91e7d30a548adb10a4dfb52e70026`。
十一项为 scalar/pair/nested/forward/two_inputs/public_argument/identity/repeated/
zero_input/empty_helper/plain_return。空返回只适合执行结构证据，不能声称输出敏感性。

## 回归和保留策略

- 针对新契约、tracing 协议、native 前端、此前 native 密态产物、bootstrap 前端边界、
  augmented assignment、v22 历史付费证据及保留策略的94项回归：94通过、0失败、0跳过。
- 额外 provider/rotation/新契约离线组合：52项，50通过、2个未指定旧证据路径的条件跳过。
  provider fixture 不算付费请求；两组测试有重叠，不能相加为独立测试数。
- 三次五例诊断批共清理 `10184700150` 字节（约9.49 GiB）可重新生成的随机私钥/评估密钥；
  原密钥已不可恢复，源码、IR、HEVM/CST、输入、reference、解密结果和日志均保留。
- 全部修改保留在 `feat/agent-dsl-correctness` working tree；HEAD 仍为
  `4995e7cadedf2bfb9104658b5638662ecf6a1d0a`，无commit/push/PR或分支切换。

人工 sandbox 重放（无付费API）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 1650s python3 scripts/baseline/run_checked_native_goldens.py
```

## 下一门禁：逐项真实 Agent

旧 v19–v22 的付费证据不能计为新原生函数契约的覆盖。新付费启动此前被执行环境审批
拒绝，需要明确外发范围及预算配置后才能重试，不能换入口绕过拒绝。
拟限11例，DeepSeek官方 API、deepseek-flash/high、API并发10/native并发2，
1200秒/请求，384000 tokens/响应，最多3轮语义修复和每轮3次传输重试：
最多44轮生成、176次HTTP尝试。上限不是预估费用；超时可能重复计费，无货币费用承诺。
外发仅合成模型、公开常量/layout/规则和受限修复诊断，不外发环境文件、私钥、测试输入数组。

取得该范围确认后，先冻结11例模型/构造要求及逐项审计器，再运行真实Agent并分别报告
首次/修复后生成、编译、执行、数值正确率及结构覆盖。当前并未声称该批已经准备完整或启动。
