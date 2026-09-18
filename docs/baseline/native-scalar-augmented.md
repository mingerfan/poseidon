# Native helper 标量增强赋值：实际运算、别名和密态证据

> 后续独立 v10 契约已接通数组 name-target 原地运算及 view/copy 别名模型，见
> [数组原地运算与别名](native-array-mutation.md)。本页保留 v9 标量契约和当时的验证边界。

2026-09-17：接通 `+=`、`-=`、`*=` 的标量 Expr name-target 增强赋值，
并验证与现有公开循环、native helper、星号实参、数组读取的组合。
7 个正确人工程序通过真实编译/密态差分；2 个错误数学表达在数值比较层正确拒绝。
本轮没有付费 API 调用；不是完整 DSL 语义证明，也不是新增 Agent 生成覆盖。

## 为什么需要区分“重新绑定”和“原地修改”

当前本地 Hecate `expr.py` 的 metaclass 将 `__iadd__`、`__isub__`、`__imul__`
绑定到生成新 Expr 的正向 binary method。已有前端修复保留了减法左右顺序。
本轮没有再修改该依赖文件。

```python
alias = x
x *= c0
# x 指向新的 Expr；alias 仍指向旧 Expr。
```

若旧 Expr 同时保存在 `np.array([x], dtype=object)` 中，那个数组单元也保持不变。
但 `array += other` 走 NumPy 原地 ufunc，会修改共享存储。实际 frontend 测试验证了
两种行为不同。因此当前新契约只开放标量 Expr 的局部名称增强赋值，不能把数组原地写入
简单改写成 `array = array + other`。数组/view 别名感知的写入仍是剩余任务。

## 输入输出与各层职责

- 新请求：`hecate-native-function-synthesis-v9`，类型契约 `hecate-native-functions-v6`。
- 入口：单例和批量的 `--native-scalar-augmented`；继承公开循环、数组只读操作/非原地算术、
  星号参数及 native 调用。旧 v1..v8 请求规则和已冻结构造要求不变。
- AST 检查：目标必须是已绑定局部名称；两侧为 c/p Expr，至少一侧为密文。
  结果为密文。公开常量名称与循环归纳变量不可重写。
- Plain 局部变量可以通过与密文运算变成密文；不允许纯 Plain/Plain 运算。
- trusted AST dispatcher 先读取左值，再求值右侧一次，执行实际 `value +=/-=/*= other`，
  不执行候选 Python，也不把 helper 内联成另一个人为定义的语言。
- tracing 记录函数名、目标、运算、源码位置、新 Expr 身份与旧 Expr 未改变的观察。
  helper trace 一次、调用多次；这些记录不是运行时密态指令次数。
- 下游仍由 Dacapo 产生 Earth/CKKS/HEVM，现有 SEAL HEVM CPU 实际执行。
- batch continuation 禁止把旧批次切换到此契约；需要新请求和新批次身份。

尚不允许：数组增强赋值、下标/属性写入、除法/幂增强赋值、未绑定名称、
密文控制流、Empty accumulator 或隐式 bootstrap。

## 人工密态验证

固定模型为 `1.5*x + 0.375` 和 period-four rotate-and-sum，两种数学图，
不是七个模型家族。每例四组固定输入，不以解密中间值回填 reference。

| 程序 | 验证重点 | 最大绝对误差 |
|---|---|---:|
| add | c/c 与 c/p 连续 `+=` | 4.9230e-10 |
| subtract | `-=` 的左右顺序 | 4.6632e-10 |
| multiply_alias | `*=` 重新绑定，保留名称别名 | 7.1036e-10 |
| array_alias | 数组单元仍保存旧 Expr | 2.7108e-10 |
| plain_left | p/c 减法及 p→c 局部类型变化 | 6.4341e-10 |
| nested_loop | helper、星号实参及公开循环组合 | 4.1990e-10 |
| sum4 | 公开循环中的 rotation 归约 | 2.1926e-7 |

7 正例共 28 组输入、112 个输出值；最大绝对误差 `2.1925893989305223e-7`，
加权 MAE `1.0757379169100046e-8`。

`wrong_alias` 故意把旧值误写为新值，最大差异 `0.5000000002882093`；
`wrong_order` 故意减去偏置而非加上，最大差异 `0.7500000000958422`。
两例都能编译和执行，但在 numerical_comparison 层被拒绝，不能计为正确候选通过。
独立证据测试还与两个错误程序各自的数学公式比较，以排除仅靠执行异常拒绝反例。

配置：显式 `seal-cpu-eva-w45-v1`；默认 waterline40 不变。
SEAL4.0.0、N32768、14×60-bit 模数、tc128；容差仍是 `1e-5+1e-4*abs(reference)`。
不运行 bootstrap 或 Poseidon GPU，不改变算法和安全参数。

结果：`/home/lhy/poseidon-work/results/native-augmented-goldens-4_7ws610/report.json`。
SHA256：`86a35b5452cc88221b302610109cb34193b8d33d1d705041361e1bd457c33474`。

九例密钥清理全部完成，共释放 6,110,820,090 字节（约5.69 GiB）。删除的原密钥不可恢复，
需要时可重新生成；模型、请求、候选、输入/reference、IR、HEVM/CST、解密结果和日志保留。

## 测试与复现

`test_native_scalar_augmented.py` 包含 4 个静态/接口测试和 2 个实际 frontend 测试。
`test_native_augmented_evidence.py` 独立检查冻结输入、响应→tracing 源码一致性、
产物哈希、重解析 HEVM、compiler 配置、真实 augmented/call 观察、reference、
解密数组及密钥清理。旧契约仍拒绝新增语法，旧案例不会被重新分类。

综合回归 221 项通过，0 失败、0 跳过，覆盖上述新增测试、旧 native/数组/循环、
compiler 配置、provider/batch、旧 v22、bootstrap 前端边界、语义清单与保留策略。
回归不调用付费 API；已完成的 11 项付费结果仍作为历史证据独立审计。

```bash
cd '/mnt/d/Code Space/Poseidon'
# 重新执行人工密态批次；没有模型 API 请求
timeout -k 5s 3050s python3 scripts/baseline/run_native_augmented_goldens.py
```

当前完整 DSL 目标仍未完成：新增语法真实 Agent 逐项覆盖、数组写入/view 别名、
native 与旧构造契约的组合、高层 helper、真实 bootstrap 等仍需继续。
所有修改本地未提交，无安装、凭据修改、分支切换或远程写操作。
