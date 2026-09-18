# Native helper object 数组算术：真实编译与密态验证

2026-09-17。本轮扩展已接入可生成候选的契约、静态检查、隔离 tracing 和批量入口；
8 个正确人工程序均通过真实 SEAL HEVM CPU 密态差分，1 个错误广播索引反例
在数值比较层被拒绝。**本轮没有付费 API 调用，不能算真实 Agent 生成覆盖。**

## 位置、语义与版本

上游依据是固定 Dacapo checkout 的
`python/hecate/hecate/expr.py` 中 `hecateMetaBinary`、`resolveType` 和
`Func`：NumPy object 运算逐单元调用真实 Expr 重载，native helper
通过现有 c/p 参数和静态内联 IR 返回 Expr。没有新增 array-valued 参数 ABI。

每个 object 数组单元表示一个独立 Expr；Expr 自己才持有 packed slots。
NumPy 数组广播改变的是外层 Expr 配对，不是密文 slot 布局，也不执行 rotation。

新版本明确区分三个标识：

- 请求：`hecate-native-function-synthesis-v6`。
- Agent 校验契约：`hecate-native-functions-v4`。
- native 类型计划：`decorated-functions-core-v4`。
- CLI：`--native-array-arithmetic`，包含此前 readonly 数组及 positional-star 语法。
- 旧 v1-v5 请求文本和 nf/na exercise 要求不变；旧批次不能原地切换新契约。

允许非原地数组 `+`、`-`、`*` 和一元 `-`，遵循 NumPy trailing-axis 广播。
二元运算左侧必须为 object ndarray，右侧可为 object ndarray 或单个 Expr。
每个广播配对至少含一个密文，一元负号的非空单元全为密文。
运算前检查形状兼容性及已有 rank<=4、cell<=16 的资源边界，再接触真实 frontend。

真实 frontend 检查确认：

1. `array * Expr` 由 NumPy 逐单元分发；`Expr + array` 则先进入 Hecate
   `resolveType` 并失败。后者明确拒绝，不把换序或自造广播当成上游语义。
2. 0 维 object 数组运算返回单个 Expr，不保留 ndarray 包装。
3. 非原地运算生成新存储，不改变输入数组单元或别名。
4. 空数组遵循 NumPy 的零长度维广播，只有结构证据，不含数值贡献。

仍不允许 array 作为单个 native c/p 形参、Plain/Plain 运算、Plain 负号、
原地写入、任意 ufunc、除法、矩阵乘法或幂运算。后两者未因 array 支持被自动开放。
候选仍是受限 AST 数据，由可信解释器分发；不 exec/eval 候选 Python。

## 新增真实密态案例

| 案例 | 检查点 | 结果 |
|---|---|---|
| vector | array×公开系数数组、array+array、array+Plain | 通过 |
| broadcast | (2,1) 与 (1,2) 的有序减法广播 | 通过 |
| zero_dim | 0 维 object 乘法，结果作为单 Expr 使用 | 通过 |
| negate | object 数组逐单元负号 | 通过 |
| mixed | 混合 c/p 单元，分别执行 c×p 和 p×c | 通过 |
| nested | helper 返回运算结果，继续数组运算并展开调用另一个 helper | 通过 |
| square | array 中 c×c，再乘公开系数、加偏置 | 通过 |
| empty | 可达空数组 helper，零长度广播和负号 | 通过 |
| wrong_broadcast | 故意选错广播结果的列，程序仍合法可编译 | 数值比较正确拒绝 |

7 个正确案例的 reference 为 `1.5*x + .375`，square 为 `.5*x**2 + .125`；
不是 8 个模型家族。空数组案例的运算只证明结构可用，不宣称输出影响。
负号案例检查 tracing/执行正确性，但双负号可能被优化消除，不等同于
每个后端 Negate opcode 都得到执行覆盖。

每例采用四组固定输入（零、有符号、固定种子随机、范围边界）。
8 个正确程序共 32 组输入、128 个输出值：

- 最大绝对误差：`2.6455489188226267e-8`。
- MAE：`5.048379663093183e-9`。
- 非零 reference 最大相对误差：`1.3805121775523794e-7`。
- 门限不变：`abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
- 错误索引反例计算成 `1.75*x+.375`；最大绝对误差
  `0.25000000960984714`，独立审计也核对了这一错误公式。

后端为 upstream SEAL HEVM CPU、SEAL 4.0.0、N32768、14×60-bit 模数、
tc128 检查、waterline40；不是 Poseidon GPU。没有 bootstrap、参数缩减或
decrypt-and-reencrypt 模拟。

## 可复现证据

批次：
`/home/lhy/poseidon-work/results/native-array-arithmetic-goldens-l742dym9/report.json`

SHA256：
`9de9d7a55ca6223eab7d4a7c752e9fc90d95a446ed13cce72ee5431d9e6cf7cd`

复现（人工程序，不调用 API）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 5s 1200s python3 scripts/baseline/run_native_array_arithmetic_goldens.py
```

审计测试 `test_native_array_arithmetic_evidence.py` 检查冻结数据哈希、
请求/响应/实际 tracing 源码一致性、重新计算的类型计划、真实调用位置、
Earth/CKKS/HEVM/CST 哈希、四组真实密态执行、独立 reference 及解密数组；
对反例要求在 numerical_comparison 层失败，不能用静态拒绝替代。

首轮 36 项定向测试通过。随后 145 项综合回归全部通过，无失败、无跳过；
包括新增真实 frontend/证据检查、既有数组/starred 契约、旧 v22 与 11 项
真实付费 native-function 证据、bootstrap 前端边界和保留策略。两组不相加计数。

9 个案例的可再生密钥共 6,110,820,089 bytes（约 5.69 GiB）已按运行器策略清理，
清理记录完整。原随机密钥不可恢复；新的密钥可以重生成。
保留源码、模型、reference、响应、IR、HEVM/CST、解密结果和日志；未删除旧批次。

## 未完成范围

本轮 agent_calls=0，live_agent_validated=false，full_semantics_proven=false。
后续需要针对这些新构造建立逐项 Agent 要求及输出影响检查，再进行获批的真实生成测试。
此前已获批的 native-function 11 项不能覆盖本轮新增语法。
native 与旧 v22 普通 helper/控制流的组合、未验证的上游高层 poly helper、
真实 bootstrap 等仍未完成；完整目标保持进行中。

全部修改留在本地 working tree，不 commit、push 或创建 PR。
