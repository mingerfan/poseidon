# v22 对象数组一元运算：确定性密态基线

后续进展（2026-09-15）：独立8例真实Agent付费验收已完成，见
[object-unary-agent.md](object-unary-agent.md)。下文保留本人工基线当时的范围、
0次API记录和历史下一门禁，不将人工结果追认为Agent生成结果。

## 结论与目标边界

新增独立 request-v22 / AST-contract-v21 / construction-schema16，
通过 `--object-unary` 显式启用。8个正确人工程序经过真实Hecate tracing、
Dacapo Earth/CKKS编译、HEVM/CST、SEAL CPU密态执行和解密差分全部通过；
2个错误符号/错误复制语义反例都被数值比较拒绝。

本轮 **0次API调用**；不是新语法的真实Agent生成验收。
此前v21的15例付费记录不能用来证明v22生成能力。完整DSL目标仍未完成，
下一门禁是冻结v22逐项要求并运行独立付费Agent案例，而非将人工答案算作Agent输出。

## 为什么补这一层

当前上游 `third_party/dacapo/python/hecate/hecate/expr.py` 的
`toInnerUnary={"neg":13}` 为Expr生成__neg__，没有__pos__。
Python/NumPy对象数组会逐元素调用这些方法。此前本地normalizer拒绝整个对象数组一元运算，
即使该数组里的cipher本身支持取负，造成实际可表达程序缺口。

实现位置：

- `object_arrays.unary`：受限对象存储遍历、结果分配、零维标量返回；不执行候选对象钩子。
- `function_construction.object_unary_value`：按cipher/Plain/公开数值类型分派，并记录可信事件。
- cipher取负发出原有Hecate UnaryOp，不新增假密态算子。
- 已知公开Plain取负沿用公开常量折叠，但保留Plain类型；不得由float/int当成解密值读取。
- `candidate_contract`、`hecate_contract`、`candidate_trace`、grammar审计与单例/批量入口：
  新版本显式接线，旧v19/v20/v21请求保持不变。

这是对象存储中的表达式操作，不是slot rotation，不改变packing，
也不等同于实现新的GPU backend或性能优化。

## 已核实的具体规则

| 输入/操作 | 行为 |
|---|---|
| `-array`、`np.negative(array)` 的cipher单元 | 调用Hecate取负 |
| 公开int/float/bool对象单元的一元正/负 | 匹配Python对象算术，bool结果为int |
| `+array`、`np.positive(array)` 的cipher或Plain单元 | 拒绝；上游Expr无__pos__，不能静默当恒等 |
| 已知构造Plain取负 | 保留Plain类型的公开常量折叠 |
| Empty/None单元 | 拒绝，不当作零 |
| 零维对象数组结果 | 返回标量或符号Expr |
| 正维对象数组结果 | 新存储；原数组、切片及其别名不被修改 |
| 操作数表达式 | 恰好求值一次 |
| named ufunc调用 | 一个显式对象数组参数；拒绝out/where/dtype/casting等关键字 |

边界：仍是rank<=4、存储<=128、展开操作和公开数值有界的现有安全契约。
本轮np.negative/positive入口不接受裸cipher或普通numeric.Array；
未声称覆盖NumPy所有dtype、参数和广播输出模式。除法、幂、矩阵乘法、归约、
bootstrap等没有借此扩展开放。

固定NumPy1.25.2的真实oracle确认：
0-D对象数组的一元运算返回标量；正维返回新ndarray；
Expr取负产生上游unary opcode13，Expr取正以及Empty/None一元操作抛TypeError。
对照使用仓库实际metaclass/Expr方法和记录式C-ABI替身，不把它当成真正FHE执行。
后续独立密态测试补足真实执行证据。

## 密态测试

所有正确模型的reference均为 `1.5*x+0.375`，输入shape[4]，
每例四组固定输入。人工源码独立写出，不修改reference以追随解密值。

| 人工案例 | 核心语义 | 最大绝对误差 |
|---|---|---:|
| neg | cipher对象数组取负 | 1.848014e-8 |
| zero | 零维结果是Expr | 2.440469e-8 |
| view | 切片取负结果与原存储独立 | 2.050977e-8 |
| positive | 公开一元加结果新存储 | 2.730511e-8 |
| negative_call | np.negative对象入口 | 2.368903e-8 |
| positive_call | np.positive对象入口 | 1.982584e-8 |
| plain | 已知Plain取负且不暴露为数字 | 1.102760e-8 |
| mixed | cipher、float、bool混合对象单元 | 6.880852e-9 |

正例32组密态输入、128个输出标量比较；加权MAE=5.582414469487013e-9，
最大绝对误差2.7305111505171453e-8。
含反例合计40组输入、160个标量比较。

- wrong_neg把取负错误地当作恒等，最大绝对误差约1.0，被numerical_comparison拒绝。
- wrong_copy让公开一元加结果错误地别名共享，最大绝对误差约0.5，被同一门限拒绝。
- neg/zero/view/negative_call/mixed的实际HEVM均含opcode2；
  上游 `lib/Runtime/SEAL_HEVM.cpp` 将它解释为NegateC。
- 公开positive和已知Plain折叠案例没有NegateC，不以公开常量运算冒充密态opcode。

固定SEAL4.0.0、degree32768、[60]*14、tc128；
门限仍为 `abs(actual-reference)<=1e-5+1e-4*abs(reference)`。
没有GPU执行、bootstrap、decrypt-and-reencrypt或参数降级。

## 证据与复现

报告：
`/home/lhy/poseidon-work/results/object-unary-goldens-aj3eucw2/report.json`

SHA256：
`b41c8dee5b88a64b64bf8ab1b066510dbf1f724d9bc5266b5f423004fe034695`

原始模型、权重、测试输入、request、trace payload、normalized source、
construction metadata、IR、HEVM/CST、解密值和日志均保留。
`test_object_unary_evidence.py`重新计算文件哈希、静态检查、artifact parsing、
独立reference、保存解密数组及误差，并检查实际NegateC。

运行新的人工实验（不收费）：

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 10s 3300s env PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/baseline/run_object_unary_goldens.py
```

查看v22普通批量计划（不读凭据、不收费；不是定向覆盖承诺）：

```bash
timeout -k 3s 30s python3 scripts/baseline/run_agent_batch.py --plan --object-unary
```

在既有固定Nix/Python环境、scripts/baseline目录中设置
`POSEIDON_OBJECT_UNARY_REPORT`为上述报告，再运行：

```bash
python -m unittest test_object_unary test_object_unary_evidence
```

所有相关回归378项：359通过、19条件跳过、0失败。
本轮人工密态证据和此前v20/v21付费/人工证据显式开启，不在跳过项中。
一次实现期测试夹具使用不支持的list下标+=，已改成普通下标赋值；
没有为了测试通过扩大无关语义。

本轮临时随机测试密钥清理6,789,800,100 bytes（约6.32GiB）；
不可恢复旧随机密钥，可重新生成。API凭据和所有结果证据保留。
WSL可用空间约189.17GiB。未安装依赖、修改系统配置或运行付费批次。

## 本地状态与下一步

分支feat/agent-dsl-correctness、HEAD4995e7cadedf2bfb9104658b5638662ecf6a1d0a保持。
已有C++/GPU和Dacapo修改不动，新增/修改仅在本地scripts与docs。
无commit、push或PR；建议之后经批准按“一元语义与契约”“密态基线与审计资料”拆分本地commit。

下一步：冻结实际对象一元事件、零维、cipher/Plain/公开单元、别名独立性等逐项要求，
再让DeepSeek独立生成并走同一密态差分链。旧版本构造清单不能改契约重用。
上游helpers、一般packing、其余对象算术和FHE管理操作等完整目标继续保留。
