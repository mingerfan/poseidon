# Native 对象数组原地运算：类型堆、共享存储与真实密态结果

2026-09-17，本轮新增独立请求 `hecate-native-function-synthesis-v10` / 类型契约
`hecate-native-functions-v7`。支持 ndarray 局部名称的 `+=`、`-=`、`*=`，
保留普通别名、切片/转置/reshape 视图和独立复制的区别。
13 个人工正确程序通过真实编译/密态差分，2 个数学错误程序在数值比较层拒绝。
没有新付费 API 调用，不能计为 Agent 新构造覆盖或完整 DSL 语义证明。

## 系统位置与语义

入口是 `run_candidate.py` / `run_agent_batch.py --native-array-mutation`。
新契约继承标量增强赋值、公开循环、native 调用、对象数组读取与非原地算术；
旧请求文本和旧案例身份保持不变，旧批次不能通过恢复选项改用新契约。

`native_array_alias.py` 是可信静态类型模块。它只持有 c/p 单元标签和有限整数地址网格，
不持有真实明文测试值或密文，不执行候选 Python：

- `alias = array` 保留同一数组对象；基本切片、转置及部分 reshape 共享存储。
- 更新 view 时，基数组及其他重叠 view 的单元类型一起变化。
- `copy()`、`flatten()` 有独立存储；`reshape` 是否共享必须依 NumPy stride 判断。
- `np.array(existing_array, dtype=object)` 分配新存储，但默认 `order='K'` 可以保留
  Fortran 顺序；非原地 ufunc 结果也不能一律假定 C 连续。这会影响后续 reshape 的别名关系。
- native helper 每次调用重建独立返回数组，不能把一个调用结果的变化传播到另一个调用。
- 原地运算的 broadcast 必须能写回已有 shape，不得扩张目标；每个参与运算的单元对
  必须至少包含一个密文。所有检查通过后才更新类型堆，失败不留下部分类型写入。
- 零维 ndarray 原地运算仍是同一个 ndarray；不同于零维非原地 object ufunc 返回标量 Expr。
- 已提取出来的标量 Expr 不会因数组单元替换而变化。
- 空数组更新只有结构意义，不存在数值输出单元影响证据。

当前仍保留数组大小16单元、rank4、有限循环与资源上限。
另外累计别名诊断的绑定/单元规模有4096单位上限，防止小 AST 膨胀成无界元数据。
下标/属性赋值、列表 mutation、array-valued native 参数、任意 NumPy API、
密文依赖控制流和真实 bootstrap 不因此开放。

## 真实 tracing 观察与诊断修复

可信 dispatcher 调用实际 NumPy ndarray `__iadd__/__isub__/__imul__`，不改写为普通赋值。
它记录函数、源码位置、数组 identity、shape、共享存储关系和被替换单元位置，
并验证目标数组 identity/shape 不变、旧 Expr 对象没有被就地修改。
这些真实观察与静态类型堆的结构预测独立对照。

本轮修复两个通过测试发现的问题：

1. 原观察代码用 `isinstance(value, Plain)` 判断 c/p，在 `callee_fresh` 上失败。
   当前 Hecate `Func` 将 p/c 参数和调用返回值都包装成 Python `Expr`，类名不是 IR 类型。
   已去掉这个错误的观察标签。真实观察只报告实际别名/单元变化；c/p 标签来自静态检查，
   由实际 c/p native 调用签名和编译器检查验证，不伪称从 Python 类读到了 IR 类型。
2. 类型堆初版把新存储一律设为 C 顺序。专门的 order-K 回归在修改前明确失败，
   修正后实际 frontend 对照及两例密态补测通过。最初13例报告保持原样，不覆写成新结果。

## 人工密态结果

两个独立批次，使用同一显式 `seal-cpu-eva-w45-v1` 和既定模型/reference：

| 构造 | 验证 |
|---|---|
| 普通别名、切片更新、转置更新 | 更新传播到共享单元，包括 p→c 转变 |
| copy 与 flatten | 独立存储不被源数组更新影响 |
| reshape view / reshape copy | 区分共享与复制 |
| 左右重叠切片减法 | 实际 NumPy 原地语义和正确数值顺序 |
| 零维数组、空数组 | identity、shape 及空结果边界 |
| 两次 native helper 返回 | 返回数组相互独立，c/p 类型不串扰 |
| 循环内原地乘法 | 重复更新可被别名读取 |
| 转置数组构造与 ufunc 的 order-K 布局 | 后续 reshape 不错误地当作 view |

所有正确程序对应 `1.5*x+0.375`，这是13种构造案例，不是13个模型家族。
共52组输入、208个正例输出值：最大绝对误差 `1.0235708947092803e-9`，
加权 MAE `1.740450797638609e-10`。

`wrong_alias` 故意读错结果，最大差异约0.5；`wrong_overlap` 将减法写成加法，
最大差异约4.0。两者实际编译、加密执行后在 numerical_comparison 层被拒绝。
独立审计也比较了它们各自的错误数学公式，排除仅靠执行异常拒绝的情况。

配置与边界：SEAL4.0.0、N32768、14×60-bit 模数、tc128；默认 waterline40 不变。
门限保持 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。
后端为 upstream SEAL HEVM CPU；无 Poseidon GPU、bootstrap 或 decrypt/re-encrypt 模拟。

原13例（11正例、2反例）：
`/home/lhy/poseidon-work/results/native-array-mutation-goldens-c9d3npba/report.json`。
SHA256：`402c93fe919c083f8e625f842e22398f1c7f77b4a5c4af6a44c676798c468a77`。

额外2例 order-K 正例：
`/home/lhy/poseidon-work/results/native-array-layout-goldens-c52z6uhz/report.json`。
SHA256：`0e52aa102a5fc8a7c9dce0859a7eb669bd8ca25ae836ce07bcc323a36f3215b1`。

15次运行的临时密钥全部清理，累计释放10,184,700,150字节（约9.49 GiB）。
原密钥不可恢复，但可重新生成；保留输入/reference、候选源码、IR、HEVM/CST、
解密结果、日志和报告，WSL可用空间约189.07 GiB。

## 测试和复现

`test_native_array_mutation.py` 包含类型/alias、原子更新、shape拒绝、旧契约隔离、
provider/CLI/batch边界、资源限制及实际frontend全案例对照。
`test_native_array_mutation_evidence.py` 重新计算静态计划、观察投影、模型明文公式、
解密误差并重解析HEVM，检查冻结输入/产物哈希、真实调用和临时密钥清理。

综合回归231项通过，0失败、0跳过，涵盖新增别名/布局与真实产物检查，
旧native、scalar augmented、数组/循环、compiler配置、provider/batch、v22、
bootstrap前端边界、语义清单和保留策略。历史付费结果仅离线审计，没有重复收费。

```bash
cd '/mnt/d/Code Space/Poseidon'
# 人工真实密态验证，无模型API；两个批次分开保留
timeout -k 5s 4300s python3 scripts/baseline/run_native_array_mutation_goldens.py
timeout -k 5s 700s python3 scripts/baseline/run_native_array_mutation_goldens.py --layout-only
```

完整目标仍未完成。下一类实质缺口包括数组下标写入、native与旧构造契约组合、
新增构造真实Agent逐项覆盖、高层helper、真实bootstrap。
修改保留在本地working tree；未安装依赖、修改凭据、切换分支、commit、push或创建PR。
