# 原生helper星号位置参数：接通已有语义，不新增数组ABI

> 后续新增 v8 逐项构造门禁、真实展开观察和独立审计，见
> [星号构造逐项覆盖](native-star-construction-coverage.md)。下面保留 v5 人工验证记录；
> 两批均不是付费 Agent 生成证据。

2026-09-17，本轮接通受检Hecate候选中的 `helper(*items)` 参数展开。
8个人工正例经真实隔离tracing、Dacapo编译和SEAL HEVM CPU执行后全部通过，
一个展开顺序错误的反例在数值比较层被拒绝。本轮没有付费API调用。
完整DSL目标仍未完成；新增契约未取得真实Agent生成证据。

## 为什么不是直接增加“数组参数”

Confirmed fact：当前 `third_party/dacapo/python/hecate/hecate/expr.py` 的
`Func.__init__` 只接受c/p签名，每个位置对应一个native标量Expr参数。
`Func.__call__` 对每个实参调用 `resolveType`，然后传给C++ createCall。
`resolveType(np.ndarray)` 走Plain构造，随后转换为浮点数组；它不是Expr容器参数ABI。
真实frontend负例确认，将 `np.array([x],dtype=object)` 整体传入密文helper会失败。
公开数值向量作为Plain内容和“装有若干Expr的对象数组”是不同概念，不能混淆。

`helper(*array)` 则由Python先迭代容器，把每个元素依次变成位置实参。
当这些元素都是匹配c/p签名的Expr时，现有Func/createCall即可处理，无需改C++。
本轮首先用直接Python调用真实Hecate验证该事实，再接通可信AST解释器。

## 系统位置、输入输出与保留的语义

- 输入：候选中已声明helper的调用AST、类型明确的容器和固定c/p签名。
- 静态检查：按从左到右顺序分析普通参数与每个星号段，只展开一层；
  最终实参类型及数量必须与helper签名完全一致，最多16个展开参数。
- 真实dispatch：可信解释器按相同顺序求值，逐段extend，再调用真实Hecate Func。
  不exec/eval候选Python，不删除decorator或绕过native调用。
- 输出：原有Expr/list/tuple/object-array返回方式不变；下游仍由现有compiler
  负责level、scale、rescale、relinearization和HEVM生成。
- 新请求 `hecate-native-function-synthesis-v5`，独立类型契约 `hecate-native-functions-v3`。
  CLI为 `--native-starred`，支持单例及批量驱动；旧请求字节/语法和冻结练习要求不变。

允许列表/元组、一维对象数组、切片结果、显式展平结果、helper返回容器的展开，
以及普通位置实参和多个星号段混合。空可迭代容器按实际第一轴长度贡献零参数。

保留Python第一轴迭代语义：0-D不可展开；二维非空数组迭代产生行数组，
不是标量Expr，因此不能静默flatten成参数。需要传单元时应明确写 `.flatten()`。
第一轴长度为0的多维空数组确实产生零参数，不因rank本身而被错误拒绝。
不会把一个密文里的slot解释成多个函数参数。

未新增kwargs/**、生成器、可变参数函数定义、星号return/list构造、
NumPy方法或rotate中的星号实参。原生数组写入/算术及与旧v22构造的组合仍需后续工作。
Expr数组整体参数不是当前上游native ABI已支持的能力，未来如要引入应作为明确接口设计，
不能声称现有接口已经具备。

## 实验与测试

| 正例 | 验证内容 |
|---|---|
| list | 列表顺序和混合c/p位置 |
| tuple | 元组位置展开 |
| array | 一维对象数组展开 |
| reverse | 负步长切片后按正确参数顺序调用 |
| matrix_flatten | 显式转置+展平后展开 |
| nested | helper返回容器再传入另一helper |
| multiple | 普通实参与多个星号段混合 |
| empty | 空容器展开后调用无参数helper |

数学reference保持 `0.5*x+x+0.375`，固定四组输入/案例，不从密态中间值回填。
8正例共32组输入执行、128个输出值：

- 最大绝对误差 `2.009584942896936e-8`。
- MAE `4.8631986404503585e-9`。
- 容差不变：`1e-5+1e-4*abs(reference)`。
- 顺序错误反例额外4组输入/16个输出值，最大绝对误差 `2.999999999296681`，
  在numerical_comparison层按预期拒绝，不能计为正确候选通过。

后端为upstream SEAL HEVM CPU，不是Poseidon GPU。
SEAL4.0.0、N32768、14×60-bit模数、tc128，未修改profile、reference或容差，不执行bootstrap。

批次：`/home/lhy/poseidon-work/results/native-starred-goldens-9hghp1ci/report.json`。
SHA256：`3cd723b2c20a3e3d664427dda5b8a1796eb5105625917266478e7aaee88eb12a`。

`test_native_starred_evidence.py` 独立复核响应/trace payload一致性、冻结输入哈希、
重算的类型检查、真实native调用位置、编译产物哈希、独立明文公式和解密数组。
本轮综合回归137项通过，0失败、0跳过，包含已有数组门禁、旧v22和11项付费函数证据。
新特性真实Agent生成尚未测试，不混入这些回归计数。

本批9例临时密钥清理均完成，删除6,110,820,090字节（约5.69GiB）；原密钥不可恢复。
程序、IR、HEVM/CST、输入、reference、解密结果与日志保留；WSL剩余约189.10GiB。

## 复现与剩余目标

```bash
cd '/mnt/d/Code Space/Poseidon'
# 真实人工密态测试；不调用模型API
timeout -k 5s 3050s python3 scripts/baseline/run_native_starred_goldens.py
```

新增能力不等于模型已学会它；随后仍需要单独的真实Agent生成证据。
上一轮10项数组付费范围询问尚未收到具体答复，本轮没有借自动goal继续消息启动收费请求。
可继续推进的实现缺口包括原生容器操作与旧构造的组合、未验证的高层helper、真实bootstrap等。
不把单个契约支持或有限测试替代“完整DSL语义”目标。

本地修改包括typed core、请求版本/CLI、tracing、batch传递、8正例1反例及证据测试。
没有修改上游C++/Python frontend实现、安装依赖、修改.env或改变安全参数。
保持当前分支及用户已有修改，不commit/push/创建PR。
建议未来将星号调用支持、回归和报告作为一个独立本地提交，须先得到用户提交授权。
