# Bootstrap 容器前端：真实 IR 数据流修复

## 结论与范围

本轮修复了上游 Python 前端 `unaryFactory` 丢弃容器元素新表达式的问题。
真实 Hecate tracing 证明：修复前 list/tuple/一维对象数组最终输出的 Earth IR
没有 bootstrap；修复后各输入元素的 bootstrap 结果确实位于函数返回依赖路径上。
二维和零维对象数组原先报错，现在也能正确 trace 并通过 Earth IR 验证。

**这不是 bootstrap 密态算法的实现或执行验收。**
没有运行 HEVM bootstrap，没有生成FHE密钥，没有解密重加密，没有付费API调用。
Agent 的 bootstrap 仍被拒绝，语义清单保持 `forbidden_backend_gap`。
未降低安全参数、取消验证或将模拟当作FHE。完整DSL目标未完成。

## 系统位置与根因

位置：`third_party/dacapo/python/hecate/hecate/expr.py: unaryFactory`。
上游：Hecate/应用 helper 调用 `hc.bootstrap(value)`。
下游：Python表达式对象持有C ABI value ID；`createUnary(opcode=0)`创建Earth bootstrap。

原来容器分支只给循环局部变量 `tt` 赋新 Expr，最终返回的原容器仍持有旧Expr。
因此即便C ABI曾创建bootstrap，返回值也不使用它；保存后的IR中这些死节点消失。
这不是密态数值误差、SEAL或GPU问题，而是frontend构造数据流错误。

最小示例：

```python
value = np.array([x, y], dtype=object)
result = hc.bootstrap(value)
return result[0] + result[1]
```

修复前：`return add(x,y)`；修复后：`return add(bootstrap(x),bootstrap(y))`。
只计bootstrap操作个数不足以验证；诊断沿返回SSA值反向追踪依赖，要求操作实际可达。
诊断只解析本脚本固定的单函数Earth样例，不作为通用MLIR分析器。

## 容器契约与局部修复

| 输入 | 返回/写回行为 |
|---|---|
| Expr | 返回新Expr，原Expr对象不修改 |
| list | 先验证所有单元，再生成新Expr并写回同一list，保持容器身份 |
| tuple | 返回含新Expr的tuple，原tuple不修改 |
| 可写dtype=object ndarray | 按flat逻辑顺序替换单元，保持shape和容器身份，包括0-D、矩阵和非连续view |
| 空的受支持容器 | 不凭空创建bootstrap操作 |
| 非Expr单元、嵌套list、数值ndarray、只读数组或其他容器 | 在生成IR及修改存储之前明确拒绝 |

这里的容器储存的是表达式对象，不是一个密文内部的slot向量。
list/array的别名会看到替换后的单元；单独保存的原Expr仍保持旧value ID。
这延续了原容器路径返回self的身份约定，并修复其没有写回结果的错误。

## 真实前端实验

每个样例独立进程加载实际`expr.py`和现有`libHecateFrontend.so`，
保存Earth IR，再运行实际`hecate-opt --verify-each`进行IR校验。
使用现有固定profile；未执行Earth到HEVM的完整优化或bootstrap后端。
IR中的bootstrap targetLevel仍是前端占位属性，不能视为实际可执行参数安排。

| 形式 | 修复前 | 修复后返回路径上的bootstrap数 |
|---|---|---:|
| scalar | 通过 | 1 |
| list | trace成功但bootstrap消失 | 2 |
| tuple | trace成功但bootstrap消失 | 2 |
| 1-D ndarray | trace成功但bootstrap消失 | 2 |
| matrix ndarray | AttributeError：数组行没有obj属性 | 2 |
| 0-D ndarray | TypeError：无法迭代零维数组 | 1 |

修复前报告：
`/home/lhy/poseidon-work/results/bootstrap-frontend-4dttyrsw/report.json`

SHA256：`5cc7490cc85c6a1f9b7584d2ea9aeac7a83b659fb2abb67f689dd1f3c5138e97`。

修复后报告：
`/home/lhy/poseidon-work/results/bootstrap-frontend-cjsowzax/report.json`

SHA256：`2603fd11f6f713bb35047ef4ba75402e080b6b2468f53efc21b7606173d87894`。

## 兼容性与失败记录

- `expr.py`中已有的增强赋值操作数顺序修复保留。
- 重新运行真实加法/减法/乘法普通写法与增强赋值的六次trace+编译；
  三对HEVM/CST均逐字节一致。
- 证据：`/home/lhy/poseidon-work/results/frontend-augmented-7zici9lg/report.json`，
  SHA256 `2f7e488bdaf2471f914cc9f6d049b4771cb3f0033863a32d7e2c9f4c2b6531bb`。
- 最初诊断器漏了通用MLIR打印格式的引号，并且未给hecate-opt传入其要求的
  CKKS配置，导致诊断解析失败和编译器配置读取失败；两者已在测试工具中修正。
  初始trace/错误日志保留于`bootstrap-frontend-cczx93um`，不计为bootstrap修复成功证据。
- 后续诊断对子进程禁用core dump，避免失败生成大体积中间文件。
- 旧付费Agent结果未覆盖，当前v22批次的离线审计仍通过；没有新付费调用。
- 最终相关回归95项：92通过、3项历史密态证据测试因未启用对应环境变量而跳过，
  0失败。新增bootstrap前端的真实前后IR证据及增强赋值编译证据均显式启用；
  跳过不算通过。

## 文件与复验

- 本地上游修复：`third_party/dacapo/python/hecate/hecate/expr.py`。
- 独立补丁：`scripts/baseline/patches/dacapo-bootstrap-containers.patch`，
  只含本次bootstrap hunk，不混入原有增强赋值修改；reverse --check已通过。
- 诊断：`scripts/baseline/probe_bootstrap_frontend.py`。
- 单元及真实证据回归：`scripts/baseline/test_bootstrap_frontend.py`。
- 语义清单只增加前端证据，仍拒绝Agent/bootstrap密态执行。

```bash
cd '/mnt/d/Code Space/Poseidon'
timeout -k 3s 450s python3 scripts/baseline/probe_bootstrap_frontend.py
timeout -k 3s 350s python3 scripts/baseline/probe_frontend_augmented.py
```

没有安装依赖、sudo、改变.env、提交或推送；分支和固定gitlink不变。
`einops==0.7.0`安装仍待批准，与本轮修复无依赖关系。
下一步仍需分别验证完整函数调用、上游高层helper和真正的bootstrap执行后端，
不能把此次前端修复当作它们已经完成。
