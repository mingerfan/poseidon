# R8e2：修正 R8e 新增测试夹具的漏写，然后跑完 R8e

## 0. 诊断（Claude 于 2026-10-21 复核）

`tests/test_r8e_tuner.py::test_03_prefix_rollback` 生成的源码中，`act(...)` 漏写了必填属性 `method`，所以 parse 时报 `E_ATTR_MISSING`。这是夹具本身的笔误，与实现无关。补上之后，测试的逻辑与 R8e §3.3 一致：
- 前 2 个激活从 127 降到 63，后 6 个从 63 降到 31，粗筛全部通过；
- 全量验证要求所有 deg ≥ 63，所以前缀二分保留前 2 项、撤销 6 项。

另外，AGENTS.md 已经新增一条例外：本轮自己新写的代码或测试中的实现层面笔误，可以直接修正后继续，并在汇报中列出。以后同类问题不需要停下等 spec。

## 1. 修改

只改 test_03_prefix_rollback 中的这一行：

```python
source += f'    x = act(x, fn="relu", method="cheb", deg={127 if index < 2 else 63}, domain=(-8.,8.))\n'
```

## 2. 然后

按 R8e 原 spec 继续执行：
1. §3 的全部离线测试和 R1–R8c 的回归，全部通过；
2. §4 的真实运行：4 个开发模型，LLM 调用合计不超过 24 次；
3. §5 的汇报。

STATUS.md：R8e 改为完成（或者如实记录真实运行的结果），新增 R8e2 一行。
