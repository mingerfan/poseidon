# R8d2：修正 R6 旧测试的过时前提，然后跑完 R8d

## 0. 诊断（Claude 于 2026-10-21 复核）

`tests/test_r6_lower.py::test_errors_include_op_id` 中的 `'act'` 子例，用 `examples/resnet20.ptdsl`（其中的激活都是 `minimax_comp`）来构造一次"act 处的 lowering 失败"。这个前提只在 R8d 之前成立：当时 minimax_comp 还没有实现。

这项测试真正要检验的是：**lowering 在 act 处失败时，LoweringError 带有正确的 op_id、lineno，并且消息中含有 "op N"**。R8d 的实现是正确的，需要改的是这个子例的构造方式。实现文件一律不改。

## 1. 修改：只改 `pt_agent/tests/test_r6_lower.py` 中的 test_errors_include_op_id

把 programs 列表中的第一项改为：先解析 resnet20.ptdsl，再把第一个 act op 的定义域改成不对称，触发 `act_minimax` 的 KernelError，由 lower 转成 LoweringError。

```python
act_program = parse((EXAMPLES/'resnet20.ptdsl').read_text())
first_act = next(op for op in act_program.ops if op.kind == 'act')
first_act.attrs['domain'] = (-1.0, 2.0)     # minimax_comp requires a symmetric domain
programs = [
    (act_program, self.graph.weights, 'act'),
    ...其余 3 项不变...
]
```

如果 `op.attrs` 是不可变映射，就用 `dataclasses.replace` 生成新的 op，替换进 `act_program.ops` 中的对应位置。其余断言不变：
- `error.exception.op_id == op.id`；
- `error.exception.lineno == op.lineno`；
- `f'op {op.id}'` 出现在错误消息中。

另外，在 'act' 子例中增加一条断言：`'symmetric' in str(error.exception)`。

## 2. 然后

1. R1–R8d 的测试全部通过。
2. 按 R8d §4 运行 `experiments/r8d_methods.py`，汇报内容同 R8d §5。
3. STATUS.md：R8d 改为完成，新增 R8d2 一行，删除"阻塞"栏中的对应条目。

## 3. 不要做

不修改 kernels、lower、ops_spec 等实现文件，也不修改 examples/ 下的程序。
