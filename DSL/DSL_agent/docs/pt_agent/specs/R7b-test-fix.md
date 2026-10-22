# R7b：修正 wrong_stride 测试的判定条件，然后完成 R7 §8 的真实运行

R7 被阻塞，原因是 R7 spec §7 第 2 条写的预期不完整，**实现没有问题**。

Claude 于 2026-10-21 复现了这个用例：把第一个 conv 的 stride 改成 (2,2) 之后，typecheck 仍然通过（后续形状自洽），但规则规划器写死的 `slice(..., end=32, ...)` 不再等于新的高度 16。lowering 因此判定"不是合法的下采样链"，evaluate 返回：

```json
{"stage": "lower", "feasible": false, "exact": null, "error": "op 21 line 24: slice requires the downsample slice/slice/pad chain"}
```

这是正确的反馈：带有 op 编号、行号和原因。

## 1. 修改：只改 `pt_agent/tests/test_r7_agent.py` 中的 wrong_stride 用例

判定条件改为：

```python
self.assertFalse(fb["feasible"])
self.assertTrue(fb["stage"] in ("typecheck", "lower") or (fb["exact"] is not None and not fb["exact"]["ok"]))
```

stage 为 "lower" 时，还要断言 `"line" in fb["error"]`。

不修改任何实现文件。

## 2. 然后

1. R7 与 R1–R6b2 的测试全部通过。
2. 按 R7 spec §8 的顺序执行：先运行 `python3 -m pt_agent.llm --smoke`，成功后再运行 agent 的完整流程（`--stages plan,check,mock,count,emit,compile,test --final`）。
3. 汇报内容同 R7 §8。STATUS.md 中 R7 改为完成（或者如实标记 agent 不可行），新增 R7b 一行，并删除"阻塞"栏中的对应条目。
