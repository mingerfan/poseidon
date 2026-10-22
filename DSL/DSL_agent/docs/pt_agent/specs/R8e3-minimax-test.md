# R8e3：把 R8e 第 4 项测试改成确定性的，然后跑完 R8e

## 0. 诊断（Claude 于 2026-10-21 复核）

`test_04_minimax_keeps_degrees_and_symmetry` 要检验的是**结构性质**：精调不改变 minimax_comp 的方法和阶数，定义域保持对称，并且只有 domain 类的改动。它却依赖真实验证集上的统计量：在 n_val=50 的 ctx 上，cnn_plain 的一致率为 49/50 = 98%，低于 99%，所以 tune 原样返回，feasible=False。

R8e spec 没有规定这一项用多少张图，这是 spec 的疏漏，实现没有问题。改成与第 3 项相同的做法：用假的 validate、estimate 和 _exact_check 隔离统计量，结果确定，耗时也从约 8 分钟降到 1 秒以内。真实数据上的 minimax 精调效果，由 §4 的真实运行（n_val=1000）检验。

## 1. 修改：只改 `pt_agent/tests/test_r8e_tuner.py` 中的 test_04（整个函数体替换）

```python
def test_04_minimax_keeps_degrees_and_symmetry(self):
    source = '@fhe.program(slots=16384)\ndef net(v: Cipher[1, 2]):\n    x = pack(v, layout="flat")\n'
    for _ in range(3):
        source += '    x = act(x, fn="relu", method="minimax_comp", degrees=(15, 15, 27), domain=(-8.0, 8.0))\n'
    source += '    return x\n'
    original = parse(source)
    ctx = SimpleNamespace(weights={}, profile='qwen48')

    def fake_validate(program, context, n=None):
        acts = [op for op in program.ops if op.kind == 'act']
        total = 200 if n is not None else 1000
        return dict(agree=total, total=total, feasible=True, diverged=False, max_u=.8,
                    rows=[dict(obs_min=-3.0, obs_max=2.0) for _ in acts])

    def fake_cost(program, weights, **kwargs):
        acts = [op for op in program.ops if op.kind == 'act']
        return dict(latency_pred_us=100.0, boots_pred=len(acts),
                    hotspots=[dict(line=op.lineno, call_lines=list(op.call_lines), latency_share=.1) for op in acts])

    with patch('pt_agent.planners.tuner.validate', side_effect=fake_validate), \
            patch('pt_agent.planners.tuner.estimate', side_effect=fake_cost), \
            patch('pt_agent.planners.tuner._exact_check', return_value={'ok': True}):
        result = tune(original, ctx)
    self.assertTrue(result.feasible)
    self.assertEqual(result.coarse_evals, 0)          # 没有 cheb 激活，不做降阶尝试
    acts = [op for op in result.program.ops if op.kind == 'act']
    self.assertEqual(len(acts), 3)
    for op in acts:
        self.assertEqual(op.attrs['method'], 'minimax_comp')
        self.assertEqual(tuple(op.attrs['degrees']), (15, 15, 27))
        self.assertIsNone(op.attrs['deg'])
        self.assertEqual(tuple(op.attrs['domain']), (-3.75, 3.75))   # max(|-3|, |2|) / TARGET_U(0.8)
    self.assertEqual(len(result.changes), 3)
    self.assertTrue(all(change['kind'] == 'domain' for change in result.changes))
```

如果文件顶部还没有导入 SimpleNamespace 和 patch，按第 3 项的写法补上。不再使用的 R8D 路径常量、get、build_context 这些导入，如果没有其他测试使用，可以删除。

## 2. 然后

按 R8e 原 spec 继续：§3 全部离线测试、R1–R8c 回归、§4 的 4 个模型真实运行（LLM 调用合计不超过 24 次）、§5 汇报。STATUS.md：R8e 改为完成（或如实记录真实运行的结果），新增 R8e3 一行。

## 3. 不要做

不修改 tuner、agent 等实现文件，也不修改其他测试。
