# R8a2：跟踪时临时包装 functional 池化函数，跑完 R8a

## 0. 问题与已验证的修复（Claude 于 2026-10-21 实测）

- 在 PyTorch 2.0.1 中，`torch.fx.symbolic_trace` 跟踪到 `F.avg_pool2d(out, out.size()[3])` 时，`out.size()[3]` 是一个 Proxy，而 C 实现的 `avg_pool2d` 会立即检查参数类型，因此报 `TypeError: … must be tuple of ints, not Proxy`。这是 R8a spec 遗漏的前置问题，R8a 的实现本身没有错。
- `Tracer(autowrap_functions=…)` 无效，因为它只替换以全局名字直接调用的函数，而 Dacapo 代码是通过 `F.avg_pool2d` 这种属性方式调用的。
- **已验证有效的做法**：跟踪期间，临时把 `torch.nn.functional` 上的几个池化函数替换为包装函数。参数中有 Proxy 时，包装函数调用 `proxy.tracer.create_proxy("call_function", 原函数, args, kwargs)`，把这次调用记成一个图节点，节点的 target 仍是原函数；否则照常调用原函数。跟踪结束后（包括出错时），在 finally 中还原。
- 实测对 Dacapo 的 ResNet20-SiLU 有效：跟踪后的 GraphModule 与直接运行的输出误差为 0.0；图中出现 `size → getitem → avg_pool2d(…, getitem)` 和 `view(avg_pool2d, size(…, 0), -1)`；ShapeProp 正常，函数也已还原。

## 1. 修改：只改 `pt_agent/intake/torch_frontend.py`

```python
PROXY_AWARE_FUNCTIONALS = ("avg_pool2d", "max_pool2d", "adaptive_avg_pool2d")

@contextlib.contextmanager
def _proxy_aware_functionals():
    saved = {n: getattr(F, n) for n in PROXY_AWARE_FUNCTIONALS}
    def make(orig):
        def wrapper(*args, **kwargs):
            proxies = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.fx.Proxy)]
            if proxies:
                return proxies[0].tracer.create_proxy("call_function", orig, args, kwargs)
            return orig(*args, **kwargs)
        return wrapper
    try:
        for n, f in saved.items():
            setattr(F, n, make(f))
        yield
    finally:
        for n, f in saved.items():
            setattr(F, n, f)
```

`import_module` 中原来的 `gm = torch.fx.symbolic_trace(module)` 改为：

```python
tracer = torch.fx.Tracer()
with _proxy_aware_functionals():
    graph = tracer.trace(module)
gm = torch.fx.GraphModule(tracer.root, graph)
```

原有的 TraceError、TypeError 到 IntakeError 的转换保持不变。

## 2. 测试

在 `tests/test_r8a_models.py` 中：

1. 新增断言：调用 `import_module` 之后，`torch.nn.functional.avg_pool2d` 仍然是原来的函数对象（导入前先取 `id` 保存下来，事后比较）。即使 import_module 抛出异常，函数也必须已经还原（用一个会触发 IntakeError 的模块验证）。
2. R8a §8 中原先失败的 resnet20_silu 导入测试和 size/shape 测试，现在应全部通过。
3. R1–R7c 的测试全部通过。

## 3. 然后

按 R8a spec §7 完整运行 `r8a_dev`：SiLU 的归一化已经选定为 ImageNet（训练集验证子集上准确率 100%），然后训练两个小模型，最后对 4 个模型跑规则基线。汇报内容同 R8a §7/§9。STATUS.md 中 R8a 改为完成，新增 R8a2 一行，并删除"阻塞"栏中的对应条目。
