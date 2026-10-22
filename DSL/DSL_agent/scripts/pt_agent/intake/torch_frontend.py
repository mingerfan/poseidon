"""FX symbolic tracing and shape propagation into R3's static core graph."""

import contextlib
import copy

import torch
from torch.nn import functional as F
from torch.fx.passes.shape_prop import ShapeProp

from . import decompose
from .graph import Graph, TensorInfo
from .torch_mapping import Mapper, Unsupported


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


class IntakeError(Exception):
    def __init__(self, entries: list[dict]):
        self.entries = entries
        super().__init__("\n".join(f"{entry['fx_node']} [{entry['op']}] {entry['target']}: {entry['reason']}" for entry in entries))


def import_module(module: torch.nn.Module, example_inputs: tuple, name: str) -> Graph:
    module = copy.deepcopy(module).eval().double()
    example_inputs = tuple(value.to(dtype=torch.float64) for value in example_inputs)
    try:
        tracer = torch.fx.Tracer()
        with _proxy_aware_functionals():
            graph = tracer.trace(module)
        gm = torch.fx.GraphModule(tracer.root, graph)
    except (torch.fx.proxy.TraceError, TypeError) as exc:
        raise IntakeError([dict(fx_node="<trace>", op="trace", target=type(module).__name__,
                                reason="dynamic control flow or untraceable Python: " + str(exc))]) from exc
    with torch.no_grad():
        ShapeProp(gm).propagate(*example_inputs)
    graph = Graph(name, [], {}, {}, [], [])
    mapper = Mapper(graph, gm)
    errors = []
    output = None
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if hasattr(meta, "shape"):
            graph.tensors[node.name] = TensorInfo(node.name, tuple(int(v) for v in meta.shape),
                                                  "input" if node.op == "placeholder" else "act")
        if node.op == "placeholder":
            graph.inputs.append(node.name)
        elif node.op == "output":
            output = node
        else:
            try:
                mapper.map(node)
            except Unsupported as exc:
                errors.append(dict(fx_node=node.name, op=node.op, target=repr(node.target), reason=str(exc)))
    if errors:
        raise IntakeError(errors)

    def outputs(value):
        if isinstance(value, torch.fx.Node):
            result = mapper.aliases.get(value, value.name)
            return [result] if isinstance(result, str) else list(result)
        if isinstance(value, (tuple, list)):
            return [name for item in value for name in outputs(item)]
        raise IntakeError([dict(fx_node=output.name, op="output", target=repr(output.target), reason="output must contain tensors")])

    graph.outputs = outputs(output.args[0])
    return decompose.run(graph)
