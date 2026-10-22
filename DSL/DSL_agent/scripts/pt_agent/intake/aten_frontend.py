"""Real-tensor ATen tracing into the static core vocabulary."""

import copy
import operator

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import ShapeProp

from . import decompose
from .graph import Graph, TensorInfo
from .torch_frontend import IntakeError
from .torch_mapping import Mapper, Unsupported, arg, axis, pair


CONTROL_FLOW_REASON = "data-dependent control flow: traced graph differs from module on other inputs"
_ACTS = {f"aten.{name}.default": name for name in
         ("relu", "silu", "sigmoid", "tanh", "hardswish", "exp")}
_IDENTITIES = {f"aten.{name}.default" for name in
               ("clone", "alias", "detach", "contiguous", "lift_fresh_copy")}
_BINARIES = {"aten.add.Tensor": "add", "aten.add_.Tensor": "add",
             "aten.sub.Tensor": "sub", "aten.mul.Tensor": "mul"}


def _trace_error(module, reason):
    return IntakeError([dict(fx_node="<trace>", op="trace", target=type(module).__name__, reason=reason)])


def _different(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
                or bool(torch.any(torch.abs(actual-expected) > 1e-9*(1+torch.abs(expected)))))
    if isinstance(expected, (tuple, list)):
        return (not isinstance(actual, type(expected)) or len(actual) != len(expected)
                or any(_different(a, b) for a, b in zip(actual, expected)))
    if isinstance(expected, dict):
        return (not isinstance(actual, dict) or actual.keys() != expected.keys()
                or any(_different(actual[key], expected[key]) for key in expected))
    return actual != expected


class _AtenMapper(Mapper):
    def __init__(self, graph, module, names):
        super().__init__(graph, module)
        self.parameter_names = names
        self.absorbed = set()
        for node in module.graph.nodes:
            if str(node.target) == "aten.t.default" and node.args[0].op == "get_attr":
                if node.users and all(self.linear_weight(user) is node.args[0] for user in node.users):
                    self.absorbed.add(node)

    @staticmethod
    def linear_weight(node):
        target = str(node.target)
        if target not in ("aten.addmm.default", "aten.mm.default"):
            return None
        value = node.args[2 if target == "aten.addmm.default" else 1]
        if (isinstance(value, torch.fx.Node) and str(value.target) == "aten.t.default"
                and isinstance(value.args[0], torch.fx.Node) and value.args[0].op == "get_attr"):
            return value.args[0]
        return None

    def tuple_result(self, node, op, attrs):
        for user in node.users:
            if user.target is not operator.getitem or user.args[1] not in (0, 1, 2):
                raise Unsupported("tuple result requires getitem[0]")
            if user.args[1] != 0 and user.users:
                raise Unsupported("only getitem[0] may have consumers")
        first = next((user.name for user in node.users if user.args[1] == 0), node.name+".0")
        shape = tuple(int(dim) for dim in node.meta["tensor_meta"][0].shape)
        self.emit(node, op, [self.activation(node.args[0])], attrs, names=[first], shapes=[shape])
        self.aliases[node] = [first, node.name+".1", node.name+".2"]

    def map(self, node):
        if node.op == "get_attr":
            value = self.module
            for part in node.target.split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor):
                raise Unsupported("get_attr must hold a tensor")
            name = self.parameter_names.get(value.data_ptr(), "const."+node.target)
            self.aliases[node] = self.weight(name, value)
            return
        if node in self.absorbed or not node.users:
            return
        if node.op != "call_function":
            raise Unsupported("aten op not in mapping table")
        if node.target is operator.getitem:
            self.getitem(node)
            return
        target = str(node.target)
        x = node.args[0] if node.args else None
        if target == "aten.convolution.default":
            if arg(node, 6, "transposed"):
                raise Unsupported("transposed convolution")
            attrs = dict(weight=self.get_weight(node.args[1]), stride=pair(node.args[3]),
                         padding=pair(node.args[4]), dilation=pair(node.args[5]), groups=int(node.args[8]))
            if node.args[2] is not None:
                attrs['bias'] = self.get_weight(node.args[2])
            self.emit(node, "conv2d", [self.activation(x)], attrs)
        elif target in ("aten.native_batch_norm.default", "aten._native_batch_norm_legit_no_training.default"):
            if target == "aten.native_batch_norm.default" and node.args[5]:
                raise Unsupported("BatchNorm requires training=False")
            attrs = dict(weight=self.get_weight(node.args[1]), bias=self.get_weight(node.args[2]),
                         mean=self.get_weight(node.args[3]), var=self.get_weight(node.args[4]),
                         eps=float(node.args[7 if target == "aten.native_batch_norm.default" else 6]))
            self.tuple_result(node, "batchnorm", attrs)
        elif target in _ACTS or target == "aten.gelu.default":
            fn = _ACTS.get(target, "gelu_tanh" if node.kwargs.get("approximate", "none") == "tanh" else "gelu")
            self.emit(node, "act", [self.activation(x)], dict(fn=fn))
        elif target in _BINARIES:
            if arg(node, 2, "alpha", 1) != 1:
                raise Unsupported("add/sub requires alpha=1")
            self.binary(node, _BINARIES[target], x, node.args[1])
        elif target == "aten.rsub.Scalar":
            if arg(node, 2, "alpha", 1) != 1:
                raise Unsupported("add/sub requires alpha=1")
            self.emit(node, "sub", [self.activation(x)], dict(reverse=True, scalar=float(node.args[1])))
        elif target == "aten.div.Tensor":
            self.divide(node, x, node.args[1])
        elif target == "aten.neg.default":
            self.emit(node, "mul", [self.activation(x)], dict(scalar=-1.0))
        elif target in ("aten.addmm.default", "aten.mm.default"):
            weight = self.linear_weight(node)
            if weight is not None:
                if node.kwargs.get('alpha', 1) != 1 or node.kwargs.get('beta', 1) != 1:
                    raise Unsupported("linear addmm requires alpha=beta=1")
                attrs = dict(weight=self.get_weight(weight))
                if target == "aten.addmm.default":
                    attrs['bias'] = self.get_weight(node.args[0])
                self.emit(node, "linear", [self.activation(node.args[1] if target == "aten.addmm.default" else x)], attrs)
            elif target == "aten.mm.default":
                self.emit(node, "matmul", [self.activation(x), self.activation(node.args[1])], {})
            else:
                raise Unsupported("linear addmm requires t(weight get_attr)")
        elif target == "aten.bmm.default":
            self.emit(node, "matmul", [self.activation(x), self.activation(node.args[1])], {})
        elif target == "aten.t.default":
            self.emit(node, "permute", [self.ref(x)], dict(dims=(1, 0)))
        elif target == "aten.expand.default":
            if tuple(node.meta['tensor_meta'].shape) != self.shape(x):
                raise Unsupported("broadcast expand")
            self.emit(node, "identity", [self.activation(x)], {})
        elif target in ("aten.view.default", "aten._unsafe_view.default", "aten.reshape.default"):
            self.emit(node, "reshape", [self.activation(x)], dict(shape=tuple(int(v) for v in node.meta['tensor_meta'].shape)))
        elif target == "aten.permute.default":
            self.emit(node, "permute", [self.activation(x)], dict(dims=tuple(node.args[1])))
        elif target == "aten.transpose.int":
            dims = list(range(len(self.shape(x))))
            first, second = axis(node.args[1], len(dims)), axis(node.args[2], len(dims))
            dims[first], dims[second] = dims[second], dims[first]
            self.emit(node, "permute", [self.activation(x)], dict(dims=tuple(dims)))
        elif target == "aten.slice.Tensor":
            dim = axis(node.args[1], len(self.shape(x)))
            length = self.shape(x)[dim]
            start, end, step = slice(arg(node, 2, 'start', 0), arg(node, 3, 'end', length),
                                     arg(node, 4, 'step', 1)).indices(length)
            if (start, end, step) == (0, length, 1):
                self.emit(node, "identity", [self.activation(x)], {})
            else:
                self.emit(node, "slice", [self.activation(x)], dict(axis=dim, start=start, end=end, step=step))
        elif target == "aten.constant_pad_nd.default":
            if arg(node, 2, 'value', 0) != 0:
                raise Unsupported("pad requires value=0")
            self.emit(node, "pad", [self.activation(x)], dict(pads=tuple(node.args[1])))
        elif target == "aten.avg_pool2d.default":
            if (arg(node, 4, 'ceil_mode', False) or not arg(node, 5, 'count_include_pad', True)
                    or arg(node, 6, 'divisor_override', None) is not None):
                raise Unsupported("avg_pool2d requires ceil_mode=False, count_include_pad=True and no divisor_override")
            kernel = pair(node.args[1])
            attrs = dict(kernel=kernel, stride=pair(arg(node, 2, 'stride', []) or kernel),
                         padding=pair(arg(node, 3, 'padding', [0, 0])))
            self.emit(node, "avgpool", [self.activation(x)], attrs)
        elif target == "aten._adaptive_avg_pool2d.default":
            self.emit(node, "adaptive_avgpool", [self.activation(x)], dict(output_size=pair(node.args[1])))
        elif target in ("aten.mean.dim", "aten.sum.dim_IntList"):
            dims = node.args[1]
            dims = (dims,) if isinstance(dims, int) else tuple(dims)
            self.emit(node, "mean" if target == "aten.mean.dim" else "sum", [self.activation(x)],
                      dict(axes=tuple(axis(dim, len(self.shape(x))) for dim in dims),
                           keepdim=bool(arg(node, 2, 'keepdim', False))))
        elif target == "aten.cat.default":
            tensors = node.args[0]
            self.emit(node, "concat", [self.activation(tensor) for tensor in tensors],
                      dict(axis=axis(arg(node, 1, 'dim', 0), len(self.shape(tensors[0])))))
        elif target in ("aten.split.Tensor", "aten.split_with_sizes.default"):
            dim = axis(arg(node, 2, 'dim', 0), len(self.shape(x)))
            metas = node.meta['tensor_meta']
            names = [next((user.name for user in node.users if user.target is operator.getitem
                           and user.args[1] == index), f'{node.name}.{index}') for index in range(len(metas))]
            self.emit(node, "split", [self.activation(x)],
                      dict(axis=dim, sizes=tuple(int(meta.shape[dim]) for meta in metas)),
                      names=names, shapes=[tuple(int(v) for v in meta.shape) for meta in metas])
            self.aliases[node] = names
        elif target == "aten._softmax.default":
            if node.args[2]:
                raise Unsupported("softmax requires half_to_float=False")
            self.emit(node, "softmax", [self.activation(x)], dict(axis=axis(node.args[1], len(self.shape(x)))))
        elif target == "aten.native_layer_norm.default":
            attrs = dict(normalized_shape=tuple(node.args[1]), weight=self.get_weight(node.args[2]),
                         bias=self.get_weight(node.args[3]), eps=float(node.args[4]))
            self.tuple_result(node, "layernorm", attrs)
        elif target in _IDENTITIES:
            self.emit(node, "identity", [self.ref(x)], {})
        elif target == "aten.max_pool2d_with_indices.default":
            raise Unsupported("max pooling has no polynomial lowering in v1")
        else:
            raise Unsupported("aten op not in mapping table")


def import_module_aten(module, example_inputs: tuple, name: str) -> Graph:
    m = copy.deepcopy(module).eval().double()
    inputs = tuple(value.to(dtype=torch.float64) for value in example_inputs)
    try:
        with torch.no_grad():
            gm = make_fx(m, tracing_mode="real")(*inputs)
            generator = torch.Generator().manual_seed(20261022)
            for _ in range(3):
                samples = tuple(torch.randn(value.shape, generator=generator, dtype=torch.float64)
                                for value in inputs)
                if _different(gm(*samples), m(*samples)):
                    raise _trace_error(m, CONTROL_FLOW_REASON)
            ShapeProp(gm).propagate(*inputs)
    except IntakeError:
        raise
    except Exception as exc:
        reason = CONTROL_FLOW_REASON if 'data-dependent control flow' in str(exc) else str(exc)
        raise _trace_error(m, reason) from exc
    names = {}
    for key, value in list(m.named_parameters())+list(m.named_buffers()):
        names.setdefault(value.data_ptr(), key)
    graph = Graph(name, [], {}, {}, [], [])
    mapper = _AtenMapper(graph, gm, names)
    errors = []
    output = None
    for node in gm.graph.nodes:
        meta = node.meta.get('tensor_meta')
        if hasattr(meta, 'shape'):
            graph.tensors[node.name] = TensorInfo(node.name, tuple(int(v) for v in meta.shape),
                                                  'input' if node.op == 'placeholder' else 'act')
        if node.op == 'placeholder':
            graph.inputs.append(node.name)
        elif node.op == 'output':
            output = node
        else:
            try:
                mapper.map(node)
            except Unsupported as exc:
                errors.append(dict(fx_node=node.name, op=node.op, target=str(node.target), reason=str(exc)))
    if errors:
        raise IntakeError(errors)

    def outputs(value):
        if isinstance(value, torch.fx.Node):
            resolved = mapper.aliases.get(value, value.name)
            return [resolved] if isinstance(resolved, str) else list(resolved)
        if isinstance(value, (tuple, list)):
            return [name for item in value for name in outputs(item)]
        raise IntakeError([dict(fx_node=output.name, op='output', target=str(output.target),
                                reason='output must contain tensors')])

    graph.outputs = outputs(output.args[0])
    return decompose.run(graph)
