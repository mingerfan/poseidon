"""Only the module, function and method mappings enumerated in R3."""

import operator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .graph import Node, TensorInfo


class Unsupported(ValueError):
    pass


_MISSING = object()


def arg(node, index, name, default=_MISSING):
    if len(node.args) > index:
        return node.args[index]
    if name in node.kwargs:
        return node.kwargs[name]
    if default is _MISSING:
        raise Unsupported(f"missing argument {name}")
    return default


def pair(value):
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return tuple(value)
    raise Unsupported("expected a static integer pair")


def axis(value, rank):
    if not isinstance(value, int) or not -rank <= value < rank:
        raise Unsupported("axis must be a static valid integer")
    return value % rank


class Mapper:
    def __init__(self, graph, module):
        self.graph = graph
        self.module = module
        self.aliases = {}
        self.constant_counts = {}
        self.static = {}

    def resolve(self, value):
        if isinstance(value, torch.fx.Node):
            return self.static.get(value, value)
        if isinstance(value, tuple):
            return tuple(self.resolve(item) for item in value)
        if isinstance(value, list):
            return [self.resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self.resolve(item) for key, item in value.items()}
        return value

    def value(self, node, index, name, default=_MISSING):
        return self.resolve(arg(node, index, name, default))

    def ref(self, value):
        if isinstance(value, torch.fx.Node) and value in self.static:
            raise Unsupported("size value used as tensor")
        if not isinstance(value, torch.fx.Node):
            raise Unsupported("expected a tensor node")
        result = self.aliases.get(value, value.name)
        if not isinstance(result, str):
            raise Unsupported("tuple-valued tensor requires split getitem")
        return result

    def activation(self, value):
        name = self.ref(value)
        if name not in self.graph.tensors or self.graph.tensors[name].kind == "weight":
            raise Unsupported("expected an activation tensor")
        return name

    def shape(self, value):
        return self.graph.tensors[self.ref(value)].shape

    def weight(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.array(value, dtype=np.float64, copy=True)
        array.setflags(write=False)
        self.graph.weights[name] = array
        self.graph.tensors[name] = TensorInfo(name, tuple(array.shape), "weight")
        return name

    def constant(self, node, value):
        index = self.constant_counts.get(node.name, 0)
        self.constant_counts[node.name] = index + 1
        return self.weight(f"const.{node.name}.{index}", value)

    def parameter(self, path, name, value):
        return None if value is None else self.weight(f"{path}.{name}", value)

    def get_weight(self, value):
        if value is None:
            return None
        if isinstance(value, torch.fx.Node) and value in self.static:
            raise Unsupported("size value used as tensor")
        if not isinstance(value, torch.fx.Node) or value.op != "get_attr":
            raise Unsupported("functional weights must come from get_attr")
        name = self.ref(value)
        if name not in self.graph.weights:
            raise Unsupported("get_attr must contain a tensor weight")
        return name

    def emit(self, node, op, inputs, attrs, *, names=None, shapes=None):
        if names is None:
            names = [node.name]
        if shapes is None:
            shapes = [tuple(int(v) for v in node.meta["tensor_meta"].shape)]
        for name, shape in zip(names, shapes):
            self.graph.tensors[name] = TensorInfo(name, tuple(shape), "act")
        self.graph.nodes.append(Node(f"n{len(self.graph.nodes)}", op, list(inputs), list(names), attrs,
                                     f"{node.name}:{node.target!r}"))
        self.aliases[node] = names[0] if len(names) == 1 else list(names)

    def binary(self, node, op, left, right):
        def is_activation(value):
            return isinstance(value, torch.fx.Node) and self.ref(value) not in self.graph.weights

        la, ra = is_activation(left), is_activation(right)
        if la and ra:
            self.emit(node, op, [self.activation(left), self.activation(right)], {})
            return
        if not (la or ra):
            raise Unsupported("binary operation requires an activation")
        value, const = (left, right) if la else (right, left)
        if isinstance(const, (int, float)):
            attrs = dict(scalar=float(const))
        elif isinstance(const, torch.fx.Node) and self.ref(const) in self.graph.weights:
            attrs = dict(const=self.ref(const))
        elif isinstance(const, (torch.Tensor, np.ndarray)):
            attrs = dict(const=self.constant(node, const))
        else:
            raise Unsupported("constant operand must be a scalar or tensor")
        if op == "sub":
            attrs["reverse"] = not la
        self.emit(node, op, [self.activation(value)], attrs)

    def divide(self, node, numerator, denominator):
        if isinstance(denominator, torch.fx.Node):
            name = self.ref(denominator)
            if name not in self.graph.weights:
                raise Unsupported("activation tensor as divisor is unsupported")
            reciprocal = self.constant(node, 1.0 / self.graph.weights[name])
            attrs = dict(const=reciprocal)
        elif isinstance(denominator, (int, float)):
            attrs = dict(scalar=1.0 / float(denominator))
        elif isinstance(denominator, (torch.Tensor, np.ndarray)):
            value = denominator.detach().cpu().numpy() if isinstance(denominator, torch.Tensor) else denominator
            attrs = dict(const=self.constant(node, 1.0 / value))
        else:
            raise Unsupported("divisor must be a scalar or constant tensor")
        self.emit(node, "mul", [self.activation(numerator)], attrs)

    def module_node(self, node):
        module = self.module.get_submodule(node.target)
        typ = type(module)
        x = arg(node, 0, "input")
        source = self.activation(x)
        rank = len(self.shape(x))
        param = lambda name, value: self.parameter(node.target, name, value)
        if typ is nn.Conv2d:
            if module.padding_mode != "zeros" or isinstance(module.padding, str):
                raise Unsupported("Conv2d requires zero padding with an integer or tuple padding")
            op, attrs = "conv2d", dict(weight=param("weight", module.weight), bias=param("bias", module.bias),
                                       stride=pair(module.stride), padding=pair(module.padding),
                                       dilation=pair(module.dilation), groups=module.groups)
        elif typ is nn.Linear:
            op, attrs = "linear", dict(weight=param("weight", module.weight), bias=param("bias", module.bias))
        elif typ in (nn.BatchNorm2d, nn.BatchNorm1d):
            if not module.track_running_stats or module.training:
                raise Unsupported("BatchNorm requires running statistics and eval mode")
            weight = module.weight if module.weight is not None else np.ones(module.num_features)
            bias = module.bias if module.bias is not None else np.zeros(module.num_features)
            op, attrs = "batchnorm", dict(weight=param("weight", weight), bias=param("bias", bias),
                                          mean=param("running_mean", module.running_mean),
                                          var=param("running_var", module.running_var), eps=float(module.eps))
        elif typ in (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Hardswish):
            fn = {nn.ReLU: "relu", nn.GELU: "gelu", nn.SiLU: "silu", nn.Sigmoid: "sigmoid",
                  nn.Tanh: "tanh", nn.Hardswish: "hardswish"}[typ]
            if typ is nn.GELU and module.approximate == "tanh":
                fn = "gelu_tanh"
            op, attrs = "act", dict(fn=fn)
        elif typ is nn.AvgPool2d:
            if module.ceil_mode or not module.count_include_pad or module.divisor_override is not None:
                raise Unsupported("AvgPool2d requires ceil_mode=False, count_include_pad=True and no divisor_override")
            op, attrs = "avgpool", dict(kernel=pair(module.kernel_size), stride=pair(module.stride or module.kernel_size),
                                       padding=pair(module.padding))
        elif typ is nn.AdaptiveAvgPool2d:
            op, attrs = "adaptive_avgpool", dict(output_size=pair(module.output_size))
        elif typ is nn.Flatten:
            op, attrs = "flatten", dict(start_dim=axis(module.start_dim, rank), end_dim=axis(module.end_dim, rank))
        elif typ in (nn.Dropout, nn.Identity):
            op, attrs = "identity", {}
        elif typ is nn.Softmax:
            op, attrs = "softmax", dict(axis=axis(module.dim, rank))
        elif typ is nn.LayerNorm:
            op, attrs = "layernorm", dict(weight=param("weight", module.weight), bias=param("bias", module.bias),
                                          normalized_shape=tuple(module.normalized_shape), eps=float(module.eps))
        else:
            raise Unsupported(f"unsupported module type {typ.__name__}")
        self.emit(node, op, [source], attrs)

    def getitem(self, node):
        value, index = node.args
        if isinstance(value, torch.fx.Node) and value in self.static:
            self.static[node] = self.static[value][self.resolve(index)]
            return
        if value in self.aliases and isinstance(self.aliases[value], list):
            if not isinstance(index, int):
                raise Unsupported("split getitem requires an integer output index")
            self.aliases[node] = self.aliases[value][index]
            return
        source = self.activation(value)
        shape = list(self.shape(value))
        if not isinstance(index, tuple) or any(not isinstance(item, slice) and item is not Ellipsis for item in index):
            raise Unsupported("tensor getitem requires a tuple of slices or Ellipsis; integer indexing is unsupported")
        if sum(item is Ellipsis for item in index) > 1:
            raise Unsupported("getitem permits at most one Ellipsis")
        indices = []
        for item in index:
            if item is Ellipsis:
                indices.extend([slice(None)] * (len(shape)-len(index)+1))
            else:
                indices.append(item)
        indices.extend([slice(None)] * (len(shape)-len(indices)))
        slices = []
        for dim, item in enumerate(indices):
            start, end, step = item.indices(shape[dim])
            if (start, end, step) != (0, shape[dim], 1):
                slices.append((dim, start, end, step))
        if not slices:
            self.emit(node, "identity", [source], {})
        for position, (dim, start, end, step) in enumerate(slices):
            shape[dim] = len(range(start, end, step))
            name = node.name if position == len(slices)-1 else f"{node.name}.slice{position}"
            self.emit(node, "slice", [source], dict(axis=dim, start=start, end=end, step=step),
                      names=[name], shapes=[tuple(shape)])
            source = name

    def tensor_operation(self, node, operation, method=False):
        x = arg(node, 0, "input")
        source = self.activation(x)
        rank = len(self.shape(x))
        if operation in ("mean", "sum"):
            dims = self.value(node, 1, "dim", None)
            if dims is None:
                raise Unsupported(f"{operation} requires an explicit dim")
            dims = (dims,) if isinstance(dims, int) else tuple(dims)
            attrs = dict(axes=tuple(axis(dim, rank) for dim in dims), keepdim=bool(self.value(node, 2, "keepdim", False)))
        elif operation == "reshape":
            requested = self.resolve(node.args[1:] if method else arg(node, 1, "shape"))
            if method and len(requested) == 1 and isinstance(requested[0], (tuple, list)):
                requested = requested[0]
            if not isinstance(requested, (tuple, list)) or not all(isinstance(dim, int) for dim in requested):
                raise Unsupported("reshape requires static integer dimensions")
            attrs = dict(shape=tuple(int(v) for v in node.meta["tensor_meta"].shape))
        elif operation == "flatten":
            attrs = dict(start_dim=axis(self.value(node, 1, "start_dim", 0), rank),
                         end_dim=axis(self.value(node, 2, "end_dim", -1), rank))
        elif operation in ("permute", "transpose"):
            if operation == "transpose":
                first = axis(self.value(node, 1, "dim0"), rank)
                second = axis(self.value(node, 2, "dim1"), rank)
                dims = list(range(rank))
                dims[first], dims[second] = dims[second], dims[first]
                operation = "permute"
            else:
                dims = self.resolve(node.args[1:] if method else arg(node, 1, "dims"))
                if method and len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                    dims = dims[0]
            attrs = dict(dims=tuple(axis(dim, rank) for dim in dims))
        elif operation == "softmax":
            attrs = dict(axis=axis(self.value(node, 1, "dim", None), rank))
        elif operation in ("split", "chunk"):
            dim = axis(self.value(node, 2, "dim", 0), rank)
            metas = node.meta["tensor_meta"]
            sizes = tuple(int(meta.shape[dim]) for meta in metas)
            names = [f"{node.name}.{i}" for i in range(len(sizes))]
            self.emit(node, "split", [source], dict(axis=dim, sizes=sizes), names=names,
                      shapes=[tuple(int(v) for v in meta.shape) for meta in metas])
            # A one-result split is still a tuple-valued FX node.
            self.aliases[node] = names
            return
        else:
            raise Unsupported(f"unsupported tensor operation {operation}")
        self.emit(node, operation, [source], attrs)

    def function_node(self, node):
        target = node.target
        if target is getattr and arg(node, 1, "name") == "shape":
            source = arg(node, 0, "object")
            self.static[node] = tuple(int(dim) for dim in source.meta["tensor_meta"].shape)
            return
        binaries = {operator.add: "add", torch.add: "add", operator.sub: "sub", torch.sub: "sub",
                    operator.mul: "mul", torch.mul: "mul"}
        if target in binaries:
            if node.kwargs.get("alpha", 1) != 1:
                raise Unsupported("add/sub requires alpha=1")
            self.binary(node, binaries[target], arg(node, 0, "input"), arg(node, 1, "other"))
            return
        if target in (operator.truediv, torch.div):
            if node.kwargs.get("rounding_mode") is not None:
                raise Unsupported("division with rounding_mode is unsupported")
            self.divide(node, arg(node, 0, "input"), arg(node, 1, "other"))
            return
        if target in (operator.neg, torch.neg):
            self.emit(node, "mul", [self.activation(arg(node, 0, "input"))], dict(scalar=-1.0))
            return
        if target is operator.getitem:
            self.getitem(node)
            return
        activations = {F.relu: "relu", torch.relu: "relu", F.gelu: "gelu", F.silu: "silu",
                       torch.sigmoid: "sigmoid", F.sigmoid: "sigmoid", torch.tanh: "tanh",
                       F.hardswish: "hardswish", torch.exp: "exp"}
        if target in activations:
            fn = activations[target]
            if target is F.gelu and arg(node, 1, "approximate", "none") == "tanh":
                fn = "gelu_tanh"
            self.emit(node, "act", [self.activation(arg(node, 0, "input"))], dict(fn=fn))
            return
        tensor_ops = {torch.flatten: "flatten", torch.mean: "mean", torch.sum: "sum", F.softmax: "softmax",
                      torch.softmax: "softmax", torch.permute: "permute", torch.transpose: "transpose",
                      torch.reshape: "reshape", torch.split: "split", torch.chunk: "chunk"}
        if target in tensor_ops:
            self.tensor_operation(node, tensor_ops[target])
            return
        if target in (torch.cat, torch.concat):
            tensors = arg(node, 0, "tensors")
            if len(tensors) < 2:
                raise Unsupported("concat requires at least two tensors")
            dim = axis(self.value(node, 1, "dim", 0), len(self.shape(tensors[0])))
            self.emit(node, "concat", [self.activation(tensor) for tensor in tensors], dict(axis=dim))
            return
        if target in (torch.matmul, operator.matmul, torch.bmm):
            self.emit(node, "matmul", [self.activation(arg(node, 0, "input")), self.activation(arg(node, 1, "other"))], {})
            return
        if target is F.pad:
            if arg(node, 2, "mode", "constant") != "constant" or arg(node, 3, "value", None) not in (None, 0):
                raise Unsupported("pad requires constant mode with value=0")
            self.emit(node, "pad", [self.activation(arg(node, 0, "input"))], dict(pads=tuple(self.value(node, 1, "pad"))))
            return
        if target is F.avg_pool2d:
            if arg(node, 4, "ceil_mode", False) or not arg(node, 5, "count_include_pad", True) or arg(node, 6, "divisor_override", None) is not None:
                raise Unsupported("avg_pool2d requires ceil_mode=False, count_include_pad=True and no divisor_override")
            kernel = pair(self.value(node, 1, "kernel_size"))
            attrs = dict(kernel=kernel, stride=pair(self.value(node, 2, "stride", None) or kernel), padding=pair(self.value(node, 3, "padding", 0)))
            self.emit(node, "avgpool", [self.activation(arg(node, 0, "input"))], attrs)
            return
        if target is F.adaptive_avg_pool2d:
            self.emit(node, "adaptive_avgpool", [self.activation(arg(node, 0, "input"))], dict(output_size=pair(self.value(node, 1, "output_size"))))
            return
        if target in (F.linear, F.conv2d):
            attrs = dict(weight=self.get_weight(arg(node, 1, "weight")), bias=self.get_weight(arg(node, 2, "bias", None)))
            op = "linear" if target is F.linear else "conv2d"
            if op == "conv2d":
                attrs.update(stride=pair(arg(node, 3, "stride", 1)), padding=pair(arg(node, 4, "padding", 0)),
                             dilation=pair(arg(node, 5, "dilation", 1)), groups=int(arg(node, 6, "groups", 1)))
            self.emit(node, op, [self.activation(arg(node, 0, "input"))], attrs)
            return
        if target is F.layer_norm:
            attrs = dict(normalized_shape=tuple(arg(node, 1, "normalized_shape")), weight=self.get_weight(arg(node, 2, "weight", None)),
                         bias=self.get_weight(arg(node, 3, "bias", None)), eps=float(arg(node, 4, "eps", 1e-5)))
            self.emit(node, "layernorm", [self.activation(arg(node, 0, "input"))], attrs)
            return
        raise Unsupported(f"unsupported function {target!r}")

    def method_node(self, node):
        method = node.target
        if method == "size":
            shape = tuple(int(dim) for dim in node.args[0].meta["tensor_meta"].shape)
            dim = self.value(node, 1, "dim", None)
            self.static[node] = shape if dim is None else shape[dim]
        elif method in ("add", "sub", "mul"):
            if node.kwargs.get("alpha", 1) != 1:
                raise Unsupported("add/sub requires alpha=1")
            self.binary(node, method, node.args[0], arg(node, 1, "other"))
        elif method == "div":
            if node.kwargs.get("rounding_mode") is not None:
                raise Unsupported("division with rounding_mode is unsupported")
            self.divide(node, node.args[0], arg(node, 1, "other"))
        elif method in ("relu", "sigmoid", "tanh", "exp"):
            self.emit(node, "act", [self.activation(node.args[0])], dict(fn=method))
        elif method == "contiguous":
            self.emit(node, "identity", [self.activation(node.args[0])], {})
        elif method in ("mean", "sum", "view", "reshape", "flatten", "permute", "transpose", "softmax", "split", "chunk"):
            self.tensor_operation(node, "reshape" if method == "view" else method, method=True)
        else:
            raise Unsupported(f"unsupported method {method}")

    def map(self, node):
        if node.op == "get_attr":
            value = self.module
            for part in node.target.split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor):
                raise Unsupported("get_attr must hold a tensor")
            self.aliases[node] = self.weight(node.target, value)
        elif node.op == "call_module":
            self.module_node(node)
        elif node.op == "call_function":
            self.function_node(node)
        elif node.op == "call_method":
            self.method_node(node)
        else:
            raise Unsupported(f"unsupported FX node kind {node.op}")
