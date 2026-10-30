"""NumPy float64 reference; exact GELU uses math.erf through np.vectorize."""

from collections import Counter
import math

import numpy as np

from .intake.graph import Graph


def _conv2d(x, weight, bias, stride, padding, dilation, groups):
    n, cin, _, _ = x.shape
    cout, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, ((kh-1)*dh+1, (kw-1)*dw+1), axis=(2, 3))
    windows = windows[:, :, ::sh, ::sw, ::dh, ::dw]
    ho, wo = windows.shape[2:4]
    cg, og = cin // groups, cout // groups
    parts = []
    for group in range(groups):
        columns = windows[:, group*cg:(group+1)*cg].transpose(0, 2, 3, 1, 4, 5).reshape(n*ho*wo, cg*kh*kw)
        kernel = weight[group*og:(group+1)*og].reshape(og, cg*kh*kw)
        result = (columns @ kernel.T).reshape(n, ho, wo, og).transpose(0, 3, 1, 2)
        parts.append(result)
    result = parts[0] if groups == 1 else np.concatenate(parts, axis=1)
    return result if bias is None else result + bias.reshape(1, -1, 1, 1)


def _act(x, fn):
    if fn == "relu":
        return np.maximum(x, 0)
    if fn == "gelu":
        return 0.5*x*(1+np.vectorize(math.erf, otypes=[np.float64])(x/math.sqrt(2)))
    if fn == "gelu_tanh":
        return 0.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+0.044715*x**3)))
    if fn == "sigmoid":
        return 1/(1+np.exp(-x))
    if fn == "silu":
        return x/(1+np.exp(-x))
    if fn == "hardswish":
        return x*np.clip(x+3, 0, 6)/6
    if fn == "tanh":
        return np.tanh(x)
    if fn == "exp":
        return np.exp(x)
    raise ValueError(f"Unknown activation {fn}")


def _evaluate(node, values, batch):
    attrs = node.attrs
    args = [values[name] for name in node.inputs]
    x = args[0]
    op = node.op
    weight = lambda key: values[attrs[key]] if attrs.get(key) is not None else None
    if op == "conv2d":
        return _conv2d(x, weight("weight"), weight("bias"), attrs["stride"], attrs["padding"], attrs["dilation"], attrs["groups"])
    if op == "linear":
        result = x @ weight("weight").T
        return result if weight("bias") is None else result + weight("bias")
    if op == "matmul":
        return np.matmul(x, args[1])
    if op in ("add", "sub", "mul"):
        y = args[1] if len(args) == 2 else (values[attrs["const"]] if "const" in attrs else attrs["scalar"])
        if op == "add":
            return x+y
        if op == "mul":
            return x*y
        return y-x if attrs.get("reverse", False) else x-y
    if op == "batchnorm":
        shape = (1, -1) + (1,)*(x.ndim-2)
        return ((x-weight("mean").reshape(shape))/np.sqrt(weight("var").reshape(shape)+attrs["eps"])
                * weight("weight").reshape(shape)+weight("bias").reshape(shape))
    if op == "layernorm":
        axes = tuple(range(x.ndim-len(attrs["normalized_shape"]), x.ndim))
        result = (x-np.mean(x, axis=axes, keepdims=True))/np.sqrt(np.var(x, axis=axes, keepdims=True)+attrs["eps"])
        if weight("weight") is not None:
            result = result*weight("weight")
        if weight("bias") is not None:
            result = result+weight("bias")
        return result
    if op == "act":
        return _act(x, attrs["fn"])
    if op == "softmax":
        result = np.exp(x-np.max(x, axis=attrs["axis"], keepdims=True))
        return result/np.sum(result, axis=attrs["axis"], keepdims=True)
    if op == 'maxpool':
        import torch
        from torch.nn import functional as F
        return F.max_pool2d(torch.from_numpy(x), attrs['kernel'], attrs['stride'],
                            attrs['padding'], attrs.get('dilation', (1, 1)), attrs.get('ceil_mode', False)).numpy()
    if op == "avgpool":
        ph, pw = attrs["padding"]
        padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, attrs["kernel"], axis=(2, 3))
        sh, sw = attrs["stride"]
        return np.mean(windows[:, :, ::sh, ::sw], axis=(-2, -1))
    if op == "adaptive_avgpool":
        h, w = x.shape[-2:]
        oh, ow = attrs["output_size"]
        result = np.empty(x.shape[:-2]+(oh, ow), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                window = x[..., i*h//oh:((i+1)*h+oh-1)//oh, j*w//ow:((j+1)*w+ow-1)//ow]
                result[..., i, j] = np.mean(window, axis=(-2, -1))
        return result
    if op == "flatten":
        start, end = attrs["start_dim"], attrs["end_dim"]
        return x.reshape(x.shape[:start]+(math.prod(x.shape[start:end+1]),)+x.shape[end+1:])
    if op == "reshape":
        if attrs["shape"][0] != 1:
            raise ValueError("reference reshape requires a static leading dimension of 1")
        return x.reshape((batch,)+attrs["shape"][1:])
    if op == "permute":
        return np.transpose(x, attrs["dims"])
    if op == "concat":
        return np.concatenate(args, axis=attrs["axis"])
    if op == "slice":
        indices = [slice(None)]*x.ndim
        indices[attrs["axis"]] = slice(attrs["start"], attrs["end"], attrs["step"])
        return x[tuple(indices)]
    if op == "split":
        return np.split(x, np.cumsum(attrs["sizes"])[:-1], axis=attrs["axis"])
    if op == "pad":
        pads = [(0, 0)]*x.ndim
        for i in range(len(attrs["pads"])//2):
            pads[-i-1] = attrs["pads"][2*i:2*i+2]
        return np.pad(x, pads)
    if op in ("mean", "sum"):
        function = np.mean if op == "mean" else np.sum
        return function(x, axis=attrs["axes"], keepdims=attrs["keepdim"])
    if op == "identity":
        return x
    raise ValueError(f"Unknown core op {op}")


def reference_forward(graph: Graph, inputs: dict[str, np.ndarray], *, keep: bool = False) -> dict[str, np.ndarray]:
    if getattr(graph, 'intake_stats', {}).get('transformer_normalized'):
        from .intake.graph_exec import forward
        result=forward(graph,inputs,keep=keep)
        return {name: value.detach().cpu().numpy() if hasattr(value,'detach') else value
                for name,value in result.items()}
    values = dict(graph.weights)
    for name in graph.inputs:
        array = np.asarray(inputs[name], dtype=np.float64)
        if array.shape[1:] != graph.tensors[name].shape[1:]:
            raise ValueError(f"Input shape differs for {name}")
        values[name] = array
    batch = len(values[graph.inputs[0]])
    uses = Counter(name for node in graph.nodes for name in node.inputs)
    uses.update(graph.outputs)
    for node in graph.nodes:
        result = _evaluate(node, values, batch)
        results = result if node.op == "split" else [result]
        values.update(zip(node.outputs, results))
        if not keep:
            for name in node.inputs:
                uses[name] -= 1
                if uses[name] == 0 and name not in graph.weights:
                    values.pop(name, None)
    return values if keep else {name: values[name] for name in graph.outputs}


def approximation_points(graph):
    """Stable ordered point keys; activation keys retain the frozen spelling."""
    result=[]
    for node in graph.nodes:
        if node.op in ('act','maxpool'):
            result.append((node,node.inputs[0] if node.op=='act' else node.outputs[0]+'::differences'))
        elif node.op in ('layernorm','rmsnorm'):
            result.append((node,node.outputs[0]+'::rsqrt'))
        elif node.op=='softmax':
            result.extend((node,node.outputs[0]+'::'+role) for role in ('exp','inv'))
    return result


def approximation_values(graph, values):
    from .kernels.cnn import DifferenceStats, maxpool_tensor
    result = {}
    for node, name in approximation_points(graph):
        if node.op == 'act':
            result[name] = values[node.inputs[0]]
        elif node.op in ('layernorm','rmsnorm'):
            x=values[node.inputs[0]]
            centered=x if node.op=='rmsnorm' else x-x.mean(-1,keepdims=True)
            result[name]=((centered*centered).mean(-1,keepdims=True)+node.attrs['eps']).reshape(1,-1)
        elif node.op=='softmax':
            x=values[node.inputs[0]];a=node.attrs
            keep=np.ones(x.shape,dtype=bool)
            if a.get('keep')=='causal':
                keep &= np.arange(x.shape[-1])[None,:] <= np.arange(x.shape[-2])[:,None]
            if a.get('mask') is not None:
                keep &= values[a['mask']].reshape((len(x),)+(1,)*(x.ndim-2)+(x.shape[-1],)).astype(bool)
            result[name]=(x[keep].reshape(1,-1) if name.endswith('::exp') else
                          (np.exp(x)*keep).sum(-1,keepdims=True).reshape(1,-1))
        else:
            stats = DifferenceStats()
            a = node.attrs
            maxpool_tensor(values[node.inputs[0]], kernel=a['kernel'], stride=a['stride'],
                           pad=a['padding'], apply=lambda x: np.maximum(x, 0), observer=stats.add)
            result[name] = stats.value
    return result


def calibrate(graph: Graph, inputs: dict[str, np.ndarray], batch: int = 100) -> dict[str, dict]:
    if batch <= 0:
        raise ValueError("batch must be positive")
    count = len(inputs[graph.inputs[0]])
    result = {}
    for start in range(0, count, batch):
        values = reference_forward(graph, {name: array[start:start+batch] for name, array in inputs.items()}, keep=True)
        for name, tensor in graph.tensors.items():
            if tensor.kind != "act":
                continue
            value = values[name]
            low, high = float(np.min(value)), float(np.max(value))
            absolute = float(np.max(np.abs(value)))
            percentile = float(np.percentile(np.abs(value), 99.9))
            if name not in result:
                result[name] = dict(min=low, max=high, absmax=absolute, p999=percentile, count=int(value.size))
            else:
                old = result[name]
                old.update(min=min(old["min"], low), max=max(old["max"], high), absmax=max(old["absmax"], absolute),
                           p999=max(old["p999"], percentile), count=old["count"]+int(value.size))
        pool_names = {name for node, name in approximation_points(graph) if node.op != 'act'}
        if pool_names:
            for name, value in approximation_values(graph, values).items():
                if name not in pool_names:
                    continue
                low, high = float(value.min()), float(value.max())
                absolute = max(abs(low), abs(high))
                if name not in result:
                    result[name] = dict(min=low, max=high, absmax=absolute)
                else:
                    old = result[name]
                    old.update(min=min(old['min'], low), max=max(old['max'], high), absmax=max(old['absmax'], absolute))
    return result
