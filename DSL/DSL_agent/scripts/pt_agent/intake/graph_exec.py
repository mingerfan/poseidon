"""Float64 PT-Graph semantic interpreter, independent of FHE lowering.

This interpreter checks imports only. It supplies neither candidate selection
nor a replacement for encrypted execution.
"""

import math

import numpy as np


def forward(graph, inputs, *, keep=False):
    import torch
    from torch.nn import functional as F

    values = {name: torch.as_tensor(np.array(value), dtype=torch.float64)
              for name, value in graph.weights.items()}
    for name in graph.inputs:
        value = inputs[name]
        values[name] = (value.detach().to(dtype=torch.float64) if torch.is_tensor(value)
                        else torch.as_tensor(np.asarray(value), dtype=torch.float64))
    batch = len(values[graph.inputs[0]])
    for node in graph.nodes:
        a = node.attrs
        args = [values[name] for name in node.inputs]
        x = args[0]
        w = lambda key: values[a[key]] if a.get(key) is not None else None
        op = node.op
        if op == 'conv2d':
            result = F.conv2d(x, w('weight'), w('bias'), a['stride'], a['padding'], a['dilation'], a['groups'])
        elif op == 'linear':
            result = F.linear(x, w('weight'), w('bias'))
        elif op == 'matmul':
            result = torch.matmul(x, args[1])
        elif op in ('add', 'sub', 'mul'):
            y = args[1] if len(args) > 1 else w('const') if 'const' in a else a['scalar']
            result = x+y if op == 'add' else x*y if op == 'mul' else y-x if a.get('reverse') else x-y
        elif op == 'batchnorm':
            result = F.batch_norm(x, w('mean'), w('var'), w('weight'), w('bias'), training=False, eps=a['eps'])
        elif op == 'layernorm':
            result = F.layer_norm(x, a['normalized_shape'], w('weight'), w('bias'), a['eps'])
        elif op == 'act':
            fn = a['fn']
            functions = dict(relu=F.relu, silu=F.silu, sigmoid=torch.sigmoid, tanh=torch.tanh,
                             hardswish=F.hardswish, exp=torch.exp, rsqrt=torch.rsqrt,
                             sqrt=torch.sqrt, reciprocal=torch.reciprocal, erf=torch.erf)
            if fn in functions:
                result = functions[fn](x)
            elif fn in ('gelu', 'gelu_tanh'):
                result = F.gelu(x, approximate='tanh' if fn == 'gelu_tanh' else 'none')
            elif fn == 'pow':
                result = torch.pow(x, a['exponent'])
            else:
                raise ValueError(f'{node.id}: unknown act function {fn}')
        elif op == 'softmax':
            result = torch.softmax(x, dim=a['axis'])
        elif op == 'maxpool_indices':
            result = F.max_pool2d(x, a['kernel'], a['stride'], a['padding'], a['dilation'], a['ceil_mode'], return_indices=True)[1]
        elif op == 'maxpool':
            result = F.max_pool2d(x, a['kernel'], a['stride'], a['padding'], a['dilation'], a['ceil_mode'])
        elif op == 'avgpool':
            result = F.avg_pool2d(x, a['kernel'], a['stride'], a['padding'])
        elif op == 'adaptive_avgpool':
            result = F.adaptive_avg_pool2d(x, a['output_size'])
        elif op == 'flatten':
            result = torch.flatten(x, a['start_dim'], a['end_dim'])
        elif op in ('reshape', 'broadcast'):
            shape = tuple(a['shape'])
            # Graphs describe batch=1. The optional batched reference extension
            # only changes an existing leading singleton, never other axes.
            if batch != 1 and shape and shape[0] == 1:
                shape = (batch,) + shape[1:]
            result = x.reshape(shape) if op == 'reshape' else x.expand(shape)
        elif op == 'permute':
            result = x.permute(a['dims'])
        elif op == 'concat':
            result = torch.cat(args, dim=a['axis'])
        elif op == 'slice':
            indices = [slice(None)] * x.ndim
            indices[a['axis']] = slice(a['start'], a['end'], a['step'])
            result = x[tuple(indices)]
        elif op == 'split':
            result = torch.split(x, a['sizes'], dim=a['axis'])
        elif op == 'pad':
            result = F.pad(x, a['pads'])
        elif op in ('mean', 'sum'):
            function = torch.mean if op == 'mean' else torch.sum
            result = function(x, dim=a['axes'], keepdim=a['keepdim'])
        elif op == 'identity':
            result = x
        elif op == 'cast':
            result = x.to(getattr(torch, a['dtype'].removeprefix('torch.')))
        elif op == 'compare':
            y = args[1] if len(args) > 1 else w('const') if 'const' in a else a['scalar']
            result = getattr(torch, a['predicate'])(x, y)
        elif op == 'where':
            result = torch.where(x.bool(), args[1], args[2])
        elif op == 'slice_scatter':
            result = torch.slice_scatter(x, args[1], dim=a['axis'], start=a['start'], end=a['end'], step=a['step'])
        else:
            raise ValueError(f'{node.id}: no reference semantics for {op}')
        outputs = result if op == 'split' else [result]
        values.update(zip(node.outputs, outputs))
    return values if keep else {name: values[name].detach().cpu().numpy() for name in graph.outputs}


reference_forward = forward
