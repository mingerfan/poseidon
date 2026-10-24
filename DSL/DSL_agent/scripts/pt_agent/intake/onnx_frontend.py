"""ONNX semantic import with user-provided shapes and static constant folding."""

import copy
from pathlib import Path

import numpy as np


def import_graph(path, example_inputs):
    import onnx
    from onnx import helper, numpy_helper, shape_inference
    from onnx.reference import ReferenceEvaluator
    from .graph import Graph, Node, TensorInfo
    from .torch_frontend import IntakeError
    from . import decompose

    model = onnx.load(str(path), load_external_data=True)
    constants = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in constants]
    if set(example_inputs) != {value.name for value in inputs}:
        raise ValueError('ONNX sample keys differ from graph inputs')
    for value in inputs:
        array = example_inputs[value.name]
        if len(array) != 1:
            raise ValueError('ONNX import requires a single original sample')
        value.type.tensor_type.ClearField('shape')
        for dim in array.shape:
            value.type.tensor_type.shape.dim.add().dim_value = int(dim)
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    original_nodes = len(model.graph.node)
    known_shapes = {value.name: tuple(int(dim.dim_value) for dim in value.type.tensor_type.shape.dim)
                    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]
                    if all(dim.HasField('dim_value') for dim in value.type.tensor_type.shape.dim)}
    # Exporters often construct static Pad/Reshape arguments through several
    # tensor operations. Materialize these constants before the second shape
    # inference so their downstream activation shapes can also be inferred.
    remaining = []
    folded_nodes = 0
    for node in model.graph.node:
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if node.op_type == 'Shape' and node.input[0] in known_shapes:
            values = [np.asarray(known_shapes[node.input[0]], dtype=np.int64)[
                attrs.get('start', 0):attrs.get('end', None)]]
        elif node.op_type == 'Constant' or all(name in constants for name in node.input if name):
            values = ReferenceEvaluator(node).run(None, {name: constants[name] for name in node.input if name})
        else:
            remaining.append(node)
            continue
        folded_nodes += 1
        for name, value in zip(node.output, values):
            if name:
                constants[name] = value
                known_shapes[name] = value.shape
                model.graph.initializer.append(numpy_helper.from_array(value, name))
    del model.graph.node[:]
    model.graph.node.extend(remaining)
    # Discard the first pass's symbolic intermediate dimensions. Inference
    # merges existing annotations and otherwise keeps symbols even after their
    # defining constant expressions have become concrete initializers.
    del model.graph.value_info[:]
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    graph = Graph(Path(path).stem.replace('-', '_').replace('.', '_'), [], {}, {},
                  [value.name for value in inputs], [value.name for value in model.graph.output])
    shapes = {}
    for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        dims = value.type.tensor_type.shape.dim
        if all(dim.HasField('dim_value') for dim in dims):
            shapes[value.name] = tuple(int(dim.dim_value) for dim in dims)
    for name, value in constants.items():
        shapes[name] = value.shape
    for name, shape in shapes.items():
        graph.tensors[name] = TensorInfo(name, shape, 'input' if name in graph.inputs else 'act')

    def weight(name, value):
        array = np.array(value, dtype=np.float64, copy=True)
        array.setflags(write=False)
        graph.weights[name] = array
        graph.tensors[name] = TensorInfo(name, tuple(array.shape), 'weight')
        return name

    for name, value in constants.items():
        weight(name, value)

    def emit(node, op, args, attrs, outputs=None, output_shapes=None):
        outputs = outputs or list(node.output[:1])
        for index, name in enumerate(outputs):
            shape = output_shapes[index] if output_shapes is not None else shapes[name]
            graph.tensors[name] = TensorInfo(name, tuple(shape), 'act')
        graph.nodes.append(Node(f'n{len(graph.nodes)}', op, list(args), outputs, attrs,
                                f'{node.name or node.output[0]}:onnx.{node.op_type}'))

    def binary(node, op):
        left, right = node.input
        if left not in constants and right not in constants:
            emit(node, op, [left, right], {})
        else:
            value, const = (left, right) if right in constants else (right, left)
            attrs = dict(const=weight(const, constants[const]))
            if op == 'sub':
                attrs['reverse'] = left in constants
            emit(node, op, [value], attrs)

    errors = []
    def spatial_pads(attrs, shape, kernel, strides, dilations):
        mode = attrs.get('auto_pad', b'NOTSET')
        if mode in (b'NOTSET', b''):
            return tuple(attrs.get('pads', (0, 0, 0, 0)))
        if mode == b'VALID':
            return (0, 0, 0, 0)
        if mode not in (b'SAME_UPPER', b'SAME_LOWER'):
            raise ValueError(f'Unknown ONNX auto_pad mode {mode!r}')
        totals = [max(0, ((size+stride-1)//stride-1)*stride+(k-1)*d+1-size)
                  for size, k, stride, d in zip(shape[-2:], kernel, strides, dilations)]
        before = tuple((value+(mode == b'SAME_LOWER'))//2 for value in totals)
        return before+tuple(value-start for value, start in zip(totals, before))
    acts = dict(Relu='relu', Gelu='gelu', Sigmoid='sigmoid', Tanh='tanh', HardSwish='hardswish',
                Sqrt='sqrt', Reciprocal='reciprocal', Erf='erf', Exp='exp')
    for node in model.graph.node:
        a = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        op, x = node.op_type, node.input[0] if node.input else None
        try:
            if op == 'Shape':
                value = np.asarray(shapes[x], dtype=np.int64)[a.get('start', 0):a.get('end', None)]
                constants[node.output[0]] = value
                weight(node.output[0], value)
                continue
            if op == 'Constant' or all(name in constants for name in node.input if name):
                values = ReferenceEvaluator(node).run(None, {name: constants[name] for name in node.input if name})
                for name, value in zip(node.output, values):
                    constants[name] = value
                    shapes[name] = value.shape
                    weight(name, value)
                continue
            if op == 'Conv':
                strides, dilations = tuple(a.get('strides', (1, 1))), tuple(a.get('dilations', (1, 1)))
                pads = spatial_pads(a, shapes[x], constants[node.input[1]].shape[-2:], strides, dilations)
                if pads[:2] != pads[2:]:
                    shape = list(shapes[x]); shape[-2] += pads[0]+pads[2]; shape[-1] += pads[1]+pads[3]
                    temp = node.output[0]+'.prepad'
                    emit(node, 'pad', [x], dict(pads=(pads[1], pads[3], pads[0], pads[2])), [temp], [shape])
                    shapes[temp] = tuple(shape); x = temp; pads = (0, 0, 0, 0)
                emit(node, 'conv2d', [x], dict(weight=node.input[1],
                    bias=node.input[2] if len(node.input) > 2 else None,
                    stride=strides, padding=pads[:2], dilation=dilations, groups=int(a.get('group', 1))))
            elif op == 'BatchNormalization':
                emit(node, 'batchnorm', [x], dict(zip(('weight', 'bias', 'mean', 'var'), node.input[1:5]),
                                                eps=float(a.get('epsilon', 1e-5))))
            elif op == 'LayerNormalization':
                dim = int(a.get('axis', -1)) % len(shapes[x])
                emit(node, 'layernorm', [x], dict(normalized_shape=shapes[x][dim:],
                    weight=node.input[1] if len(node.input) > 1 else None,
                    bias=node.input[2] if len(node.input) > 2 else None,
                    eps=float(a.get('epsilon', 1e-5))))
            elif op == 'Gemm':
                if a.get('transA', 0) or node.input[1] not in constants:
                    raise ValueError('Gemm requires nontransposed activations and static weights')
                kernel = constants[node.input[1]]
                kernel = kernel if a.get('transB', 0) else kernel.T
                kernel = kernel*float(a.get('alpha', 1))
                key = weight(node.input[1]+'.linear', kernel)
                bias = (weight(node.input[2]+'.linear', constants[node.input[2]]*float(a.get('beta', 1)))
                        if len(node.input) > 2 and node.input[2] else None)
                emit(node, 'linear', [x], dict(weight=key, bias=bias))
            elif op == 'MatMul':
                right = node.input[1]
                if right in constants and constants[right].ndim == 2:
                    emit(node, 'linear', [x], dict(weight=weight(right+'.linear', constants[right].T), bias=None))
                else:
                    emit(node, 'matmul', list(node.input), {})
            elif op in ('Add', 'Sub', 'Mul'):
                binary(node, op.lower())
            elif op == 'Div':
                if node.input[1] in constants:
                    key = weight(node.input[1]+'.reciprocal', 1/constants[node.input[1]])
                    emit(node, 'mul', [x], dict(const=key))
                else:
                    name = node.output[0]+'.reciprocal'
                    emit(node, 'act', [node.input[1]], dict(fn='reciprocal'), [name], [shapes[node.input[1]]])
                    emit(node, 'mul', [x, name], {})
            elif op in acts:
                emit(node, 'act', [x], dict(fn=acts[op]))
            elif op == 'Pow':
                exponent = constants[node.input[1]]
                if exponent.size != 1:
                    raise ValueError('Pow requires a constant scalar exponent')
                emit(node, 'act', [x], dict(fn='pow', exponent=float(exponent.item())))
            elif op == 'Neg':
                emit(node, 'mul', [x], dict(scalar=-1.0))
            elif op in ('MaxPool', 'AveragePool'):
                pads = tuple(a.get('pads', (0, 0, 0, 0)))
                if pads[:2] != pads[2:]:
                    raise ValueError('asymmetric pool padding has no core mapping')
                attrs = dict(kernel=tuple(a['kernel_shape']), stride=tuple(a.get('strides', (1, 1))), padding=pads[:2])
                if op == 'MaxPool':
                    attrs.update(dilation=tuple(a.get('dilations', (1, 1))), ceil_mode=bool(a.get('ceil_mode', 0)))
                elif a.get('ceil_mode', 0) or not a.get('count_include_pad', 0) and any(pads):
                    raise ValueError('AveragePool padding/count semantics have no core mapping')
                emit(node, 'maxpool' if op == 'MaxPool' else 'avgpool', [x], attrs)
            elif op == 'GlobalAveragePool':
                emit(node, 'adaptive_avgpool', [x], dict(output_size=(1, 1)))
            elif op == 'Flatten':
                dim = int(a.get('axis', 1)) % len(shapes[x])
                emit(node, 'flatten', [x], dict(start_dim=dim, end_dim=len(shapes[x])-1))
            elif op in ('Reshape', 'Unsqueeze', 'Squeeze'):
                emit(node, 'reshape', [x], dict(shape=shapes[node.output[0]]))
            elif op == 'Expand':
                emit(node, 'broadcast', [x], dict(shape=shapes[node.output[0]]))
            elif op == 'Transpose':
                emit(node, 'permute', [x], dict(dims=tuple(a.get('perm', reversed(range(len(shapes[x])))))))
            elif op in ('ReduceMean', 'ReduceSum'):
                axes = constants[node.input[1]].tolist() if len(node.input) > 1 else a.get('axes', range(len(shapes[x])))
                emit(node, 'mean' if op == 'ReduceMean' else 'sum', [x],
                     dict(axes=tuple(int(axis) % len(shapes[x]) for axis in axes), keepdim=bool(a.get('keepdims', 1))))
            elif op == 'Concat':
                emit(node, 'concat', list(node.input), dict(axis=int(a['axis']) % len(shapes[x])))
            elif op == 'Slice':
                starts = constants[node.input[1]].tolist()
                ends = constants[node.input[2]].tolist()
                axes = constants[node.input[3]].tolist() if len(node.input) > 3 else list(range(len(starts)))
                steps = constants[node.input[4]].tolist() if len(node.input) > 4 else [1]*len(starts)
                shape, source = list(shapes[x]), x
                for i, (start, end, axis, step) in enumerate(zip(starts, ends, axes, steps)):
                    axis %= len(shape)
                    start, end, step = slice(start, end, step).indices(shape[axis])
                    shape[axis] = len(range(start, end, step))
                    name = node.output[0] if i == len(starts)-1 else node.output[0]+f'.slice{i}'
                    emit(node, 'slice', [source], dict(axis=axis, start=start, end=end, step=step), [name], [shape])
                    source = name
            elif op == 'Gather' and node.input[1] in constants:
                indices = constants[node.input[1]]
                if indices.size != 1:
                    raise ValueError('nonstatic/noncontiguous Gather has no core mapping')
                dim = int(a.get('axis', 0)) % len(shapes[x])
                index = int(indices.item()) % shapes[x][dim]
                shape = list(shapes[x]); shape[dim] = 1
                temp = node.output[0]+'.slice'
                emit(node, 'slice', [x], dict(axis=dim, start=index, end=index+1, step=1), [temp], [shape])
                emit(node, 'reshape', [temp], dict(shape=shapes[node.output[0]]))
            elif op == 'Pad':
                pads = constants[node.input[1]].astype(int).tolist()
                if a.get('mode', b'constant') != b'constant' or (len(node.input) > 2 and node.input[2] and float(constants[node.input[2]]) != 0):
                    raise ValueError('Pad requires zero constant mode')
                rank = len(pads)//2
                reversed_pads = tuple(v for dim in reversed(range(rank)) for v in (pads[dim], pads[rank+dim]))
                emit(node, 'pad', [x], dict(pads=reversed_pads))
            elif op == 'Softmax':
                emit(node, 'softmax', [x], dict(axis=int(a.get('axis', -1)) % len(shapes[x])))
            elif op == 'Where':
                emit(node, 'where', list(node.input), {})
            elif op == 'Cast':
                dtype = np.dtype(helper.tensor_dtype_to_np_dtype(a['to'])).name
                emit(node, 'cast', [x], dict(dtype=dtype))
            elif op == 'Identity':
                emit(node, 'identity', [x], {})
            elif op == 'Dropout':
                if len(node.input) > 2 and (node.input[2] not in constants or bool(constants[node.input[2]])):
                    raise ValueError('Dropout training mode is stochastic and has no static inference semantics')
                emit(node, 'identity', [x], {})
                if len(node.output) > 1 and node.output[1]:
                    constants[node.output[1]] = np.ones(shapes[x], dtype=bool)
                    weight(node.output[1], constants[node.output[1]])
            else:
                raise ValueError('ONNX op has no semantic mapping')
        except (ValueError, KeyError, NotImplementedError) as exc:
            errors.append(dict(fx_node=node.name or node.output[0], op=op, target='onnx.'+op, reason=str(exc)))
    if errors:
        raise IntakeError(errors)
    graph = decompose.run(graph)
    graph.intake_stats = dict(original_nodes=original_nodes,
                             folded_nodes=folded_nodes + sum(all(name in constants for name in node.output)
                                              for node in model.graph.node), original_unmapped={})
    return graph


def import_model(path, samples):
    import onnx
    from .probe import ImportedModel, model_info
    from ..data.samples import SampleSplits
    if samples.kind != 'tensor':
        raise ValueError('ONNX requires the original preprocessed tensor samples')
    split = samples.split()
    model = onnx.load(str(path), load_external_data=True)
    initializer_names = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializer_names]
    arrays, _ = samples.tensors(split['calibration'][:1], names=[value.name for value in inputs])
    graph = import_graph(path, arrays)
    info = model_info(dict(type='onnx', path=str(path)), graph, split=split,
                      has_labels=samples.labels is not None,
                      dtypes={value.name: str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))
                              for value in inputs})
    return ImportedModel(info, SampleSplits(samples, info), model)


def normalize_for_comparison(graph):
    """U1 cross-source normalization: fold BN, merge shape op names, omit identity.

    No normalization is applied to Agent graphs or to saved frozen programs.
    """
    graph = copy.deepcopy(graph)
    aliases, retained = {}, []
    for node in graph.nodes:
        node.inputs = [aliases.get(name, name) for name in node.inputs]
        if node.op == 'identity':
            aliases[node.outputs[0]] = node.inputs[0]
            continue
        if node.op == 'batchnorm':
            conv = next((old for old in retained if old.outputs == node.inputs and old.op == 'conv2d'), None)
            if conv is not None and len(graph.consumers(node.inputs[0])) == 1:
                a, weights = node.attrs, graph.weights
                scale = weights[a['weight']]/np.sqrt(weights[a['var']]+a['eps'])
                kernel = weights[conv.attrs['weight']]*scale.reshape(-1, 1, 1, 1)
                bias = weights[conv.attrs['bias']] if conv.attrs.get('bias') else np.zeros_like(scale)
                bias = (bias-weights[a['mean']])*scale+weights[a['bias']]
                key = conv.attrs['weight']+'.bn_folded'
                graph.weights[key], graph.weights[key+'.bias'] = kernel, bias
                conv.attrs.update(weight=key, bias=key+'.bias')
                conv.outputs = list(node.outputs)
                continue
        if node.op == 'flatten':
            node.op = 'reshape'
            node.attrs = dict(shape=graph.tensors[node.outputs[0]].shape)
        retained.append(node)
    graph.nodes = retained
    graph.outputs = [aliases.get(name, name) for name in graph.outputs]
    return graph
