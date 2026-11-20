"""Expand explicitly declared framework rounding into reference graph atoms.

This preserves original floating-point reference behavior for import checks.
The returned graph is never substituted for an original model's reference or
passed to the polynomial planner as a separately chosen model.
"""

import copy


def expand_reference_precision(graph, precisions):
    """Apply source-proven intermediate dtypes at selected normalization nodes.

    ``precisions`` maps existing node IDs to ``torch.float32``.  Each entry
    refers to an RMSNorm or masked attention softmax whose original framework
    performs its reduction in that dtype and returns float64.  Original weight
    arrays stay shared; only graph metadata and primitive nodes are created.
    """
    from .graph import Graph, Node, TensorInfo
    if any(value != 'torch.float32' for value in precisions.values()):
        raise ValueError('Only explicit original float32 intermediate rounding is supported')
    present = {node.id: node for node in graph.nodes}
    if not set(precisions).issubset(present):
        raise ValueError('An explicit precision node is absent from the original graph')
    tensors = copy.deepcopy(graph.tensors)
    nodes, expanded = [], []
    allocated = set(tensors)
    for node in graph.nodes:
        if node.id not in precisions:
            nodes.append(Node(node.id, node.op, list(node.inputs), list(node.outputs),
                              copy.deepcopy(node.attrs), node.src))
            continue
        if node.op not in ('rmsnorm', 'softmax') or len(node.outputs) != 1:
            raise ValueError('Intermediate dtype requires one RMSNorm or softmax output')
        if node.op == 'softmax' and 'keep' not in node.attrs:
            raise ValueError('Only original masked attention softmax has a source-proven dtype here')
        out = node.outputs[0]
        shape = tensors[node.inputs[0]].shape

        def emit(suffix, op, inputs, attrs, output_shape, *, final=False):
            name = out if final else out + '.framework_' + suffix
            if not final and name in allocated:
                raise ValueError('Reference intermediate name collides: ' + name)
            allocated.add(name)
            tensors[name] = TensorInfo(name, tuple(output_shape), 'act')
            nodes.append(Node(node.id + '.framework_' + suffix, op, list(inputs), [name],
                              attrs, node.src))
            return name

        cast = emit('input32', 'cast', [node.inputs[0]], {'dtype': 'torch.float32'}, shape)
        if node.op == 'rmsnorm':
            if set(node.attrs) != {'weight', 'eps'} or len(node.inputs) != 1:
                raise ValueError('Original RMSNorm must retain its original weight and epsilon')
            reduced = shape[:-1] + (1,)
            square = emit('square', 'act', [cast], {'fn': 'pow', 'exponent': 2.0}, shape)
            mean = emit('mean', 'mean', [square], {'axes': (-1,), 'keepdim': True}, reduced)
            shifted = emit('epsilon', 'add', [mean], {'scalar': node.attrs['eps']}, reduced)
            inverse = emit('inverse', 'act', [shifted], {'fn': 'rsqrt'}, reduced)
            scaled = emit('normalize32', 'mul', [cast, inverse], {}, shape)
            wide = emit('output64', 'cast', [scaled], {'dtype': 'torch.float64'}, shape)
            emit('weight', 'mul', [wide], {'const': node.attrs['weight']}, shape, final=True)
        else:
            normalized = emit('softmax32', 'softmax', [cast] + list(node.inputs[1:]),
                              copy.deepcopy(node.attrs), shape)
            emit('output64', 'cast', [normalized], {'dtype': 'torch.float64'}, shape, final=True)
        expanded.append(dict(node_id=node.id, op=node.op, intermediate_dtype=precisions[node.id],
                             output_dtype='torch.float64', source=node.src))
    result = Graph(graph.name, nodes, tensors, dict(graph.weights), list(graph.inputs), list(graph.outputs))
    result.intake_stats = dict(getattr(graph, 'intake_stats', {}),
                              reference_precision_expansions=expanded,
                              source_proven_intermediate_rounding=True)
    return result
