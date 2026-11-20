"""Explicit source-only VGG pool/conv fusion; no exporter or CLI integration.

The original graph and every original parameter remain intact. Derived weights
implement a real-number identity. Binary64 end-to-end acceptance and HEIR layout
success require separate measurements using the unchanged criteria.
"""
from collections import Counter
import copy
from dataclasses import is_dataclass, replace
import math


MODEL = 'vgg16_silu_avgpool'
VARIANT = 'vgg-four-nonoverlapping-pool-conv-k6-v1'
PAIRS = (
    ('n6', 'n7', 'conv_2_1.Conv2d.weight'),
    ('n13', 'n14', 'conv_3_1.Conv2d.weight'),
    ('n23', 'n24', 'conv_4_1.Conv2d.weight'),
    ('n33', 'n34', 'conv_5_1.Conv2d.weight'),
)


def _pair(value, expected):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(x) is int for x in value) and tuple(value) == expected)


def _shape(tensor):
    shape = tuple(tensor.shape)
    if not shape or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError('Positive static tensor dimensions required')
    return shape


def find_pairs(graph):
    """Validate exactly the four known source edges without reading weight values."""
    if graph.name != MODEL or len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError('Only the single-input/output original VGG graph is supported')
    ids = [node.id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        raise ValueError('Original node identifiers must be unique')
    wanted = {pool for pool, _, _ in PAIRS}
    if {node.id for node in graph.nodes if node.op == 'avgpool'} != wanted:
        raise ValueError('Exactly the four original local VGG pools are required')
    positions = {node.id: index for index, node in enumerate(graph.nodes)}
    pairs = []
    for pool_id, conv_id, weight_name in PAIRS:
        index = positions[pool_id]
        pool = graph.nodes[index]
        if index + 1 >= len(graph.nodes) or graph.nodes[index + 1].id != conv_id:
            raise ValueError('Pool and its original convolution must be adjacent')
        conv = graph.nodes[index + 1]
        if (len(pool.inputs) != 1 or len(pool.outputs) != 1
                or conv.op != 'conv2d' or conv.inputs != pool.outputs or len(conv.outputs) != 1
                or pool.outputs[0] in graph.outputs or pool.outputs[0] in graph.inputs):
            raise ValueError('One direct original pool-to-convolution edge is required')
        users = [n.id for n in graph.nodes for name in n.inputs if name == pool.outputs[0]]
        if users != [conv_id]:
            raise ValueError('Pool output must have exactly one use by the adjacent convolution')
        a, b = pool.attrs, conv.attrs
        if (set(a) - {'kernel', 'stride', 'padding', 'ceil_mode', 'count_include_pad', 'divisor_override'}
                or not _pair(a.get('kernel'), (2, 2)) or not _pair(a.get('stride'), (2, 2))
                or not _pair(a.get('padding'), (0, 0))
                or a.get('ceil_mode', False) is not False
                or a.get('count_include_pad', True) is not True
                or a.get('divisor_override') is not None):
            raise ValueError('Pool requires kernel2/stride2/pad0, floor and the original divisor')
        bias_name = weight_name.removesuffix('weight') + 'bias'
        if (set(b) - {'weight', 'bias', 'stride', 'padding', 'dilation', 'groups'}
                or b.get('weight') != weight_name or b.get('bias') != bias_name
                or not _pair(b.get('stride'), (1, 1)) or not _pair(b.get('padding'), (1, 1))
                or not _pair(b.get('dilation'), (1, 1)) or type(b.get('groups')) is not int
                or b['groups'] != 1):
            raise ValueError('Original named groups1/dilation1/stride1/pad1 convolution required')
        try:
            before, pooled, after = (_shape(graph.tensors[name])
                                    for name in (pool.inputs[0], pool.outputs[0], conv.outputs[0]))
            weight = tuple(graph.weights[weight_name].shape)
            bias = tuple(graph.weights[bias_name].shape)
            declared_weight = _shape(graph.tensors[weight_name])
        except (KeyError, AttributeError) as exc:
            raise ValueError('Original tensor and parameter metadata must be present') from exc
        if (len(before) != 4 or before[0] != 1 or any(n % 2 for n in before[2:])
                or pooled != (1, before[1], before[2] // 2, before[3] // 2)
                or len(after) != 4 or after[0] != 1 or after[2:] != pooled[2:]
                or weight != (after[1], before[1], 3, 3) or weight != declared_weight
                or bias != (after[1],)):
            raise ValueError('Original even pool geometry and convolution parameter shapes differ')
        derived = 'derived-exact-pool-conv:' + pool_id + ':' + conv_id
        if derived in graph.weights or derived in graph.tensors:
            raise ValueError('Preserve any existing derived identifier; fusion is single-use')
        pairs.append(dict(pool_id=pool_id, conv_id=conv_id, input=pool.inputs[0],
            removed_tensor=pool.outputs[0], output=conv.outputs[0],
            input_shape=list(before), pooled_shape=list(pooled), output_shape=list(after),
            source_weight=weight_name, source_bias=bias_name, source_weight_shape=list(weight),
            derived_weight=derived, derived_weight_shape=[weight[0], weight[1], 6, 6],
            pool_attrs=copy.deepcopy(a), conv_attrs=copy.deepcopy(b),
            bias_added_once=True, intermediary_operations=0))
    return pairs


def derive_weight(weight):
    """Return an independent readonly6x6 kernel, keeping the source untouched."""
    import numpy as np
    if (not isinstance(weight, np.ndarray) or weight.dtype != np.dtype('float64')
            or weight.ndim != 4 or weight.shape[2:] != (3, 3)
            or min(weight.shape) <= 0 or not np.isfinite(weight).all()):
        raise ValueError('Finite original float64 OI3x3 weights are required')
    derived = np.empty((*weight.shape[:2], 6, 6), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', under='ignore'):
        scaled = weight * .25
        for a in range(2):
            for b in range(2):
                derived[:, :, a::2, b::2] = scaled
    if not np.isfinite(derived).all():
        raise ValueError('Derived convolution coefficients must stay finite')
    derived.flags.writeable = False
    return derived, dict(source_shape=list(weight.shape), source_dtype=str(weight.dtype),
        derived_shape=list(derived.shape), derived_dtype=str(derived.dtype),
        source_elements=int(weight.size), derived_elements=int(derived.size),
        source_bytes=int(weight.nbytes), derived_bytes=int(derived.nbytes),
        source_weight_modified=False, derived_readonly=True, coefficient=.25,
        rule='Wf[o,c,2*r+a,2*s+b]=W[o,c,r,s]/4; r,s=0..2; a,b=0..1')


def _derived_tensor(template, name, shape):
    if is_dataclass(template):
        return replace(template, name=name, shape=tuple(shape))
    tensor = copy.copy(template)
    tensor.name, tensor.shape = name, tuple(shape)
    return tensor


def fuse_graph(graph):
    """Produce a new in-memory graph and explicit derived-constant provenance."""
    import numpy as np
    pairs = find_pairs(graph)
    for name, weight in graph.weights.items():
        if (not isinstance(weight, np.ndarray) or weight.dtype != np.dtype('float64')
                or weight.flags.writeable):
            raise ValueError('Original graph parameters must already be readonly float64 arrays: ' + name)
    result = copy.copy(graph)
    result.nodes = copy.deepcopy(graph.nodes)
    result.tensors = copy.deepcopy(graph.tensors)
    result.weights = dict(graph.weights)
    result.inputs, result.outputs = list(graph.inputs), list(graph.outputs)
    deleted = {row['pool_id'] for row in pairs}
    nodes = {node.id: node for node in result.nodes}
    for row in pairs:
        derived, inventory = derive_weight(graph.weights[row['source_weight']])
        conv = nodes[row['conv_id']]
        conv.inputs = [row['input']]
        conv.attrs.update(weight=row['derived_weight'], stride=(2, 2), padding=(2, 2),
                          dilation=(1, 1), groups=1)
        result.weights[row['derived_weight']] = derived
        result.tensors[row['derived_weight']] = _derived_tensor(
            graph.tensors[row['source_weight']], row['derived_weight'], derived.shape)
        del result.tensors[row['removed_tensor']]
        row['constant'] = inventory
        row['removed_pool_dense_constant_bytes'] = row['input_shape'][1] ** 2 * 4 * 8
    result.nodes = [node for node in result.nodes if node.id not in deleted]
    old_operators = dict(sorted(Counter(node.op for node in graph.nodes).items()))
    new_operators = dict(sorted(Counter(node.op for node in result.nodes).items()))
    source_bytes = sum(row['constant']['source_bytes'] for row in pairs)
    derived_bytes = sum(row['constant']['derived_bytes'] for row in pairs)
    pool_bytes = sum(row['removed_pool_dense_constant_bytes'] for row in pairs)
    metadata = dict(variant=VARIANT, model=MODEL, pair_count=4, pairs=pairs,
        original_graph_nodes=len(graph.nodes), fused_graph_nodes=len(result.nodes),
        original_operator_counts=old_operators, fused_operator_counts=new_operators,
        original_graph_modified=False, original_weights_modified=False,
        original_weight_keys_retained=list(graph.weights), derived_constant_count=4,
        source_fused_weights_bytes=source_bytes, derived_weight_storage_bytes=derived_bytes,
        new_graph_additional_storage_bytes=derived_bytes,
        net_fused_weight_expansion_bytes=derived_bytes-source_bytes,
        removed_export_pool_constant_bytes=pool_bytes,
        estimated_net_export_tensor_bytes=derived_bytes-source_bytes-pool_bytes,
        estimated_export_scope='Tensor payload relative to existing independent diagonal-pool export only; MLIR/layout expansion and compiler peak unmeasured',
        activation_ids_preserved=[node.id for node in graph.nodes if node.op == 'act'],
        real_number_equivalence=True,
        equivalence='Nonoverlapping full-cover2x2 pool followed immediately by original3x3 stride1 pad1 conv; fused6x6 stride2 pad2; original bias once',
        binary64_strict_math_passed=None, real_training_probes_run=0,
        HEIR_compile_attempts=0, HEIR_layout_success=None, task_accuracy_acceptance=None,
        original_model_state_unchanged=True, source_weight_values_serialized=False,
        exporter_or_pipeline_integrated=False)
    return result, metadata
