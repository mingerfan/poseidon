"""Versioned, pre-generation channel partitioning of public spatial graphs.

This produces another mathematical graph, never DSL or a candidate answer.
Original operator parameters and tensor entries are retained exactly. The caller
must bind both graphs and evaluate the original function independently. No
compiler/runtime feedback, model name, sample, or reference selects a partition.
"""
import copy
import hashlib
import json

from benchmark_graph import validate, require

VERSION = 'poseidon-pool-only-channel-normalization-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def normalize(model, *, channels_per_part=1, maximum_nodes=1024):
    require(type(channels_per_part) is int and 1 <= channels_per_part <= 16,
            'Finite spatial channel partition size')
    require(type(maximum_nodes) is int and 1 <= maximum_nodes <= 1024,
            'Finite spatial normalization graph budget')
    checked = validate(model)
    original_bytes = canonical(model)
    result = copy.deepcopy(model)
    result['nodes'] = []
    occupied = set(checked['shapes']) | {n['id'] for n in model['nodes']}
    serial = 0
    constants_by_value = {canonical(v): k for k, v in result['constants'].items()}
    changes = []

    def fresh():
        nonlocal serial
        while True:
            value = 'spatial' + str(serial)
            serial += 1
            if value not in occupied and value + 'v0' not in occupied:
                occupied.add(value); occupied.add(value + 'v0')
                return value

    def emit(op, refs, attrs, *, old=None):
        ident = old['id'] if old is not None else fresh()
        outputs = copy.deepcopy(old['outputs']) if old is not None else [ident + 'v0']
        result['nodes'].append(dict(id=ident, op=op, inputs=list(refs), attrs=copy.deepcopy(attrs), outputs=outputs))
        require(len(result['nodes']) <= maximum_nodes, 'Spatial normalization node budget')
        return outputs[0]

    def public(value):
        key = canonical(value)
        if key not in constants_by_value:
            name = fresh()
            result['constants'][name] = copy.deepcopy(value)
            constants_by_value[key] = name
        return constants_by_value[key]

    for node in model['nodes']:
        op = node['op']
        if op not in ('avg_pool1d', 'avg_pool2d'):
            result['nodes'].append(copy.deepcopy(node))
            require(len(result['nodes']) <= maximum_nodes, 'Spatial normalization node budget')
            continue
        rank = 1 if op.endswith('1d') else 2
        shape = checked['shapes'][node['inputs'][0]]
        axis = len(shape) - rank - 1
        channels = checked['shapes'][node['outputs'][0]][axis]
        if channels <= channels_per_part:
            result['nodes'].append(copy.deepcopy(node))
            require(len(result['nodes']) <= maximum_nodes, 'Spatial normalization node budget')
            continue
        convolution = False
        groups = node['attrs']['groups'] if convolution else 1
        output_per_group = channels // groups
        input_per_group = shape[axis] // groups
        start = 0
        slices = {}
        pieces = []
        records = []
        while start < channels:
            group = start // output_per_group
            stop = min(channels, start + channels_per_part, (group + 1) * output_per_group)
            lo, hi = (group * input_per_group, (group + 1) * input_per_group) if convolution else (start, stop)
            key = (lo, hi)
            if key not in slices:
                slices[key] = node['inputs'][0] if lo == 0 and hi == shape[axis] else emit(
                    'slice', [node['inputs'][0]], dict(axis=axis, start=lo, stop=hi, step=1))
            refs = [slices[key]]
            attrs = copy.deepcopy(node['attrs'])
            if convolution:
                refs.append(public(model['constants'][node['inputs'][1]][start:stop]))
                if len(node['inputs']) == 3:
                    refs.append(public(model['constants'][node['inputs'][2]][start:stop]))
                attrs['groups'] = 1
            pieces.append(emit(op, refs, attrs))
            records.append(dict(output_channels=[start, stop], input_channels=[lo, hi],
                                source_weight=node['inputs'][1] if convolution else None))
            start = stop
        while len(pieces) > 8:
            pieces = [emit('concat', pieces[i:i+8], dict(axis=axis)) if len(pieces[i:i+8]) > 1
                      else pieces[i] for i in range(0, len(pieces), 8)]
        emit('concat', pieces, dict(axis=axis), old=node)
        changes.append(dict(node_id=node['id'], operator=op, axis=axis, partitions=records))
    used = {ref for node in result['nodes'] for ref in node['inputs']}
    result['constants'] = {k: v for k, v in result['constants'].items() if k in used}
    normalized = validate(result)
    require(model['inputs'] == result['inputs'] and model['outputs'] == result['outputs'],
            'Spatial normalization changed public signature')
    require(all(normalized['shapes'][value] == dimensions for value, dimensions in checked['shapes'].items()
                if value in normalized['shapes']), 'Spatial normalization changed original value shapes')
    require(canonical(model) == original_bytes, 'Spatial normalization mutated input')
    receipt = dict(format=VERSION, original_model_sha256=hashlib.sha256(original_bytes).hexdigest(),
        normalized_model_sha256=hashlib.sha256(canonical(result)).hexdigest(),
        channels_per_part=channels_per_part, maximum_nodes=maximum_nodes,
        original_nodes=len(model['nodes']), normalized_nodes=len(result['nodes']), changes=changes,
        original_public_values_only=True, model_name_dispatch=False, DSL_generated=False,
        compiler_feedback_used=False, numerical_reference_used=False,
        resource_limits_increased=False, numerical_equivalence_requires_independent_check=True)
    receipt['binding'] = hashlib.sha256(canonical(receipt)).hexdigest()
    return result, receipt
