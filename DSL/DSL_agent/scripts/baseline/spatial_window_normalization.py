"""Generic pre-generation window tiling, preserving the full mathematical graph.

Zero-padded boundary adaptation is deliberately left to the original operator.
For unpadded pools, each output tile reads exactly its original kernel windows,
including overlapping halos and gaps. No input/weight/reference/DSL is generated.
"""
import copy
import hashlib
import itertools
import json
import math

from benchmark_graph import validate, require

VERSION = 'poseidon-spatial-window-normalization-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def normalize(model, *, maximum_tile_output_elements=64, maximum_nodes=1024):
    require(type(maximum_tile_output_elements) is int and 1 <= maximum_tile_output_elements <= 256,
            'Finite spatial output tile size')
    require(type(maximum_nodes) is int and 1 <= maximum_nodes <= 1024, 'Finite spatial normalization node budget')
    checked = validate(model)
    original_bytes = canonical(model)
    result = copy.deepcopy(model); result['nodes'] = []
    occupied = set(checked['shapes']) | {node['id'] for node in model['nodes']}
    serial = 0
    changes, unchanged = [], []

    def fresh():
        nonlocal serial
        while True:
            name = 'window' + str(serial); serial += 1
            if name not in occupied and name + 'v0' not in occupied:
                occupied.add(name); occupied.add(name + 'v0')
                return name

    def emit(op, refs, attrs, *, old=None):
        ident = old['id'] if old is not None else fresh()
        outputs = copy.deepcopy(old['outputs']) if old is not None else [ident + 'v0']
        result['nodes'].append(dict(id=ident, op=op, inputs=list(refs), attrs=copy.deepcopy(attrs), outputs=outputs))
        require(len(result['nodes']) <= maximum_nodes, 'Spatial window normalization node budget')
        return outputs[0]

    def concat(pieces, axis, *, old=None):
        require(bool(pieces), 'Nonempty tile concatenation')
        while len(pieces) > 8:
            pieces = [emit('concat', pieces[i:i+8], dict(axis=axis)) if len(pieces[i:i+8]) > 1
                      else pieces[i] for i in range(0, len(pieces), 8)]
        if len(pieces) == 1 and old is None:
            return pieces[0]
        require(len(pieces) >= 2, 'Changed pool requires at least two final pieces')
        return emit('concat', pieces, dict(axis=axis), old=old)

    for node in model['nodes']:
        if node['op'] not in ('avg_pool1d', 'avg_pool2d'):
            result['nodes'].append(copy.deepcopy(node))
            require(len(result['nodes']) <= maximum_nodes, 'Spatial window normalization node budget')
            continue
        rank = 1 if node['op'].endswith('1d') else 2
        source = checked['shapes'][node['inputs'][0]]
        target = checked['shapes'][node['outputs'][0]]
        leading = math.prod(target[:-rank])
        reason = ('already_within_tile_size' if math.prod(target) <= maximum_tile_output_elements else
                  'padding_requires_boundary_adapter' if any(node['attrs']['padding']) else
                  'leading_dimensions_exceed_tile_size' if leading > maximum_tile_output_elements else None)
        if reason is not None:
            result['nodes'].append(copy.deepcopy(node))
            require(len(result['nodes']) <= maximum_nodes, 'Spatial window normalization node budget')
            unchanged.append(dict(node_id=node['id'], reason=reason))
            continue
        budget = maximum_tile_output_elements // leading
        extent = [1] * rank
        for axis in reversed(range(rank)):
            extent[axis] = min(target[len(target)-rank+axis], budget)
            budget = max(1, budget // extent[axis])
        axes = list(range(len(source)-rank, len(source)))
        starts = [list(range(0, target[axis], width)) for axis, width in zip(axes, extent)]
        tile_values, tile_records, slice_cache = {}, [], {}
        for coordinates in itertools.product(*starts):
            value = node['inputs'][0]; current_shape = list(source); spans = []
            for local_axis, (axis, start) in enumerate(zip(axes, coordinates)):
                stop = min(start + extent[local_axis], target[axis])
                stride = node['attrs']['stride'][local_axis]
                kernel = node['attrs']['kernel'][local_axis]
                lo, hi = start * stride, (stop-1) * stride + kernel
                require(0 <= lo < hi <= current_shape[axis], 'Pool tile exceeds original valid source')
                if lo != 0 or hi != current_shape[axis]:
                    key = value, axis, lo, hi
                    if key not in slice_cache:
                        slice_cache[key] = emit('slice', [value], dict(axis=axis, start=lo, stop=hi, step=1))
                    value = slice_cache[key]
                    current_shape[axis] = hi - lo
                spans.append(dict(axis=axis, output=[start,stop], source=[lo,hi]))
            pooled = emit(node['op'], [value], node['attrs'])
            tile_values[coordinates] = pooled
            volume = leading * math.prod(s['output'][1]-s['output'][0] for s in spans)
            require(volume <= maximum_tile_output_elements, 'Pool tile volume exceeds finite bound')
            tile_records.append(dict(coordinates=list(coordinates), spans=spans, output_elements=volume))
        require(len(tile_values) >= 2, 'Oversized pool did not partition')
        # Reassemble C-order spatial axes from the innermost axis outwards.
        for local_axis in reversed(range(rank)):
            groups = {}
            for coord, value in tile_values.items():
                key = coord[:local_axis]
                groups.setdefault(key, []).append((coord[local_axis], value))
            combined = {}
            for key, entries in groups.items():
                entries.sort()
                final = local_axis == 0
                if len(entries) == 1 and final:
                    # This can only occur if an inner axis alone was tiled;
                    # preserve the original name with its final inner concat.
                    require(len(starts[0]) == 1 and rank == 2, 'Unexpected single outer tile')
                    last = result['nodes'][-1]
                    require(last['outputs'] == [entries[0][1]] and last['op'] == 'concat', 'Missing final inner concatenation')
                    last['id'] = node['id']; last['outputs'] = copy.deepcopy(node['outputs'])
                    combined[key] = node['outputs'][0]
                else:
                    combined[key] = concat([v for _,v in entries], axes[local_axis], old=node if final else None)
            tile_values = combined
        changes.append(dict(node_id=node['id'], operator=node['op'], source_shape=source, target_shape=target,
            tile_spatial_extent=extent, tiles=tile_records, padding_unchanged=node['attrs']['padding'],
            stride_unchanged=node['attrs']['stride'], kernel_unchanged=node['attrs']['kernel']))
    normalized = validate(result)
    require(model['inputs'] == result['inputs'] and model['outputs'] == result['outputs']
            and model['constants'] == result['constants'], 'Window tiling changed public model signature/weights')
    require(all(normalized['shapes'][name] == shape for name,shape in checked['shapes'].items()),
            'Window tiling changed an original value shape')
    require(canonical(model) == original_bytes, 'Window tiling mutated input')
    receipt = dict(format=VERSION, original_model_sha256=hashlib.sha256(original_bytes).hexdigest(),
        normalized_model_sha256=hashlib.sha256(canonical(result)).hexdigest(), maximum_nodes=maximum_nodes,
        maximum_tile_output_elements=maximum_tile_output_elements, changes=changes, unchanged=unchanged,
        original_nodes=len(model['nodes']), normalized_nodes=len(result['nodes']), original_weights_preserved=True,
        original_input_and_output_shapes_preserved=True, model_name_dispatch=False, numerical_reference_used=False,
        compiler_feedback_used=False, DSL_generated=False, resource_limits_increased=False,
        scope='unpadded spatial pools only; no backend/CKKS or first-output certificate')
    receipt['binding'] = hashlib.sha256(canonical(receipt)).hexdigest()
    return result, receipt
