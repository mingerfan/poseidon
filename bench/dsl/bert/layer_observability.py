"""Public layer-boundary metadata; no ciphertext snapshots or GPU execution."""
from __future__ import annotations

import re

_BOUNDARY = re.compile(r'^layer(\d+)\.ffn_ln\.output\.group(\d+)$')


def collect_layer_boundaries(graph, lowerer, plan, outputs):
    """Bind each reachable encoder's output hints to actual physical values.

    This is a plan audit, not per-layer timing or empirical execution counts.
    No intermediate value is added to final_outputs, so liveness is unchanged.
    """
    descriptions = {v['id']: v for v in plan['values']}
    producers = {}
    boots = []
    for phase in ('initialization', 'execution', 'finalization'):
        for row in plan[phase]:
            for ident in row.get('outputs', [row.get('output')]):
                if ident is not None:
                    producers[ident] = row['ordinal']
            if row.get('op') == 'boot':
                boots.append(row['ordinal'])
    layers = {}
    for node in graph.reachable(outputs):
        if node.op != 'boot':
            continue
        match = _BOUNDARY.fullmatch(str(node.data))
        if not match:
            continue
        layer, group = map(int, match.groups())
        physical = lowerer.lowered[node.index]
        value = descriptions[physical.id]
        layers.setdefault(layer, []).append(dict(group=group, value_id=physical.id,
            producer_ordinal=producers.get(physical.id), level=value['level'],
            scale_log2=value['scale_log2'], components=value['components']))
    calls = {str(c['call_id']): c for c in lowerer.prepared_calls}
    query_inputs = {}
    for row in plan['execution']:
        if row.get('op') != 'ApiCall' or row['attrs']['callee'] != 'bert.compact_linear':
            continue
        call = calls[str(row['attrs']['call_id'])]
        match = re.fullmatch(r'layer(\d+)\.q_weight', call['matrix_id'])
        if match and call['output_tile'] == 0:
            query_inputs[int(match.group(1))] = row['inputs']
    rows = []
    previous_values = None
    previous_end = -1
    for layer in sorted(layers):
        values = sorted(layers[layer], key=lambda v: v['group'])
        groups = [v['group'] for v in values]
        if groups != list(range(len(values))) or not values:
            raise ValueError('layer boundary groups are not unique and contiguous')
        if any(v['producer_ordinal'] is None for v in values):
            raise ValueError('layer boundary has no physical producer')
        end = max(v['producer_ordinal'] for v in values)
        if end <= previous_end:
            raise ValueError('physical layer boundaries are out of execution order')
        inputs = query_inputs.get(layer)
        rows.append(dict(layer=layer, label='layer' + str(layer), values=values,
            input_device_values=inputs,
            consumes_previous_boundary_directly=(None if previous_values is None else inputs == previous_values),
            end_ordinal=end, planned_native_boot_calls=sum(previous_end < x <= end for x in boots),
            boundary_matches_input=all(v['level'] == lowerer.initial and
                abs(v['scale_log2'] - lowerer.scales[lowerer.initial]) < 1e-9 for v in values)))
        previous_end = end
        previous_values = [v['value_id'] for v in values]
    if rows and [r['layer'] for r in rows] != list(range(len(rows))):
        raise ValueError('layer indices are not contiguous from zero')
    return dict(format_version=1, kind='physical-encoder-layer-boundaries',
        timing_measured=False, ciphertext_snapshots_retained=False, layers=rows,
        note='SSA/ordinal metadata only; final_outputs and release schedule are unchanged')

