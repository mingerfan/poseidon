"""Export actual small RuntimePlan dependency windows around saved boot sites.

Only execution/value metadata is streamed. No initialization or external tensor
payload is read, and source-layer labels are left unmapped.
"""
from pathlib import Path

from .native_component_matrix import array_items


def bootstrap_windows(plan, inventory, *, radius=6):
    if type(radius) is not int or not 1 <= radius <= 16:
        raise ValueError('A bounded operation window is required')
    plan = Path(plan)
    if Path(inventory['source_runtime_plan']).resolve() != plan.resolve():
        raise ValueError('The saved inventory must bind this exact actual plan')
    sites = inventory['bootstrap_sites']
    if not sites or len(sites) != inventory['boots']:
        raise ValueError('The complete saved bootstrap inventory is required')
    selected = sorted({0, len(sites)//2, len(sites)-1})
    intervals = [(max(0, sites[i]['op_index']-radius),
                  min(inventory['exec_ops'], sites[i]['op_index']+radius+1)) for i in selected]
    needed = {index for start, end in intervals for index in range(start, end)}
    operations, total = {}, 0
    for index, op in enumerate(array_items(plan, 'execution')):
        if index in needed:
            attrs = op.get('attrs') or {}
            operations[index] = dict(execution_index=index, ordinal=op['ordinal'], op=op['op'],
                inputs=list(op.get('inputs', [])), output=op['output'],
                attrs={key: attrs[key] for key in ('operator_profile', 'implementation', 'rotation')
                       if key in attrs and type(attrs[key]) in (str, int, bool)})
        total = index+1
    if total != inventory['exec_ops'] or set(operations) != needed:
        raise ValueError('Actual execution windows differ from the accepted complete inventory')
    ids = {value for op in operations.values() for value in [*op['inputs'], op['output']]}
    values = {}
    for value in array_items(plan, 'values'):
        if value['id'] in ids:
            values[value['id']] = {key: value[key] for key in
                ('id', 'kind', 'type', 'level', 'scale_log2', 'components', 'ntt', 'context')
                if key in value and (value[key] is None or type(value[key]) in (str, int, float, bool))}
    if set(values) != ids:
        raise ValueError('Every exported operand must have original value metadata')
    windows = []
    for site_index, (start, end) in zip(selected, intervals):
        site = sites[site_index]
        nodes = [operations[index] for index in range(start, end)]
        target = operations[site['op_index']]
        if (target['op'] != 'boot' or target['ordinal'] != site['ordinal']
                or [values[v]['level'] for v in target['inputs']] != site['input_levels']
                or values[target['output']]['level'] != site['output_level']):
            raise ValueError('The selected real bootstrap location or physical levels differ')
        producers, edges, external = {}, [], set()
        for node in nodes:
            for position, operand in enumerate(node['inputs']):
                producer = producers.get(operand)
                if producer is None:
                    external.add(operand)
                edges.append(dict(value=operand, input_position=position,
                    source_execution_index=producer, target_execution_index=node['execution_index']))
            producers[node['output']] = node['execution_index']
        used = {v for node in nodes for v in [*node['inputs'], node['output']]}
        windows.append(dict(bootstrap_inventory_index=site_index, selected_bootstrap=dict(site),
            op_start=start, op_end=end, nodes=nodes, operand_edges=edges,
            values=[values[v] for v in sorted(used)], external_inputs=sorted(external),
            semantic_layer_label='unmapped',
            dependency_scope='Actual serialized operand reads and preceding writes inside this window; external predecessors are explicit'))
    return dict(source_runtime_plan=str(plan.absolute()), model=inventory['model'], system=inventory['system'],
        param_group=inventory['param_group'], complete_plan_ops=total, complete_plan_boots=inventory['boots'],
        selection='First, middle and last bootstrap in each separate compiled program',
        windows=windows, numerical_or_model_execution=False, external_payloads_read=False,
        inferred_semantic_layer_labels=0)
