"""Describe output-reachable user-graph semantics, not merely DSL spelling.

Schema-1 catalog IDs are intentionally not reconstructed or counted as free-form
graph evidence. This analysis never evaluates a model or proves equivalence.
"""
from collections import Counter

from model_graph import OPS, array_shape, input_specs, validate_graph

OP_SEMANTICS = {
    'add':'add', 'multiply':'multiply', 'subtract':'subtract', 'negate':'negate',
    'square':'polynomial', 'power':'polynomial', 'linear':'linear', 'flatten':'flatten',
    'rotate':'rotation', 'conv1d':'conv', 'conv2d':'conv', 'avg_pool1d':'pool', 'avg_pool2d':'pool',
}


def dimensions(shape):
    return 'x'.join(map(str, shape))


def analyze_graph(data):
    checked = validate_graph(data)
    nodes = {node['id']: node for node in data['nodes']}
    active = set()
    pending = [data['output']]
    while pending:
        name = pending.pop()
        if name in nodes and name not in active:
            active.add(name)
            pending.extend(nodes[name]['inputs'])
    constants = data['constants']
    features = {f'inputs.{len(input_specs(data))}', f'output.shape.{dimensions(checked["output_shape"])}'}
    features.update('input.shape.' + dimensions(spec['shape']) for spec in input_specs(data))
    operators = Counter()
    linear_depth = {spec['name']:0 for spec in input_specs(data)}
    consumers = {}
    linear_shapes, spatial = [], []
    for node in data['nodes']:
        if node['id'] not in active:
            continue
        op = node['op']
        operators[op] += 1
        for value in set(node['inputs']):
            if value not in constants:
                consumers.setdefault(value, set()).add(node['id'])
        linear_depth[node['id']] = max((linear_depth.get(v, 0) for v in node['inputs']), default=0) + (op == 'linear')
        if op == 'linear':
            out_width, in_width = array_shape(constants[node['weight']])
            linear_shapes.append([in_width, out_width])
            features.add(f'linear.shape.{in_width}x{out_width}')
            features.add(f'linear.output_width.{out_width}')
        if op == 'power':
            features.add(f'power.exponent.{node["exponent"]}')
        if op == 'rotate':
            features.add(f'rotation.step.{node["step"]}')
        if op in ('add', 'multiply', 'subtract') and node['inputs'][1] in constants:
            shape = array_shape(constants[node['inputs'][1]])
            kind = 'scalar' if not shape else 'length1' if shape == (1,) else 'exact.' + dimensions(shape)
            features.add('broadcast.' + kind)
        if op.startswith('conv') or op.startswith('avg_pool'):
            rank = 1 if op.endswith('1d') else 2
            spec = dict(op=op, stride=list(node['stride']), padding=list(node['padding']))
            if op.startswith('conv'):
                spec.update(groups=node.get('groups', 1), dilation=list(node.get('dilation', [1]*rank)),
                            weight_shape=list(array_shape(constants[node['weight']])))
                features.add(f'{op}.groups.{spec["groups"]}')
                features.add(f'{op}.dilation.{dimensions(spec["dilation"])}')
            else:
                spec.update(kernel=list(node['kernel']), count_include_pad=node['count_include_pad'])
                features.add(f'{op}.count_include_pad.{str(spec["count_include_pad"]).lower()}')
            features.add(f'{op}.stride.{dimensions(spec["stride"])}')
            features.add(f'{op}.padding.{dimensions(spec["padding"])}')
            spatial.append(spec)
    depth = linear_depth.get(data['output'], 0)
    features.add(f'graph.linear_depth.{depth}')
    if any(len(users) > 1 for users in consumers.values()):
        features.add('graph.fanout')
    return dict(operators=dict(sorted(operators.items())), features=sorted(features),
                linear_shapes=linear_shapes, spatial_parameters=spatial,
                output_reachable_nodes=[n['id'] for n in data['nodes'] if n['id'] in active],
                dead_nodes=[n['id'] for n in data['nodes'] if n['id'] not in active],
                logical_input_count=len(input_specs(data)), linear_depth=depth,
                interpretation='Output reachability, not a proof of numerical influence or correctness')


def summarize_models(rows):
    """Rows must already be numerically audited; this function does not execute tests."""
    from dsl_semantic_inventory import SEMANTICS
    if set(OP_SEMANTICS) != set(OPS):
        raise ValueError('Every supported graph operator needs a semantic-inventory mapping')
    return dict(
        operator_matrix=[dict(operator=op, semantic=OP_SEMANTICS[op],
            support_status=SEMANTICS[OP_SEMANTICS[op]]['status'],
            tests=SEMANTICS[OP_SEMANTICS[op]]['tests'],
            observed_cases=[r['evidence'] for r in rows if op in r['model_semantics']['operators']]) for op in sorted(OPS)],
        feature_matrix=[dict(feature=feature, observed_cases=[r['evidence'] for r in rows
            if feature in r['model_semantics']['features']]) for feature in sorted({f for r in rows for f in r['model_semantics']['features']})],
        tests_executed_by_summary=False, full_model_domain_verified=False)
