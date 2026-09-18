"""Export historical fixtures as self-contained user graphs, never as Agent output.

Family dispatch is confined to this compatibility exporter. The exported schema
contains actual public arrays and connections and needs no catalog at execution.
Historical descriptors, reports, IDs and success rates are not rewritten.
"""
import copy

from model_graph import validate_graph


def expand_legacy(descriptor):
    from model_catalog import build_model, validate_descriptor
    family, _ = validate_descriptor(descriptor)
    if descriptor['schema'] != 1:
        raise ValueError('This exporter accepts only the historical schema-1 fixtures')
    model, shape = build_model(descriptor)
    constants, nodes = {}, []

    def constant(name, tensor):
        constants[name] = tensor.detach().cpu().tolist()
        return name

    def node(name, op, *inputs, **attributes):
        nodes.append(dict(id=name, op=op, inputs=list(inputs), **attributes))
        return name

    if family in ('affine', 'polynomial', 'fanout', 'residual'):
        w, b = constant('w', model.w), constant('b', model.b)
        if family == 'polynomial':
            square = node('squared', 'square', 'x')
            quadratic = node('quadratic', 'multiply', square, w)
            linear = node('linear_term', 'multiply', 'x', b)
            output = node('sum_terms', 'add', quadratic, linear)
            output = node('out', 'add', output, constant('c', model.c))
        else:
            product = node('weighted', 'multiply', 'x', w)
            branch = node('branch', 'add', product, b)
            output = branch
            if family != 'affine':
                c = constant('c', model.c)
                squared = node('branch_squared', 'square', branch)
                if family == 'fanout':
                    input_square = node('input_squared', 'square', 'x')
                    other = node('other_branch', 'multiply', input_square, c)
                    output = node('out', 'add', squared, other)
                else:
                    scaled = node('scaled_branch', 'multiply', squared, c)
                    output = node('out', 'add', 'x', scaled)
    else:
        output = node('flat', 'flatten', 'x') if family == 'flatten_linear' else 'x'
        for index, layer in enumerate(model.layers):
            weight = constant(f'weight{index}', layer.weight)
            bias = constant(f'bias{index}', layer.bias)
            output = node(f'linear{index}', 'linear', output, weight=weight, bias=bias)
            if index+1 != len(model.layers):
                output = node(f'activation{index}', 'square', output)
    result = dict(schema=2, id='usergraph-'+descriptor['id'], input_shape=shape,
                  constants=constants, nodes=nodes, output=output)
    validate_graph(result)
    return result


def user_graph_suite():
    """Versioned new suite; original expanded_model_suite.manifest stays unchanged."""
    from expanded_model_suite import entries, FAMILIES, digest
    cases, provenance = [], []
    for item in entries():
        original = item['descriptor']
        graph = expand_legacy(original) if original['schema'] == 1 else copy.deepcopy(original)
        validate_graph(graph)
        cases.append(graph)
        provenance.append(dict(id=graph['id'], family=item['family'], configuration=item['configuration'],
            source_descriptor=copy.deepcopy(original), source_sha256=digest(original), graph_sha256=digest(graph)))
    return dict(schema=1, cases=cases), dict(version='user-graph-suite-v2', families=list(FAMILIES),
        entries=provenance, graph_set_sha256=digest(cases), historical_reports_unchanged=True,
        agent_generation_validated=False, source='trusted_fixture_export_not_agent_inference')


if __name__ == '__main__':
    import json
    manifest, provenance = user_graph_suite()
    print(json.dumps(dict(manifest=manifest, provenance=provenance), separators=(',', ':'), allow_nan=False))
