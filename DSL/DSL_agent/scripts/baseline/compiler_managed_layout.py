"""Pre-output geometry selection from registered layouts, without DSL answers."""
import copy
import math
from benchmark_graph import validate, graph_resource_profile, require
from unified_capacity_layout import layout, WIDE_PERIODS, physical_limits
from unified_capacity_spatial import geometry, OPS, MAX_ROUTE_GROUPS
from public_basis_packing import VERSION
from public_weight_context import POLICY, REFERENCE_POLICY


def select(model, options):
    result = copy.deepcopy(options)
    explicit = result.get('capacity_packing_version')
    if explicit is not None and explicit != VERSION:
        return result
    check = validate(model)
    spatial = [geometry(n, check['shapes']) for n in model['nodes'] if n['op'] in OPS]
    if not spatial and explicit != VERSION:
        return result
    work = max([math.prod(g['target']) * math.prod(g['kernel']) *
                (g['source'][1] // g['groups'] if g['conv'] else 1) for g in spatial] or [0])
    period = result.get('chunk_period')
    if explicit is None:
        default = period if period is not None else layout(model)['input_slot_period']
        if work <= MAX_ROUTE_GROUPS * default:
            return result
    if period is None:
        profile = graph_resource_profile(model)
        _, maximum_inputs, maximum_outputs = physical_limits(profile, 'unified-capacity-chunked-inputs-v4')
        candidates = [p for p in WIDE_PERIODS if work <= MAX_ROUTE_GROUPS * p and
                      sum(math.ceil(math.prod(s['shape']) / p) for s in model['inputs']) <= maximum_inputs and
                      sum(math.ceil(math.prod(check['shapes'][o['value']]) / p) for o in model['outputs']) <= maximum_outputs]
        require(candidates, 'No registered whole-graph geometry period')
        period = min(candidates)
    result['chunk_period'] = period
    result['capacity_packing_version'] = VERSION
    result.setdefault('public_weight_policy', REFERENCE_POLICY)
    require(result['public_weight_policy'] in (POLICY,REFERENCE_POLICY), 'Public basis uses a registered hash/shape descriptor policy')
    return result
