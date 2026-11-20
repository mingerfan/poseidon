"""Calibration/analytical domains for the existing square-normalize algorithm.

All public parameters derive from original score observations.  Raw unshifted
exp-sums are retained as diagnostics and are not inputs of this decomposition.
No candidate is frozen and no continuous polynomial certificate is claimed.
"""

import copy
import math
import threading

from ..margin import minimum_domain
from . import transformer_rule


VARIANT = 'minimum-domain-public-shift-square-v1'
_RULE_LOCK = threading.RLock()


def _interval(record, name):
    try:
        lo, hi = float(record['min']), float(record['max'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Missing original calibration interval: '+name) from exc
    if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
        raise ValueError('Invalid original calibration interval: '+name)
    return lo, hi


def derive_policy(calibration, key_count):
    """Keep the original algorithm and derive a full signed-domain envelope.

    The existing adjustment helper supplies its public shift and step-inverse
    margin.  The registered minimum_domain handles positive, negative and
    crossing-zero observations.  Exponential approximation error is explicitly
    included in the initial inverse interval; its bound remains a fit contract
    until the registered numerical fitting diagnostics validate it.
    """
    if type(key_count) is not int or key_count < 1:
        raise ValueError('A positive upper bound on effective key count is required')
    lo, hi = _interval(calibration, 'softmax scores')
    if lo == hi:
        raise ValueError('A constant score interval needs a separately recorded exact decomposition')
    baseline = transformer_rule.squared_softmax_adjustments(calibration, key_count)
    raw_domain = minimum_domain(lo, hi)
    width = raw_domain[1]-raw_domain[0]
    steps = min(max(0, math.ceil(math.log2(width))), 8)
    shift = baseline['shift']
    a, b = ((value-shift)*(2.**-steps) for value in raw_domain)
    lower, upper_exp = math.exp(a), math.exp(b)
    scale = key_count*upper_exp
    if not all(math.isfinite(v) and v > 0 for v in (lower, upper_exp, scale)):
        raise ValueError('The existing steps0..8 family has no finite positive scaled exponential range')
    exp_error = transformer_rule.RELATIVE_TARGET/(2**steps)
    inverse_error = transformer_rule.RELATIVE_TARGET
    exact_initial = (lower/scale, 1.)
    robust_initial = ((1-exp_error)*exact_initial[0], 1+exp_error)
    if robust_initial[0] <= 0:
        raise ValueError('A representable positive inverse lower bound is required')
    inverse_domain = (math.nextafter(robust_initial[0], -math.inf),
                      math.nextafter(robust_initial[1], math.inf))
    if inverse_domain[0] <= 0 or not all(map(math.isfinite, inverse_domain)):
        raise ValueError('The outward-rounded inverse interval must remain finite and positive')
    step_domain = tuple(baseline['step_inv_domain'])
    squared_bound = ((1-inverse_error)**2/key_count, (1+inverse_error)**2)
    if not step_domain[0] <= squared_bound[0] <= squared_bound[1] <= step_domain[1]:
        raise ValueError('The original step-inverse domain does not cover the relative-error envelope')
    return dict(variant=VARIANT, key_count=key_count, original_score_interval=(lo, hi),
        minimum_raw_domain=raw_domain, shift=shift, steps=steps, inv_scale=scale,
        exp_domain=(a, b), inv_domain=inverse_domain, step_inv_domain=step_domain,
        exp_relative_error_contract=exp_error, inverse_relative_error_contract=inverse_error,
        proof=dict(at_least_one_effective_key_required=True, maximum_effective_keys=key_count,
            exact_exp_range=(lower, upper_exp), exact_scaled_initial_sum=exact_initial,
            relative_exp_error_scaled_initial_sum=robust_initial,
            exact_square_sum=(1/key_count, 1.), relative_inverse_error_square_sum=squared_bound,
            initial_inverse_rounding='one outward float64 ULP on each analytically derived endpoint',
            exponent_error_requires_registered_fit_evidence=True,
            continuous_polynomial_error_certificate=False),
        baseline_adjustment=baseline,
        parameter_changes_from_baseline=['minimum-domain signed interval envelope',
            'steps derived from that envelope', 'initial inverse includes exponential relative-error contract'])


def build_bridge(graph, calibration, *, raw_issues=()):
    rows, required = [], set()
    for node in graph.nodes:
        if node.op == 'act':
            required.add(node.inputs[0])
        elif node.op in ('rmsnorm', 'layernorm'):
            required.add(node.outputs[0]+'::rsqrt')
        elif node.op == 'softmax':
            key = node.outputs[0]+'::exp'
            required.add(key)
            shape = tuple(graph.tensors[node.inputs[0]].shape)
            if len(shape) not in (3, 4) or node.attrs.get('keep') not in ('all', 'causal'):
                raise ValueError('Only the original token/score attention semantics are supported')
            policy = derive_policy(calibration.get(key, {}), shape[-1])
            rows.append(dict(node_id=node.id, output=node.outputs[0], score_input=node.inputs[0],
                score_shape=list(shape), keep=node.attrs['keep'], mask=node.attrs.get('mask'),
                source_calibration=key, policy=policy,
                approximation_sources=dict(exp='original training score interval and fixed public affine transform',
                    inv='positive exponential sum, effective key count and explicit exponential fit-error contract',
                    step_inv='Cauchy bound on squared positive probabilities and inverse fit-error contract')))
    for key in required:
        _interval(calibration.get(key, {}), key)
    return dict(variant=VARIANT, graph=graph.name, softmax_nodes=len(rows), rows=rows,
        required_original_points=sorted(required), required_original_points_ready=True,
        selected_softmax_roles=sum(2+int(row['policy']['steps']>0) for row in rows),
        raw_issues=copy.deepcopy(list(raw_issues)), raw_unshifted_inverse_used=False,
        actual_score_coverage_measured=False, exp_fit_error_validated=False,
        polynomial_quality_measured=False, candidate_frozen=False)


def observe_softmax(scores, mask, row, *, reference=None):
    """Compare the selected exact decomposition with an independent oracle."""
    import numpy as np
    from ..kernels.stable_softmax_reference import fixed_shift_squaring
    if reference is not None:
        fixed_shift_squaring = reference
    x = np.asarray(scores)
    if x.dtype != np.float64 or tuple(x.shape) != tuple(row['score_shape']) or not np.isfinite(x).all():
        raise ValueError('Original finite float64 score shape differs from its source metadata')
    keep = np.ones(x.shape, dtype=bool)
    if row['keep'] == 'causal':
        keep &= np.arange(x.shape[-1])[None, :] <= np.arange(x.shape[-2])[:, None]
    if row['mask'] is not None:
        key_mask = np.asarray(mask)
        if (key_mask.dtype != np.float64 or key_mask.shape != (len(x), x.shape[-1])
                or not np.isin(key_mask, (0., 1.)).all()):
            raise ValueError('The original binary attention mask differs')
        keep &= key_mask.reshape((len(x),)+(1,)*(x.ndim-2)+(x.shape[-1],)).astype(bool)
    policy = row['policy']
    observed, stages = fixed_shift_squaring(x, keep, shift=policy['shift'], steps=policy['steps'])
    masked = np.where(keep, x, -np.inf)
    exponents = np.exp(masked-masked.max(-1, keepdims=True))
    expected = exponents/exponents.sum(-1, keepdims=True)
    difference = np.abs(observed-expected)
    tolerance = 1e-12+1e-12*np.abs(expected)
    domains = []

    def record(role, low, high, domain, source):
        passed = domain[0] <= low <= high <= domain[1]
        domains.append(dict(role=role, observed_min=low, observed_max=high, domain=list(domain),
                            passed=bool(passed), source=source, exact_decomposition_observation=True))
        if not passed:
            raise ValueError('The selected stable decomposition left its declared '+role+' domain')

    record('exp', stages['initial_exponent_min'], stages['initial_exponent_max'],
           policy['exp_domain'], 'real original training scores after fixed public transform')
    initial = stages['stages'][0]
    record('inv', initial['sum_min']/policy['inv_scale'], initial['sum_max']/policy['inv_scale'],
           policy['inv_domain'], 'real scaled exponential sum plus analytical fit-error envelope')
    for stage in stages['stages'][1:]:
        record('step_inv['+str(stage['index'])+']', stage['sum_min'], stage['sum_max'],
               policy['step_inv_domain'], 'real square sum and registered analytical inverse domain')
    result = dict(node_id=row['node_id'], output=row['output'], passed=bool(np.all(difference <= tolerance)),
        max_absolute_difference=float(difference.max()), max_tolerance_ratio=float((difference/tolerance).max()),
        criterion='1e-12+1e-12*abs(reference)', finite_elements=int(observed.size), domains=domains,
        effective_key_min=int(keep.sum(-1).min()), effective_key_max=int(keep.sum(-1).max()),
        source='same original real score tensor; independent float64 stable softmax reference',
        original_framework_float32_reference_replaced=False, approximation_fit_or_quality_measured=False)
    if not result['passed']:
        raise ValueError('The stable decomposition differs from the independent float64 softmax')
    return observed, result


def plan_with_bridge(graph, calibration, bridge, *, slots=16384, placement='depth-dp', profile='qwen48',
                     fit_observer=None, policy_builder=None, approximation_selector=None):
    """Use the existing trusted planner with explicit new algorithm parameters."""
    expected = (policy_builder or build_bridge)(graph, calibration, raw_issues=bridge.get('raw_issues', ()))

    def plain(value):
        # Persisted JSON uses lists for both original tuple/list containers.
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        return value

    if plain(bridge['rows']) != plain(expected['rows']) or not bridge.get('actual_score_coverage_measured'):
        raise ValueError('Count planning requires the source-matched measured bridge')
    policies = {(tuple(row['policy']['original_score_interval']), row['policy']['key_count']): row['policy']
                for row in bridge['rows']}

    def adjusted(observations, key_count):
        policy = policies.get((_interval(observations, 'score interval'), key_count))
        if policy is None:
            raise ValueError('A planner softmax lacks its measured source policy')
        return {key: tuple(policy[key]) if key.endswith('_domain') else policy[key]
                for key in ('shift', 'steps', 'inv_scale', 'exp_domain',
                            'inv_domain', 'step_inv_domain')}

    with _RULE_LOCK:
        original = transformer_rule.squared_softmax_adjustments
        original_selector = transformer_rule.select_approximation

        def selected(fn, domain, **kwargs):
            request = dict(fn=fn, domain=list(domain), settings=copy.deepcopy(kwargs))
            if fit_observer is not None:
                fit_observer('begin', request)
            try:
                report = (approximation_selector or original_selector)(fn, domain, **kwargs)
            except BaseException as exc:
                if fit_observer is not None:
                    fit_observer('error', dict(request, error_type=type(exc).__name__, error=str(exc)))
                raise
            if fit_observer is not None:
                fit_observer('end', dict(request, report=report))
            return report

        transformer_rule.squared_softmax_adjustments = adjusted
        transformer_rule.select_approximation = selected
        try:
            return transformer_rule.plan(graph, calibration, slots=slots, placement=placement, profile=profile)
        finally:
            transformer_rule.squared_softmax_adjustments = original
            transformer_rule.select_approximation = original_selector
