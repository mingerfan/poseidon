"""Explicit precision budgets using original domains and fitting evidence."""

import copy
import math

from . import stable_softmax_bridge as original


SQUARE_VARIANT = 'unit-width-public-square-v2'
NEWTON_VARIANT = 'newton-budget7-v1'
MAX_STEPS = 11
MAX_NEWTON = 7


def unit_width_policy(observations, key_count):
    from ..margin import minimum_domain
    lo, hi = original._interval(observations,'softmax scores')
    if lo == hi:
        raise ValueError('A constant interval requires an explicit exact decomposition')
    domain = minimum_domain(lo,hi)
    width = domain[1]-domain[0]
    steps = max(0,math.ceil(math.log2(width)))
    if steps > MAX_STEPS:
        raise ValueError('The explicit unit-width variant supports required k<=11')
    policy = original.derive_policy(observations,key_count)
    previous = copy.deepcopy(policy)
    shift = policy['shift']
    a,b = ((value-shift)*2.**-steps for value in domain)
    low,high = math.exp(a),math.exp(b)
    scale = key_count*high
    epsilon = original.transformer_rule.RELATIVE_TARGET/2**steps
    exact = (low/scale,1.)
    robust = ((1-epsilon)*exact[0],1+epsilon)
    policy.update(variant=SQUARE_VARIANT,steps=steps,inv_scale=scale,exp_domain=(a,b),
        inv_domain=(math.nextafter(robust[0],-math.inf),math.nextafter(robust[1],math.inf)),
        exp_relative_error_contract=epsilon,maximum_steps=MAX_STEPS,
        maximum_initial_exponent_width=1.,actual_initial_exponent_width=b-a,
        previous_variant=previous,additional_square_rounds=steps-previous['steps'])
    policy['proof'].update(exact_exp_range=(low,high),exact_scaled_initial_sum=exact,
        relative_exp_error_scaled_initial_sum=robust,continuous_polynomial_error_certificate=False)
    if not (0 < policy['inv_domain'][0] < policy['inv_domain'][1]
            and all(math.isfinite(value) for value in policy['inv_domain'])
            and 0 < b-a <= 1.+1e-15):
        raise ValueError('The explicit unit-width inverse envelope is not finite and positive')
    return policy


def build_unit_width_bridge(graph, calibration, *, raw_issues=()):
    result = original.build_bridge(graph,calibration,raw_issues=raw_issues)
    result['variant'] = SQUARE_VARIANT
    result['maximum_steps'] = MAX_STEPS
    for row in result['rows']:
        row['policy'] = unit_width_policy(calibration[row['source_calibration']],
                                         row['policy']['key_count'])
    result['additional_square_rounds'] = sum(row['policy']['additional_square_rounds'] for row in result['rows'])
    return result


def extend_newton_budget(report):
    """Admit diagnostic iterations6/7 as a separately named allowed budget.

    This uses recorded numerical histories, rather than asserting a continuous
    fitting certificate.  Domains, initial degrees, fits and targets stay as
    recorded. Existing report objects and old legal candidates remain intact.
    """
    result = copy.deepcopy(report)
    fn = result['fn']
    if fn not in ('inv','rsqrt'):
        return result
    target = result['target']
    if not isinstance(target,(int,float)) or not math.isfinite(target) or target <= 0:
        raise ValueError('The original positive fitting target is required')
    added = []
    seen = set()
    existing = {(candidate.get('fit'),candidate.get('deg0'),candidate.get('iters')):candidate
                for candidate in result['candidates']
                if candidate.get('method')=='newton' and candidate.get('precision_variant')==NEWTON_VARIANT}
    for candidate in result['candidates']:
        if candidate.get('method') != 'newton' or candidate.get('iters') != 5:
            continue
        profile = candidate.get('profile')
        if profile is None:
            continue
        identity = (candidate['fit'],candidate['deg0'])
        if identity in seen:
            raise ValueError('Duplicate original Newton diagnostic profile')
        seen.add(identity)
        low,high = profile['minimum_initial_z'],profile['maximum_initial_z']
        convergence_upper = 2. if fn == 'inv' else math.sqrt(3.)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high < convergence_upper):
            continue
        history = profile.get('newton_model',{}).get('history',[])
        for iterations in range(6,min(MAX_NEWTON,len(history)-1)+1):
            error = history[iterations]
            if not isinstance(error,(int,float)) or not math.isfinite(error) or error < 0:
                raise ValueError('Recorded Newton diagnostics must remain finite and nonnegative')
            extended = dict(method='newton',fit=candidate['fit'],deg0=candidate['deg0'],
                iters=iterations,depth=profile['depth']+(2 if fn=='inv' else 3)*iterations,
                relative_error=error,feasible=error<=target,profile=profile,
                precision_variant=NEWTON_VARIANT,source='existing diagnostic history beyond original allowed max5',
                continuous_error_certificate=False)
            identity = (candidate['fit'],candidate['deg0'],iterations)
            if identity in existing:
                if existing[identity] != extended:
                    raise ValueError('Existing extended candidate differs from its original numerical history')
            else:
                added.append(extended)
    result['candidates'].extend(added)
    accepted = [candidate for candidate in result['candidates'] if candidate.get('feasible')]
    choice = min(accepted,key=lambda candidate:(candidate['depth'],candidate['relative_error'],
        candidate['method'],candidate.get('deg',candidate.get('deg0',0)),candidate.get('iters',0),candidate['fit'])) if accepted else None
    result['selected'] = ({key:choice[key] for key in ('method','deg','deg0','iters','fit') if key in choice}
                          if choice is not None else None)
    result['selected_depth'] = choice['depth'] if choice is not None else None
    result['status'] = 'selected' if choice is not None else 'infeasible'
    result.update(precision_variant=NEWTON_VARIANT,old_maximum_newton_iterations=5,
                  maximum_newton_iterations=MAX_NEWTON,additional_candidates=len(added),
                  continuous_error_certificate=False)
    return result


def recorded_selector(records, fallback, *, newton_budget=5, observe=None):
    """Reuse matching original fit requests and record each fresh request."""
    if newton_budget not in (5,7):
        raise ValueError('The explicitly supported Newton budgets are5 and7')
    cache = {}

    def key(fn,domain,settings):
        return (fn,tuple(map(float,domain)),float(settings.get('target',.001)),
                bool(settings.get('exp_ladder',False)))

    for item in records:
        if item.get('kind') != 'end':
            continue
        if item.get('settings',{}).get('context') is not None:
            raise ValueError('Contextual reports require their separate original selector')
        identity = key(item['fn'],item['domain'],item.get('settings',{}))
        report = item['report']
        if (report['fn'] != item['fn'] or tuple(report['domain']) != identity[1]
                or report['target'] != identity[2]):
            raise ValueError('Recorded fitting request and numerical report differ')
        if identity in cache and cache[identity] != report:
            raise ValueError('Repeated exact fitting requests have different recorded numerical evidence')
        cache.setdefault(identity,report)
    statistics = dict(reused_reports=0,fresh_fit_calls=0,newton_budget=newton_budget)

    def select(fn,domain,**settings):
        identity = key(fn,domain,settings)
        if identity in cache:
            report = copy.deepcopy(cache[identity])
            statistics['reused_reports'] += 1
            source = 'source-matched recorded request/report'
        else:
            statistics['fresh_fit_calls'] += 1
            if observe is not None:
                observe()
            report = fallback(fn,domain,**settings)
            source = 'fresh original registered fitting function'
        if (report['fn'] != fn or tuple(map(float,report['domain'])) != identity[1]
                or report['target'] != identity[2]):
            raise ValueError('The selected fitting report differs from its exact request')
        if newton_budget == 7:
            report = extend_newton_budget(report)
        report = copy.deepcopy(report)
        report['fitting_evidence_source'] = source
        return report

    return select,statistics
