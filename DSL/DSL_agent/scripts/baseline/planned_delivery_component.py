"""Delivery entry with frozen generic planning and original backend preflight.

The ordinary approved Provider and request contract are unchanged. The chosen
layout is already in its public request. Compiler feedback never enters delivery.
"""
from pathlib import Path
import time

import delivery_component as delivery
from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_planning import verify_plan


def generator_manifest(version, source_paths, *, plans, backend_identity=None, **kwargs):
    require(type(plans) is list and bool(plans), 'Frozen pre-generation plans required')
    import first_output_planning
    sources = [Path(__file__).absolute(), Path(first_output_planning.__file__).absolute(), *source_paths]
    capabilities = {sha(canonical(p['constraints'])): p['constraints'] for p in plans}
    expected = (next(iter(capabilities)) if len(capabilities) == 1 else sha(canonical(capabilities)))
    require(kwargs.get('capabilities_sha256', expected) == expected, 'Frozen capabilities differ from planning')
    kwargs['capabilities_sha256'] = expected
    mechanism = delivery.generator_manifest(version, sources, **kwargs)
    frozen = {}
    for plan in plans:
        checked_bound(plan)
        require(plan['request_id'] not in frozen, 'Duplicate pre-generation request')
        frozen[plan['request_id']] = plan
    return bound(dict({k: v for k, v in mechanism.items() if k != 'binding'},
        planning_contract='poseidon-pre-generation-plan-v1', pre_generation_plans=frozen,
        planned_backend_identity=backend_identity))


def _check_slot(request, mechanism, plan, output):
    delivery.verify_generator(mechanism); checked_bound(plan)
    require(sha(canonical(plan)) == mechanism['evaluation_policy_sha256'], 'Frozen delivery slots changed')
    slot = plan['slots'].get(request['request_id'])
    require(slot is not None and slot['request_sha256'] == sha(canonical(request))
            and slot['delivery_path'] == str(Path(output).absolute()), 'Exactly one frozen output slot required')
    require(slot['split'] != 'acceptance' or plan['independent_seal_verified'], 'Independent acceptance seal missing')
    import first_output_planning
    require(mechanism.get('planning_contract') == 'poseidon-pre-generation-plan-v1'
            and all(str(p) in mechanism['source_sha256'] for p in
                [Path(__file__).absolute(), Path(first_output_planning.__file__).absolute()]), 'Planning entry absent from frozen mechanism')
    frozen = mechanism['pre_generation_plans'][request['request_id']]
    verify_plan(frozen, request)
    return frozen


def _blocked(request, mechanism, plan, output, planning, failure):
    folder = delivery._folder(output)
    write_once(folder/'request.json', canonical(request)); write_once(folder/'generator.json', canonical(mechanism))
    write_once(folder/'delivery-plan.json', canonical(plan))
    write_once(folder/'pre-generation-plan.json', canonical(planning))
    write_once(folder/'attempt.json', canonical(bound(dict(format='poseidon-pre-generation-admission-v1',
        generations=0, HTTP_attempts=0, compilation=0, encrypted_execution=0, post_output_feedback=0))))
    receipt = bound(dict(format=delivery.VERSION, mode='delivery_generation', origin='pre_generation_admission',
        request_id=request['request_id'], mechanism_binding=mechanism['binding'], first_generation_index=None,
        post_output_repairs=0, real_Agent=False, external_evaluation_performed=False,
        request_sha256=sha(read_bytes(folder/'request.json')), generator_sha256=sha(read_bytes(folder/'generator.json')),
        delivery_plan_sha256=sha(read_bytes(folder/'delivery-plan.json')),
        response_sha256=None, source_sha256=None, source_bytes=0, status=failure,
        first_output_eligible=False, provider_metrics=None, planning_binding=planning['binding']))
    write_once(folder/'delivery.json', canonical(receipt))
    return receipt


def _admission_path(output):
    target = Path(output).absolute()
    return target.with_name(target.name+'.admission')


def verify_planned_delivery(output, receipt, mechanism):
    """Require the frozen pre-call host record, including for current assessment."""
    require(mechanism.get('planning_contract') == 'poseidon-pre-generation-plan-v1', 'Planning contract')
    planning = mechanism['pre_generation_plans'][receipt['request_id']]
    if receipt['origin'] == 'pre_generation_admission':
        require(receipt['first_output_eligible'] is False and receipt['source_sha256'] is None,
                'Blocked request cannot become an Agent output')
        return planning
    path = _admission_path(output)/'admission.json'
    record = checked_bound(delivery.strict_json(read_bytes(path).decode()))
    attempt = checked_bound(delivery.strict_json(read_bytes(Path(output)/'attempt.json').decode()))
    require(record['format'] == 'poseidon-delivery-pre-Provider-admission-v1'
            and record['request_id'] == receipt['request_id']
            and record['mechanism_binding'] == mechanism['binding']
            and record['planning_binding'] == planning['binding']
            and record['backend_identity'] == mechanism['planned_backend_identity']
            and record['recorded_at_ns'] <= attempt['started_at_ns']
            and record['Provider_calls'] == record['compiler_calls'] == record['encrypted_calls'] == 0,
            'Original backend admission did not precede this delivery')
    require(not receipt['real_Agent'] or record['test_only'] is False, 'Fixture backend cannot establish real Agent admission')
    return record


def generate_once(request, provider, output, *, mechanism, plan, backend=None,
                  backend_identity=None, batch=None, checkpoint=None, test_only=False):
    """Call original Provider at most once, after planning and backend admission.

    An unavailable SDK needs no Provider object and no credential-loader call.
    The trusted host constructs its approved Provider only after a prepared plan.
    """
    request = delivery.public_request(request)
    planning = _check_slot(request, mechanism, plan, output)
    require(type(test_only) is bool and (test_only or not planning['test_only']), 'Fixture plan cannot authorize a live delivery')
    if planning['status'] != 'prepared':
        return _blocked(request, mechanism, plan, output, planning, planning['status'])
    require(backend is not None and backend_identity is not None, 'Original backend admission required')
    require(backend_identity == mechanism.get('planned_backend_identity'), 'Backend identity differs from generator freeze')
    if not test_only:
        from normal_cpu_overlay_backend import trusted_backend
        trusted_backend(backend,approved_overlay_required=
            planning['constraints'].get('SDK_format')=='poseidon-normal-cpu-approved-overlay-v1')
    require(backend.identity_manifest() == backend_identity, 'Frozen backend identity changed before Provider')
    try:
        backend.check_request(request, 'numerical')
    except ValueError:
        return _blocked(request, mechanism, plan, output, planning, 'backend_blocked')
    require(backend.identity_manifest() == backend_identity, 'Backend changed during admission')
    require(provider is not None, 'Admitted delivery needs its approved fresh Provider')
    if test_only:
        require(not isinstance(provider._transport, delivery.HTTPSTransport), 'Fixture admission cannot authorize live transport')
    admission = _admission_path(output)
    require(admission.parent.resolve() == admission.parent, 'Indirect admission storage')
    if not test_only:
        require(batch is not None and any(admission.is_relative_to(a['path']) for a in batch.record()['artifacts']),
                'Admission outside accounted batch')
    admission.mkdir(mode=0o700)
    write_once(admission/'admission.json', canonical(bound(dict(format='poseidon-delivery-pre-Provider-admission-v1',
        request_id=request['request_id'], mechanism_binding=mechanism['binding'], planning_binding=planning['binding'],
        backend_identity=backend_identity, recorded_at_ns=time.time_ns(), test_only=test_only,
        Provider_calls=0, compiler_calls=0, encrypted_calls=0))))
    result = delivery.generate_once(request, provider, output, mechanism=mechanism, plan=plan,
        batch=batch, checkpoint=checkpoint)
    write_once(Path(output)/'pre-generation-plan.json', canonical(planning))
    verify_planned_delivery(output, result, mechanism)
    return result
