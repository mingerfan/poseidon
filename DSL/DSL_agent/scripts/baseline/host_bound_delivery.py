"""Explicit host/public request binding for original, single-output delivery.

The immutable existing Provider projection and original compiler keep their
different data representations. This adapter freezes their relationship before
generation. It never repairs a response or modifies a compiler input.
"""
import copy
from pathlib import Path
import time

import delivery_component as delivery
from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from deepseek_provider import public_request, HTTPSTransport
from first_output_planning import verify_plan
from first_output_assessment import evaluate_witness, BLOCKED

VERSION = 'poseidon-host-bound-first-output-v1'
BINDING = 'poseidon-frozen-host-and-public-request-v1'
ADMISSION = 'poseidon-host-bound-pre-Provider-admission-v1'


def _json(path):
    from first_output_backend import _json as read_host_json
    from public_weight_context import TRUSTED_BYTES
    read_bytes(path, TRUSTED_BYTES)  # Existing 4MiB host bound; response cap is unchanged.
    return read_host_json(path)


def _full_request(value):
    from component_contract import reconstruct_request
    require(type(value) is dict, 'Full host request required')
    reconstructed = reconstruct_request(copy.deepcopy(value))
    require(canonical(reconstructed) == canonical(value), 'Full host request reconstruction changed')
    return value


def freeze_host_request(path):
    """Bind a reviewed full public mathematical request and its exact projection."""
    path = Path(path).absolute()
    require(path.suffix == '.json', 'Full host request JSON asset required')
    raw = read_bytes(path)
    host = _full_request(_json(path))
    wire = public_request(copy.deepcopy(host))
    require(wire['request_id'] == host['request_id'], 'Projection changed request identity')
    return bound(dict(format=BINDING, path=str(path), sha256=sha(raw), request_id=host['request_id'],
        host_canonical_sha256=sha(canonical(host)), public_canonical_sha256=sha(canonical(wire))))


def resolve_host(binding, wire=None, *, mechanism=None):
    checked_bound(binding)
    require(set(binding) == {'format', 'path', 'sha256', 'request_id', 'host_canonical_sha256',
                            'public_canonical_sha256', 'binding'} and binding['format'] == BINDING,
            'Frozen host/public binding fields')
    require(freeze_host_request(binding['path']) == binding, 'Frozen complete host request changed')
    if mechanism is not None:
        require(mechanism['source_sha256'].get(binding['path']) == binding['sha256'],
                'Host asset absent from complete generator freeze')
    host = _json(binding['path'])
    if wire is not None:
        require(canonical(wire) == canonical(public_request(copy.deepcopy(host))),
                'Public delivery differs from frozen host projection')
    return host


def generator_manifest(version, source_paths, *, host_bindings, plans, assessment_paths,
                       backend_identity, baseline_binding, evaluation_policy_sha256,
                       provider_config):
    """Freeze the full mechanism, host assets, public slots and external locations."""
    import first_output_planning
    require(type(host_bindings) is list and type(plans) is list and bool(host_bindings), 'Host request/planning set')
    bindings = {b['request_id']: b for b in host_bindings}
    planning = {p['request_id']: p for p in plans}
    require(len(bindings) == len(host_bindings) == len(plans) == len(planning)
            and set(bindings) == set(planning) == set(assessment_paths), 'Unique matching request set')
    frozen_paths = {}
    capabilities = {}
    for key, binding in bindings.items():
        host = resolve_host(binding)
        verify_plan(planning[key], host)
        output = Path(assessment_paths[key]).absolute()
        require(output.parent.resolve() == output.parent, 'Indirect external assessment path')
        frozen_paths[key] = str(output)
        capability = planning[key]['constraints']
        capabilities[sha(canonical(capability))] = capability
    capability_hash = next(iter(capabilities)) if len(capabilities) == 1 else sha(canonical(capabilities))
    sources = [Path(__file__).absolute(), Path(first_output_planning.__file__).absolute(),
               *source_paths, *(Path(b['path']) for b in bindings.values())]
    original = delivery.generator_manifest(version, sources, baseline_binding=baseline_binding,
        capabilities_sha256=capability_hash, evaluation_policy_sha256=evaluation_policy_sha256,
        provider_config=provider_config)
    return bound(dict({k: v for k, v in original.items() if k != 'binding'}, host_bound_contract=VERSION,
        host_request_bindings=bindings, host_pre_generation_plans=planning,
        host_backend_identity=copy.deepcopy(backend_identity), host_assessment_paths=frozen_paths))


def _mechanism(mechanism):
    delivery.verify_generator(mechanism)
    require(mechanism.get('host_bound_contract') == VERSION and 'planning_contract' not in mechanism,
            'Explicit host-bound delivery mechanism required')
    require(str(Path(__file__).absolute()) in mechanism['source_sha256'], 'Host-bound entry absent from source freeze')
    keys = set(mechanism['host_request_bindings'])
    require(keys == set(mechanism['host_pre_generation_plans']) == set(mechanism['host_assessment_paths']),
            'Host-bound mechanism request sets differ')
    return mechanism


def _slot(host, mechanism, plan, output):
    _mechanism(mechanism); checked_bound(plan)
    require(sha(canonical(plan)) == mechanism['evaluation_policy_sha256'], 'Delivery policy changed')
    key = host['request_id']
    binding = mechanism['host_request_bindings'][key]
    full = resolve_host(binding, mechanism=mechanism)
    require(canonical(host) == canonical(full), 'Caller supplied another host request')
    wire = public_request(copy.deepcopy(full))
    slot = plan['slots'][key]
    require(slot['request_sha256'] == sha(canonical(wire))
            and slot['delivery_path'] == str(Path(output).absolute()), 'Unique frozen public delivery slot')
    require(slot['split'] != 'acceptance' or plan['independent_seal_verified'], 'Independent seal missing')
    planning = mechanism['host_pre_generation_plans'][key]
    verify_plan(planning, full)
    return full, wire, binding, planning


def _admission_path(output):
    path = Path(output).absolute()
    return path.with_name(path.name + '.host-admission')


def generate_once(host_request, provider, output, *, mechanism, plan, backend=None,
                  backend_identity=None, batch=None, checkpoint=None, test_only=False):
    """Delivery only: pre-call admission, one original Provider call, exact bytes."""
    host, wire, binding, planning = _slot(host_request, mechanism, plan, output)
    require(type(test_only) is bool and (test_only or not planning['test_only']), 'Fixture cannot authorize live delivery')
    if planning['status'] != 'prepared':
        from planned_delivery_component import _blocked
        return _blocked(wire, mechanism, plan, output, planning, planning['status'])
    require(backend is not None and backend_identity == mechanism['host_backend_identity'], 'Frozen original backend required')
    if not test_only:
        from normal_cpu_overlay_backend import trusted_backend
        trusted_backend(backend, approved_overlay_required=
            planning['constraints'].get('SDK_format') == 'poseidon-normal-cpu-approved-overlay-v1')
    require(backend.identity_manifest() == backend_identity, 'Backend changed before host admission')
    try:
        backend.check_request(copy.deepcopy(host), 'numerical')
    except ValueError:
        from planned_delivery_component import _blocked
        return _blocked(wire, mechanism, plan, output, planning, 'backend_blocked')
    require(backend.identity_manifest() == backend_identity, 'Backend changed during host admission')
    require(provider is not None, 'Admitted request needs a fresh original Provider')
    if test_only:
        require(not isinstance(provider._transport, HTTPSTransport), 'Fixture cannot authorize network transport')
    admission = _admission_path(output)
    require(admission.parent.resolve() == admission.parent, 'Indirect host admission output')
    if not test_only:
        require(batch is not None and any(admission.is_relative_to(a['path']) for a in batch.record()['artifacts']),
                'Host admission outside accounted batch')
    admission.mkdir(mode=0o700)
    write_once(admission/'admission.json', canonical(bound(dict(format=ADMISSION,
        request_id=host['request_id'], mechanism_binding=mechanism['binding'], host_binding=binding['binding'],
        host_sha256=binding['host_canonical_sha256'], public_sha256=binding['public_canonical_sha256'],
        planning_binding=planning['binding'], backend_identity=backend_identity, recorded_at_ns=time.time_ns(),
        test_only=test_only, Provider_calls=0, compiler_calls=0, encrypted_calls=0))))
    receipt = delivery.generate_once(host, provider, output, mechanism=mechanism, plan=plan,
                                    batch=batch, checkpoint=checkpoint)
    verify_host_delivery(output, receipt['binding'])
    return receipt


def verify_host_delivery(folder, expected_binding):
    receipt, wire, source, mechanism = delivery.verify_delivery(folder, expected_binding)
    _mechanism(mechanism)
    key = receipt['request_id']
    binding = mechanism['host_request_bindings'][key]
    host = resolve_host(binding, wire, mechanism=mechanism)
    verify_plan(mechanism['host_pre_generation_plans'][key], host)
    attempt = checked_bound(_json(Path(folder)/'attempt.json'))
    if receipt['origin'] == 'pre_generation_admission':
        require(receipt['first_output_eligible'] is False and source is None
                and attempt['generations'] == attempt['HTTP_attempts'] == 0, 'Blocked request cannot become an output')
        return receipt, host, source, mechanism
    require(receipt['origin'] == 'provider_first_output' and receipt['post_output_repairs'] == 0,
            'Host-bound mode accepts only the original prospective output')
    record = checked_bound(_json(_admission_path(folder)/'admission.json'))
    planning = mechanism['host_pre_generation_plans'][key]
    require(record['format'] == ADMISSION and record['request_id'] == key
            and record['mechanism_binding'] == mechanism['binding'] and record['host_binding'] == binding['binding']
            and record['host_sha256'] == binding['host_canonical_sha256']
            and record['public_sha256'] == binding['public_canonical_sha256']
            and record['planning_binding'] == planning['binding']
            and record['backend_identity'] == mechanism['host_backend_identity']
            and record['recorded_at_ns'] <= attempt['started_at_ns']
            and record['Provider_calls'] == record['compiler_calls'] == record['encrypted_calls'] == 0,
            'Host admission did not precede this original Provider call')
    require(attempt['generations'] == attempt['HTTP_attempts'] == 1 and attempt['retries'] == 0
            and attempt['feedback'] == [] and attempt['compilation'] == attempt['encrypted_execution'] == 0,
            'Delivery used feedback or another generation')
    require(not receipt['real_Agent'] or record['test_only'] is False, 'Fixture admission cannot certify a real Agent')
    return receipt, host, source, mechanism


def assess_once(folder, expected_binding, policy, output, evaluator):
    """External-only evaluation with the frozen full request and unedited DSL.

The original witness comparisons/verifiers remain authoritative. No evaluation
or numeric result is returned to a Provider. Every frozen request owns exactly
one assessment directory; later development work is reported elsewhere.
"""
    receipt, host, source, mechanism = verify_host_delivery(folder, expected_binding)
    checked_bound(policy)
    require(sha(canonical(policy)) == mechanism['evaluation_policy_sha256'], 'Frozen external policy changed')
    key = receipt['request_id']; slot = policy['slots'][key]
    require(slot['request_sha256'] == mechanism['host_request_bindings'][key]['public_canonical_sha256'],
            'External policy belongs to another public request')
    numeric_policy = checked_bound(slot['policy'])
    path = Path(output).absolute()
    require(str(path) == mechanism['host_assessment_paths'][key]
            and path.parent.resolve() == path.parent, 'Unique frozen external assessment location')
    path.mkdir(mode=0o700)
    write_once(path/'policy.json', canonical(policy))
    write_once(path/'attempt.json', canonical(bound(dict(format=VERSION, mode='external_first_output_assessment',
        delivery_binding=expected_binding, source_sha256=receipt['source_sha256'],
        external_evaluation_calls=1 if source is not None else 0, feedback_to_Agent=0))))
    result = dict(format=VERSION, mode='external_first_output_assessment', request_id=key,
        delivery_binding=expected_binding, mechanism_binding=mechanism['binding'],
        host_request_binding=mechanism['host_request_bindings'][key]['binding'],
        source_sha256=receipt['source_sha256'], source_unchanged=True, delivery_origin=receipt['origin'],
        first_output_eligible=receipt['first_output_eligible'], post_output_repairs=0, status=receipt['status'],
        raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False,
        input_groups_evaluated=0, compared_values=0, inputs=[], original_function_to_declared_approximation=[],
        domain_verified=False, backend_binding_verified=False, native_values_binding_verified=False,
        independent_math_binding_verified=False)
    if source is not None:
        try:
            witness = evaluator(copy.deepcopy(host), source)
            verify_host_delivery(folder, expected_binding)
            write_once(path/'witness.json', canonical(witness))
            numeric = evaluate_witness(witness, numeric_policy, receipt['source_sha256'])
            result.update(numeric)
            if not numeric['test_only']:
                from first_output_numerical_evidence import verify_backend_witness, verify_native_rows, verify_mathematical_rows
                try:
                    verify_backend_witness(witness, host, source)
                    result['backend_binding_verified'] = True
                except (ValueError, KeyError, TypeError, IndexError, OSError):
                    result.update(raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
                        failure_category='backend_blocked', evidence_failure='actual_compiler_input_binding_missing_or_invalid')
                if result['backend_binding_verified'] and witness.get('inputs'):
                    try:
                        verify_native_rows(witness, host, source)
                        result['native_values_binding_verified'] = True
                        verify_mathematical_rows(witness, numeric_policy, host, source)
                        result['independent_math_binding_verified'] = True
                    except (ValueError, KeyError, TypeError, IndexError, OSError):
                        result.update(raw_numeric_passed=False, raw_joint_passed=False,
                            failure_category='backend_blocked', evidence_failure='native_or_independent_math_binding_missing_or_invalid')
                if (not result['backend_binding_verified'] or witness.get('inputs') and not
                        (result['native_values_binding_verified'] and result['independent_math_binding_verified'])):
                    result['unverified_numeric_claims'] = {k: result[k] for k in
                        ('input_groups_evaluated', 'compared_values', 'inputs', 'original_function_to_declared_approximation', 'domain_verified')}
                    result.update(input_groups_evaluated=0, compared_values=0, inputs=[],
                                  original_function_to_declared_approximation=[], domain_verified=False)
            eligible = receipt['first_output_eligible'] and not numeric['test_only']
            for name in ('compile', 'numeric', 'joint'):
                result['first_'+name+'_passed'] = eligible and result['raw_'+name+'_passed']
            result['status'] = ('passed' if result['raw_joint_passed'] else result['failure_category']
                if result['failure_category'] in BLOCKED else 'numeric_failed' if result['raw_compile_passed'] else 'compile_failed')
        except Exception:
            result.update(status='evaluation_failed', raw_compile_passed=False, raw_numeric_passed=False,
                raw_joint_passed=False, first_compile_passed=False, first_numeric_passed=False,
                first_joint_passed=False, input_groups_evaluated=0, compared_values=0, inputs=[])
            try:
                verify_host_delivery(folder, expected_binding)
            except (ValueError, OSError):
                result.update(status='integrity_rejected', source_unchanged=False)
    saved = bound(result)
    write_once(path/'assessment.json', canonical(saved))
    return saved
