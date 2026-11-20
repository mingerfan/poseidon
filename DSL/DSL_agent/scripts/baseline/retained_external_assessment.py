"""Retained three-layer external assessment; never generates or repairs DSL.

Real-slot proof and program-complex bytes are distinct ABIs. The latter stays
closed until the actual compiler/layout/input/process adapter is independently
audited; JSON flags and a byte bridge cannot grant native eligibility.
"""
import copy
import math
from pathlib import Path
import time

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha, write_once
from first_output_backend import _json
from first_output_assessment import evaluate_witness, BLOCKED
import first_output_numerical_evidence as backend_evidence
import native_dual_reference_evidence as native_evidence
import retained_math_evidence as mathematical_evidence
import retained_first_output_mode as mode
import shared_execution

VERSION = 'poseidon-retained-three-layer-external-assessment-v1'


def _resource_state(row, scope, proof, mechanism_binding, minimum_started_ns):
    """Pure control seam; the live reader obtains rows from original Coordinator."""
    checked_bound(scope)
    require(type(row) is dict and row.get('status') in ('succeeded', 'failed', 'interrupted') and row.get('active') is None,
            'Original native batch must be terminal')
    plan = row['plan']; require(row['plan_binding'] == sha(canonical(plan)) and plan['source_binding'] == mechanism_binding,
                              'Native batch belongs to another frozen generator')
    require(set(plan['stages']) == set(row['used']) == set(shared_execution.KINDS)
            and all(type(plan['stages'][k]) is int and type(row['used'][k]) is int
                    and plan['stages'][k] == row['used'][k] == int(k == 'qualify') for k in shared_execution.KINDS),
            'Original external qualifier once; no Provider or restored allowance')
    require(type(row.get('finished')) in (int, float) and math.isfinite(row['finished']) and 0 < row['finished'] <= time.time(),
            'Original native terminal timestamp')
    require(type(minimum_started_ns) is int and minimum_started_ns > 0 and type(row.get('started')) in (int, float)
            and math.isfinite(row['started']) and int(row['started']*10**9) >= minimum_started_ns,
            'Native evaluation must start after this original delivery was sealed')
    root = Path(scope['native_evidence_root']); evidence = Path(proof['evidence']).absolute()
    require(root.resolve() == root and evidence.resolve() == evidence and evidence != root and evidence.is_relative_to(root),
            'Native evidence outside pre-generation frozen output scope')
    accounted = False
    for entry in row['artifacts']:
        path = Path(entry['path'])
        if root == path or root.is_relative_to(path):
            st = path.stat(); require(st.st_dev == entry['device'] and st.st_ino == entry['inode'], 'Native accounted root was replaced')
            accounted = True
    require(accounted, 'Native output outside original shared artifacts')
    uncertain = row.get('uncertain_stage') is not None
    last = row.get('last_stage', {})
    certain = (not uncertain and last.get('kind') == 'qualify' and type(last.get('number')) is int and last['number'] == 1
               and last.get('outcome') == 'completed' and last.get('execution_uncertain') is False)
    if row['status'] == 'interrupted':
        owner = row.get('owner'); require(type(owner) is dict and type(owner.get('pid')) is int, 'Original interrupted owner')
        require(shared_execution.process(owner['pid']) != owner, 'Interrupted native owner remains live')
    return bound(dict(format='poseidon-retained-native-terminal-scope-v1', original_batch_plan_binding=row['plan_binding'],
        status=row['status'], native_terminal_certain=certain, original_allowances_preserved=True,
        native_evidence=str(evidence), ledger_mutations=0, API=0))


def verify_resource_scope(coordinator, selected, proof, mechanism_binding, minimum_started_ns):
    require(type(coordinator) is shared_execution.Coordinator, 'Original shared Coordinator required')
    scope = selected['native_mode']; snapshot, _ = coordinator.snapshot()
    require(scope['external_batch_name'] in snapshot['batches'], 'Frozen native batch absent')
    return _resource_state(snapshot['batches'][scope['external_batch_name']], scope, proof, mechanism_binding, minimum_started_ns)


def check_witness(witness, policy, request, source, selected, *, mechanism_binding, coordinator=None,
                  minimum_started_ns=None, test_only=False):
    """Bind actual ordinary native cells and retained math before numerical credit."""
    require(type(test_only) is bool, 'Explicit external fixture scope')
    checked_bound(selected); checked_bound(policy)
    require(selected['numeric_policy_sha256'] == sha(canonical(policy)), 'External numeric policy differs from frozen mode')
    spec = checked_bound(policy['retained_mathematical_producer'])
    require(selected['mathematical_producer_binding'] == spec['binding'], 'Retained producer selection differs')
    if not test_only:
        require(spec['test_only'] is False and witness.get('test_only') is False, 'Fixture evidence cannot count as actual external evaluation')
    result = dict(raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        test_only=test_only or spec['test_only'] or witness.get('test_only') is not False,
        input_groups_evaluated=0, compared_values=0, inputs=[], original_function_to_declared_approximation=[],
        domain_verified=False, backend_binding_verified=False, native_values_binding_verified=False,
        independent_math_binding_verified=False, native_resource_binding_verified=False,
        native_reference_comparisons_passed=False, failure_category='backend_blocked', evidence_failures=[])
    abi = selected['native_mode']['abi']
    require(abi in (mode.REAL, mode.COMPLEX), 'Unselected native ABI')
    if abi == mode.COMPLEX:
        result['evidence_failures'].append('complex_compiler_layout_input_host_adapter_not_yet_audited')
        result.update(complex_decoder_called=False, unverified_CKKS_values_consumed=False)
        return result
    try:
        proof = backend_evidence.verify_backend_witness(witness, request, source)
        require(proof['backend_identity_sha256'] == selected['native_backend_identity_sha256'], 'Actual native backend differs from generator freeze')
        if test_only:
            resource = dict(native_terminal_certain=False, fixture_only=True)
        else:
            resource = verify_resource_scope(coordinator, selected, proof, mechanism_binding, minimum_started_ns)
            result['native_resource_binding_verified'] = True
        result['backend_binding_verified'] = True
        numeric = evaluate_witness(witness, policy, sha(source))
        result.update(numeric)
        result['test_only'] = test_only or spec['test_only'] or numeric['test_only']
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        result['evidence_failures'].append('actual_compiler_backend_or_frozen_native_batch_binding_invalid')
        return result
    if not result['raw_compile_passed']:
        return result
    try:
        mathematical = mathematical_evidence.verify_mathematical_rows(witness, policy, request, source)
        require(mathematical['test_only'] is spec['test_only'] and mathematical['all_policy_inputs_covered'] is True,
                'All frozen retained mathematical inputs required')
        result['independent_math_binding_verified'] = True
        result['independent_mathematical_record'] = mathematical
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        result['evidence_failures'].append('retained_math_domain_application_or_full_input_binding_invalid')
    try:
        pairs = native_evidence.verify_native_rows(witness, request, source)
        result['native_values_binding_verified'] = True
        result['native_reference_comparisons_passed'] = pairs['passed']
        result['native_reference_pairs'] = pairs
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        result['evidence_failures'].append('original_real_slot_native_cells_or_reference_binding_invalid')
    native_certain = test_only or resource['native_terminal_certain']
    if not native_certain:
        result['evidence_failures'].append('native_terminal_execution_uncertain')
    verified = (result['independent_math_binding_verified'] and result['native_values_binding_verified'] and native_certain)
    if not verified:
        result['unverified_numeric_claims'] = {k: result[k] for k in ('input_groups_evaluated', 'compared_values', 'inputs',
            'original_function_to_declared_approximation', 'domain_verified')}
        result.update(input_groups_evaluated=0, compared_values=0, inputs=[], original_function_to_declared_approximation=[], domain_verified=False)
    result['raw_numeric_passed'] = bool(result['raw_numeric_passed'] and verified and result['native_reference_comparisons_passed'])
    result['raw_joint_passed'] = bool(result['raw_compile_passed'] and result['raw_numeric_passed'])
    if result['evidence_failures']:
        result['failure_category'] = 'backend_blocked'
    return result


def _delivery(folder, binding, mechanism, *, coordinator, delivery_batch, test_only):
    import prepared_first_delivery
    import lossless_host_bound_delivery
    receipt, host, source, actual = lossless_host_bound_delivery.verify_host_delivery(folder, binding)
    require(actual == mechanism, 'External generator changed')
    if source is not None:
        prepared_first_delivery.verify_preserved(folder, binding)
    if test_only:
        require(receipt['real_Agent'] is False, 'Fixture cannot consume or certify a live delivery')
        return receipt, host, source
    require(type(coordinator) is shared_execution.Coordinator, 'Original delivery ledger required')
    if source is not None:
        from completed_delivery_startup import admit
        *_, startup = admit(folder, binding, coordinator, delivery_batch)
        require(startup['eligible_for_external_assessment'] is True, 'Original single completed delivery is not admitted')
    else:
        snapshot, _ = coordinator.snapshot(); row = snapshot['batches'][delivery_batch]
        require(row['status'] in ('succeeded', 'failed', 'interrupted') and row['active'] is None
                and row['plan']['source_binding'] == mechanism['binding'], 'Original failure batch remains active or differs')
    return receipt, host, source


def assess_once(folder, expected_binding, policy, output, evaluator, *, coordinator=None, delivery_batch=None, test_only=False):
    """External-only, one immutable slot; saved failures cannot be overwritten."""
    from delivery_component import verify_delivery
    receipt, wire, source, mechanism = verify_delivery(folder, expected_binding)
    mode.verify_selection(mechanism, policy)
    receipt, host, source = _delivery(folder, expected_binding, mechanism, coordinator=coordinator,
                                     delivery_batch=delivery_batch, test_only=test_only)
    key = receipt['request_id']; selected = mechanism['retained_first_output_slots'][key]
    path = Path(output).absolute()
    require(str(path) == mechanism['host_assessment_paths'][key] and path.parent.resolve() == path.parent, 'One frozen external assessment slot')
    path.mkdir(mode=0o700)
    write_once(path/'policy.json', canonical(policy))
    write_once(path/'attempt.json', canonical(bound(dict(format=VERSION, delivery_binding=expected_binding,
        source_sha256=receipt['source_sha256'], external_evaluation_calls=int(source is not None), feedback_to_Agent=0,
        Provider_calls=0, source_modifications=0, recorded_at_ns=time.time_ns(), test_only=test_only))))
    result = dict(format=VERSION, mode='external_first_output_assessment', request_id=key, delivery_binding=expected_binding,
        mechanism_binding=mechanism['binding'], source_sha256=receipt['source_sha256'], source_unchanged=True,
        delivery_origin=receipt['origin'], first_output_eligible=receipt['first_output_eligible'], post_output_repairs=0,
        test_only=test_only, status=receipt['status'], raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False,
        input_groups_evaluated=0, compared_values=0, inputs=[], original_function_to_declared_approximation=[], domain_verified=False)
    if source is not None:
        try:
            witness = evaluator(copy.deepcopy(host), source)
            _delivery(folder, expected_binding, mechanism, coordinator=coordinator, delivery_batch=delivery_batch, test_only=test_only)
            write_once(path/'witness.json', canonical(witness))
            numerical = check_witness(witness, policy['slots'][key]['policy'], host, source, selected,
                mechanism_binding=mechanism['binding'], coordinator=coordinator,
                minimum_started_ns=(Path(folder)/'delivery.json').stat().st_mtime_ns, test_only=test_only)
            result.update(numerical)
            eligible = receipt['first_output_eligible'] and not result['test_only']
            for item in ('compile', 'numeric', 'joint'):
                result['first_'+item+'_passed'] = bool(eligible and result['raw_'+item+'_passed'])
            result['status'] = ('passed' if result['raw_joint_passed'] else result.get('failure_category')
                if result.get('failure_category') in BLOCKED else 'numeric_failed' if result['raw_compile_passed'] else 'compile_failed')
        except Exception:
            result.update(status='evaluation_failed', raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
                first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False,
                input_groups_evaluated=0, compared_values=0, inputs=[], original_function_to_declared_approximation=[], domain_verified=False)
            try:
                _delivery(folder, expected_binding, mechanism, coordinator=coordinator, delivery_batch=delivery_batch, test_only=test_only)
            except (ValueError, KeyError, TypeError, OSError):
                result.update(status='integrity_rejected', source_unchanged=False)
    saved = bound(result); write_once(path/'assessment.json', canonical(saved))
    return saved


def classify_retained_metadata(record):
    """Read an old reference receipt without accepting it as numerical evidence."""
    checked_bound(record)
    require(record.get('format') == 'poseidon-retained-source-declared-reference-row-v1', 'Explicit old retained-reference metadata format')
    return bound(dict(format='poseidon-retained-reference-external-classification-v1', reference_binding=record['binding'],
        case_id=record['case_id'], evidence_role='reference_rows_only', accepted_as_mathematical_record=False,
        accepted_as_native_or_compiler_record=False, first_output_success_certified=False, numeric_cells_decoded=0))
