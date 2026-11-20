"""External assessment and request denominators, never a generation/repair loop.

The trusted external runner supplies evidence from the real compiler/native
execution. Array comparisons here audit that evidence; they do not execute a
model or simulate CKKS. B owns the real-backend evidence adapter.
"""
import copy
import math
from pathlib import Path

from delivery_component import (bound, canonical, checked_bound, read_bytes,
    require, sha, verify_delivery, write_once)

VERSION = 'poseidon-first-output-external-assessment-v3'
MATH = dict(atol=1e-12, rtol=1e-12)
CKKS = dict(atol=1e-5, rtol=1e-4)
BLOCKED = {'refused', 'unsupported', 'resource_blocked', 'backend_blocked', 'provider_failed',
           'contract_rejected', 'not_started', 'evaluation_failed', 'integrity_rejected'}


def evaluation_policy(input_sha256s, *, domain_sha256, approximation_requirements,
                      application_requirements, whole_model_required=False,
                      independent_math_sources=None, original_model_sha256=None, evaluation_inputs=None):
    """All inputs, domains and application requirements freeze before generation."""
    require(type(input_sha256s) is list and bool(input_sha256s) and len(set(input_sha256s)) == len(input_sha256s), 'Distinct frozen inputs required')
    for value in [domain_sha256] + input_sha256s:
        require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Frozen data hash')
    for requirements in (approximation_requirements, application_requirements):
        require(type(requirements) is list, 'Frozen metric requirements')
        require(len({r['id'] for r in requirements}) == len(requirements), 'Duplicate metric requirement')
        for r in requirements:
            require(set(r) == {'id', 'comparison', 'limit'} and type(r['id']) is str and bool(r['id'])
                    and r['comparison'] in ('<=', '>=') and type(r['limit']) in (int, float)
                    and math.isfinite(r['limit']), 'Metric requirement')
    require(type(whole_model_required) is bool, 'Whole-model requirement')
    sources = {} if independent_math_sources is None else dict(independent_math_sources)
    for name, expected in sources.items():
        require(type(name) is str and Path(name).is_absolute() and Path(name).suffix == '.py'
                and type(expected) is str and len(expected) == 64
                and all(c in '0123456789abcdef' for c in expected), 'Frozen mathematical source identity')
    require(original_model_sha256 is None or type(original_model_sha256) is str
            and len(original_model_sha256) == 64 and all(c in '0123456789abcdef' for c in original_model_sha256),
            'Original model identity')
    if evaluation_inputs is not None:
        require(type(evaluation_inputs) is dict and evaluation_inputs.get('format')=='poseidon-fixed-evaluation-inputs-v1'
                and len(evaluation_inputs.get('cases',[]))==len(input_sha256s), 'Frozen evaluation input selection')
    result=dict(format='poseidon-first-output-numeric-policy-v1', inputs=input_sha256s,
        frozen_domain_sha256=domain_sha256, independent_math=dict(MATH), CKKS=dict(CKKS),
        approximation_requirements=approximation_requirements, application_requirements=application_requirements,
        whole_model_required=whole_model_required, independent_math_sources=sources,
        original_model_sha256=original_model_sha256)
    if evaluation_inputs is not None:result["evaluation_inputs"]=copy.deepcopy(evaluation_inputs)
    return bound(result)


def _values(value):
    require(type(value) is list and bool(value) and all(type(v) in (int, float) and math.isfinite(v) for v in value), 'Finite numeric evidence required')
    return value


def compare(reference, real, *, imag=None, atol, rtol):
    reference = _values(reference); real = _values(real)
    imag = [0.] * len(real) if imag is None else _values(imag)
    require(len(reference) == len(real) == len(imag), 'Numeric evidence shape')
    errors = [math.hypot(a-r, i) for r, a, i in zip(reference, real, imag)]
    require(all(math.isfinite(e) for e in errors), 'Nonfinite comparison')
    return dict(compared_values=len(errors), failed_values=sum(e > atol+rtol*abs(r) for e, r in zip(errors, reference)),
                max_absolute_error=max(errors), RMSE=math.hypot(*errors)/math.sqrt(len(errors)))


def _metrics_pass(requirements, values):
    require(type(values) is dict, 'Metric evidence')
    for r in requirements:
        v = values.get(r['id'])
        if type(v) not in (int, float) or not math.isfinite(v):
            return False
        if not (v <= r['limit'] if r['comparison'] == '<=' else v >= r['limit']):
            return False
    return True


def evaluate_witness(witness, policy, source_sha256):
    checked_bound(policy)
    require(policy.get('format') == 'poseidon-first-output-numeric-policy-v1'
            and policy['independent_math'] == MATH and policy['CKKS'] == CKKS, 'Fixed numerical thresholds required')
    require(type(witness) is dict and witness.get('source_sha256') == source_sha256, 'Runner evaluated another DSL')
    compiled = (witness.get('compiled_validated') is True and witness.get('real_compiler') is True
                and witness.get('compiler_source_sha256') == source_sha256)
    result = dict(raw_compile_passed=compiled, raw_numeric_passed=False, raw_joint_passed=False,
                  input_groups_evaluated=0, compared_values=0, inputs=[], test_only=witness.get('test_only') is not False,
                  original_function_to_declared_approximation=[], domain_verified=False,
                  failure_category=witness.get('failure_category'))
    if not compiled:
        return result
    rows = witness.get('inputs', [])
    require(type(rows) is list, 'Input evidence list')
    observed = [r['input_sha256'] for r in rows]
    require(len(set(observed)) == len(observed) and set(observed) <= set(policy['inputs']), 'Duplicate or undeclared input evidence')
    for row in rows:
        require(row.get('source_sha256') == source_sha256, 'Input used a replacement DSL')
        original = row['original_reference']; declared = row['declared_reference']
        approximation = compare(original, declared, atol=0., rtol=0.)
        mathematical = compare(declared, row['dsl_mathematical_output'], **MATH)
        ckks = compare(declared, row['CKKS_real'], imag=row.get('CKKS_imag'), **CKKS)
        result['original_function_to_declared_approximation'].append(dict(input_sha256=row['input_sha256'], **approximation))
        domain = row.get('inside_frozen_domain') is True and row.get('domain_sha256') == policy['frozen_domain_sha256']
        application = _metrics_pass(policy['application_requirements'], row.get('application_metrics', {}))
        approximation_ok = _metrics_pass(policy['approximation_requirements'],
            dict(max_absolute_error=approximation['max_absolute_error'], RMSE=approximation['RMSE']))
        whole = not policy['whole_model_required'] or row.get('whole_model_output') is True
        passed = domain and application and approximation_ok and whole and mathematical['failed_values'] == ckks['failed_values'] == 0
        result['inputs'].append(dict(input_sha256=row['input_sha256'], mathematical=mathematical, CKKS=ckks,
            domain_passed=domain, approximation_passed=approximation_ok, application_passed=application,
            whole_model_output_required=policy['whole_model_required'], whole_model_passed=whole, passed=passed))
        result['compared_values'] += ckks['compared_values']
    result['input_groups_evaluated'] = len(rows)
    result['domain_verified'] = bool(rows) and all(r['domain_passed'] for r in result['inputs'])
    complete = set(observed) == set(policy['inputs']) and bool(rows)
    native = (witness.get('encrypted_execution') is True and witness.get('real_CKKS') is True
              and witness.get('intermediate_decryptions') == 0 and witness.get('reference_or_tolerance_changed') is False)
    result['raw_numeric_passed'] = complete and native and all(r['passed'] for r in result['inputs'])
    result['raw_joint_passed'] = compiled and result['raw_numeric_passed']
    return result


def assess_once(delivery, expected_binding, policy, output, evaluator):
    """Call an external trusted evaluator once with the exact frozen source bytes.

    No result returns to a Provider. A failed or interrupted assessment cannot
    reuse this output directory; it remains separate from development repairs.
    """
    receipt, request, source, mechanism = verify_delivery(delivery, expected_binding)
    checked_bound(policy)
    require(mechanism['evaluation_policy_sha256'] == sha(canonical(policy)), 'Evaluation policy differs from generator freeze')
    per_request_policy = policy
    if policy.get('format') == 'poseidon-first-output-delivery-plan-v1':
        slot = policy['slots'].get(receipt['request_id'])
        require(slot is not None and slot['request_sha256'] == sha(canonical(request)), 'Evaluation request slot mismatch')
        per_request_policy = checked_bound(slot['policy'])
    p = Path(output).absolute()
    require(p.parent.resolve() == p.parent, 'Indirect assessment parent')
    p.mkdir(mode=0o700)
    write_once(p/'policy.json', canonical(policy))
    write_once(p/'attempt.json', canonical(bound(dict(format=VERSION, delivery_binding=expected_binding,
        source_sha256=receipt['source_sha256'], external_evaluation_calls=1 if source is not None else 0, feedback_to_Agent=0))))
    result = dict(format=VERSION, mode='external_first_output_assessment', delivery_binding=expected_binding,
        request_id=receipt['request_id'], mechanism_binding=receipt['mechanism_binding'],
        source_sha256=receipt['source_sha256'], source_unchanged=True, delivery_origin=receipt['origin'],
        first_output_eligible=receipt['first_output_eligible'], post_output_repairs=0,
        status=receipt['status'], raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False, input_groups_evaluated=0)
    result.update(backend_binding_verified=False, native_values_binding_verified=False,
                  independent_math_binding_verified=False)
    if source is not None:
        try:
            if mechanism.get('planning_contract') is not None:
                from planned_delivery_component import verify_planned_delivery
                verify_planned_delivery(delivery, receipt, mechanism)
            witness = evaluator(copy.deepcopy(request), source)
            verify_delivery(delivery, expected_binding)
            write_once(p/'witness.json', canonical(witness))
            numeric = evaluate_witness(witness, per_request_policy, receipt['source_sha256'])
            result.update(numeric)
            if not numeric['test_only']:
                from first_output_numerical_evidence import (verify_backend_witness,
                    verify_native_rows, verify_mathematical_rows)
                try:
                    verify_backend_witness(witness, request, source)
                    result['backend_binding_verified'] = True
                except (ValueError, KeyError, TypeError, OSError):
                    result.update(raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
                        failure_category='backend_blocked', evidence_failure='actual_compiler_input_binding_missing_or_invalid',
                        input_groups_evaluated=0)
                if result['backend_binding_verified'] and witness.get('inputs'):
                    try:
                        verify_native_rows(witness, request, source, policy=per_request_policy)
                        result['native_values_binding_verified'] = True
                        verify_mathematical_rows(witness, per_request_policy, request, source)
                        result['independent_math_binding_verified'] = True
                    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
                        from ordinary_complete_numerical_evidence import CompleteNativeProofUnavailable
                        result.update(raw_numeric_passed=False, raw_joint_passed=False,
                            failure_category='evaluation_failed' if isinstance(error,CompleteNativeProofUnavailable) else 'backend_blocked',
                            evidence_failure='ordinary_complete_execution_proof_adapter_pending' if isinstance(error,CompleteNativeProofUnavailable) else 'native_or_independent_math_binding_missing_or_invalid')
                if (not result['backend_binding_verified'] or witness.get('inputs') and
                        not (result['native_values_binding_verified'] and result['independent_math_binding_verified'])):
                    # Keep the runner's exact claims in witness.json. Unverified
                    # cells cannot be reported as evaluated input/value counts.
                    result['unverified_numeric_claims'] = {key: result[key] for key in
                        ('input_groups_evaluated', 'compared_values', 'inputs',
                         'original_function_to_declared_approximation', 'domain_verified')}
                    result.update(input_groups_evaluated=0, compared_values=0, inputs=[],
                        original_function_to_declared_approximation=[], domain_verified=False)
            eligible = receipt['first_output_eligible'] and not numeric['test_only']
            for key in ('compile', 'numeric', 'joint'):
                result['first_'+key+'_passed'] = eligible and result['raw_'+key+'_passed']
            result['status'] = ('passed' if result['raw_joint_passed'] else
                                result['failure_category'] if result['failure_category'] in BLOCKED else
                                'numeric_failed' if result['raw_compile_passed'] else 'compile_failed')
        except Exception:
            result.update(status='evaluation_failed', raw_compile_passed=False,
                raw_numeric_passed=False, raw_joint_passed=False, first_compile_passed=False,
                first_numeric_passed=False, first_joint_passed=False, input_groups_evaluated=0)
            # Integrity failures are distinguished from other external failures.
            try:
                verify_delivery(delivery, expected_binding)
            except (ValueError, OSError):
                result.update(status='integrity_rejected', source_unchanged=False)
    saved = bound(result)
    write_once(p/'assessment.json', canonical(saved))
    return saved


def summarize_requests(cohort, assessments):
    """One row per frozen request; failures and missing outcomes stay in denominator.

    Keep each split/tier/version separate. A cohort seal is external: the caller
    must not label known/tuned requests as independent unseen acceptance.
    """
    checked_bound(cohort)
    require(cohort.get('format') == 'poseidon-first-output-cohort-v1', 'Frozen cohort required')
    rows = cohort['requests']
    keys = [r['request_id'] for r in rows]
    require(len(set(keys)) == len(keys), 'Duplicate frozen request')
    require(type(assessments) is dict and set(assessments) <= set(keys), 'Unknown request outcome')
    groups = {}
    for row in rows:
        require(row['split'] in ('development', 'regression', 'acceptance') and row['tier'] in ('operators', 'medium', 'complete'), 'Benchmark split/tier')
        if row.get('unseen'):
            require(row['split'] == 'acceptance' and cohort.get('independent_seal_verified') is True, 'Unseen evidence requires independent seal')
        group = (row['split'], row['tier'], bool(row.get('unseen')))
        groups.setdefault(group, []).append(row)
    def totals(selected):
        result = dict(models=len({r['model_sha256'] for r in selected}), requests=len(selected),
            planned_input_groups=sum(len(r['input_sha256s']) for r in selected), evaluated_input_groups=0,
            first_compile_passed=0, first_numeric_passed=0, first_joint_passed=0, statuses={})
        for row in selected:
            outcome = assessments.get(row['request_id'])
            status = 'not_started' if outcome is None else outcome['status']
            if outcome is not None:
                checked_bound(outcome)
                require(outcome['request_id'] == row['request_id'] and outcome['mechanism_binding'] == cohort['mechanism_binding'], 'Cohort version/request mismatch')
                require(outcome['mode'] == 'external_first_output_assessment', 'Development results cannot replace first-output outcomes')
                eligible = outcome['first_output_eligible'] and outcome['source_unchanged'] and outcome['post_output_repairs'] == 0
                for key in ('compile', 'numeric', 'joint'):
                    result['first_'+key+'_passed'] += int(eligible and outcome['first_'+key+'_passed'] is True)
                result['evaluated_input_groups'] += outcome['input_groups_evaluated']
            result['statuses'][status] = result['statuses'].get(status, 0)+1
        for key in ('compile', 'numeric', 'joint'):
            result['first_'+key+'_rate'] = result['first_'+key+'_passed']/len(selected) if selected else None
        return result
    return bound(dict(format='poseidon-first-output-scorecard-v1', mode='external_benchmark_report',
        mechanism_binding=cohort['mechanism_binding'], cohort_binding=cohort['binding'],
        overall_descriptive_only=totals(rows),
        strata=[dict(split=k[0], tier=k[1], unseen=k[2], **totals(v)) for k,v in sorted(groups.items())]))
