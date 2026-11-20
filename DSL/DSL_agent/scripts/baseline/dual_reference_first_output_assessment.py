"""Frozen external dual-reference assessment; no Provider or repair calls."""
import copy
from pathlib import Path
import lossless_host_bound_delivery as host_entry
from lossless_host_bound_delivery import verify_host_delivery
from delivery_component import bound,canonical,checked_bound,require,sha,read_bytes,write_once
from first_output_assessment import evaluate_witness,BLOCKED

VERSION='poseidon-external-dual-reference-assessment-v1'


def source_manifest():
    import native_dual_reference_evidence
    import first_output_numerical_evidence
    import first_output_assessment
    import first_output_backend
    paths=[Path(__file__).absolute(),*[Path(m.__file__).absolute()for m in [host_entry,native_dual_reference_evidence,
        first_output_numerical_evidence,first_output_assessment,first_output_backend]]]
    return {str(p):sha(read_bytes(p))for p in paths}


def bind_assessment(mechanism):
    host_entry._mechanism(mechanism)
    require('external_assessment_contract'not in mechanism,'External mode must be selected once before generation')
    sources=source_manifest()
    require(all(mechanism['source_sha256'].get(p)==v for p,v in sources.items()),'External assessment absent from generator freeze')
    return bound(dict({k:v for k,v in mechanism.items()if k!='binding'},
        external_assessment_contract=VERSION,external_assessment_sources=sources))


def verify_selection(mechanism):
    host_entry._mechanism(mechanism)
    require(mechanism.get('external_assessment_contract')==VERSION
        and mechanism.get('external_assessment_sources')==source_manifest(),'Pre-generation external mode/source selection changed')
    require(all(mechanism['source_sha256'].get(p)==v for p,v in source_manifest().items()),'Unfrozen external dependency')
    return mechanism


def assess_once(folder, expected_binding, policy, output, evaluator):
    """External-only evaluation with the frozen full request and unedited DSL.

The original witness comparisons/verifiers remain authoritative. No evaluation
or numeric result is returned to a Provider. Every frozen request owns exactly
one assessment directory; later development work is reported elsewhere.
"""
    receipt, host, source, mechanism = verify_host_delivery(folder, expected_binding)
    verify_selection(mechanism)
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
        independent_math_binding_verified=False, native_reference_comparisons_passed=False, native_reference_pairs=None)
    if source is not None:
        try:
            witness = evaluator(copy.deepcopy(host), source)
            verify_host_delivery(folder, expected_binding)
            write_once(path/'witness.json', canonical(witness))
            numeric = evaluate_witness(witness, numeric_policy, receipt['source_sha256'])
            result.update(numeric)
            if not numeric['test_only']:
                from first_output_numerical_evidence import verify_backend_witness, verify_mathematical_rows
                from native_dual_reference_evidence import verify_native_rows
                try:
                    verify_backend_witness(witness, host, source)
                    result['backend_binding_verified'] = True
                except (ValueError, KeyError, TypeError, IndexError, OSError):
                    result.update(raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
                        failure_category='backend_blocked', evidence_failure='actual_compiler_input_binding_missing_or_invalid')
                if result['backend_binding_verified'] and witness.get('inputs'):
                    try:
                        pairs = verify_native_rows(witness, host, source)
                        write_once(path/'native-reference-pairs.json', canonical(pairs))
                        result['native_reference_pairs'] = dict(path=str(path/'native-reference-pairs.json'), sha256=sha(canonical(pairs)), binding=pairs['binding'])
                        result['native_reference_comparisons_passed'] = pairs['passed']
                        result['native_values_binding_verified'] = True
                        if not pairs['passed']:
                            result.update(raw_numeric_passed=False, raw_joint_passed=False)
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
