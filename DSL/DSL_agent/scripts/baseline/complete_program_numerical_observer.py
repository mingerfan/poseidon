"""External full-period numeric observations, separate from native authority.

The public retained entry verifies the independent mathematical records, then
reads the fixed B decoder. It never generates, compiles, repairs, decrypts or
certifies execution. Compiler/host/input/key/resource evidence is still needed
by the external first-output assessor before any delivery credit is possible.
"""
import itertools
import math
from pathlib import Path
import struct

from delivery_component import bound, canonical, checked_bound, read_bytes, require, sha
from first_output_assessment import CKKS, MATH, compare, _metrics_pass
import derived_compiler_program_adapter as adapter

FORMAT = 'poseidon-complete-program-complex-numeric-observations-v1'
CASE = 'poseidon-complete-program-complex-case-observation-v1'
MISSING_NATIVE = ['actual whole compiler/frontend host provenance',
    'actual raw-to-declared-to-encrypted input provenance',
    'same-key final-only host and original resource/terminal proof']


def source_manifest():
    import first_output_assessment
    import retained_math_evidence
    import retained_math_producer
    result = adapter.source_manifest()
    for module in (first_output_assessment, retained_math_evidence, retained_math_producer):
        p = Path(module.__file__).absolute(); result[str(p)] = sha(read_bytes(p))
    p = Path(__file__).absolute(); result[str(p)] = sha(read_bytes(p))
    return result


def _references(target, values):
    """Original named order, including logical aliases, never equal-value merge."""
    require(type(values) is list and len(values) == target['logical_values']
        and all(type(v) in (int, float) and math.isfinite(v) for v in values),
        'Complete finite declared reference in original named output order')
    by_name, cursor = {}, 0
    for row in target['logical_IO']['outputs']:
        size = math.prod(row['shape']); selected = values[cursor:cursor+size]; cursor += size
        require(row['name'] not in by_name and len(selected) == size, 'Distinct complete output reference')
        if row['alias_of'] is not None:
            require(row['alias_of'] in by_name
                and canonical(selected) == canonical(by_name[row['alias_of']]),
                'A model-value alias must retain the same original reference')
        by_name[row['name']] = selected
    require(cursor == len(values), 'No omitted or surplus named references')
    return by_name


def _stats():
    return dict(compared_values=0, compared_real_and_imaginary_components=0,
        failed_values=0, nonfinite_native_values=0, nonfinite_errors=0,
        max_absolute_error=0., maximum_real_absolute_error=0.,
        maximum_imaginary_absolute_error=0., _scale=0., _scaled_squares=0.)


def _record(stats, reference, real, imaginary):
    stats['compared_values'] += 1; stats['compared_real_and_imaginary_components'] += 2
    if not (math.isfinite(real) and math.isfinite(imaginary)):
        stats['nonfinite_native_values'] += 1; stats['failed_values'] += 1
        return
    real_error = abs(real-reference); error = math.hypot(real-reference, imaginary)
    if not math.isfinite(error):
        stats['nonfinite_errors'] += 1; stats['failed_values'] += 1
        return
    stats['failed_values'] += int(error > CKKS['atol']+CKKS['rtol']*abs(reference))
    stats['max_absolute_error'] = max(stats['max_absolute_error'], error)
    stats['maximum_real_absolute_error'] = max(stats['maximum_real_absolute_error'], real_error)
    stats['maximum_imaginary_absolute_error'] = max(stats['maximum_imaginary_absolute_error'], abs(imaginary))
    if error:
        if error > stats['_scale']:
            stats['_scaled_squares'] = 1.+stats['_scaled_squares']*(stats['_scale']/error)**2
            stats['_scale'] = error
        else:
            stats['_scaled_squares'] += (error/stats['_scale'])**2


def _finish(stats):
    value = {k:v for k,v in stats.items() if not k.startswith('_')}
    invalid = bool(stats['nonfinite_native_values'] or stats['nonfinite_errors'])
    # Nonfinite native/error cells remain failures. Never replace their metrics
    # with zero or serialize NaN/Infinity as if those were finite observations.
    for key in ('max_absolute_error', 'maximum_real_absolute_error', 'maximum_imaginary_absolute_error'):
        if invalid: value[key] = None
    rmse = stats['_scale']*math.sqrt(stats['_scaled_squares']/stats['compared_values']) if stats['compared_values'] else None
    value['RMSE'] = None if invalid or rmse is None or not math.isfinite(rmse) else rmse
    value['all_finite_errors'] = not invalid
    return value


def compare_decoded(target, expected_target_binding, byte_receipt, expected_decoder_binding,
                    cells, declared_reference):
    """Pure observation seam. Live callers obtain cells only from pinned B.

    Metadata and every streamed raw byte cell are checked. This function grants
    no native, mathematical, compiler, SDK, Bootstrap or first-output authority.
    """
    checked_bound(target, expected_target_binding); checked_bound(byte_receipt)
    require(target['format'] == adapter.TARGET
        and target['full_decoder_contract_or_actual_host_input_math_CKKS_or_first_delivery_qualification'] is False
        and target['every_periodic_copy_included'] is True, 'Explicit complete-period derived target role')
    require(byte_receipt['format'] == 'poseidon-B-program-complex-decoder-byte-receipt-v2'
        and byte_receipt['decoder_source_sha256'] == adapter.DECODER_SHA
        and byte_receipt['decoder_contract_binding'] == expected_decoder_binding
        and byte_receipt['identity'] == target['physical_identity']
        and byte_receipt['outputs'] == target['decoder_outputs']
        and byte_receipt['input_slot_period'] == target['input_slot_period']
        and byte_receipt['compact_periodic_copies_are_complete'] is True
        and byte_receipt['raw_nonfinite_bits_preserved'] is True,
        'Original decoder/target/case/key/period scope')
    require(byte_receipt['logical_values'] == target['logical_values']
        and byte_receipt['explicitly_recorded_alias_values'] == target['periodic_alias_values'],
        'All named logical and complete periodic aliases')
    references = _references(target, declared_reference)
    _, decoder = adapter.pinned_B_modules()
    physical = byte_receipt['physical_files']; files = {}
    require(type(physical) is list and 1 <= len(physical) <= 16, 'Bounded original physical finals')
    for row in physical:
        require(row['name'] not in files and row['slots'] == 16384 and row['bytes'] == 262144,
            'Distinct original complex file extent')
        require(type(row['ciphertext_sha256']) is str and len(row['ciphertext_sha256']) == 64
            and all(c in '0123456789abcdef' for c in row['ciphertext_sha256']),
            'Original final ciphertext identity')
        files[row['name']] = row
    decoder.validate_outputs(target['decoder_outputs'], set(files), target['input_slot_period'])
    stats = _stats(); by_output = {name:_stats() for name in references}
    selectors = decoder.selector_cells(target['decoder_outputs']); missing = object()
    for expected, actual in itertools.zip_longest(selectors, cells, fillvalue=missing):
        require(expected is not missing and actual is not missing, 'Every selected cell once; no surplus or truncation')
        require(type(actual) is dict and all(type(actual.get(k)) is type(v) and actual[k] == v for k,v in expected.items()),
            'Exact original named output, slot, alias ordinal and both byte offsets')
        require(actual['ciphertext_sha256'] == files[actual['file']]['ciphertext_sha256'], 'Stream ciphertext origin changed')
        raw = actual['raw_cell_hex_le']
        require(type(raw) is str and len(raw) == 32, 'One original LE complex byte cell')
        try:
            payload = bytes.fromhex(raw); real, imaginary = struct.unpack('<dd', payload)
            same = struct.pack('<dd', actual['real'], actual['imaginary']) == payload
        except (ValueError, TypeError, OverflowError, struct.error):
            raise ValueError('Invalid original complex cell bytes') from None
        require(type(actual['real']) is float and type(actual['imaginary']) is float and same,
            'Reported components differ from original raw bytes')
        finite = math.isfinite(real) and math.isfinite(imaginary)
        require(type(actual['both_components_finite']) is bool and actual['both_components_finite'] is finite,
            'Nonfinite raw bits must remain explicit')
        reference = references[actual['output']][actual['logical_index']]
        _record(stats, reference, real, imaginary); _record(by_output[actual['output']], reference, real, imaginary)
    total = target['logical_values']+target['periodic_alias_values']
    require(stats['compared_values'] == total and 2*total == target['complex_components_with_periodic_aliases'],
        'Original complete component accounting')
    return bound(dict(format=CASE, derived_target_binding=target['binding'], decoder_byte_receipt_binding=byte_receipt['binding'],
        physical_identity=target['physical_identity'], source_sha256=target['source_sha256'], request_id=target['request_id'],
        declared_reference_sha256=sha(canonical(declared_reference)), logical_values=target['logical_values'],
        logical_alias_values=target['logical_alias_cells'], periodic_alias_values=target['periodic_alias_values'],
        CKKS=dict(CKKS), comparison=_finish(stats), named_outputs={name:_finish(value) for name,value in by_output.items()},
        all_selected_cells_consumed=True, CKKS_observations_within_tolerance=stats['failed_values'] == 0,
        reference_is_not_a_native_core_substitute=True, independent_mathematical_records_verified_here=False,
        compiler_host_input_samekey_resource_or_first_delivery_verified=False,
        raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False))


def _case_scope(case, request, source, input_binding):
    """Assessor-selected identities are distinct from self-bound decoder claims."""
    require(type(case) is dict and set(case) == {'input_sha256','target','expected_target_binding',
        'derived_target','expected_derived_target_binding','expected_request','expected_identity',
        'expected_generation_receipt_binding','expected_trace_binding','expected_target_inputs','original_input_binding'},
        'Closed external case selection')
    require(case['input_sha256'] == input_binding['physical_input_sha256']
        and case['original_input_binding'] == input_binding, 'Original raw/logical/physical case selection')
    contract = checked_bound(case['target'], case['expected_target_binding'])
    target = checked_bound(case['derived_target'], case['expected_derived_target_binding'])
    require(target['source_sha256'] == sha(source) and target['request_id'] == request['request_id'], 'Original frozen request/DSL target')
    require(contract['request'] == case['expected_request']
        and contract['physical_identity'] == case['expected_identity'] == target['physical_identity']
        and contract['inputs'] == case['expected_target_inputs'], 'Externally selected target/request/input/key identity')
    first = contract['first_source']
    require(first['source']['sha256'] == sha(source)
        and first['generation_receipt_binding'] == case['expected_generation_receipt_binding']
        and first['trace_binding'] == case['expected_trace_binding'], 'Original generation/source trace commitment')
    require(contract['outputs'] == target['target_outputs']
        and contract['compile']['io_map_binding'] == target['logical_IO']['binding']
        and contract['compile']['compiler_dag_binding'] == target['compiler_program_map']['binding']
        and contract['compile']['input_slot_period'] == target['input_slot_period'],
        'Complete source-derived target and compact program map')
    return contract, target


def _numerical_layers(policy, row, result):
    require(policy['independent_math'] == MATH and policy['CKKS'] == CKKS, 'Original numerical thresholds')
    checked_bound(result)
    require(result['format'] == CASE and result['CKKS'] == CKKS
        and result['declared_reference_sha256'] == sha(canonical(row['declared_reference'])),
        'Same original declared reference for independent math and complex comparison')
    approximation = compare(row['original_reference'], row['declared_reference'], atol=0., rtol=0.)
    dsl_math = compare(row['declared_reference'], row['dsl_mathematical_output'], **MATH)
    domain = row['inside_frozen_domain'] is True and row['domain_sha256'] == policy['frozen_domain_sha256']
    application = _metrics_pass(policy['application_requirements'], row['application_metrics'])
    approximate_ok = _metrics_pass(policy['approximation_requirements'], approximation)
    whole = not policy['whole_model_required'] or row['whole_model_output'] is True
    return dict(original_function_to_declared_approximation=approximation, independent_DSL_mathematics=dsl_math,
        domain_passed=domain, approximation_passed=approximate_ok, application_passed=application,
        whole_model_passed=whole, numerical_observations_passed=bool(domain and application and approximate_ok and whole
            and dsl_math['failed_values'] == 0 and result['CKKS_observations_within_tolerance']))


def observe_retained(request, source, policy, mathematical_witness, cases):
    """Full frozen-input numeric observations; native eligibility stays separate.

    Both reference provenance and independent DSL math use the existing retained
    producer verifier. No mathematical model or preprocessing is reexecuted.
    Missing cases reject the observation request; they cannot shrink a cohort.
    """
    import retained_math_evidence
    checked_bound(policy); spec = checked_bound(policy['retained_mathematical_producer'])
    require(policy['independent_math'] == MATH and policy['CKKS'] == CKKS, 'Unchanged independent math/CKKS tolerances')
    require(type(cases) is list and len(cases) == len(policy['inputs']) and bool(cases), 'Every frozen input requires a complex case')
    require([c['input_sha256'] for c in cases] == policy['inputs']
        and len(set(policy['inputs'])) == len(policy['inputs']), 'Original distinct case order; no omitted, duplicate or replacement input')
    bindings = spec['input_bindings']; require(len(bindings) == len(cases), 'Complete retained input binding scope')
    selected = [_case_scope(c, request, source, binding) for c,binding in zip(cases, bindings)]
    mathematical = retained_math_evidence.verify_mathematical_rows(mathematical_witness, policy, request, source)
    require(mathematical['all_policy_inputs_covered'] is True and mathematical['input_groups'] == len(cases),
        'All independently verified mathematical inputs required')
    rows = {r['input_sha256']:r for r in mathematical_witness['inputs']}
    require(set(rows) == set(policy['inputs']), 'Original mathematical reference coverage')
    bridge, _ = adapter.pinned_B_modules(); observations = []
    for case,(contract,target) in zip(cases, selected):
        row = rows[case['input_sha256']]
        require(sha(read_bytes(contract['first_source']['source']['path'])) == sha(source), 'Preserved original DSL file')
        from candidate_contract import strict_json
        model_bytes = read_bytes(contract['model']['path'])
        require(sha(model_bytes) == contract['model']['sha256']
            and canonical(strict_json(model_bytes.decode('utf-8'))) == canonical(request['model']),
            'Original full requested model, including its weights and operators')
        bridge_receipt, byte_receipt, cells = bridge.decode(contract, case['expected_target_binding'], adapter.DECODER, adapter.DECODER_SHA)
        result = compare_decoded(target, case['expected_derived_target_binding'], byte_receipt,
            contract['decoder_binding'], cells(include_aliases=True), row['declared_reference'])
        observations.append(dict(input_sha256=case['input_sha256'], original_input_binding=case['original_input_binding'],
            byte_bridge_binding=bridge_receipt['binding'], complex_observation=result,
            **_numerical_layers(policy, row, result)))
    return bound(dict(format=FORMAT, mode='external_numeric_observations_pending_actual_native_witnesses',
        request_id=request['request_id'], source_sha256=sha(source), numeric_policy_sha256=sha(canonical(policy)),
        mathematical_provenance=mathematical, input_groups=len(observations), inputs=observations,
        complete_frozen_inputs_covered=True, independent_math=dict(MATH), CKKS=dict(CKKS),
        numerical_observations_passed=all(r['numerical_observations_passed'] for r in observations),
        test_only=spec['test_only'], required_external_native_evidence=list(MISSING_NATIVE),
        compiler_host_input_samekey_resource_or_first_delivery_verified=False,
        raw_compile_passed=False, raw_numeric_passed=False, raw_joint_passed=False,
        first_compile_passed=False, first_numeric_passed=False, first_joint_passed=False,
        Provider_calls=0, native_execution_or_private_loader_calls=0, source_modifications=0,
        default_first_delivery_or_Bootstrap_or_SDK_gate_changed=False))
