"""Bind reported numerical cells to retained native and independent math data.

This verifier executes neither a model nor a candidate. It cannot turn a JSON
success flag into real compilation/CKKS evidence. Independent producers and
original-model identity must be frozen in the pre-generation evaluation policy.
"""
from pathlib import Path

from delivery_component import (canonical, checked_bound, read_bytes, require, sha)
from first_output_backend import _json, verify_qualification_binding



def native_input_samples(request, payload):
    """Read the native serializer's declared list or wide-layout envelope."""
    from poseidon_validation import execution_contract
    from cipher_abi import physical_input_names
    import math
    contract=execution_contract(request['layout'])
    if contract is None:
        require(type(payload) is list, 'Original native input list required')
        samples=payload
    else:
        require(type(payload) is dict and set(payload)==set(contract)|{'samples'}
                and all(type(payload[k]) is type(v) and payload[k]==v for k,v in contract.items()),
                'Original wide native input envelope required')
        samples=payload['samples']
    period=request['layout']['input_slot_period'];count=len(physical_input_names(request['layout']))
    require(type(samples) is list and 1<=len(samples)<=1024, 'Finite native input group count')
    require(all(type(sample) is list and len(sample)==count and
                all(type(vector) is list and len(vector)==period and
                    all(type(value) in (int,float) and math.isfinite(value) for value in vector)
                    for vector in sample) for sample in samples), 'Declared native input ciphertext shapes')
    return samples


def verify_backend_witness(witness, request, source):
    proof = witness.get('backend_binding')
    require(type(proof) is dict, 'Actual compiler input binding missing')
    verify_qualification_binding(proof, request, source)
    require(witness.get('compiled_validated') is proof['real_compilation_observed']
            and witness.get('real_compiler') is proof['real_compilation_observed'], 'Compiler observations differ from actual evidence')
    require(witness.get('encrypted_execution') is proof['real_encrypted_execution_observed']
            and witness.get('real_CKKS') is proof['real_encrypted_execution_observed'], 'CKKS observations differ from actual evidence')
    return proof


def verify_native_rows(witness, request, source, *, policy=None):
    proof = verify_backend_witness(witness, request, source)
    require(proof['real_compilation_observed'] and proof['real_encrypted_execution_observed'], 'Actual compiler and native outputs required')
    if (type(policy)is dict and 'native_output_profile'in policy) or 'native_output_profile'in witness:
        from ordinary_complete_numerical_evidence import verify_rows
        return verify_rows(witness,request,source,policy,proof)
    root = Path(proof['evidence'])
    report = _json(root/'report.json')
    attempt = report['attempts'][0]
    from poseidon_input_groups import audit as audit_input_groups
    audit_input_groups(root/'attempt-00/output')
    execution = _json(root/'attempt-00/output/execution.json')
    inputs = native_input_samples(request, _json(root/'attempt-00/output/inputs.json'))
    outputs = execution.get('outputs')
    reference = attempt.get('comparison', {}).get('reference')
    actual = attempt.get('comparison', {}).get('actual')
    require(type(inputs) is list and type(outputs) is list and type(reference) is list
            and type(actual) is list and len(inputs) == len(outputs) == len(reference) == len(actual),
            'Native input/output/reference cells missing')
    selectors = request.get('layout', {}).get('output_selectors')
    require(type(selectors) is list and bool(selectors)
            and all(type(s) is list and len(s) == 2 and all(type(i) is int and i >= 0 for i in s) for s in selectors),
            'Explicit audited output decoder required')
    # This profile exports decoded real slots. Other/native decoders need their
    # own recorded mapping; no guessed shape, index, or imaginary component.
    native = {}
    for sample, output, expected, observed in zip(inputs, outputs, reference, actual):
        key = sha(canonical(sample))
        require(key not in native, 'Duplicate qualification input group')
        decoded = [output[ciphertext][slot] for ciphertext, slot in selectors]
        require(decoded == observed, 'Recorded native decode differs from comparison cells')
        native[key] = dict(actual=decoded, reference=expected)
    rows = witness.get('inputs')
    require(type(rows) is list and bool(rows), 'Numerical input rows')
    seen = set()
    for row in rows:
        key = row['input_sha256']
        require(key in native and key not in seen, 'Reported input is not a unique native input')
        seen.add(key)
        require(row['CKKS_real'] == native[key]['actual'], 'Reported CKKS values differ from actual native outputs')
        require(row['declared_reference'] == native[key]['reference'], 'Reported reference differs from original frozen comparison')
        imaginary = row.get('CKKS_imag')
        require(imaginary is None, 'This recorded real-output decoder has no complex-output witness')
    require(type(witness.get('intermediate_decryptions')) is int and witness['intermediate_decryptions'] == 0
            and witness.get('reference_or_tolerance_changed') is False, 'Original execution/reference boundary')
    return proof


def verify_mathematical_rows(witness, policy, request, source):
    """Require the independently produced three-layer mathematical record.

    The source list binds the trusted producer code, not a dynamically loaded
    candidate/plugin. A frozen producer still needs a live acceptance run before
    its scope is certified; this verifier never re-signs historical certificates.
    """
    if 'retained_mathematical_producer' in policy:
        from retained_math_evidence import verify_mathematical_rows as verify_retained
        return verify_retained(witness, policy, request, source)
    frozen = policy.get('independent_math_sources')
    original = policy.get('original_model_sha256')
    require(type(frozen) is dict and len(set(frozen.values())) >= 2, 'Independent mathematical producer sources were not frozen')
    require(type(original) is str and len(original) == 64, 'Original mathematical model identity missing')
    for name, expected in frozen.items():
        path = Path(name).absolute()
        require(path.suffix == '.py' and sha(read_bytes(path)) == expected, 'Frozen mathematical producer changed')
    descriptor = witness.get('mathematical_evidence')
    require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256'}, 'Independent mathematical evidence descriptor')
    path = Path(descriptor['path']).absolute()
    backend = witness['backend_binding']
    require(path.suffix == '.json' and path.is_relative_to(Path(backend['evidence']).parent), 'Mathematical evidence outside assessment storage')
    require(sha(read_bytes(path)) == descriptor['sha256'], 'Independent mathematical record changed')
    evidence = checked_bound(_json(path))
    require(evidence.get('format') == 'poseidon-three-layer-mathematical-evidence-v1'
            and evidence.get('request_id') == request['request_id']
            and evidence.get('source_sha256') == sha(source)
            and evidence.get('policy_sha256') == sha(canonical(policy))
            and evidence.get('original_model_sha256') == original
            and evidence.get('declared_model_sha256') == sha(canonical(request['model'])), 'Mathematical model/source/policy binding')
    require(evidence.get('producer_sources') == frozen and evidence.get('candidate_python_executed') is False
            and evidence.get('CKKS_outputs_used_as_reference') is False
            and evidence.get('test_only') is False, 'Independent mathematical provenance required')
    if 'mathematical_producer' in policy:
        spec = checked_bound(policy['mathematical_producer'])
        require(spec.get('format') == 'poseidon-frozen-mathematical-producer-v1'
                and spec.get('producer_sources') == frozen
                and spec.get('original_model_sha256') == original
                and spec.get('declared_model_sha256') == sha(canonical(request['model']))
                and sha(canonical(spec['input_domain'])) == policy['frozen_domain_sha256']
                and evidence.get('producer_spec_binding') == spec['binding']
                and evidence.get('reference_origin') == spec['reference_kind'], 'Frozen mathematical producer/model/domain selection')
        assets = spec.get('producer_assets')
        require(type(assets) is dict and all(Path(name).name != '.env'
                and sha(read_bytes(name)) == expected for name, expected in assets.items()),
                'Frozen mathematical non-code dependency changed')
    rows = evidence.get('inputs')
    require(type(rows) is list, 'Independent mathematical rows')
    by_input = {row['input_sha256']: row for row in rows}
    require(len(by_input) == len(rows), 'Duplicate mathematical input group')
    fields = ('source_sha256', 'original_reference', 'declared_reference', 'dsl_mathematical_output',
              'inside_frozen_domain', 'domain_sha256', 'application_metrics', 'whole_model_output')
    for row in witness['inputs']:
        recorded = by_input.get(row['input_sha256'])
        require(type(recorded) is dict and all(key in recorded and row.get(key) == recorded[key] for key in fields),
                'Reported mathematical/domain/application cells differ from independent record')
    return dict(path=str(path), sha256=descriptor['sha256'], binding=evidence['binding'])
