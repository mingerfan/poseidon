"""Recompute saved manual-candidate evidence; no inference or FHE execution.

The graph reference is evaluated from the original input arrays, never from DSL
outputs. Successful counterexample detection is not counted as a correct model.
"""
import io
import json
from pathlib import Path

from audit_agent_lineage import RESULTS, metadata, read, require


def recompute_arrays(model, inputs, reference, actual, recorded, expected_pass):
    import numpy as np
    from model_graph import evaluate_reference, input_specs, validate_graph
    from seal_cpu_golden import compare
    require(type(expected_pass) is bool, 'Expected result must be explicit')
    validate_graph(model)
    specs = input_specs(model)
    expected_shape = (4, 4) if model['schema'] == 2 else (4, len(specs), 4)
    require(inputs.shape == expected_shape and inputs.dtype == np.float64 and
            reference.dtype == np.float64 and actual.dtype == np.float64,
            'Unexpected original input or output shape/dtype')
    independent = []
    for batch in inputs:
        supplied = (batch.reshape(specs[0]['shape']).tolist() if model['schema'] == 2 else
                    {s['name']: batch[i].reshape(s['shape']).tolist() for i, s in enumerate(specs)})
        independent.append(evaluate_reference(model, supplied))
    independent = np.asarray(independent, dtype=np.float64)
    require(reference.shape == independent.shape and
            np.allclose(reference, independent, rtol=1e-12, atol=1e-12),
            'Saved reference differs from independent original-input graph evaluation')
    compared = compare(actual, reference, 1e-5, 1e-4)
    require(compared == recorded, 'Saved numerical metrics differ from actual arrays')
    require(compared['passed'] is expected_pass, 'Unexpected numerical outcome')
    return compared


def audit_manual_case(run, expected_pass, *, root=RESULTS, golden_source=None):
    import numpy as np
    from candidate_contract import validate_candidate, request_input_names, request_rotations
    from model_semantic_coverage import analyze_graph
    from seal_artifact_gate import inspect_artifacts
    from zero_evidence import verify_zero_execution
    run, root = Path(run), Path(root).resolve()
    require(not run.is_symlink() and run.resolve().parent == root, 'Manual evidence outside results root')
    run = run.resolve()
    report, report_hash = metadata(run/'report.json', root)
    require(report.get('agent_calls') == 0 and report.get('llm_generation_validated') is False,
            'Manual audit cannot relabel Agent evidence')
    require(report.get('backend') == 'upstream_SEAL_HEVM_CPU' and report.get('poseidon_gpu_validated') is False,
            'Unexpected backend claim')
    require(report.get('tolerance') == dict(atol=1e-5, rtol=1e-4), 'Frozen tolerance changed')
    params = report['parameters']
    require(params['seal_version'] == '4.0.0' and params['polynomial_degree'] == 32768 and
            params['security_check'] == 'tc128' and params['parameters_set'] is True and
            params['modulus_bits'] == [60]*14, 'Security parameters changed')
    require(len(report['attempts']) == 1, 'Expected one explicit manual program')
    attempt = report['attempts'][0]
    require(attempt['index'] == 0 and all(attempt.get(k) is True for k in
            ('parsed', 'source_parsed', 'checked', 'compiled', 'executed')), 'Incomplete native execution evidence')
    if expected_pass:
        require(report['status'] == 'passed' and attempt['numerically_correct'] is True, 'Missing positive pass')
    else:
        require(report['status'] != 'passed' and attempt.get('failure_layer') == 'numerical_comparison',
                'Counterexample must reach decrypted numerical comparison')
    frozen = report['frozen_hashes']
    require({'arrays.npz','weights.npz','request.json','model.json','attempt-00/trace-payload.json'} <= set(frozen),
            'Missing frozen model, inputs or trace payload')
    for name, digest in frozen.items():
        require(read(run/name, run)[1] == digest, 'Frozen input hash mismatch')
    model, _ = metadata(run/'model.json', run)
    request, _ = metadata(run/'request.json', run)
    payload, payload_hash = metadata(run/'attempt-00/trace-payload.json', run)
    require(model == request['model'] and request == payload['request'], 'Model/request/trace identity mismatch')
    candidate = payload['candidate']
    require(validate_candidate(candidate, request) == attempt['static_check'], 'Static contract differs from traced payload')
    require(read(run/'attempt-00/candidate.py', run)[0].decode('utf-8') == candidate['hecate_source'],
            'Loose candidate source differs from frozen trace payload')
    if golden_source is not None:
        require(candidate['hecate_source'] == golden_source, 'Traced program differs from declared manual golden')
    output = run/'attempt-00/output'
    hashes = attempt['artifact_hashes']
    require({'lowered._hecate_golden.hevm','_hecate_golden.cst','lowered.ckks.mlir','lowered.earth.mlir',
             'trace-evidence.json','arrays.npz'} <= set(hashes), 'Missing immutable compiler or runtime-input artifact')
    for name, digest in hashes.items():
        require(read(output/name, output)[1] == digest, 'Compiled artifact hash mismatch')
    trace, _ = metadata(output/'trace-evidence.json', output)
    require(trace == attempt['trace'] and trace['frontend'] == 'real_Hecate' and
            trace['candidate_python_executed'] is False and trace['request_id'] == request['request_id'],
            'Missing safe real tracing evidence')
    gate = inspect_artifacts(read(output/'lowered._hecate_golden.hevm', output)[0],
                            read(output/'_hecate_golden.cst', output)[0],
                            rotation_steps=request_rotations(request), expected_inputs=len(request_input_names(request)))
    require(gate == attempt['artifact_gate'], 'Reparsed HEVM/CST differs from recorded artifact gate')
    require('--verify-each' in attempt['compile_command'], 'Compiler verification was disabled')
    execution, _ = metadata(output/'execution.json', output)
    require(execution == attempt['execution'] and execution['encrypted_execution'] is True and
            execution['bootstrap_executed'] is False and execution['input_batches'] == 4,
            'Missing four saved real encrypted executions')
    key_check = execution.get('rotation_key_check')
    if key_check is None:
        # The v1 arithmetic runs predate the key-file probe. Do not invent it or
        # let a newer contract silently lose its required key-file evidence.
        require(request['task'] == 'hecate-function-synthesis-v1', 'Missing required rotation-key evidence')
        key_evidence = 'not_recorded_in_legacy_v1'
    else:
        require(key_check['actual_key_file_verified'] is True and
                key_check['required_steps'] == gate['rotation_steps'], 'Rotation key evidence mismatch')
        key_evidence = 'actual_key_file_verified'
    verify_zero_execution(request['layout'], execution)
    arrays_raw, arrays_hash = read(run/'arrays.npz', run)
    actual_raw, actual_hash = read(output/'decrypted.npy', output)
    runtime_raw, _ = read(output/'arrays.npz', output)
    with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as arrays, \
         np.load(io.BytesIO(runtime_raw), allow_pickle=False) as runtime:
        require(set(runtime.files) == {'inputs'} and np.array_equal(arrays['inputs'], runtime['inputs']),
                'Runtime must receive unchanged original inputs, not reference answers')
        compared = recompute_arrays(model, arrays['inputs'], arrays['reference'],
            np.load(io.BytesIO(actual_raw), allow_pickle=False), attempt['comparison'], expected_pass)
    require(metadata(run/'report.json', root)[1] == report_hash, 'Report changed during read-only audit')
    return dict(case=model['id'], evidence=str(run), report_sha256=report_hash,
        arrays_sha256=arrays_hash, decrypted_sha256=actual_hash, trace_payload_sha256=payload_hash,
        expected_numerical_pass=expected_pass, actual_numerical_pass=compared['passed'],
        compared_values=compared['compared_values'], input_batches=4,
        max_absolute_error=compared['max_absolute_error'], mae=compared['mae'],
        model_semantics=analyze_graph(model), hevm_opcode_counts=gate['opcode_counts'],
        independent_reference_recomputed=True, compiled_artifacts_reparsed=True,
        original_inputs_only_in_runtime=True, manual_source_matched=golden_source is not None,
        rotation_key_evidence=key_evidence,
        agent_calls=0, new_fhe_executions=0)
