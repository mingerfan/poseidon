"""Recompute unified candidate references, artifacts and native/public witnesses.

Derived from the historical r6 auditor; that evidence source remains unchanged.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from benchmark_graph import digest, require, samples
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference
from benchmark_runner import ROOT
from unified_graph_contract import validate_request, validate_candidate
from unified_public_coverage import verify_trace_coverage
from candidate_contract import request_input_names, request_rotations
from cipher_abi import artifact_options
from compiler_configuration import verify_artifact_configuration
from seal_artifact_gate import inspect_artifacts
from seal_cpu_golden import compare, PROFILE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_files(folder, hashes):
    for name, expected in hashes.items():
        path = folder / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe evidence path')
        require(path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink(),
                'Indirect evidence path')
        require(sha(path) == expected, 'Changed evidence: ' + name)


def verify_candidate(folder, *, live_approval=None, validation_level="numerical"):
    require(validation_level in ("compiled", "numerical"), "Audit validation level")
    report = read(folder / 'report.json')
    require(report.get("validation_level", "numerical") == validation_level, "Audit level mismatch")
    if live_approval is None:
        require(report['status'] == 'passed' and report['agent_calls'] == 0,
                'Expected manual passed candidate')
        require(not report['llm_generation_validated'], 'Not an Agent result')
    else:
        from agent_response_provenance import verify_answers
        verify_answers(folder, report, read(folder/'request.json'), live_approval)
    verify_files(ROOT / 'scripts/baseline', report['source_hashes'])
    verify_files(folder, report['frozen_hashes'])
    require(read(folder / 'key-cleanup-outcome.json')['complete'], 'Key cleanup incomplete')
    request = read(folder / 'request.json'); validate_request(request)
    model = read(folder / 'model.json')
    require(model == request['model'], 'Model/request mismatch')
    passed = [a for a in report['attempts'] if a.get('status') == 'passed']
    require(len(passed) == 1, 'Expected one successful manual attempt')
    attempt = passed[0]; attempt_dir = folder / ('attempt-%02d' % attempt['index'])
    out = attempt_dir / 'output'
    verify_files(out, attempt['artifact_hashes'])
    require(attempt["compiled"] and attempt["trace"]["frontend"] == "real_Hecate", "Missing real compilation")
    if validation_level == "numerical":
        require(attempt["executed"] and attempt["numerically_correct"], "Missing real execution")
        require(read(out / "execution.json") == attempt["execution"] and
                attempt["execution"]["encrypted_execution"], "Execution binding")
    else:
        require(attempt.get("compiled_validated") is True and not attempt["executed"] and
                not attempt["numerically_correct"], "Compiled-only evidence mismatch")
    checked = validate_candidate(dict(schema=1, request_id=request['request_id'],
                                      hecate_source=(attempt_dir / 'candidate.py').read_text()), request)
    if 'construction_exercise' not in request:
        coverage = recorded = None
    elif request.get('construction_profile') == 'hecate-unified-public-v1':
        coverage = verify_trace_coverage(checked['construction_exercise'],
                                         read(out / 'public-construction-events.json'))
        recorded=attempt['public_expression_coverage']
    else:
        from unified_native_coverage import verify_trace_coverage as native_coverage
        records={key:read(out/name) for key,name in (
            ('storage','native-array-events.json'),('star','native-star-events.json'),
            ('augmented','native-augmented-events.json'),('mutation','native-array-mutation-events.json'),
            ('calls','native-call-events.json'))}
        coverage=native_coverage(checked['construction_exercise'],records)
        recorded=attempt['packed_native_coverage']
    require(coverage == recorded, 'Directed trace mismatch')
    helper_coverage=None
    helper_trace=None
    if 'upstream_helpers' in request:
        from upstream_candidate_helpers import verify_sources,verify_events
        from poly_dependencies import verify as verify_dependency
        require(dict(sources=verify_sources(),dependency=verify_dependency())==report['upstream_helper_environment'],
                'Helper environment binding')
        helper_trace=verify_events(read(out/'upstream-calls.json'),request,(attempt_dir/'candidate.py').read_text())
        require(helper_trace==attempt['upstream_helper_trace'],'Actual helper trace mismatch')
        if 'upstream_exercise' in request:
            from upstream_helper_coverage import verify_trace
            helper_coverage=verify_trace(checked['upstream_exercise'],read(out/'upstream-calls.json'))
            require(helper_coverage==attempt['upstream_helper_coverage'],'Helper contribution witness mismatch')
        if request['upstream_helpers']['profile'] in ('upstream-poly-bn-silu-v2','upstream-poly-concat-bn-silu-v3','upstream-poly-spatial-v4','upstream-poly-spatial-mapped-v5','upstream-poly-fused-spatial-v6','upstream-poly-downsample-v7','upstream-poly-virtual-prefix-v8','upstream-poly-chunked-virtual-v9'):
            from upstream_adapters.constants import canonicalize
            recorded=read(out/'constant-layout.json')
            require(recorded==attempt['upstream_constant_layout'],'Constant bridge record binding')
            require(set(recorded)=={p.name for p in out.glob('*.cst')},'Constant bridge file coverage')
            for name,item in recorded.items():
                require(item['original_file']==name+'.upstream-original','Original constant identity')
                normalized,record=canonicalize((out/item['original_file']).read_bytes(),request['layout']['input_slot_period'])
                require(normalized==(out/name).read_bytes() and item==dict(record,original_file=item['original_file']),
                        'Non-lossless upstream constants')
    comparison = None
    period = request["layout"]["input_slot_period"]
    if validation_level == "numerical":
        expected_inputs = []; references = []
        period = request['layout']['input_slot_period']
        for inputs in samples(model, 4):
            first = mathematical(model, inputs); second = torch_reference(model, inputs)
            for name in first:
                np.testing.assert_allclose(first[name], second[name], atol=1e-12, rtol=1e-12)
            references.append(np.concatenate([first[o['name']].reshape(-1) for o in model['outputs']]))
            # Independent reconstruction, not the preparation helper.
            parts=[]
            for spec in model['inputs']:
                flat=inputs[spec['name']].reshape(-1)
                if 'logical_inputs' in request['layout']:
                    for start in range(0,len(flat),period):
                        part=flat[start:start+period]
                        parts.append(np.pad(part,(0,period-len(part))))
                else:parts.append(np.pad(flat,(0,period-len(flat))))
            expected_inputs.append(np.stack(parts))
        packed = np.stack(expected_inputs)
        if len(request['layout']['inputs']) == 1:
            packed = packed[:,0,:]
        with np.load(folder / 'arrays.npz', allow_pickle=False) as data:
            require(np.array_equal(data['inputs'], packed) and
                    np.array_equal(data['reference'], np.stack(references)), 'Reference/input mismatch')
            comparison = compare(np.load(out / 'decrypted.npy', allow_pickle=False),
                                 data['reference'], 1e-5, 1e-4)
        require(comparison == attempt['comparison'] and comparison['passed'], 'Numerical mismatch')
    gate = inspect_artifacts((out / 'lowered._hecate_golden.hevm').read_bytes(),
                             (out / '_hecate_golden.cst').read_bytes(),
                             rotation_steps=request_rotations(request),
                             expected_inputs=len(request_input_names(request)),
                             **artifact_options(request['layout']))
    verify_artifact_configuration(request, gate, sha(PROFILE))
    require(gate == attempt['artifact_gate'], 'Artifact gate mismatch')
    from seal_artifact_gate import verify_execution_binding
    if validation_level == "numerical":
        verify_execution_binding(gate,attempt["execution"],report)
    # Include post-execution arrays/cleanup as well as the originally frozen artifacts.
    files = {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    return dict(evidence=str(folder), files=files, request_id=request['request_id'],
                model_sha256=digest(model), exercise=request.get('construction_exercise',{}).get('id'), upstream_helper_trace=helper_trace,
                comparison=comparison, coverage=coverage, helper_coverage=helper_coverage, period=period,
                inputs=len(model['inputs']), outputs=len(model['outputs']))

