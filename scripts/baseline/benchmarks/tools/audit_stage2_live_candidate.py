"""Independently audit a live Agent answer and its retained FHE/witness evidence.

Shares the reviewed mathematical and artifact checks with the manual auditor;
does not relabel manual results, edit evidence, call a provider or execute DSL.
"""
import sys
import os
import shlex
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
if __name__ == "__main__" and "--inside" not in sys.argv:
    from hecate_python_env import VENV, enter_nix
    raise SystemExit(enter_nix(
        'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
        shlex.join([str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--inside"]),
        seconds=300))
from stage2_agent_provenance import verify_answers
MANUAL_AUDITOR_SOURCE_SHA256 = "3e1bfe3bb79d353c24a7e6807372e235c819ffddd4e2466a7b6f30e380b00383"

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


def verify_candidate(folder, paid):
    report = read(folder / 'report.json')
    provenance = verify_answers(folder, report, read(folder/'request.json'), paid)
    verify_files(ROOT / 'scripts/baseline', report['source_hashes'])
    verify_files(folder, report['frozen_hashes'])
    require(read(folder / 'key-cleanup-outcome.json')['complete'], 'Key cleanup incomplete')
    request = read(folder / 'request.json'); validate_request(request)
    model = read(folder / 'model.json')
    require(model == request['model'], 'Model/request mismatch')
    passed = [a for a in report['attempts'] if a.get('status') == 'passed']
    require(len(passed) == 1, 'Expected one successful Agent attempt')
    attempt = passed[0]; attempt_dir = folder / ('attempt-%02d' % attempt['index'])
    out = attempt_dir / 'output'
    verify_files(out, attempt['artifact_hashes'])
    require(all(attempt[k] for k in ('compiled','executed','numerically_correct')),
            'Missing real execution')
    require(attempt['trace']['frontend'] == 'real_Hecate', 'Not real tracing')
    require(read(out / 'execution.json') == attempt['execution'] and
            attempt['execution']['encrypted_execution'], 'Execution binding')
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
    verify_execution_binding(gate,attempt["execution"],report)
    # Include post-execution arrays/cleanup as well as the originally frozen artifacts.
    files = {str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
    return dict(evidence=str(folder), files=files, provenance=provenance, request_id=request['request_id'],
                model_sha256=digest(model), exercise=request.get('construction_exercise',{}).get('id'), upstream_helper_trace=helper_trace,
                comparison=comparison, coverage=coverage, helper_coverage=helper_coverage, period=period,
                inputs=len(model['inputs']), outputs=len(model['outputs']))


def main():
    import argparse
    from benchmark_runner import strict_file, dump
    from stage2_agent_pilot_plan import check_binding
    from semantic_benchmark_execution import runtime_sources
    from workspace_paths import RESULTS
    from hecate_python_env import VENV
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--case-id",required=True)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:
        p.error("Pinned pure Nix Python required")
    if a.output.exists() or not a.output.resolve().is_relative_to(RESULTS.resolve()):
        p.error("New platform result output required")
    if a.evidence.is_symlink() or a.evidence.resolve().parent!=RESULTS.resolve():
        p.error("Direct platform evidence directory required")
    plan=strict_file(a.plan,8*1024**2);check_binding(plan)
    require(runtime_sources()==plan["source_hashes"],"Current source differs from evaluated plan")
    for name,hsh in plan["proposal_files"].items():
        require(sha(ROOT/name)==hsh,"Proposal dependency changed")
    spec=next((c for c in plan["cases"] if c["id"]==a.case_id),None)
    require(spec is not None,"Case is outside approved proposal")
    require(read(a.evidence/"request.json")==spec["request"],"Changed request")
    require(digest(read(a.evidence/"model.json"))==spec["model_sha256"],"Changed model")
    audit=verify_candidate(a.evidence,plan["paid_configuration"])
    audit.update(format="poseidon-stage2-live-agent-audit-v1",case_id=a.case_id,
                 plan_binding=plan["binding"],status="passed",
                 audit_source_sha256=sha(Path(__file__)),
                 provenance_source_sha256=sha(Path(__file__).with_name("stage2_agent_provenance.py")),
                 new_encrypted_executions=0,new_paid_calls=0,stage2_complete=False)
    require(runtime_sources()==plan["source_hashes"],"Source changed during audit")
    audit["binding"]=digest(audit);dump(a.output,audit)
    print(json.dumps({k:audit[k] for k in ("case_id","status","comparison","provenance","binding")},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
