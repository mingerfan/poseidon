"""Audit targeted native-call evidence. Never generates candidates or runs FHE.

Paid is the default and rejects manual/replay results. --manual explicitly audits
the separate golden cohort; it never upgrades those results to Agent evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, metadata, read, require, audit_case
from candidate_contract import validate_candidate, strict_json, canonical
from deepseek_provider import public_request
from native_function_exercises import CATALOG, EXERCISES, descriptor, verify_trace_coverage
from run_agent_batch import case_metrics, summarize


def audit_run(run, name, *, live):
    import numpy as np
    from seal_cpu_golden import compare
    run = Path(run).resolve()
    require(run.parent == RESULTS, 'Case must be a direct child of results')
    report, report_hash = metadata(run/'report.json',RESULTS)
    require(report['status'] == 'passed', 'Run did not pass')
    require((report.get('llm_generation_validated') is True and report.get('agent_calls',0) > 0
             and report.get('provider') == 'deepseek_api') if live else
            (report.get('llm_generation_validated') is False and report.get('agent_calls') == 0
             and report.get('provider') == 'scripted_replay'), 'Evidence generation kind mismatch')
    require(report['backend'] == 'upstream_SEAL_HEVM_CPU' and not report['poseidon_gpu_validated'],
            'Unexpected execution backend')
    require(report['tolerance'] == dict(atol=1e-5,rtol=1e-4),'Frozen tolerance changed')
    params = report['parameters']
    require(params['security_check'] == 'tc128' and params['polynomial_degree'] == 32768
            and params['modulus_bits'] == [60]*14 and params['parameters_set']
            and params['seal_version'] == '4.0.0', 'Security profile changed')
    require({'model.json','request.json','arrays.npz','weights.npz'} <= set(report['frozen_hashes']),
            'Missing frozen reference data')
    for filename,value in report['frozen_hashes'].items():
        require(read(run/filename,run)[1] == value,'Frozen input changed')
    model,_ = metadata(run/'model.json',run)
    require(model == descriptor(name),'Frozen model changed')
    request,_ = metadata(run/'request.json',run)
    public_request(request)
    require(request['model'] == model and request['construction_exercise']['id'] == name, 'Request model mismatch')
    attempt = next(a for a in reversed(report['attempts']) if a.get('numerically_correct'))
    directory = run/('attempt-%02d' % attempt['index'])
    payload,payload_hash = metadata(directory/'trace-payload.json',run)
    require(str((directory/'trace-payload.json').relative_to(run)) in report['frozen_hashes'],'Missing frozen payload')
    require(payload['request'] == request,'Traced request mismatch')
    candidate = payload['candidate']
    response = strict_json(read(directory/'response.txt',run)[0].decode())
    require(response == candidate,'Provider/replay response differs from traced source')
    require(read(directory/'candidate.py',run)[0].decode() == candidate['hecate_source'],'Candidate sidecar mismatch')
    check = validate_candidate(candidate,request)
    require(canonical(check) == canonical(attempt['static_check']),'Recomputed static coverage changed')
    out = directory/'output'
    hashes = attempt['artifact_hashes']
    require({'native-call-events.json','trace-evidence.json','candidate_trace.mlir','lowered.ckks.mlir',
             'lowered._hecate_golden.hevm','_hecate_golden.cst'} <= set(hashes),'Missing real frontend/compiler evidence')
    for filename,value in hashes.items():
        require(read(out/filename,out)[1] == value,'Frozen compiled artifact changed')
    trace,_ = metadata(out/'trace-evidence.json',out)
    require(trace == attempt['trace'] and trace == dict(frontend='real_Hecate',candidate_python_executed=False,
            construction='validated_native_AST_to_Hecate_functions',request_id=request['request_id']),
            'Invalid trusted trace evidence')
    calls,_ = metadata(out/'native-call-events.json',out)
    native = verify_trace_coverage(check['construction_exercise'],calls)
    require(native == attempt['native_call_coverage'],'Saved native call-site audit changed')
    execution,_ = metadata(out/'execution.json',out)
    require(execution == attempt['execution'] and execution['encrypted_execution']
            and execution['input_batches'] == 4 and not execution['bootstrap_executed'],
            'Missing actual four-batch encrypted execution')
    require(execution['encrypted_input_count'] == (2 if name == 'nf-two-inputs' else 1),
            'Encrypted input count mismatch')
    with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
        inputs, reference = arrays['inputs'], arrays['reference']
        independent = (inputs[:,0,:]-inputs[:,1,:] if name == 'nf-two-inputs' else inputs*.5+inputs+.375)
        require(np.allclose(reference,independent,atol=1e-15,rtol=0),'Independent model reference mismatch')
    _, decrypted_hash = read(out/'decrypted.npy',out)
    actual = np.load(out/'decrypted.npy',allow_pickle=False)
    comparison = compare(actual,reference,1e-5,1e-4)
    require(comparison == attempt['comparison'] and comparison['passed'],'Numerical evidence mismatch')
    cleanup,_ = metadata(run/'key-cleanup-outcome.json',run)
    require(cleanup['complete'] and not (run/'private-keys').exists(),'Key cleanup incomplete')
    return dict(id=name,evidence=str(run),report_sha256=report_hash,trace_payload_sha256=payload_hash,
        decrypted_sha256=decrypted_hash,exercise=check['construction_exercise'],native_trace=native,
        comparison=comparison,freed_key_bytes=cleanup['freed_bytes'],live_agent=live)


def audit(batch, *, manual=False):
    root = Path(batch).resolve()
    report,report_hash = metadata(root/'report.json',RESULTS)
    names = list(EXERCISES)
    if manual:
        require(report.get('generator') == 'manual_golden' and report.get('agent_calls') == 0
                and report['status'] in ('passed','failed'),'Not a terminal manual golden cohort')
        require([r['case'] for r in report['cases']] == ['exercise-'+n for n in names]
                and not any(r['counterexample'] for r in report['cases']),'Wrong manual cohort')
    else:
        require(report['status'] in ('passed','completed_with_failures') and report.get('native_functions') is True,
                'Not a terminal live native-function batch')
        require(report.get('native_function_exercises') == dict(schema=1,
                catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest()),'Frozen native catalog mismatch')
        require([r['descriptor'] for r in report['cases']] == [descriptor(n) for n in names],'Frozen cohort changed')
        require(summarize(report['cases']) == report['summary'],'Batch metrics changed')
    passed = []; failed = []
    for name,row in zip(names,report['cases']):
        if manual:
            if not row.get('matched_expected'):
                failed.append(dict(id=name,layer=row.get('failure_layer'))); continue
            run = Path(row['run'])
        else:
            run = Path(row['evidence'])
            raw,_ = metadata(run/'report.json',RESULTS)
            require(case_metrics(raw) == row['metrics'],'Case metrics changed')
            if not row['metrics']['passed']:
                failed.append(dict(id=name,layer=row['metrics']['failure_layer'])); continue
            audit_case(row)
        passed.append(audit_run(run,name,live=not manual))
    matrix = [dict(feature=f,cases=[p['id'] for p in passed if f in p['exercise']['required']],
                   evidence_kind='trace_structural_only' if f == 'call.empty_return' else 'finite_output_influence')
              for f in sorted({f for spec in EXERCISES.values() for f in spec['required_features']})]
    values = sum(p['comparison']['compared_values'] for p in passed)
    return dict(schema=1,status='covered' if len(passed) == len(names) else 'incomplete',
        evidence_kind='manual_golden' if manual else 'live_agent',batch=str(root),batch_sha256=report_hash,
        planned=len(names),passed=len(passed),failed=failed,feature_matrix=matrix,cases=passed,
        compared_values=values,input_executions=4*len(passed),
        max_absolute_error=max((p['comparison']['max_absolute_error'] for p in passed),default=None),
        weighted_mae=sum(p['comparison']['mae']*p['comparison']['compared_values'] for p in passed)/values if values else None,
        freed_key_bytes=sum(p['freed_key_bytes'] for p in passed),new_api_calls=0,new_fhe_executions=0,
        live_agent_cases=0 if manual else len(passed),poseidon_gpu_validated=False,all_upstream_semantics_proven=False)


def main():
    from hecate_python_env import ROOT,VENV,enter_nix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--manual',action='store_true')
    parser.add_argument('--inside',action='store_true')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT,'Wrong workspace')
    if not args.inside:
        command = 'LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--inside',
             *(['--manual'] if args.manual else [])])
        return enter_nix(command,seconds=240)
    require(Path(sys.prefix) == VENV,'Pinned audit environment required')
    result = audit(args.batch,manual=args.manual)
    output = Path(tempfile.mkdtemp(prefix='native-exercise-audit-',dir=RESULTS))
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','feature_matrix')},indent=2))
    print('Audit evidence:',output)
    return 0 if result['status'] == 'covered' else 1


if __name__ == '__main__': raise SystemExit(main())
