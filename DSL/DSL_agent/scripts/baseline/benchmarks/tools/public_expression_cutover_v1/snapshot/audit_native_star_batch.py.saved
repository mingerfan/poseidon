"""Recheck saved starred-construction evidence; no API or FHE execution.

Live mode is the default. Manual mode also requires the real wrong-reverse
counterexample, and never upgrades manual source to Agent evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, metadata, read, require, audit_case
from candidate_contract import canonical, strict_json, validate_candidate
from deepseek_provider import public_request
from native_star_exercises import CATALOG,EXERCISES,descriptor,verify_trace_coverage
from run_agent_batch import case_metrics,summarize


def audit_run(run,name,*,live,wrong=False):
    import numpy as np
    from seal_cpu_golden import compare
    run = Path(run).resolve()
    require(run.parent == RESULTS,'Case outside results')
    report,report_hash = metadata(run/'report.json',RESULTS)
    require(report['status'] == ('repair_budget_exhausted' if wrong else 'passed'),'Unexpected case status')
    require((report['llm_generation_validated'] is True and report['agent_calls'] > 0
             and report['provider'] == 'deepseek_api') if live else
            (report['llm_generation_validated'] is False and report['agent_calls'] == 0
             and report['provider'] == 'scripted_replay'),'Generation kind mismatch')
    from compiler_configuration import configuration
    require(report.get('compiler_configuration')==configuration('seal-cpu-eva-w45-v1') and report.get('waterline')==45,
            'Starred cohort requires explicit compiler45 identity')
    require(report['backend'] == 'upstream_SEAL_HEVM_CPU' and not report['poseidon_gpu_validated'],'Backend mismatch')
    require(report['tolerance'] == dict(atol=1e-5,rtol=1e-4),'Frozen tolerance changed')
    params = report['parameters']
    require(params['seal_version']=='4.0.0' and params['security_check']=='tc128' and params['parameters_set']
            and params['polynomial_degree']==32768 and params['modulus_bits']==[60]*14,'Security profile changed')
    require({'arrays.npz','weights.npz','model.json','request.json'} <= set(report['frozen_hashes']),'Missing frozen inputs')
    for name_,value in report['frozen_hashes'].items(): require(read(run/name_,run)[1]==value,'Frozen input changed')
    model,_ = metadata(run/'model.json',run)
    require(model==descriptor(name),'Model changed')
    req,_ = metadata(run/'request.json',run); public_request(req)
    require(req['model']==model and req['construction_exercise']['id']==name,'Request/model mismatch')
    require(req.get('compiler_configuration')==report['compiler_configuration'],'Request compiler configuration mismatch')
    attempt = report['attempts'][-1]
    folder = run/('attempt-%02d'%attempt['index']);out = folder/'output'
    payload,payload_hash = metadata(folder/'trace-payload.json',run)
    require(str((folder/'trace-payload.json').relative_to(run)) in report['frozen_hashes'],'Missing frozen trace payload')
    require(payload['request']==req,'Traced request changed')
    candidate = strict_json(read(folder/'response.txt',run)[0].decode())
    require(candidate==payload['candidate'],'Response differs from traced candidate')
    require(read(folder/'candidate.py',run)[0].decode()==candidate['hecate_source'],'Candidate sidecar mismatch')
    checked = validate_candidate(candidate,req)
    require(canonical(checked)==canonical(attempt['static_check']),'Static starred coverage changed')
    hashes = attempt['artifact_hashes']
    require({'native-star-events.json','native-call-events.json','native-function-plan.json',
             'trace-evidence.json','candidate_trace.mlir','lowered.ckks.mlir',
             'lowered._hecate_golden.hevm','_hecate_golden.cst'} <= set(hashes),'Missing actual compiler evidence')
    for filename,value in hashes.items(): require(read(out/filename,out)[1]==value,'Artifact changed')
    from seal_artifact_gate import inspect_artifacts
    from candidate_contract import request_rotations,request_input_names
    from compiler_configuration import verify_artifact_configuration,PROFILE_SHA256
    gate=inspect_artifacts(read(out/'lowered._hecate_golden.hevm',out)[0],read(out/'_hecate_golden.cst',out)[0],
        rotation_steps=request_rotations(req),expected_inputs=len(request_input_names(req)))
    require(gate==attempt['artifact_gate'],'Actual artifact gate mismatch')
    verify_artifact_configuration(req,gate,PROFILE_SHA256)
    plan,_ = metadata(out/'native-function-plan.json',out)
    require(canonical(plan)==canonical(checked['native_functions']),'Actual native plan mismatch')
    trace,_ = metadata(out/'trace-evidence.json',out)
    require(trace==attempt['trace'] and trace==dict(frontend='real_Hecate',candidate_python_executed=False,
            construction='validated_native_AST_to_Hecate_functions',request_id=req['request_id']),'Trace protocol mismatch')
    observed,_ = metadata(out/'native-star-events.json',out)
    coverage = verify_trace_coverage(checked['construction_exercise'],observed)
    require(coverage==attempt['native_star_coverage'],'Actual starred coverage mismatch')
    execution,_ = metadata(out/'execution.json',out)
    require(execution==attempt['execution'] and execution['encrypted_execution']
            and execution['input_batches']==4 and execution['encrypted_input_count']==1
            and not execution['bootstrap_executed'],'Actual FHE execution missing')
    with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
        inputs,reference = arrays['inputs'],arrays['reference']
        require(np.allclose(reference,inputs*.5+inputs+.375,atol=1e-15,rtol=0),'Independent reference mismatch')
    _,decrypted_hash = read(out/'decrypted.npy',out)
    comparison = compare(np.load(out/'decrypted.npy',allow_pickle=False),reference,1e-5,1e-4)
    require(comparison==attempt['comparison'] and comparison['passed']==(not wrong),'Numerical evidence mismatch')
    if wrong: require(attempt['failure_layer']=='numerical_comparison','Wrong reversed argument formula not rejected by numbers')
    cleanup,_ = metadata(run/'key-cleanup-outcome.json',run)
    require(cleanup['complete'] and not (run/'private-keys').exists(),'Key cleanup incomplete')
    return dict(id=name,evidence=str(run),report_sha256=report_hash,trace_payload_sha256=payload_hash,
                decrypted_sha256=decrypted_hash,exercise=checked['construction_exercise'],trace=coverage,
                comparison=comparison,live_agent=live,wrong=wrong,freed_key_bytes=cleanup['freed_bytes'])


def audit(batch,*,manual=False):
    root = Path(batch).resolve();report,report_hash = metadata(root/'report.json',RESULTS)
    names = list(EXERCISES);counterexample = None
    if manual:
        require(report['status']=='passed' and report.get('generator')=='manual_golden'
                and report.get('agent_calls')==0,'Not a completed manual starred cohort')
        require([r['case'] for r in report['cases']]==['exercise-'+n for n in names]+['exercise-ns-reverse']
                and [r['counterexample'] for r in report['cases']]==[False]*8+[True],'Wrong manual cohort')
        require(all(r['matched_expected'] for r in report['cases']),'Manual expectation failed')
        counterexample = audit_run(report['cases'][-1]['run'],'ns-reverse',live=False,wrong=True)
    else:
        require(report['status'] in ('passed','completed_with_failures') and report.get('native_starred') is True,
                'Not a terminal live starred cohort')
        require(report.get('native_star_exercises')==dict(schema=1,catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest()),
                'Starred catalog mismatch')
        require([r['descriptor'] for r in report['cases']]==[descriptor(n) for n in names],'Frozen cohort changed')
        require(summarize(report['cases'])==report['summary'],'Batch metrics changed')
    passed=[];failed=[]
    for name,row in zip(names,report['cases']):
        if not manual:
            raw,_ = metadata(Path(row['evidence'])/'report.json',RESULTS)
            require(case_metrics(raw)==row['metrics'],'Case metrics changed')
            if not row['metrics']['passed']:
                failed.append(dict(id=name,layer=row['metrics']['failure_layer']));continue
            audit_case(row)
        passed.append(audit_run(row['run'] if manual else row['evidence'],name,live=not manual))
    features = sorted({f for _,required,_ in EXERCISES.values() for f in required})
    matrix = [dict(feature=f,cases=[p['id'] for p in passed if f in p['exercise']['required']],
                   evidence_kind='trace_structural_only' if f=='star.empty' else 'finite_argument_influence') for f in features]
    values = sum(p['comparison']['compared_values'] for p in passed)
    return dict(schema=1,status='covered' if len(passed)==len(names) else 'incomplete',
        evidence_kind='manual_golden' if manual else 'live_agent',batch=str(root),batch_sha256=report_hash,
        planned=len(names),passed=len(passed),failed=failed,cases=passed,feature_matrix=matrix,counterexample=counterexample,
        input_executions=4*len(passed),compared_values=values,
        max_absolute_error=max((p['comparison']['max_absolute_error'] for p in passed),default=None),
        weighted_mae=sum(p['comparison']['mae']*p['comparison']['compared_values'] for p in passed)/values if values else None,
        freed_key_bytes=sum(p['freed_key_bytes'] for p in passed)+(counterexample['freed_key_bytes'] if counterexample else 0),
        live_agent_cases=0 if manual else len(passed),new_api_calls=0,new_fhe_executions=0,
        poseidon_gpu_validated=False,full_semantics_proven=False)


def main():
    from hecate_python_env import ROOT,VENV,enter_nix
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--manual',action='store_true')
    parser.add_argument('--inside',action='store_true');args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong workspace')
    if not args.inside:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--inside',
             *(['--manual'] if args.manual else [])])
        return enter_nix(command,seconds=240)
    require(Path(sys.prefix)==VENV,'Pinned audit environment required')
    result=audit(args.batch,manual=args.manual)
    output=Path(tempfile.mkdtemp(prefix='native-star-exercise-audit-',dir=RESULTS))
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','feature_matrix','counterexample')},indent=2))
    print('Audit evidence:',output)
    return 0 if result['status']=='covered' else 1


if __name__=='__main__': raise SystemExit(main())
