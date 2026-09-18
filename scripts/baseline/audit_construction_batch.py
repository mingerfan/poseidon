"""Re-audit targeted paid Agent evidence and enumerate each required construct.

No credentials, API calls, candidate Python execution or new FHE executions.
Additional standalone retries remain separate runs; never overwrite old failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS, audit_case, metadata, require
from audit_dsl_coverage import source_evidence
from construction_exercises import EXERCISES, descriptor, validate_exercise_request
from run_agent_batch import case_metrics, summarize


def audit(batch, additional=()):
    root=Path(batch).resolve()
    report,digest=metadata(root/'report.json',RESULTS)
    require(report['status'] in ('passed','completed_with_failures'),'Batch must be terminal')
    require(report.get('construction_exercises',{}).get('schema')==1,'Not a targeted batch')
    require([r['descriptor'] for r in report['cases']]==[descriptor(n) for n in EXERCISES],
            'Target cohort changed')
    require(summarize(report['cases'])==report['summary'],'Batch metrics changed')
    all_rows=list(report['cases'])
    sources=[dict(path=str(root/'report.json'),sha256=digest)]
    for path in additional:
        raw,hash_=metadata(Path(path),RESULTS)
        require(raw.get('provider')=='deepseek_api','Retry is not real API evidence')
        model,_=metadata(Path(path).parent/'model.json',RESULTS)
        row=dict(descriptor=model,evidence=str(Path(path).parent),metrics=case_metrics(raw),
                 status='passed' if case_metrics(raw)['passed'] else 'failed')
        all_rows.append(row)
        sources.append(dict(path=str(path),sha256=hash_))
    successful={};seen=set();calls=0;usage=dict(prompt_tokens=0,completion_tokens=0,total_tokens=0)
    freed=0;failures=[]
    for row in all_rows:
        require(row['descriptor']['id'].startswith('exercise-'),'Unexpected model ID')
        name=row['descriptor']['id'][9:]
        require(row['descriptor']==descriptor(name),'Exercise model changed')
        if 'evidence' not in row:
            failures.append(dict(case=name,layer='batch_worker'));continue
        run=Path(row['evidence'])
        require(str(run) not in seen,'Duplicate evidence run')
        seen.add(str(run))
        raw,_=metadata(run/'report.json',RESULTS)
        require(case_metrics(raw)==row['metrics'],'Per-case metrics changed')
        calls+=raw['agent_calls']
        for key in usage:usage[key]+=row['metrics']['usage'][key]
        request,_=metadata(run/'request.json',RESULTS)
        validate_exercise_request(request)
        for file,expected in raw['frozen_hashes'].items():
            require(metadata(run/file,run)[1]==expected if file.endswith('.json') else
                    hashlib.sha256((run/file).read_bytes()).hexdigest()==expected,'Frozen evidence changed')
        for attempt in raw['attempts']:
            if attempt['status']!='passed':
                failures.append(dict(case=name,index=attempt['index'],
                                     layer=attempt.get('failure_layer'),diagnostic=attempt.get('diagnostic')))
        cleanup=run/'key-cleanup-outcome.json'
        if cleanup.exists():
            outcome,_=metadata(cleanup,RESULTS)
            require(outcome['complete'],'Incomplete key cleanup')
            freed+=outcome['freed_bytes']
        if row['status']!='passed':continue
        checked=audit_case(row)
        try:
            source=source_evidence(run,raw)
        except ValueError as error:
            failures.append(dict(case=name,layer='coverage_reaudit',diagnostic=str(error)))
            continue
        attempt=next(a for a in reversed(raw['attempts']) if a.get('numerically_correct'))
        exercise=attempt['static_check']['construction_exercise']
        require(exercise['id']==name,'Exercise witness mismatch')
        successful.setdefault(name,dict(checked,exercise=exercise,source=source,
                                       first_passed=row['metrics']['first_passed'],
                                       comparison=attempt['comparison']))
    required=sorted({f for e in EXERCISES.values() for f in e['required_features']})
    matrix=[]
    for feature in required:
        rows=[name for name,row in successful.items() if feature in row['exercise']['required']]
        sensitive=[name for name in rows if feature in successful[name]['exercise']['expression_influence']]
        matrix.append(dict(feature=feature,passing_cases=rows,
                           perturbed_output_sensitive_cases=sensitive,
                           evidence_level='executed_and_expression_sensitive' if sensitive else
                                          'executed_structure_source_review_required' if rows else 'missing'))
    missing_cases=[n for n in EXERCISES if n not in successful]
    missing_features=[r['feature'] for r in matrix if not r['passing_cases']]
    count=sum(r['compared_values'] for r in successful.values())
    return dict(schema=1,status='covered' if not missing_cases and not missing_features else 'incomplete',
                new_api_calls=0,new_fhe_executions=0,api_calls_in_sources=calls,usage_in_sources=usage,
                planned_cases=len(EXERCISES),passing_cases=len(successful),missing_cases=missing_cases,
                feature_count=len(required),observed_feature_count=len(required)-len(missing_features),
                missing_features=missing_features,feature_matrix=matrix,cases=list(successful.values()),
                sources=sources,original_batch_summary=report['summary'],attempt_failures=failures,
                compared_values=count,
                weighted_mae=sum(r['comparison']['mae']*r['compared_values'] for r in successful.values())/count if count else None,
                max_absolute_error=max((r['max_absolute_error'] for r in successful.values()),default=None),
                freed_key_bytes=freed,poseidon_gpu_validated=False,all_upstream_semantics_proven=False,
                interpretation='Targeted implementation-form coverage, not free-synthesis/generalization rate. '
                    'Expression perturbations are finite probes; storage/argument/control cases need source review. '
                    'Not all variants of a construct or all possible programs.')


def main():
    from hecate_python_env import ROOT,VENV,enter_nix
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--additional-run',type=Path,action='append',default=[])
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong source directory')
    if not args.inside:
        command='LD_LIBRARY_PATH="'+chr(36)+'HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--inside',
             *[item for p in args.additional_run for item in ('--additional-run',str(p.resolve()))]])
        return enter_nix(command,seconds=600)
    result=audit(args.batch,args.additional_run)
    output=Path(tempfile.mkdtemp(prefix='construction-agent-audit-',dir=RESULTS))
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
                     ('feature_matrix','cases','attempt_failures','sources','original_batch_summary')},indent=2))
    print('Construction audit:',output)
    return 0 if result['status']=='covered' else 1


if __name__=='__main__':
    raise SystemExit(main())
