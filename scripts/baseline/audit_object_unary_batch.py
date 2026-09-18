"""Offline audit of frozen v22 paid construction cases; never calls a provider."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tempfile

from audit_agent_lineage import RESULTS,metadata,require,audit_case
from audit_dsl_coverage import source_evidence
from object_unary_exercises import CATALOG,EXERCISES,descriptor
from run_agent_batch import summarize,case_metrics


def audit(batch):
    import numpy as np
    root=Path(batch).resolve()
    report,digest=metadata(root/'report.json',RESULTS)
    require(report['status'] in ('passed','completed_with_failures'),'Batch not terminal')
    require(report.get('object_unary') is True,'Not v22 batch')
    require(report.get('object_unary_exercises')==dict(schema=1,
        catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest()),'Frozen exercise catalog changed')
    require([row['descriptor'] for row in report['cases']]==[descriptor(n) for n in EXERCISES],
            'Frozen model cohort changed')
    require(summarize(report['cases'])==report['summary'],'Batch metrics changed')
    rows=[];failed=[];freed=0
    for row in report['cases']:
        name=row['descriptor']['id'].removeprefix('exercise-')
        run=Path(row['evidence'])
        raw,_=metadata(run/'report.json',RESULTS)
        require(row['metrics']==case_metrics(raw),'Case metrics changed')
        outcome,_=metadata(run/'key-cleanup-outcome.json',RESULTS)
        require(outcome['complete'],'Key cleanup incomplete')
        freed+=outcome['freed_bytes']
        if not row['metrics']['passed']:
            failed.append(dict(id=name,layer=row['metrics']['failure_layer']))
            continue
        numeric=audit_case(row)
        source=source_evidence(run,raw)
        attempt=next(a for a in reversed(raw['attempts']) if a['numerically_correct'])
        exercise=attempt['static_check']['construction_exercise']
        require(exercise['id']==name and set(exercise['influence'])==set(EXERCISES[name]['required_features']),
                'Missing executed/influential typed unary operation')
        # The frozen instructions require a single mixed operand and a fresh
        # result of the actual view operation, not unrelated demonstrations.
        if name=='ou-mixed':
            selected={tuple(exercise['influence'][f]['span']) for f in
                      ('operator.USub','cipher_cells','public_cells','boolean_cells')}
            require(len(selected)==1,'Mixed types must influence the same unary operation')
        if name=='ou-view':
            require(exercise['influence']['input_view']['span']==
                    exercise['influence']['fresh.USub']['span'],
                    'Freshness witness must refer to the negated view')
        with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
            x=arrays['inputs']
            expected=.5*(x*x if EXERCISES[name]['family']=='quadratic' else x)+x+.375
            require(np.allclose(expected,arrays['reference'],atol=1e-12,rtol=1e-12),
                    'Independent model reference mismatch')
        rows.append(dict(numeric,exercise=exercise,source=source,comparison=attempt['comparison']))
    matrix=[dict(feature=f,cases=[r['exercise']['id'] for r in rows if f in r['exercise']['required']])
            for f in sorted({f for spec in EXERCISES.values() for f in spec['required_features']})]
    count=sum(r['compared_values'] for r in rows)
    return dict(schema=1,status='covered' if len(rows)==len(EXERCISES) else 'incomplete',
        batch=str(root),batch_sha256=digest,planned=len(EXERCISES),passed=len(rows),failed=failed,
        feature_matrix=matrix,cases=rows,original_summary=report['summary'],
        compared_values=count,max_absolute_error=max((r['max_absolute_error'] for r in rows),default=None),
        weighted_mae=sum(r['comparison']['mae']*r['compared_values'] for r in rows)/count if count else None,
        freed_key_bytes=freed,new_api_calls=0,new_fhe_executions=0,
        poseidon_gpu_validated=False,all_upstream_semantics_proven=False)


def main():
    from hecate_python_env import ROOT,VENV,enter_nix
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path)
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT,'Wrong workspace')
    if not args.inside:
        command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--inside'])
        return enter_nix(command,seconds=450)
    result=audit(args.batch)
    output=Path(tempfile.mkdtemp(prefix='object-unary-agent-audit-',dir=RESULTS))
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','feature_matrix','original_summary')},indent=2))
    print('Audit evidence:',output)
    return 0 if result['status']=='covered' else 1


if __name__=='__main__':raise SystemExit(main())
