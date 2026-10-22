"""Offline audit of the fixed six-model concat paid cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
from audit_agent_lineage import RESULTS,metadata,require,audit_case
from audit_dsl_coverage import source_evidence
from custom_batch_manifest import load_manifest
from model_graph import evaluate_reference
from model_semantic_coverage import analyze_graph
from run_agent_batch import summarize,case_metrics

MANIFEST=Path(__file__).with_name('cases')/'concat-agent-6-manifest.json'

def audit(batch, manifest=MANIFEST):
    import numpy as np
    root=Path(batch).resolve()
    report,digest=metadata(root/'report.json',RESULTS)
    require(report['status'] in ('passed','completed_with_failures'),'Batch not terminal')
    models,_,manifest_hash=load_manifest(manifest)
    require(report['custom_manifest']['sha256']==manifest_hash,'Frozen concat cohort changed')
    require([r['descriptor'] for r in report['cases']]==[r['descriptor'] for r in models],'Model order changed')
    require(report['summary']==summarize(report['cases']),'Batch metrics changed')
    rows=[];failed=[];freed=0
    for row in report['cases']:
        run=Path(row['evidence']);raw,_=metadata(run/'report.json',RESULTS)
        require(row['metrics']==case_metrics(raw),'Case metrics changed')
        cleanup,_=metadata(run/'key-cleanup-outcome.json',RESULTS)
        require(cleanup['complete'],'Private keys not cleaned')
        freed+=cleanup['freed_bytes']
        if not row['metrics']['passed']:
            failed.append(dict(id=row['descriptor']['id'],layer=row['metrics']['failure_layer']))
            continue
        numeric=audit_case(row)
        source=source_evidence(run,raw)
        semantics=analyze_graph(row['descriptor'])
        require(semantics['operators'].get('concat')==1,
                'Missing output-reachable concat')
        attempt=next(a for a in reversed(raw['attempts']) if a.get('numerically_correct'))
        with np.load(run/'arrays.npz',allow_pickle=False) as arrays:
            from concat_evidence import reference_batch
            expected=reference_batch(row['descriptor'],arrays['inputs'])
            require(np.allclose(arrays['reference'],expected,atol=1e-12,rtol=1e-12),'Independent concat reference mismatch')
        rows.append(dict(numeric,source=source,model_semantics=semantics,comparison=attempt['comparison']))
    count=sum(r['compared_values'] for r in rows)
    return dict(schema=1,status='covered' if len(rows)==len(models) else 'incomplete',batch=str(root),batch_sha256=digest,
        manifest_sha256=manifest_hash,planned=len(models),passed=len(rows),failed=failed,cases=rows,
        summary=report['summary'],compared_values=count,
        max_absolute_error=max((r['max_absolute_error'] for r in rows),default=None),
        weighted_mae=sum(r['comparison']['mae']*r['compared_values'] for r in rows)/count if count else None,
        freed_key_bytes=freed,new_api_calls=0,new_fhe_executions=0,
        upstream_packing_helper_validated=False,poseidon_gpu_validated=False,all_semantics_verified=False)

def main():
    from hecate_python_env import ROOT,VENV,enter_nix
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path);p.add_argument('--inside',action='store_true')
    p.add_argument('--manifest',type=Path,default=MANIFEST)
    args=p.parse_args();require(Path.cwd().resolve()==ROOT,'Wrong workspace')
    if not args.inside:
        cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--manifest',str(args.manifest.resolve()),'--inside'])
        return enter_nix(cmd,seconds=450)
    result=audit(args.batch,args.manifest)
    out=Path(tempfile.mkdtemp(prefix='concat-agent-audit-',dir=RESULTS))
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','summary')},indent=2))
    print('Audit evidence:',out)
    return 0 if result['status']=='covered' else 1

if __name__=='__main__':raise SystemExit(main())
