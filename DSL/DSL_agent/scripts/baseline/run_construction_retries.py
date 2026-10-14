"""Explicit, bounded fresh paid retries of selected construction exercises.

Keeps the original batch and every failed candidate. The user selects IDs;
no infinite retry, silent repair-budget increase or golden-answer fallback.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

from construction_exercises import EXERCISES,descriptor
from hecate_python_env import ROOT,WORK,VENV,enter_nix,digest
from run_agent_batch import case_metrics,save,summarize

SCRIPT=Path(__file__).resolve()


def inside(args):
    key=os.environ.pop('DEEPSEEK_API_KEY','')
    if not key or Path(sys.prefix)!=VENV or not os.environ.get('IN_NIX_SHELL'):
        raise ValueError('Pinned environment and credential required')
    os.umask(0o077)
    source=args.batch.resolve()/'report.json'
    if not source.is_relative_to(WORK/'results'):
        raise ValueError('Source batch outside results')
    previous=json.loads(source.read_text())
    if previous['status'] not in ('passed','completed_with_failures'):
        raise ValueError('Source batch still active')
    selected={r['descriptor']['id']:r for r in previous['cases']}
    for name in args.ids:
        if selected['exercise-'+name]['descriptor']!=descriptor(name):
            raise ValueError('Source model mismatch')
    root=Path(tempfile.mkdtemp(prefix='construction-retry-',dir=WORK/'results'))
    rows=[dict(descriptor=descriptor(name),status='pending') for name in args.ids]
    report=dict(schema=1,status='running',started_unix=time.time(),cases=rows,
                source_report=str(source),source_sha256=digest(source),
                runner_sha256=digest(SCRIPT),new_experiment=True,
                reason='Selected generation failures and coverage-reinspection gaps; original outcomes retained',
                config=dict(provider='deepseek',model='deepseek-flash',reasoning_effort='high',
                            max_tokens=384000,api_timeout=1200,provider_retries=3,
                            max_repairs=3,jobs=10,native_jobs=2,proxy=6478))
    def checkpoint():
        report['summary']=summarize(rows)
        report['elapsed_seconds']=time.time()-report['started_unix']
        save(root/'report.json',report)
    checkpoint()
    print('Construction retry evidence:',root,flush=True)
    def worker(index):
        row=rows[index];name=args.ids[index]
        directory=root/row['descriptor']['id'];directory.mkdir()
        save(directory/'model.json',row['descriptor'])
        command=['timeout','-k','5s','22000s',str(VENV/'bin/python'),
                 str(ROOT/'scripts/baseline/run_candidate.py'),'--inside','--deepseek',
                 '--case',str(directory/'model.json'),'--construction-exercise',name,
                 '--provider','deepseek','--model','deepseek-flash','--reasoning-effort','high',
                 '--max-tokens','384000','--api-timeout','1200','--provider-retries','3',
                 '--loopback-proxy-port','6478','--max-repairs','3']
        result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,DEEPSEEK_API_KEY=key),
                              capture_output=True,timeout=22010)
        output=result.stdout.decode('utf-8',errors='replace')
        (directory/'runner.log').write_text(output)
        paths=[line.removeprefix('Candidate evidence: ') for line in output.splitlines()
               if line.startswith('Candidate evidence: ')]
        if len(paths)!=1:raise ValueError('Missing candidate evidence')
        path=Path(paths[0])
        if path.parent!=WORK/'results' or not path.name.startswith('agent-deepseek-'):
            raise ValueError('Invalid candidate evidence')
        raw=json.loads((path/'report.json').read_text())
        metrics=case_metrics(raw)
        return index,dict(status='passed' if metrics['passed'] else 'failed',
                          metrics=metrics,evidence=str(path),exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures={pool.submit(worker,i):i for i in range(len(rows))}
        for future in as_completed(futures):
            index=futures[future]
            try:_,outcome=future.result()
            except Exception as error:
                outcome=dict(status='failed',diagnostic_type=type(error).__name__,
                             metrics=case_metrics(dict(status='infrastructure_failed',attempts=[])))
            rows[index].update(outcome);checkpoint()
            print(rows[index]['descriptor']['id'],outcome['status'],flush=True)
    report['status']='passed' if all(r['status']=='passed' for r in rows) else 'completed_with_failures'
    checkpoint();print(json.dumps(report['summary'],indent=2),flush=True)
    return 0 if report['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',choices=tuple(EXERCISES),required=True)
    parser.add_argument('--deepseek',action='store_true',required=True)
    parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT or len(args.ids)!=len(set(args.ids)) or len(args.ids)>10:
        raise ValueError('Wrong source directory or invalid bounded retry selection')
    if args.inside:return inside(args)
    from agent_credentials import load_api_key
    previous=os.environ.get('DEEPSEEK_API_KEY')
    os.environ['DEEPSEEK_API_KEY']=load_api_key(ROOT,provider='deepseek')
    try:
        command='LD_LIBRARY_PATH="'+chr(36)+'HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
            [str(VENV/'bin/python'),str(SCRIPT),'--inside','--deepseek',
             '--batch',str(args.batch.resolve()),'--ids',*args.ids])
        return enter_nix(command,seconds=22500,keep_env=('DEEPSEEK_API_KEY',))
    finally:
        if previous is None:os.environ.pop('DEEPSEEK_API_KEY',None)
        else:os.environ['DEEPSEEK_API_KEY']=previous


if __name__=='__main__':
    raise SystemExit(main())
