"""Replay two original valid Agent candidates after the Plain binding fix; no API."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK, VENV, enter_nix, digest
from seal_artifact_gate import require


def inside(batch):
    from seal_cpu_golden import dump
    os.umask(0o077)
    prior=json.loads((batch/'report.json').read_text())
    require(prior['status'] in ('passed','completed_with_failures'),'Live batch must be terminal')
    chosen=('chunked-linear-2x4','chunked-residual-3x4')
    root=Path(tempfile.mkdtemp(prefix='chunked-trace-replays-',dir=WORK/'results'))
    print('Trace replay evidence:',root,flush=True)
    report=dict(status='running',generator='saved_agent_candidate_replay',agent_calls=0,
        source_batch=str(batch),source_batch_sha256=digest(batch/'report.json'),cases=[])
    for name in chosen:
        row=next(r for r in prior['cases'] if r['descriptor']['id']==name)
        old=Path(row['evidence']).resolve()
        require(old.parent==(WORK/'results').resolve(),'Invalid original candidate path')
        child=json.loads((old/'report.json').read_text())
        require(child['attempts'][0]['failure_layer']=='dsl_trace','Not the observed original trace failure')
        original=old/'attempt-00/response.txt'
        folder=root/name;folder.mkdir()
        dump(folder/'responses.json',[original.read_text()])
        command=[str(VENV/'bin/python'),str(ROOT/'scripts/baseline/run_candidate.py'),'--inside',
            '--case',str(old/'model.json'),'--replay',str(folder/'responses.json'),'--max-repairs','0',
            '--compiler-configuration','seal-cpu-eva-w45-v1']
        item=dict(id=name,original_run=str(old),original_response_sha256=digest(original),command=command)
        report['cases'].append(item)
        try:
            with (folder/'run.log').open('w') as stream:
                code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=360).returncode
            paths=re.findall(r'^Candidate evidence: (.+)$',(folder/'run.log').read_text(),re.M)
            require(len(paths)==1,'Missing replay evidence')
            run=Path(paths[0]).resolve();require(run.parent==(WORK/'results').resolve(),'Invalid replay path')
            result=json.loads((run/'report.json').read_text())
            item.update(run=str(run),report_sha256=digest(run/'report.json'),exit_code=code,status=result['status'])
            require(json.loads((run/'request.json').read_text())==json.loads((old/'request.json').read_text()),
                    'Replay changed the original request')
            require(digest(run/'attempt-00/response.txt')==item['original_response_sha256'],'Replay altered candidate')
            require(code==0 and result['status']=='passed' and result['agent_calls']==0 and
                    not result['llm_generation_validated'],'Expected successful offline replay, not new Agent')
            item['comparison']=result['attempts'][0]['comparison']
            cleanup=json.loads((run/'key-cleanup-outcome.json').read_text())
            require(cleanup['complete'] and not (run/'private-keys').exists(),'Cleanup incomplete')
            item['freed_key_bytes']=cleanup['freed_bytes']
        except Exception as error:
            item.update(status='failed',diagnostic=str(error))
        dump(root/'report.json',report)
        print(name,item['status'],flush=True)
    report['status']='passed' if all(r['status']=='passed' for r in report['cases']) else 'failed'
    dump(root/'report.json',report)
    return 0 if report['status']=='passed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch',type=Path);parser.add_argument('--inside',action='store_true')
    args=parser.parse_args()
    require(Path.cwd().resolve()==ROOT and args.batch.resolve().parent==(WORK/'results').resolve(),'Invalid source or batch')
    if args.inside:
        require(Path(sys.prefix)==VENV,'Pinned environment required')
        return inside(args.batch)
    cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 '+shlex.join(
        [str(VENV/'bin/python'),str(Path(__file__).resolve()),str(args.batch.resolve()),'--inside'])
    return enter_nix(cmd,seconds=800)


if __name__=='__main__':raise SystemExit(main())
