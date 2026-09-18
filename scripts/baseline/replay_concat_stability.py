"""Two fresh-key offline replays of a frozen paid concat-vector response."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from audit_agent_lineage import RESULTS,metadata,read,require
from hecate_python_env import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    args=p.parse_args();require(Path.cwd().resolve()==ROOT,'Wrong workspace')
    run=args.run.resolve();old,digest=metadata(run/'report.json',RESULTS)
    model,_=metadata(run/'model.json',RESULTS)
    require(old['status']=='passed' and old['llm_generation_validated'],'Expected original passed Agent evidence')
    require(model['id']=='concat-vector','Only frozen concat-vector stability diagnostic')
    case=ROOT/'scripts/baseline/cases/concat-vector.json'
    require(case.read_bytes()==(run/'model.json').read_bytes() or json.loads(case.read_text())==model,'Model changed')
    require(len(old['attempts'])==1,'Expected first-attempt candidate')
    raw,response_hash=read(run/'attempt-00/response.txt',run)
    out=Path(tempfile.mkdtemp(prefix='concat-stability-',dir=RESULTS))
    response=out/'replay.json';response.write_text(json.dumps([raw.decode('utf-8')])+'\n')
    result=dict(status='running',original_run=str(run),original_report_sha256=digest,
        response_sha256=response_hash,new_api_calls=0,cases=[],not_new_agent_generation=True)
    print('Stability evidence:',out,flush=True)
    for i in range(2):
        cmd=['timeout','-k','5s','320s',sys.executable,str(ROOT/'scripts/baseline/run_candidate.py'),
             '--case',str(case),'--replay',str(response),'--object-unary','--max-repairs','0']
        with (out/f'replay-{i}.log').open('w') as f:
            code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=330).returncode
        matches=re.findall(r'^Candidate evidence: (.+)$',(out/f'replay-{i}.log').read_text(),re.MULTILINE)
        require(len(matches)==1,'Missing replay directory')
        replay=Path(matches[0]);r,_=metadata(replay/'report.json',RESULTS)
        require(r['agent_calls']==0 and not r['llm_generation_validated'],'Unexpected paid replay')
        require(read(replay/'request.json',replay)[1]==read(run/'request.json',run)[1],'Request changed')
        attempt=r['attempts'][0]
        require(read(replay/'attempt-00/response.txt',replay)[1]==response_hash,'Response changed')
        cleanup,_=metadata(replay/'key-cleanup-outcome.json',RESULTS)
        result['cases'].append(dict(run=str(replay),exit_code=code,status=r['status'],
            failure_layer=attempt.get('failure_layer'),comparison=attempt.get('comparison'),
            encrypted_execution=attempt.get('execution',{}).get('encrypted_execution',False),
            key_cleanup_complete=cleanup['complete'],freed_key_bytes=cleanup['freed_bytes']))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(i+1,r['status'],attempt.get('comparison',{}).get('max_absolute_error'),flush=True)
    result['status']='passed' if all(c['status']=='passed' for c in result['cases']) else 'failed'
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
