"""One bounded real case, unchanged-source resume, and stale-evidence rejection."""
import json,subprocess,hashlib,copy
from pathlib import Path
from hecate_python_env import VENV,WORK,ROOT
from benchmark_graph import require
from benchmark_runner import dump
from audit_unified_candidate import verify_candidate

out=WORK/'results/upstream-helper-r17-resume'
require(not out.exists(),'Preserve resume acceptance');out.mkdir()
case=out/'case'
base=[str(VENV/'bin/python'),'-B',str(ROOT/'scripts/benchmark.py'),'helper-directed','--inside','--execute','--limit','1']
def run(path,resume=False):
    command=base+['--output',str(path)]+(['--resume'] if resume else [])
    with (out/(path.name+('-resume' if resume else '-first')+'.log')).open('w') as log:
        return subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
require(run(case)==0,'Initial resume fixture failed')
first=json.loads((case/'report.json').read_text());folder=Path(first['rows'][0]['evidence'])
def hashes():return {str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file()}
before=hashes();require(run(case,True)==0,'Resume failed');after=hashes()
require(before==after,'Resume reran or changed execution evidence')
second=json.loads((case/'report.json').read_text());require(first['records']==second['records'],'Resume record changed')
checks=[]
for change in ['source','runner','task','candidate']:
    negative=out/('reject-'+change);negative.mkdir()
    plan=json.loads((case/'plan.json').read_text());progress=json.loads((case/'progress.json').read_text())
    if change=='source':plan['source_sha256']='0'*64
    elif change=='runner':plan['runner_sha256']='0'*64
    elif change=='task':progress['rows'][0]['task_sha256']='0'*64
    else:
        wrong=negative/'wrong-evidence';wrong.mkdir()
        request=json.loads((folder/'request.json').read_text());request['request_id']='0'*64
        dump(wrong/'request.json',request);progress['rows'][0]['evidence']=str(wrong)
    dump(negative/'plan.json',plan);dump(negative/'progress.json',progress)
    require(run(negative,True)!=0,'Stale resume was accepted: '+change)
    text=(out/(negative.name+'-resume.log')).read_text()
    expected='Resume task/source/environment changed' if change in ('source','runner') else 'Resume task hash' if change=='task' else 'Resume candidate/model identity'
    require(expected in text,'Unexpected rejection layer: '+change)
    checks.append(dict(change=change,status='rejected_as_expected',reason=expected))
require(hashes()==before,'Negative tests changed real evidence')
dump(out/'report.json',dict(schema=1,initial=first,after_resume=second,unchanged_execution_files=True,
                           negative_checks=checks,additional_encrypted_executions_on_resume=0,agent_calls=0,
                           checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print('resume: one FHE pass, unchanged evidence, four expected rejections',flush=True)
