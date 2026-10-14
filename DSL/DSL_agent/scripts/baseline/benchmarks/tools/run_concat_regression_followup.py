"""Fixed precision controls and remaining helper regressions; keep every failure."""
import argparse,json,subprocess,time,hashlib
from pathlib import Path
from hecate_python_env import ROOT,WORK,VENV
from benchmark_graph import require,digest
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate
from upstream_candidate_cases import cases,polynomial
from upstream_bn_candidate_cases import bn
from upstream_candidate_helpers import PROFILE,BN_PROFILE,CONCAT_PROFILE
from benchmark_suite import Builder
from compiler_configuration import configuration,PROFILE_SHA256
from unified_graph_contract import prepare

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
require(not out.exists() and out.resolve().is_relative_to(WORK/'results'),'Preserve evidence');out.mkdir()
sources=runtime_sources();start=time.monotonic()
plan=[dict(id='public_'+waterline+'_'+str(i),kind='precision_control',config='seal-cpu-eva-'+waterline+'-v1') for waterline in ('w40','w45') for i in range(2)]
base=cases()[0];b=Builder([(1,2)]);z=bn(b,polynomial(b,'input0'));g=b.finish(z)
for name,model,source,profile in [('silu_v1',base['model'],base['source'],PROFILE),('silu_v2',base['model'],base['source'],BN_PROFILE),('silu_v3',base['model'],base['source'],CONCAT_PROFILE),('silu_then_bn_v3',g,'@hc.func("c,c")\ndef golden(x,zero_ct):\n    return HE_BN0(HE_SiLU(x))\n',CONCAT_PROFILE)]:
    plan.append(dict(id=name,kind='remaining_helper',config='seal-cpu-eva-w40-v1',model=model,source=source,profile=profile))
dump(out/'plan.json',dict(source_sha256=digest(sources),sources=sources,tasks=plan,agent_calls=0,max_wall_seconds=900,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
rows=[]
for task in plan:
    require(runtime_sources()==sources and time.monotonic()-start<900,'Source/time gate')
    folder=out/task['id'];folder.mkdir()
    command=[str(VENV/'bin/python'),'-B','scripts/baseline/run_candidate.py','--inside','--max-repairs','0','--compiler-configuration',task['config']]
    if task['kind']=='precision_control':
        command+=['--case','scripts/baseline/cases/unified-two-input-two-output.json','--self-test','--unified-profile','public-v1']
    else:
        dump(folder/'model.json',task['model']);r=prepare(task['model'],PROFILE_SHA256,configuration(task['config']),helper_profile=task['profile'])
        dump(folder/'responses.json',[json.dumps(dict(schema=1,request_id=r['request_id'],hecate_source=task['source']))])
        command+=['--case',str(folder/'model.json'),'--replay',str(folder/'responses.json'),'--unified-helpers',task['profile']]
    with (folder/'run.log').open('w') as log:process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=min(300,900-(time.monotonic()-start)))
    paths=[x.split('Candidate evidence: ',1)[1] for x in (folder/'run.log').read_text().splitlines() if x.startswith('Candidate evidence: ')]
    require(paths,'Runner did not retain evidence');evidence=Path(paths[-1]);report=json.loads((evidence/'report.json').read_text())
    row=dict(id=task['id'],kind=task['kind'],configuration=task['config'],status=report['status'],exit_code=process.returncode,evidence=str(evidence))
    row['attempts']=report['attempts']
    if report['status']=='passed':row['audit']=verify_candidate(evidence)
    rows.append(row);dump(out/'progress.json',dict(rows=rows,seconds=time.monotonic()-start));print(task['id'],row['status'],flush=True)
require(runtime_sources()==sources,'Source changed')
dump(out/'report.json',dict(rows=rows,source_sha256=digest(sources),planned=len(plan),passed=sum(r['status']=='passed' for r in rows),failed=sum(r['status']!='passed' for r in rows),skipped=0,seconds=time.monotonic()-start,agent_calls=0))
