from pathlib import Path
import hashlib
import json
import re
import subprocess
import time

root=Path('/home/xuming/poseidon-manual-boot-20261026')
base=root/'resnet20-optimized-20261027T055506Z'
experiment=base/'placement-audit/target-scale-experiment'
out=experiment/'automatic-boot-count/fixed-optimized'
out.mkdir(parents=True,exist_ok=True)
with (base/'performance/single-state/full-resnet20/resnet20.optimized.earth.mlir').open() as f:
 header=f.readline()+f.readline()
entries=re.search(r'btp_schedule = \[(.*?)\]',header)[1]
schedule=[list(map(int,item.split(','))) for item in re.findall(r'array<i64: ([^>]*)>',entries)]
assert len(schedule)==27
(out/'original-schedule.json').write_text(json.dumps(schedule)+'\n')
source=json.loads((experiment/'automatic-boot-count/target/run.json').read_text())['command']
command=source[:]
command.remove('--dacapo')
command[command.index('-o')+1]='/dev/null'
profile='poseidon-resnet20-manual-q50-scale40'
spec=root/'alignment-tests/operator-spec.json'
physical=f'builtin.module(dacapo-precompile,func.func(materialize-ckks-physical-levels{{operator-spec={spec} boot-profile={profile} levels-per-logical-level=1}}))'
schedule_path=out/'trial-schedule.json'
command += ['--native-target-bootstrap-schedule='+str(schedule_path),'--pass-pipeline='+physical]
(out/'trial-command.json').write_text(json.dumps(command,indent=2)+'\n')

def trial(entries,label):
 schedule_path.write_text(json.dumps(entries)+'\n')
 start=time.monotonic()
 run=subprocess.run(command,capture_output=True,text=True,timeout=120)
 (out/(label+'.log')).write_text(run.stdout+run.stderr)
 result=dict(label=label,boots=len(entries),passed=run.returncode==0,returncode=run.returncode,seconds=time.monotonic()-start,schedule=entries)
 (out/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='schedule'}),flush=True)
 if run.returncode:
  print(run.stderr[-2000:],flush=True)
 return result

result=trial(schedule,'original-27')
if not result['passed']:
 raise SystemExit(1)
retained=schedule[:]
trials=[result]
for index,entry in enumerate(schedule):
 candidate=[item for item in retained if item!=entry]
 result=trial(candidate,f'delete-{index:02}-{entry[0]}')
 trials.append(result)
 if result['passed']:
  retained=candidate
 (out/'deletion-results.json').write_text(json.dumps(dict(original_boots=27,retained_boots=len(retained),schedule=retained,trials=trials),indent=2)+'\n')
(out/'retained-schedule.json').write_text(json.dumps(retained)+'\n')
print('Retained',len(retained),flush=True)
