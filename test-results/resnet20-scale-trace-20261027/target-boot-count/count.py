from pathlib import Path
import collections
import hashlib
import json
import resource
import shlex
import subprocess
import sys
import time

strategy=sys.argv[1]
assert strategy in ('threshold','target')
root=Path('/home/xuming/poseidon-manual-boot-20261026')
audit=root/'resnet20-optimized-20261027T055506Z/placement-audit'
experiment=audit/'target-scale-experiment'
out=experiment/'automatic-boot-count'/strategy
out.mkdir(parents=True,exist_ok=True)
generator=json.loads((experiment/'target-command.json').read_text())
generator[2]='resnet20'
generator[generator.index('--traced-mlir')+1]='/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc'
generator[generator.index('--output-dir')+1]=str(out)
if strategy=='threshold':
 del generator[generator.index('--scale-strategy'):generator.index('--scale-strategy')+2]
generator+=['--inline-payload-max-bytes','0','--dry-run']
command=shlex.split(subprocess.run(generator,check=True,capture_output=True,text=True).stdout.strip())
command.remove('--enable-debug-printer')
command[command.index('-o')+1]=str(out/'resnet20.optimized.mlirbc')
command+=['--emit-bytecode']
report=dict(strategy=strategy,placement=('fixed-old-optimized-schedule' if strategy=='target-fixed' else 'greedy'),input=generator[generator.index('--traced-mlir')+1],command=command,state='running')
report['compiler_sha256']=(experiment/'binary.sha256').read_text().strip()
report['operator_spec_sha256']=hashlib.sha256((root/'alignment-tests/operator-spec.json').read_bytes()).hexdigest()
path=out/'run.json'
path.write_text(json.dumps(report,indent=2)+'\n')
t=time.monotonic()
with (out/'compile.log').open('w') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
 report['pid']=process.pid
 path.write_text(json.dumps(report,indent=2)+'\n')
 print(strategy,'PID',process.pid,flush=True)
 code=process.wait()
report.update(state='passed' if code==0 else 'failed',returncode=code,seconds=time.monotonic()-t,peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
path.write_text(json.dumps(report,indent=2)+'\n')
if code:
 print((out/'compile.log').read_text()[-7000:],flush=True)
 raise SystemExit(code)
plan_path=next(out.glob('*.runtime-plan.json'))
plan=json.loads(plan_path.read_text())
desc={v['id']:v for v in plan['values']}
ops=plan['initialization']+plan['execution']+plan['finalization']
boots=[op for op in ops if op.get('op')=='boot']
for op in boots:
 before,after=desc[op['inputs'][0]],desc[op['output']]
 assert before['level']==5 and abs(before['scale_log2']-40)<1e-8,before
 assert after['level']==30 and after['scale_log2']==40,after
verify=subprocess.run([str(root/'validate-plan'),str(plan_path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True)
report.update(boots=len(boots),values=len(desc),counts=dict(collections.Counter(op.get('op',op['kind']) for op in ops)),plan_path=str(plan_path),plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),boot_contracts_passed=True,validation=dict(returncode=verify.returncode,output=verify.stdout+verify.stderr),boot_details=[dict(instruction=op,before=desc[op['inputs'][0]],after=desc[op['output']]) for op in boots])
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('command','boot_details')}),flush=True)
assert verify.returncode==0,verify.stdout+verify.stderr
