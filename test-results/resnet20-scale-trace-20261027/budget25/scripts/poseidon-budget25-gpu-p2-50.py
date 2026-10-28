from pathlib import Path
import json,os,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');a=root/'resnet20-optimized-20261027T055506Z/placement-audit';e=a/'target-scale-experiment';p=e/'budget25/relu-grid/p1-40.1-p2-50';report=p/'gpu-report.json'
while Path('/proc/4063580').exists():time.sleep(5)
c=[str(a/'variable-chain32/gpu/relu_gpu'),str(next(p.glob('*.runtime-plan.json'))),str(root/'alignment-tests/operator-spec.json'),str(report)]
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='2',POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50=str(root/'src/third_party/ckks-runtime/tests/fixtures/resnet20-manual-q50.json'))
t=time.monotonic()
with (p/'gpu-run.log').open('w') as log:r=subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=300)
(p/'gpu-command.json').write_text(json.dumps(dict(command=c,seconds=time.monotonic()-t,returncode=r.returncode),indent=2)+'\n');print('GPU P2=50',r.returncode,flush=True)
if report.exists():d=json.loads(report.read_text());print({k:v for k,v in d.items() if not isinstance(v,list)},flush=True)
else:print((p/'gpu-run.log').read_text()[-2500:])
