from pathlib import Path
import json
import os
import subprocess
import sys
import time

root = Path('/home/xuming/poseidon-manual-boot-20261026')
audit = root/'resnet20-optimized-20261027T055506Z/placement-audit'
out = audit/'variable-chain32'
name = sys.argv[1]
case = out/name
gpu = out/'gpu'
spec = json.loads((case/'operator-spec.json').read_text())
assert max(q.bit_length() for q in spec['context']['rns_moduli'])<=32
fixture = json.loads((root/'src/third_party/ckks-runtime/tests/fixtures/resnet20-manual-q50.json').read_text())
fixture['rns_moduli'] = spec['context']['rns_moduli']
assert not set(fixture['rns_moduli']) & set(fixture['p_moduli'])
(gpu/(name+'-fixture.json')).write_text(json.dumps(fixture)+'\n')
command = ['python3',str(root/'src/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'),'mlp','--operator-spec',str(case/'operator-spec.json'),'--compiler-profile',str(root/'alignment-tests/compiler.json'),'--hecate-opt',str(root/'hecate-opt'),'--traced-mlir',str(audit/'one-relu-tagged.mlir'),'--output-dir',str(gpu/(name+'-artifacts')),'--device-counts','1','--plan-format','json','--boot-placement','greedy','--waterline','40','--inline-payload-max-bytes','1048576']
compile_run = subprocess.run(command,capture_output=True,text=True)
(gpu/(name+'-compile.log')).write_text(compile_run.stdout+compile_run.stderr)
assert compile_run.returncode==0,compile_run.stderr
plan = next((gpu/(name+'-artifacts')).glob('*.runtime-plan.json'))
text = json.loads(plan.read_text())
assert not any(op.get('op')=='boot' for op in text['execution'])
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '2'
env['POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50'] = str(gpu/(name+'-fixture.json'))
env['OMP_NUM_THREADS'] = '8'
start = time.monotonic()
run = subprocess.run([str(gpu/'relu_gpu'),str(plan),str(case/'operator-spec.json'),str(gpu/(name+'-report.json'))],capture_output=True,text=True,env=env)
(gpu/(name+'-run.log')).write_text(run.stdout+run.stderr)
result = dict(name=name,exit_code=run.returncode,seconds=time.monotonic()-start,stdout=run.stdout,stderr=run.stderr)
if (gpu/(name+'-report.json')).exists():
    report = json.loads((gpu/(name+'-report.json')).read_text())
    result['report'] = {k:v for k,v in report.items() if k not in ('input','expected','actual')}
    result['samples'] = {k:report[k][:10] for k in ('input','expected','actual')}
(gpu/(name+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False),flush=True)
