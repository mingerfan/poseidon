from pathlib import Path
from collections import Counter
import ast
import json
import math
import re
import subprocess
import sys
import time

root = Path('/home/xuming/poseidon-manual-boot-20261026')
audit = root/'resnet20-optimized-20261027T055506Z/placement-audit'
out = audit/'target-scale-experiment'
inline = '--inline' in sys.argv
production = '--production' in sys.argv
case = out/('one-relu-production' if production else 'one-relu-inline' if inline else 'one-relu')
spec_path = root/'alignment-tests/operator-spec.json' if production else audit/'one-relu-full-spec.json'
command = ['python3',str(root/'src/third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py'),'mlp','--operator-spec',str(spec_path),'--compiler-profile',str(root/'alignment-tests/compiler.json'),'--hecate-opt',str(out/'hecate-target'),'--traced-mlir',str(audit/'one-relu-tagged.mlir'),'--output-dir',str(case),'--device-counts','1','--plan-format','json','--boot-placement','greedy','--waterline','40']
if inline: command += ['--inline-payload-max-bytes','1048576']
start = time.monotonic()
run = subprocess.run(command,capture_output=True,text=True)
(out/(case.name+'.log')).write_text(run.stdout+run.stderr)
print('compile',run.returncode,round(time.monotonic()-start,3),flush=True)
if run.returncode:
    print((run.stdout+run.stderr)[-8000:]);raise SystemExit(run.returncode)
plan_path = next(case.glob('*.runtime-plan.json'))
plan = json.loads(plan_path.read_text())
spec = json.loads(spec_path.read_text())
module = ast.parse((audit/'chain-waste-probe.py').read_text())
exec(compile(ast.Module(body=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='extract'],type_ignores=[]),'trace-parser','exec'))
result = extract(plan,(case/'mlp.optimized.mlir').read_text(),spec['context']['rns_moduli'],spec['boot_profiles'][0]['output_level'])
# A target-planned stage can rescale AFTER its final addition.
desc = {v['id']:v for v in plan['values']}
users = {}
for op in plan['execution']:
    for arg in op.get('inputs',[]):users.setdefault(arg,[]).append(op)
previous = 0
for stage in result['stages']:
    current = stage['id']
    while True:
        following = [op for op in users.get(current,[]) if op.get('op')=='rescale']
        if not following:break
        assert len(following)==1
        current = following[0]['output']
    value = desc[current]
    total = spec['boot_profiles'][0]['output_level']-value['level']
    stage.update(id=current,q=value['level']+1,scale=value['scale_log2'],consumed_q=total-previous,total_consumed_q=total)
    previous = total
units = [m for m in result['multiplications'] if m['unit']]
assert all(abs(m['input_scales'][1])<1e-8 or m['input_scales'][1]>=40-1e-8 for m in units),units
result['unit_scales'] = [m['input_scales'][1] for m in units]
verify = subprocess.run([str(root/'validate-plan'),str(plan_path),str(spec_path)],capture_output=True,text=True)
assert verify.returncode==0,verify.stdout+verify.stderr
result['cpp_validation'] = verify.stdout.strip()
(out/(case.name+'-summary.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='multiplications'},ensure_ascii=False),flush=True)
