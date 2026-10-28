from pathlib import Path
import json,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026')
audit=root/'resnet20-optimized-20261027T055506Z/placement-audit'
e=audit/'target-scale-experiment'
out=e/'budget25/manual-cuts';out.mkdir(parents=True,exist_ok=True)
ops=json.loads((audit/'earth-ops.json').read_text())['ops']
def peel(name):
 while ops[name]['kind'] in ('rescale','upscale','modswitch','bootstrap'):
  name=ops[name]['args'][0]
 return ops[name]
tails=[v for v in ops.values() if v['kind']=='mul' and v.get('source','').endswith('ResNet20.py') and v.get('line')==111]
assert len(tails)==19
inputs=[peel(v['args'][0]) for v in tails]
assert all(v['kind']=='add' for v in inputs)
(out/'relu-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
base=json.loads((e/'budget25/full/target/run.json').read_text())['command']
for name,items in [('manual18',inputs[1:]),('relu19',inputs)]:
 schedule=[[v['opid'],v['opid']] for v in items]
 path=out/(name+'.json');path.write_text(json.dumps(schedule)+'\n')
 target=out/name;target.mkdir(exist_ok=True)
 command=base[:];command[command.index('-o')+1]=str(target/'resnet20.optimized.mlirbc')
 command+=['--native-target-bootstrap-schedule='+str(path)]
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:
  code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120).returncode
 r=dict(name=name,boots=len(items),returncode=code,seconds=time.monotonic()-start,command=command,schedule=schedule)
 if not code:
  plan_path=next(target.glob('*.runtime-plan.json'))
  plan=json.loads(plan_path.read_text());desc={v['id']:v for v in plan['values']};boots=[o for o in plan['execution'] if o.get('op')=='boot']
  assert len(boots)==len(items)
  for boot in boots:
   assert desc[boot['inputs'][0]]['level']==5 and abs(desc[boot['inputs'][0]]['scale_log2']-40)<1e-8
  v=subprocess.run([str(root/'validate-plan'),str(plan_path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True)
  r.update(validation=v.stdout+v.stderr,validation_code=v.returncode,final=desc[plan['final_outputs'][0]])
 (out/(name+'-result.json')).write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({k:v for k,v in r.items() if k not in ('command','schedule')}),flush=True)
 if code:print((out/(name+'.log')).read_text()[-1600:],flush=True)
