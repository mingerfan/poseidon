from pathlib import Path
import json,subprocess,sys,time,hashlib,collections
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';m=e/'budget25/manual-cuts';name=sys.argv[1]
base=json.loads((e/'budget25/full18-input34/result.json').read_text())['command']
c=[x for x in base if not x.startswith('--native-target-bootstrap-schedule=') and not x.startswith('--native-target-scale-goals=') and not x.startswith('--native-input-level=')]
if name=='manual34':c+=['--native-target-scale-goals='+str(m/'manual-stem-goals.json'),'--native-input-level=33']
elif name=='manual31':c+=['--native-target-scale-goals='+str(m/'manual-conv-goals.json')]
elif name=='stem-out':
 c[c.index('/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc')]=str(e/'budget25/stem-out/stem-out.mlir');c+=['--native-target-scale-goals='+str(e/'budget25/stem-out/goals.json')]
else:assert name=='default31'
assert not any('bootstrap-schedule' in x for x in c)
out=e/'budget25/automatic'/name;out.mkdir(parents=True,exist_ok=True);c[c.index('-o')+1]=str(out/'resnet20.optimized.mlirbc');t=time.monotonic()
r=dict(name=name,command=c,fixed_boot_schedule=False,compiler_sha256=(e/'binary.sha256').read_text().strip())
with (out/'compile.log').open('w') as log:
 process=subprocess.Popen(c,stdout=log,stderr=subprocess.STDOUT);r['pid']=process.pid;(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(name,process.pid,flush=True)
 code=process.wait(timeout=180)
r.update(returncode=code,seconds=time.monotonic()-t)
if code==0:
 path=next(out.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());d={v['id']:v for v in plan['values']};boots=[o for o in plan['execution'] if o.get('op')=='boot'];r.update(boots=len(boots),input=d[plan['external_inputs'][0]],output=d[plan['final_outputs'][0]],counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 assert all(d[o['inputs'][0]]['level']==5 and abs(d[o['inputs'][0]]['scale_log2']-40)<1e-8 and d[o['output']]['level']==30 for o in boots)
 v=subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);r.update(validation=v.stdout+v.stderr,validation_code=v.returncode)
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='command'}),flush=True)
if code:print((out/'compile.log').read_text()[-2400:],flush=True)
