from pathlib import Path
import collections,hashlib,json,math,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';out=e/'budget25/input-prefix-v2';out.mkdir(exist_ok=True)
base=json.loads((e/'budget25/full18-input34/result.json').read_text())['command'];base=[x for x in base if not x.startswith('--native-target-scale-goals=') and not x.startswith('--native-target-bootstrap-schedule=') and not x.startswith('--native-input-level=')]
q=json.loads((root/'alignment-tests/operator-spec.json').read_text())['context']['rns_moduli'];goals=json.loads((e/'budget25/manual-cuts/manual-stem-goals.json').read_text());results=[]
for count in [31,32,33,34]:
 g=goals.copy();high=sum(math.log2(x) for x in q[count-3:count])
 for i in [35,39,95,99]:g[str(i)]=high
 for mode in ['fixed18','automatic']:
  case=out/f'q{count}-{mode}';case.mkdir(exist_ok=True);gf=case/'goals.json';gf.write_text(json.dumps(g,indent=2)+'\n');c=base+['--native-input-level='+str(count-1),'--native-target-scale-goals='+str(gf)];c[c.index('-o')+1]=str(case/'model.optimized.mlirbc')
  if mode=='fixed18':c+=['--native-target-bootstrap-schedule='+str(e/'budget25/manual-cuts/manual18.json')]
  r=dict(initial_q=count,mode=mode,command=c,compiler_sha256=(e/'binary.sha256').read_text().strip());t=time.monotonic();p=subprocess.run(c,capture_output=True,text=True,timeout=180);(case/'compile.log').write_text(p.stdout+p.stderr);r.update(returncode=p.returncode,seconds=time.monotonic()-t)
  if not p.returncode:
   path=next(case.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());r.update(counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest());v=subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);r.update(validation=v.stdout+v.stderr,validation_code=v.returncode);assert not v.returncode,v.stderr
  (case/'result.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);(out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print({k:v for k,v in r.items() if k!='command'},flush=True)
