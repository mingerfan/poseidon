from pathlib import Path
import collections,hashlib,json,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';out=e/'budget25/automatic/manual34-p2-50';out.mkdir(exist_ok=True)
g=json.loads((e/'budget25/manual-cuts/manual-stem-goals.json').read_text());stage=json.loads((e/'budget25/manual-cuts/stage-boundaries.json').read_text());assert len(stage)==38
for r in stage[1::2]:assert g[str(r['opid'])]==44;g[str(r['opid'])]=50
f=out/'goals.json';f.write_text(json.dumps(g,indent=2)+'\n');c=json.loads((e/'budget25/automatic/manual34/result.json').read_text())['command'];c=[x for x in c if not x.startswith('--native-target-scale-goals=')];c+=['--native-target-scale-goals='+str(f)];assert not any('bootstrap-schedule' in x for x in c);c[c.index('-o')+1]=str(out/'model.optimized.mlirbc');t=time.monotonic();p=subprocess.run(c,capture_output=True,text=True,timeout=180);(out/'compile.log').write_text(p.stdout+p.stderr);r=dict(command=c,compiler_sha256=(e/'binary.sha256').read_text().strip(),returncode=p.returncode,seconds=time.monotonic()-t)
if not p.returncode:
 path=next(out.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());r.update(counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest());v=subprocess.run([str(root/'validate-plan'),str(path),str(root/'alignment-tests/operator-spec.json')],capture_output=True,text=True);r.update(validation_code=v.returncode,validation=v.stdout+v.stderr);assert not v.returncode,v.stderr
(out/'result.json').write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='command'},flush=True)
