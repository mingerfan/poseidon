from pathlib import Path
import collections,hashlib,itertools,json,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';out=e/'budget25/relu-grid';out.mkdir(exist_ok=True)
base=json.loads((e/'budget25/one-relu-manual/command.json').read_text());base=[x for x in base if not x.startswith('--native-target-scale-goals=')]
rows=[]
for a,b in itertools.product([40,40.1,41,44,48,50],[40,41,44,48,50,60]):
 case=out/f'p1-{a}-p2-{b}';case.mkdir(exist_ok=True);g=case/'goals.json';g.write_text(json.dumps({'43':a,'85':b,'157':40})+'\n');c=base+['--native-target-scale-goals='+str(g)];c[c.index('-o')+1]=str(case/'one.optimized.mlirbc');t=time.monotonic();p=subprocess.run(c,capture_output=True,text=True,timeout=120);(case/'compile.log').write_text(p.stdout+p.stderr)
 row=dict(p1=a,p2=b,code=p.returncode,seconds=time.monotonic()-t)
 if not p.returncode:
  path=next(case.glob('*.runtime-plan.json'));plan=json.loads(path.read_text());d={v['id']:v for v in plan['values']};desc=d[plan['final_outputs'][0]];row.update(consumed_q=30-desc['level'],scale=desc['scale_log2'],counts=dict(collections.Counter(o.get('op',o['kind']) for o in plan['execution'])),plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
