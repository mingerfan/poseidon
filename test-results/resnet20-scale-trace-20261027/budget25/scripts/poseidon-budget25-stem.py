from pathlib import Path
import json,re,subprocess,time
root=Path('/home/xuming/poseidon-manual-boot-20261026');e=root/'resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment';out=e/'budget25/stem-gpu';out.mkdir(exist_ok=True)
with (e/'budget25/stem-out/raw.mlir').open() as src,(out/'stem.mlir').open('w') as dst:
 for line in src:
  match=re.match(r'\s*%(\d+) = "earth\.',line)
  if match and int(match[1])>133:continue
  if re.match(r'\s*return %',line):line=re.sub(r'%\d+','%133',line,count=1)
  dst.write(line)
goals=json.loads((e/'budget25/manual-cuts/manual-stem-goals.json').read_text());goals={k:v for k,v in goals.items() if int(k)<=135};(out/'goals.json').write_text(json.dumps(goals,indent=2)+'\n');(out/'schedule.json').write_text('[]\n')
c=json.loads((e/'budget25/full18-input34/result.json').read_text())['command'];c[c.index('-o')+1]=str(out/'stem.optimized.mlirbc');c[c.index('/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc')]=str(out/'stem.mlir');c=[x for x in c if not x.startswith('--native-target-bootstrap-schedule=') and not x.startswith('--native-target-scale-goals=')]
c=[x if not x.startswith('--runtime-plan-inline-payload-max-bytes=') else '--runtime-plan-inline-payload-max-bytes=1048576' for x in c];c+=['--native-target-bootstrap-schedule='+str(out/'schedule.json'),'--native-target-scale-goals='+str(out/'goals.json')]
t=time.monotonic();p=subprocess.run(c,capture_output=True,text=True,timeout=180);(out/'compile.log').write_text(p.stdout+p.stderr);(out/'command.json').write_text(json.dumps(c,indent=2)+'\n');print('stem compiler',p.returncode,round(time.monotonic()-t,2),flush=True)
if p.returncode:print(p.stderr[-2000:]);raise SystemExit(p.returncode)
plan=json.loads(next(out.glob('*.runtime-plan.json')).read_text());d={v['id']:v for v in plan['values']};print('stem states',d[plan['external_inputs'][0]],d[plan['final_outputs'][0]],flush=True)
