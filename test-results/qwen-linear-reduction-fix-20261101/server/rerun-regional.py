import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
rows=json.load(open(root/'remote-tests-result.json'))
source='/home/xuming/poseidon-regional-dacapo-20261101/runtime/third_party/dacapo'
for name,flags in [('DacapoRegionalScale',[]),('DacapoRegionalOptimized',['--boot-placement','optimized']),('DacapoRegionalAllCandidates',['--boot-placement','optimized','--all-candidates'])]:
 cmd=['python',str(root/'check-regional-scale.py'),'--hecate-opt',str(root/'hecate-opt'),'--source-dir',source]+flags
 start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 (root/(name+'.log')).write_text(p.stdout+p.stderr)
 rows=[r for r in rows if r['name']!=name];r=dict(name=name,returncode=p.returncode,seconds=time.monotonic()-start);rows.append(r);print(r,flush=True)
(root/'remote-tests-result.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(any(r['returncode'] for r in rows))
