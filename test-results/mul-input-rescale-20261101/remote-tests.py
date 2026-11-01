import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
rows=json.load(open('/home/xuming/poseidon-regional-dacapo-20261101/remote-tests.json'))
results=[]
for row in rows:
 if row['name']=='DacapoModulusChain':continue
 cmd=[str(root/'hecate-opt') if a.endswith('/bin/hecate-opt') else str(root/'check-regional-scale.py') if a.endswith('/check-regional-scale.py') else a for a in row['command']]
 start=time.monotonic();p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
 (root/(row['name']+'.log')).write_text(p.stdout+p.stderr)
 out=dict(name=row['name'],returncode=p.returncode,seconds=time.monotonic()-start);results.append(out);print(out,flush=True)
(root/'remote-tests-result.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(any(r['returncode'] for r in results))
