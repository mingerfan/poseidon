import json,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
out=here/'resnet-regression'
out.mkdir(exist_ok=False)
start=time.monotonic()
with (out/'compile.log').open('w') as log:
 result=subprocess.run(json.loads((here/'resnet-command.json').read_text()), stdout=log,stderr=subprocess.STDOUT,timeout=180)
record=dict(returncode=result.returncode,seconds=time.monotonic()-start)
(out/'result.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
raise SystemExit(result.returncode)
