import json
from pathlib import Path
import subprocess
import time
here = Path(__file__).resolve().parent
results = []
for row in json.loads((here / 'remote-tests.json').read_text()):
    start = time.monotonic()
    result = subprocess.run(row['command'], capture_output=True, text=True)
    (here / (row['name'] + '.log')).write_text(result.stdout + result.stderr)
    results.append(dict(name=row['name'], returncode=result.returncode,
                        seconds=time.monotonic()-start))
    print(row['name'], result.returncode, flush=True)
(here / 'remote-tests-result.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(any(row['returncode'] for row in results))
