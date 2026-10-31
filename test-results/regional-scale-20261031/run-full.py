"""Compile the same full ResNet20 with region scales or the frozen baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=('regional', 'regional-fixed40', 'regional-q34', 'threshold-greedy'), required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
baseline = Path('/home/xuming/poseidon-local-scale-step3-20261031/ablate-stages12/full-steps12/command.json')
command = json.loads(baseline.read_text())
output = here / args.variant
output.mkdir(exist_ok=True)
regional = args.variant.startswith('regional')
command[0] = str(here/'bin/hecate-opt') if regional else command[0]
command = [x.replace('--boot-placement=optimized', '--boot-placement=greedy') for x in command]
if regional:
    command += ['--native-scale-policy=regional', f'--regional-scale-report={output}/scale-trace.json']
if args.variant == 'regional-q34':
    command.append('--native-input-level=33')
if args.variant == 'regional-fixed40':
    command.append('--regional-output-scale=40')
command[command.index('-o') + 1] = str(output/'resnet20.mlirbc')
(output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
start = time.monotonic()
with (output/'compile.log').open('w') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
report = {'variant': args.variant, 'returncode': result.returncode,
          'seconds': time.monotonic()-start,
          'compiler_sha256': hashlib.sha256((Path(command[0]).parent/'hecate-opt.real' if not regional else here/'hecate-opt.real').read_bytes()).hexdigest()}
(output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(result.returncode)
