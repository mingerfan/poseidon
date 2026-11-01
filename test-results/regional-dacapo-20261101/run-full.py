"""Compile the full, unchanged ResNet20 with regional scales and DaCapo Boot."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input-q', type=int, choices=(31, 34), required=True)
parser.add_argument('--strategy', choices=('greedy', 'optimized'), default='optimized')
parser.add_argument('--timeout', type=float, default=None)
args = parser.parse_args()
here = Path(__file__).resolve().parent
out = here / f'regional-{args.strategy}-q{args.input_q}-s40'
out.mkdir(exist_ok=True)
command = [x.format(here=here, output=out) for x in
           json.loads((here/'full-command-template.json').read_text())]
command[command.index('--boot-placement=optimized')] = f'--boot-placement={args.strategy}'
if args.input_q != 31:
    command.append(f'--native-input-level={args.input_q-1}')
(out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
start = time.monotonic()
compiler_sha256 = hashlib.sha256((here/'hecate-opt.real').read_bytes()).hexdigest()
with (out/'compile.log').open('w') as log:
    try:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout)
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
report = dict(input_q=args.input_q, final_scale_required=40,
              strategy=args.strategy,
              returncode=returncode, seconds=time.monotonic()-start,
              timeout_seconds=args.timeout,
              compiler_sha256=compiler_sha256)
(out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
raise SystemExit(returncode)
