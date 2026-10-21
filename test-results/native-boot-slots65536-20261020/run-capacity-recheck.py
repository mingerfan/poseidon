"""Explicit resource-only repeat of target scale40 with a 31.25GiB pool cap."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BINARY = Path('/home/xuming/poseidon-native-boot-scale40-20261020/native-boot-scale-bench')
sys.path.insert(0, '/home/xuming/poseidon-qwen-prefix-20261019')
from nvml_sampler import Observer


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == \
    '5824d7d625439e685704bde83dc3519d369fc98797415dbafcf143a0bc8a29dc'
case = ROOT / 'capacity32000-qwen30-s40'
case.mkdir(exist_ok=False)
observer = Observer(indices=(0,))
try:
    preflight = observer.sample()
    save(case / 'gpu0-preflight.json', preflight)
    if preflight['applications']:
        raise RuntimeError('GPU0 is occupied; do not run the capacity recheck')
finally:
    observer.close()
env = {k: v for k, v in os.environ.items() if not k.startswith('POSEIDON_')}
env.update(CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
           POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='1024',
           POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='32000')
command = [str(BINARY), str(case / 'report.json'), '131072', 'qwen30', '10', '40']
save(case / 'command.json', command)
save(case / 'environment.json', {k: v for k, v in env.items()
     if k.startswith(('POSEIDON_', 'CUDA_', 'OMP_', 'OPENBLAS_'))})
save(case / 'method.json', dict(
    original='qwen30-s40', change='Only maximum pool capacity:28672MiB ->32000MiB',
    reason='Separate artificial 28GiB cap from inability to run in available GPU0 VRAM',
    initial_pool_mib=1024, maximum_pool_mib=32000,
))
started = time.monotonic()
with (case / 'run.log').open('w') as log:
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    save(case / 'launch.json', dict(pid=proc.pid,
         timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    code = proc.wait()
status = dict(eval_mod_log_scale=40, maximum_pool_mib=32000, exit_code=code,
              wall_seconds=time.monotonic() - started)
save(case / 'status.json', status)
print(json.dumps(status), flush=True)
report = json.loads((case / 'report.json').read_text())
print(json.dumps(dict(stage=report['stage'], error=report.get('error'))), flush=True)
save(ROOT / 'capacity-recheck-finished.json', status)
