"""Build a private benchmark and compare only EvalMod log scale, on GPU0."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BUILD = Path('/home/xuming/poseidon-framework-validation-20261015-6dc8cbfb/build')
SOURCE = BUILD.parent / 'src'
sys.path.insert(0, '/home/xuming/poseidon-qwen-prefix-20261019')
from nvml_sampler import Observer


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


reference = json.loads((ROOT / 'reference-provenance.json').read_text())
libraries = {name: sha(Path(name)) for name in reference['shared_libraries_sha256']}
if libraries != reference['shared_libraries_sha256']:
    raise RuntimeError('Current backend differs from the previous native Boot benchmark')
sources = {name: sha(Path(name)) for name in reference['current_sources']}
if sources != reference['current_sources']:
    raise RuntimeError('Current native Boot sources differ from the reference')
save(ROOT / 'provenance.json', dict(
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    libraries_sha256=libraries, current_sources_sha256=sources,
    benchmark_sha256=sha(ROOT / 'native-boot-scale-bench.cpp'),
    scope='65536 complex slots, N131072, Q50/P8 context; single V100 GPU0; production sources unchanged',
    reference='native-boot-timing-20261020/full58-qwen30',
))

BINARY = Path('/home/xuming/poseidon-native-boot-scale40-20261020/native-boot-scale-bench')
expected_binary_sha = '5824d7d625439e685704bde83dc3519d369fc98797415dbafcf143a0bc8a29dc'
if sha(BINARY) != expected_binary_sha:
    raise RuntimeError('Benchmark executable differs from the scale40/45 comparison')
save(ROOT / 'binary.json', dict(path=str(BINARY), sha256=sha(BINARY)))

results = []
for scale in (45, 40):
    case = ROOT / f'qwen30-s{scale}'
    case.mkdir(exist_ok=False)
    observer = Observer(indices=(0,))
    try:
        preflight = observer.sample()
        save(case / 'gpu0-preflight.json', preflight)
        if preflight['applications']:
            raise RuntimeError('GPU0 has other compute processes; do not start the test')
    finally:
        observer.close()
    env = {k: v for k, v in os.environ.items() if not k.startswith('POSEIDON_')}
    env.update(CUDA_VISIBLE_DEVICES='0', OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
               POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='1024',
               POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB='28672')
    command = [str(BINARY), str(case / 'report.json'),
               '131072', 'qwen30', '10', str(scale)]
    save(case / 'command.json', command)
    save(case / 'environment.json', {k: v for k, v in env.items()
         if k.startswith(('POSEIDON_', 'CUDA_', 'OMP_', 'OPENBLAS_'))})
    started = time.monotonic()
    with (case / 'run.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        save(case / 'launch.json', dict(pid=proc.pid,
             timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        code = proc.wait()
    status = dict(eval_mod_log_scale=scale, exit_code=code,
                  wall_seconds=time.monotonic() - started)
    save(case / 'status.json', status)
    results.append(status)
    save(ROOT / 'run-status.json', results)
    print(json.dumps(status), flush=True)
    report = json.loads((case / 'report.json').read_text())
    print(json.dumps(dict(scale=scale, stage=report.get('stage'), error=report.get('error'))), flush=True)

save(ROOT / 'finished.json', dict(
    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    binary_sha256=sha(BINARY), results=results))
