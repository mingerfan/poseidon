"""Reuse the recorded Qwen24 command and its lossless native bytecode."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

root = Path('/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1')
out = root / 'artifacts/qwen24-plaintext-streaming'
out.mkdir(exist_ok=True)
old = json.loads((root / 'artifacts/qwen24-unit-plaintext/gpu-plans/gpu4/compiler-report.json').read_text())
command = old['command'][:]
command[3] = str(root / 'toolchain/bin/hecate-opt.plaintext-streaming')
command[-1] = str(out / 'qwen24.mlir')
command += ['--runtime-plan-plaintext-schedule=stream',
            f'--runtime-plan-gpu-budget-bytes={24*2**30}',
            f'--runtime-plan-host-budget-bytes={4*2**30}',
            f'--runtime-plan-pinned-budget-bytes={512*2**20}',
            f'--runtime-plan-raw-budget-bytes={2**20}',
            f'--runtime-plan-workspace-bytes-per-op={16*2**20}',
            f'--runtime-plan-prefetch-bytes={256*2**20}']
env = os.environ.copy()
env.update(HECATE=str(root/'toolchain'), HECATE_BUILD=str(root/'toolchain'), OMP_NUM_THREADS='16')
def limits():
    resource.setrlimit(resource.RLIMIT_AS, (256*2**30,)*2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (7200, 7210))
start = time.monotonic()
with (out/'compile.log').open('w') as log:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
    (out/'running.json').write_text(json.dumps({'pid':process.pid,'command':command}, indent=2))
    _, status, usage = os.wait4(process.pid, 0)
report = {'returncode':os.waitstatus_to_exitcode(status), 'elapsed_seconds':time.monotonic()-start,
          'max_rss_MiB':usage.ru_maxrss/1024, 'command':command}
(out/'compiler-report.json').write_text(json.dumps(report, indent=2)+'\n')
