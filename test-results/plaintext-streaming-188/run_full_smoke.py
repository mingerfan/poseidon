"""Attempt full local execution after compilation, with a bounded test duration.
This uses deterministic one-hot inputs, not a trained-model numerical oracle.
"""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

root = Path('/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1')
out = root/'artifacts/qwen24-plaintext-streaming'
while not (out/'compiler-report.json').exists():
    time.sleep(10)
if json.loads((out/'compiler-report.json').read_text())['returncode'] != 0:
    raise SystemExit('full compiler failed; see compiler-report.json')
runner = '/home/xuming/poseidon-memory-step3-20261008/build/bin/poseidon_gpu_mpi_runtime_e2e'
command = [runner, str(out/'qwen24._hecate_qwen25_24layer.runtime-plan.json'),
           str(root/'profiles/operator-spec.json'), str(out/'qwen24._hecate_qwen25_24layer.bundle'),
           str(out/'runtime-result.json'), '--local', '--warmups', '0', '--iterations', '1',
           '--execution-mode', 'sequential']
env = os.environ.copy()
env.update(LD_LIBRARY_PATH='/home/xuming/poseidon-tools/openmpi/lib:/home/xuming/poseidon-tools/nccl/nvidia/nccl/lib:/usr/local/cuda-12.2/lib64',
           OMP_NUM_THREADS='2', POSEIDON_GPU_RUNTIME_INITIAL_POOL_MB='64')
def limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (384*2**30,)*2)
start = time.monotonic()
peak_rss = 0
peak_gpu = [0]*4
reason = None
with (out/'runtime.log').open('w') as log, (out/'runtime-samples.jsonl').open('w') as samples:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            break
        elapsed = time.monotonic()-start
        mem = {}
        try:
            for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    k, v = line.split(':'); mem[k]=int(v.split()[0])*1024
        except FileNotFoundError:
            pass
        peak_rss=max(peak_rss, mem.get('VmHWM',0))
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
        used=[int(v)*2**20 for v in gpu.splitlines()]
        peak_gpu=[max(a,b) for a,b in zip(peak_gpu,used)]
        samples.write(json.dumps({'seconds':elapsed,'host':mem,'gpu_used_bytes':used})+'\n');samples.flush()
        if elapsed > 600:
            reason='600 second smoke-test wall limit'
            os.killpg(process.pid, signal.SIGTERM)
            _, status, usage=os.wait4(process.pid,0)
            break
        time.sleep(2)
report={'command':command,'returncode':os.waitstatus_to_exitcode(status), 'stop_reason':reason,
        'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':max(peak_rss,usage.ru_maxrss*1024),
        'sampled_gpu_used_bytes':peak_gpu, 'memory_sample_scope':'whole device, sampled every 2 seconds',
        'complete_model_numerical_oracle':False}
(out/'runtime-smoke-report.json').write_text(json.dumps(report,indent=2)+'\n')
