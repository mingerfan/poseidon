import json, resource, subprocess, time
from pathlib import Path
root = Path('/home/xuming/poseidon-runtime-io-20261010')
old = Path('/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1')
prefix = root / 'artifacts/gpu4/qwen24'
prefix.parent.mkdir(parents=True, exist_ok=True)
options = (f'prefix={prefix} plan-id=2514 target-id=poseidon-ckks-gpu capability-version=1 '
           'operator-spec-id=poseidon-ckks-gpu-65536-estimated-v1 operator-spec-version=1 '
           'operator-spec-sha256=sha256:1f12fb5f6a7228e7373ada0d96d5115d190533a887f801a6928ba4ae14c193ff '
           'context-id=poseidon-ckks-gpu-65536-estimated ntt=true device-count=4 '
           'boot-profile=poseidon-gpu-host-boot-v1 boot-implementation=decrypt_reencrypt '
           'inline-payload-max-bytes=4096 report-io=true')
command = [str(old/'toolchain/lib/ld-linux-x86-64.so.2'), '--library-path', str(old/'toolchain/lib'),
           str(root/'hecate-opt-streaming'),
           str(root/'artifacts/gpu4/qwen24.restored.mlir'),
           f'-p=builtin.module(func.func(emit-runtime-plan{{{options}}}))',
           '--ckks-config=' + str(old/'profiles/compiler-profile.json'), '--mlir-timing', '--mlir-print-op-on-diagnostic=false', '-o', '/dev/null']
def limit():
    resource.setrlimit(resource.RLIMIT_AS, (128 * 1024**3, 128 * 1024**3))
    __import__('os').nice(10)
start = time.monotonic()
with (root/'results/stage-c-gpu4-export.log').open('w') as log:
    result = subprocess.run(command, cwd=old, stdout=log, stderr=log, preexec_fn=limit)
report = {'command': command, 'returncode': result.returncode, 'elapsed_seconds': time.monotonic()-start,
          'max_rss_bytes': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
          'input_ir_bytes': (root/'artifacts/gpu4/qwen24.restored.mlir').stat().st_size,
          'address_space_limit_bytes': 128*1024**3, 'nice': 10}
(root/'results/stage-c-gpu4-export-process.json').write_text(json.dumps(report, indent=2)+'\n')
raise SystemExit(result.returncode)
