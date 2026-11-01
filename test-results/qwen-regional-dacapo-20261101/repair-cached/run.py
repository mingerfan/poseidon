"""Time one complete Qwen compile, with a process-group deadline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', choices=('greedy', 'optimized'), required=True)
parser.add_argument('--timeout', type=float, default=900)
args = parser.parse_args()
here = Path(__file__).resolve().parent
meta = json.loads((here / 'experiment.json').read_text())
spec_path = here / 'profile/operator-spec.json'
spec = json.loads(spec_path.read_text())
boot = spec['boot_profiles'][0]
source = Path(meta['source'])
stat = source.stat()
out = here / f'regional-{args.strategy}'
out.mkdir(exist_ok=False)
command = [meta['compiler'], str(source), '--dacapo',
           f'--boot-placement={args.strategy}', '--native-scale-policy=regional',
           f'--ckks-config={here}/profile/compiler.json', '--waterline=40',
           f'--bootstrap-scale={boot["output_scale_log2"]}', '--regional-output-scale=40',
           '--compilation-progress', f'--regional-failure-report={out}/failure.json', '--mlir-disable-threading', '--mlir-timing', '--mlir-print-op-on-diagnostic=false',
           '--mlir-elide-elementsattrs-if-larger=16', '--mlir-elide-resource-strings-if-larger=16',
           '--runtime-plan-format=binary', '--runtime-plan-bundle-format=pack',
           f'--runtime-plan-physical-level-operator-spec-path={spec_path}',
           f'--runtime-plan-operator-spec-path={spec_path}', '--runtime-plan-levels-per-logical-level=1',
           f'--runtime-plan-boot-profile={boot["profile_id"]}', '--runtime-plan-boot-implementation=native',
           f'--runtime-plan-target-id={spec["target_id"]}',
           f'--runtime-plan-context-id={spec["context"]["context_id"]}',
           f'--runtime-plan-operator-spec-id={spec["spec_id"]}',
           f'--runtime-plan-operator-spec-version={spec["version"]}',
           '--runtime-plan-operator-spec-sha256=sha256:' + hashlib.sha256(spec_path.read_bytes()).hexdigest(),
           '--runtime-plan-device-counts=1', '--runtime-plan-device-count=1',
           '--runtime-plan-memory', '--runtime-plan-memory-report',
           '--runtime-plan-inline-payload-max-bytes=0', '-o', str(out/'qwen3.mlir')]
(out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
compiler_binary = Path(meta['compiler']).parents[1] / 'hecate-opt.real'
compiler_hash = hashlib.sha256(compiler_binary.read_bytes()).hexdigest()
assert compiler_hash == meta['compiler_sha256']
started = time.monotonic()
peak_rss_kib = 0
samples = []
timed_out = False
with (out/'compile.log').open('w') as log:
    process = subprocess.Popen(['/usr/bin/time', '-v', '-o', str(out/'time.log')] + command,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(strategy=args.strategy, pid=process.pid, started=True)), flush=True)
    while process.poll() is None:
        elapsed = time.monotonic() - started
        children = Path(f'/proc/{process.pid}/task/{process.pid}/children')
        if children.exists():
            for pid in children.read_text().split():
                status = Path(f'/proc/{pid}/status')
                try:
                    fields = status.read_text().splitlines()
                except FileNotFoundError:
                    continue
                rss = next((int(x.split()[1]) for x in fields if x.startswith('VmRSS:')), 0)
                peak_rss_kib = max(peak_rss_kib, rss)
        samples.append(dict(seconds=elapsed, peak_rss_kib=peak_rss_kib))
        (out/'status.json').write_text(json.dumps(samples[-1])+'\n')
        if elapsed >= args.timeout:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            break
        time.sleep(1)
    code = process.wait()
result = dict(strategy=args.strategy, returncode=124 if timed_out else code,
              timed_out=timed_out, seconds=time.monotonic()-started,
              peak_rss_kib_sampled=peak_rss_kib, compiler_sha256=compiler_hash,
              source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
              source_inode=stat.st_ino, required_output_s=40, timeout_seconds=args.timeout)
(out/'samples.json').write_text(json.dumps(samples, indent=2)+'\n')
(out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(result['returncode'])
