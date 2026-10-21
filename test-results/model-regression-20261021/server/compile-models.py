#!/usr/bin/env python3
"""Run real DaCapo planning on shared model bytecode; retain each command/log."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--strategies', nargs='+', default=['greedy', 'depth-dp', 'optimized'])
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--boot-scale', type=int, default=45)
    a = p.parse_args()
    for strategy in a.strategies:
        out = a.root / a.name / strategy
        out.mkdir(parents=True, exist_ok=False)
        command = [str(a.compiler), '--dacapo-precompile',
                   f'--boot-placement={strategy}', f'--ckks-config={a.config}',
                   '--waterline=40', f'--bootstrap-scale={a.boot_scale}',
                   '--mlir-timing', '--mlir-disable-threading',
                   '--mlir-print-op-on-diagnostic=false',
                   '--mlir-elide-elementsattrs-if-larger=16',
                   str(a.source), '-o', str(out / 'logical.mlir')]
        (out / 'command.json').write_text(json.dumps(command, indent=2)+'\n')
        start = time.monotonic()
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (out / 'compile.log').open('w') as log:
            process = subprocess.Popen(['/usr/bin/time', '-v', *command],
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            expired = False
            while True:
                code = process.poll()
                elapsed = time.monotonic() - start
                status = dict(returncode=code, elapsed_seconds=elapsed,
                              timed_out=expired, pid=process.pid)
                (out / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
                if code is not None:
                    print(json.dumps(dict(case=str(out), **status)), flush=True)
                    break
                if elapsed > a.timeout:
                    expired = True
                    # time(1) is the process parent; kill only this compilation.
                    subprocess.run(['pkill', '-TERM', '-P', str(process.pid)], check=False)
                time.sleep(2)


if __name__ == '__main__':
    main()
