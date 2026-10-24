#!/usr/bin/env python3
"""Historical comparison using compiler e01f8ee; no GPU or encrypted execution.

The experimental all mode and coefficient-scale options used here were removed
after this experiment. Use that compiler revision to reproduce these records.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--modes', nargs='+', default=['none', 'double', 'double-align', 'all'])
    p.add_argument('--strategies', nargs='+', default=['greedy', 'depth-dp', 'optimized'])
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    audit = runpy.run_path(str(root.parent / 'model-regression-20261021/audit-logical.py'))['audit']
    for mode in a.modes:
        for strategy in a.strategies:
            out = root / mode / strategy
            out.mkdir(parents=True, exist_ok=False)
            logical = out / 'logical.mlir'
            command = [str(a.compiler.resolve()), '--dacapo-precompile',
                       f'--polynomial-optimizations={mode}', f'--boot-placement={strategy}',
                       f'--ckks-config={root}/compiler-profile.json',
                       '--waterline=40', '--bootstrap-scale=45',
                       '--coefficient-min-scale=30', '--coefficient-precision-bits=24',
                       '--mlir-disable-threading', '--mlir-timing',
                       '--mlir-print-op-on-diagnostic=false',
                       '--mlir-elide-elementsattrs-if-larger=16',
                       str(a.source.resolve()), '-o', str(logical)]
            (out/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            start = time.monotonic()
            binary_sha256 = hashlib.sha256(a.compiler.read_bytes()).hexdigest()
            with (out/'compile.log').open('w') as log:
                result = subprocess.run(command,
                                        stdout=log, stderr=subprocess.STDOUT)
            status = dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-start,
                          compiler_sha256=binary_sha256)
            (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
            if result.returncode:
                raise RuntimeError(f'{mode}/{strategy}: see {out}/compile.log')
            counts = audit(logical, 1, 22, 4)
            text = logical.read_text()
            reported = re.search(r'Number of Bootstrapping: (\d+)', (out/'compile.log').read_text())
            assert reported and int(reported[1]) == counts['boot_count']
            constants = {}
            double_muls = unit_muls = 0
            for line in text.splitlines():
                m = re.search(r'(%\w+) = "ckks.encode".*payload = dense<([^>]+)>', line)
                if m:
                    try: constants[m[1]] = float(m[2])
                    except ValueError: pass
                m = re.search(r'"ckks.mulcp"\((%\w+), (%\w+)\)', line)
                if m:
                    double_muls += constants.get(m[2]) == 2
                    unit_muls += constants.get(m[2]) == 1
            counts.update(mode=mode, strategy=strategy, multiply_by_two=double_muls,
                          multiply_by_one=unit_muls,
                          logical_sha256=hashlib.sha256(logical.read_bytes()).hexdigest())
            (out/'audit.json').write_text(json.dumps(counts, indent=2)+'\n')
            with gzip.open(out/'logical.mlir.gz', 'wb') as archive:
                archive.write(logical.read_bytes())
            logical.unlink()
            print(json.dumps(dict(mode=mode, strategy=strategy, **status,
                                  **{k:counts[k] for k in ['boot_count','rescale_q','modswitch_q',
                                                          'multiply_by_two','multiply_by_one']})), flush=True)


if __name__ == '__main__':
    main()
