#!/usr/bin/env python3
"""Record repeated GPU API runs without replacing previous failure evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--expected-passes', type=int, default=20)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    prefix = args.prefix.resolve()
    report_path = prefix.with_name(prefix.name + '-status.json')
    if report_path.exists() or any(prefix.parent.glob(prefix.name + '-[0-9][0-9].log')):
        raise FileExistsError('use a new prefix to preserve existing evidence')
    report = dict(binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  repeats_requested=args.repeats, expected_passes=args.expected_passes, runs=[], passed=False,
                  scope='existing full GPU API regression suite; not a complete model test')
    for index in range(args.repeats):
        log_path = prefix.with_name(prefix.name + f'-{index:02d}.log')
        started = time.monotonic()
        with log_path.open('x') as log:
            try:
                result = subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout_seconds, env=os.environ.copy())
                code = result.returncode
                timeout = False
            except subprocess.TimeoutExpired:
                code = None
                timeout = True
        lines = log_path.read_text().splitlines()
        passes = sum(line.startswith('[PASS]') for line in lines)
        failures = [line for line in lines if line.startswith('[FAIL]')]
        run = dict(index=index, returncode=code, timeout=timeout, passes=passes, failures=failures,
                   seconds=time.monotonic()-started, log=str(log_path))
        report['runs'].append(run)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(run), flush=True)
        if code != 0 or timeout or passes != args.expected_passes or failures:
            raise SystemExit(1)
    report['passed'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
