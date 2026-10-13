#!/usr/bin/env python3
"""Sample a running model's RSS and per-process driver memory until it exits.

These are interval samples, not allocator peaks. Start before the model's main
work when possible; a late start must be reported with its first timestamp.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error('PID and interval must be positive')
    proc = Path('/proc') / str(args.pid)
    identity = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    started = time.monotonic()
    with args.output.open('x', encoding='utf-8') as output:
        while proc.exists():
            try:
                current = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                status = (proc / 'status').read_text()
            except FileNotFoundError:
                break
            if current != identity:
                break
            host = {}
            for line in status.splitlines():
                key, _, value = line.partition(':')
                if key in ('VmRSS', 'VmHWM'):
                    host[key + '_bytes'] = int(value.split()[0]) * 1024
            query = subprocess.run([
                'nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                '--format=csv,noheader,nounits'], check=True,
                text=True, capture_output=True)
            gpu = {}
            for row in csv.reader(query.stdout.splitlines()):
                pid, uuid, memory = (part.strip() for part in row)
                if int(pid) == args.pid:
                    gpu[uuid] = int(memory) * 2**20
            sample = dict(timestamp=datetime.datetime.now().astimezone().isoformat(),
                          elapsed_seconds=time.monotonic() - started,
                          pid=args.pid, host=host, gpu_process_bytes=gpu)
            output.write(json.dumps(sample) + '\n')
            output.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
