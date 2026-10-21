#!/usr/bin/env python3
"""Bound the experimental optimized compile without changing its planner."""
import json
from pathlib import Path
import subprocess
import time

status = Path('/home/xuming/poseidon-model-regression-20261021/qwen-prefix-optimized/gpu1/status.json')
while True:
    value = json.loads(status.read_text())
    if value['returncode'] is not None:
        break
    if value['elapsed_seconds'] >= 900:
        status.with_name('experiment-time-limit.json').write_text(json.dumps(dict(
            reason='Experiment compile wall-time budget; not a reported compiler error',
            limit_seconds=900, last_status=value), indent=2)+'\n')
        subprocess.run(['pkill', '-TERM', '-P', str(value['pid'])], check=True)
        break
    time.sleep(5)
