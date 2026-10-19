#!/usr/bin/env python3
"""Run recorded follow-up commands sequentially on one host/GPU allocation."""
import json
import os
from pathlib import Path
import subprocess
import sys

path = Path(sys.argv[1]).resolve()
for job in json.loads(path.read_text()):
    print(job["name"], flush=True)
    env = os.environ.copy()
    env.update(job.get("environment", {}))
    with (path.parent / (job["name"] + ".log")).open("w") as log:
        subprocess.run(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
