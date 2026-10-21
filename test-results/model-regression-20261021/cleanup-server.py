#!/usr/bin/env python3
"""Remove only this experiment's temporary checkout/data after archiving."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

workspace = Path('/home/xuming/poseidon-model-regression-20261021')
destination = Path('/home/xuming/test-results/model-regression-20261021')
for line in subprocess.check_output(['ps', '-eo', 'pid,args'], text=True).splitlines():
    if str(workspace) in line and any(name in line for name in [
            'hecate-opt --', 'hecate-opt-baseline --', 'poseidon_runtime_gpu_qwen_e2e ',
            'compile-models.py --', 'audit-plan ']):
        raise RuntimeError('Experiment process still running: '+line)
destination.mkdir(exist_ok=False)
archive = Path('/home/xuming/poseidon-model-regression-evidence-20261021.tar.gz')
shutil.copy2(archive, destination/'evidence.tar.gz')
for name in ['results.zh-CN.txt', 'validation-summary.json', 'provenance.json']:
    shutil.copy2(workspace/name, destination/name)
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
assert hashlib.sha256((destination/'evidence.tar.gz').read_bytes()).hexdigest() == digest
shutil.rmtree(workspace)
Path('/home/xuming/poseidon-model-regression-inputs-20261021.tar.gz').unlink()
archive.unlink()
assert not workspace.exists()
assert Path('/home/xuming/poseidon-waterline-20261013/qwen3-traced-bsgs/trace_qwen3.mlirbc').is_file()
(destination/'cleanup.json').write_text(json.dumps(dict(
    completed_utc=datetime.now(timezone.utc).isoformat(), temporary_workspace_removed=str(workspace),
    archive_sha256=digest, retained_results=str(destination), shared_qwen_source_preserved=True,
    physical_gpu=2, local_baseline_worktree_removed=True), indent=2)+'\n')
print('Removed experiment workspace; preserved verified evidence:', destination)
