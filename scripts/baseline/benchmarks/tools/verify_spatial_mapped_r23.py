"""Serial native acceptance after exact public-zero normalization; no API."""
import subprocess,time
from pathlib import Path
from hecate_python_env import WORK,VENV
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
from benchmark_runner import dump
start=time.monotonic();sources=runtime_sources()
for suffix,limit in [('zero-smoke',2),('native-final',48)]:
    require(runtime_sources()==sources,'Frozen sources changed')
    subprocess.run([str(VENV/'bin/python'),'-B','scripts/baseline/benchmarks/tools/run_upstream_spatial_mapped_batch.py',
        '--output',str(WORK/'results'/('upstream-spatial-r23-'+suffix)),'--limit',str(limit)],check=True,
        timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Frozen sources changed')
dump(WORK/'results/upstream-spatial-r23-native-complete.json',dict(status='passed',source_sha256=digest(sources),
    seconds=time.monotonic()-start,agent_calls=0))
