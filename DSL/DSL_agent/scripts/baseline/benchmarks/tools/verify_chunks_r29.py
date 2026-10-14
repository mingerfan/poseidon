"""Final-source serial regression and artifact audit; no provider."""
import subprocess,time,json
from pathlib import Path
from hecate_python_env import WORK,VENV
from semantic_benchmark_execution import runtime_sources
from benchmark_graph import require,digest
from benchmark_runner import dump
start=time.monotonic();sources=runtime_sources()
for tool in ("verify_chunk_units_r29.py","regress_chunks_r29.py","regress_chunk_helpers_r29.py"):
 require(runtime_sources()==sources,"Source drift")
 subprocess.run([str(VENV/"bin/python"),"-B","scripts/baseline/benchmarks/tools/"+tool],
                check=True,timeout=max(1,1800-(time.monotonic()-start)))
args=[str(VENV/"bin/python"),"-B","scripts/benchmark.py","compiler-evidence","--inside","--output",str(WORK/"results/upstream-chunks-r29-compiler-evidence.json")]
for name in ("helper-directed","regression","chunk-regression","legacy-helpers"):
 args+=["--batch-report",str(WORK/("results/upstream-chunks-r29-"+name)/"report.json")]
subprocess.run(args,check=True,timeout=300)
require(runtime_sources()==sources,"Final source drift")
dump(WORK/"results/upstream-chunks-r29-complete.json",dict(status="passed",seconds=time.monotonic()-start,source_sha256=digest(sources),agent_calls=0))
