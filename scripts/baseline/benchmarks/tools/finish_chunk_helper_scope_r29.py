"""Remaining 44 frozen helpers plus three I/O boundary controls, within existing per-batch limits."""
import subprocess,time,json,hashlib
from pathlib import Path
from benchmark_graph import require,digest
from benchmark_runner import dump
from semantic_benchmark_execution import runtime_sources
from upstream_helper_directed_cases import cases
from hecate_python_env import WORK,VENV
source=runtime_sources();start=time.monotonic()
complete=json.loads((WORK/"results/upstream-chunks-r29-complete.json").read_text())
require(complete["status"]=="passed" and complete["source_sha256"]==digest(source),"Main batch must finish first")
tested={"bench_helper_0008","bench_helper_0072","bench_helper_0080"}
remaining=[r for r in cases() if "chunk_period" not in r and r["name"] not in tested]
require(len(remaining)==44,"Unrun helper denominator")
dump(WORK/"results/upstream-chunks-r29-remainder-plan.json",dict(source_sha256=digest(source),
 remaining_helpers=[r["name"] for r in remaining],boundary_controls=3,intentional_negative_controls=1,max_wall_seconds=1800,
 native_concurrency=1,agent_calls=0,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
args=["scripts/benchmark.py","helper-directed","--inside","--execute","--output",str(WORK/"results/upstream-chunks-r29-helper-remainder")]
for row in remaining:args+=["--task-id","upstream_"+row["name"]]
subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=1400)
require(runtime_sources()==source,"Source drift")
subprocess.run([str(VENV/"bin/python"),"-B","scripts/baseline/benchmarks/tools/run_chunk_boundaries_r29.py"],check=True,
               timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==source,"Source drift")
subprocess.run([str(VENV/"bin/python"),"-B","scripts/baseline/benchmarks/tools/reject_swapped_chunks_r29.py"],check=True,
               timeout=max(1,1800-(time.monotonic()-start)))
args=["scripts/benchmark.py","compiler-evidence","--inside","--output",str(WORK/"results/upstream-chunks-r29-remainder-compiler-evidence.json")]
for name in ("helper-remainder","boundaries","negative"):
 args+=["--batch-report",str(WORK/("results/upstream-chunks-r29-"+name)/"report.json")]
subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==source,"Final source drift")
dump(WORK/"results/upstream-chunks-r29-all-helper-complete.json",dict(status="passed",seconds=time.monotonic()-start,
 source_sha256=digest(source),agent_calls=0))
