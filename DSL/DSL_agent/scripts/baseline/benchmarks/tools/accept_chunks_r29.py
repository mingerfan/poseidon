"""Frozen serial r29 acceptance. No provider calls, installs or budget increases."""
import json,subprocess,time,hashlib
from pathlib import Path
from benchmark_runner import DEFAULT,build,load,dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
from hecate_python_env import VENV,WORK
from upstream_chunk_cases import cases
source=runtime_sources();start=time.monotonic()
require(not DEFAULT.exists(),"Preserve frozen release")
build(DEFAULT)
rows,index=load(DEFAULT);require(len(rows)==1200,"Frozen corpus denominator")
selected=cases();require(len(selected)==13,"Thirteen directed chunk cases")
dump(WORK/"results/upstream-chunks-r29-acceptance-plan.json",dict(sources=source,source_sha256=digest(source),
 case_ids=[r["name"] for r in selected],max_wall_seconds=1800,agent_calls=0,native_concurrency=1,
 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
def run(args,seconds):
 require(runtime_sources()==source,"Source changed")
 subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=min(seconds,max(1,1800-(time.monotonic()-start))))
run(["scripts/baseline/benchmarks/tools/verify_chunk_units_r29.py"],600)
run(["scripts/benchmark.py","plaintext","--suite",str(DEFAULT),"--output",str(WORK/"results/upstream-chunks-r29-plaintext")],240)
args=["scripts/benchmark.py","helper-directed","--inside","--execute","--output",str(WORK/"results/upstream-chunks-r29-helper-directed")]
for row in selected:args+=["--task-id","upstream_"+row["name"]]
run(args,900)
run(["scripts/baseline/benchmarks/tools/regress_chunks_r29.py"],600)
run(["scripts/baseline/benchmarks/tools/regress_chunk_helpers_r29.py"],300)
args=["scripts/benchmark.py","compiler-evidence","--inside","--output",str(WORK/"results/upstream-chunks-r29-compiler-evidence.json")]
for name in ("helper-directed","regression","chunk-regression","legacy-helpers"):
 args+=["--batch-report",str(WORK/("results/upstream-chunks-r29-"+name)/"report.json")]
run(args,180)
require(runtime_sources()==source,"Final source drift")
dump(WORK/"results/upstream-chunks-r29-complete.json",dict(status="passed",seconds=time.monotonic()-start,source_sha256=digest(source),agent_calls=0))
