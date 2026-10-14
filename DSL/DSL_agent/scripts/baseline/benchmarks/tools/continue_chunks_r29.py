"""Resume after harness-only missing --inside; preserve source, unit pass and failure log."""
import subprocess,time,json,hashlib
from pathlib import Path
from hecate_python_env import WORK,VENV
from benchmark_runner import DEFAULT,load,dump
from benchmark_graph import require,digest
from semantic_benchmark_execution import runtime_sources
from upstream_chunk_cases import cases
plan_path=WORK/"results/upstream-chunks-r29-acceptance-plan.json"
plan=json.loads(plan_path.read_text());source=runtime_sources();require(source==plan["sources"],"Source drift")
unit=json.loads((WORK/"results/upstream-chunks-r29-unit.json").read_text())
require(unit["success"] and unit["source_sha256"]==digest(source),"Verified unit checkpoint required")
load(DEFAULT);selected=cases();require(len(selected)==13,"Thirteen chunk tasks")
# Count all time since the original frozen plan plus a conservative minute for
# its preparation. This continuation never grants a fresh full phase budget.
prior=time.time()-plan_path.stat().st_mtime+60;require(0<=prior<1800,"Original wall budget exhausted")
start=time.monotonic()
dump(WORK/"results/upstream-chunks-r29-continuation-plan.json",dict(prior_wall_seconds=prior,max_total_seconds=1800,
 source_sha256=digest(source),reason="harness omitted --inside for plaintext; no model test had started",
 runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),agent_calls=0))
def run(args,seconds):
 require(runtime_sources()==source,"Source drift")
 remaining=1800-prior-(time.monotonic()-start);require(remaining>0,"Original wall budget")
 subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=min(seconds,remaining))
run(["scripts/benchmark.py","plaintext","--inside","--suite",str(DEFAULT),"--output",str(WORK/"results/upstream-chunks-r29-plaintext")],240)
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
dump(WORK/"results/upstream-chunks-r29-complete.json",dict(status="passed",seconds=prior+time.monotonic()-start,
 source_sha256=digest(source),agent_calls=0,original_harness_failure_preserved=True))
