"""Serial frozen-source correctness batch; no API and no budget increase."""
import subprocess,json,time
from pathlib import Path
from hecate_python_env import WORK,VENV
from semantic_benchmark_execution import runtime_sources
from benchmark_graph import require,digest
from benchmark_runner import dump
source=runtime_sources();start=time.monotonic();base=WORK/"results"
steps=[
 ["scripts/baseline/benchmarks/tools/verify_packed_prefix_units_r27.py"],
 ["scripts/baseline/benchmarks/tools/preflight_packed_prefix_r27.py"],
 ["scripts/baseline/benchmarks/tools/run_packed_prefix_r27.py"],
 ["scripts/baseline/benchmarks/tools/regress_packed_prefix_r27.py"]]
for args in steps:
 require(runtime_sources()==source,"Source changed at batch boundary")
 subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=max(1,1800-(time.monotonic()-start)))
args=["scripts/benchmark.py","compiler-evidence","--inside","--output",str(base/"packed-prefix-r27-compiler-evidence.json")]
for folder in ("packed-prefix-r27-native","packed-prefix-r27-regression","packed-prefix-r27-chunk-regression"):
 args+=["--batch-report",str(base/folder/"report.json")]
subprocess.run([str(VENV/"bin/python"),"-B",*args],check=True,timeout=300)
require(runtime_sources()==source,"Final source drift")
dump(base/"packed-prefix-r27-complete.json",dict(status="passed",seconds=time.monotonic()-start,source_sha256=digest(source),agent_calls=0))
