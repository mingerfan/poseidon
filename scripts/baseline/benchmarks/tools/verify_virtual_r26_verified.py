import subprocess,time
from hecate_python_env import WORK,VENV
from benchmark_graph import require
from semantic_benchmark_execution import runtime_sources
sources=runtime_sources();start=time.monotonic()
steps=[
 ['scripts/baseline/benchmarks/tools/verify_virtual_units_r26_verified.py'],
 ['scripts/baseline/benchmarks/tools/run_upstream_virtual_batch.py','--output',str(WORK/'results/upstream-virtual-r26-verified-native')],
 ['scripts/baseline/benchmarks/tools/verify_virtual_regression_r26_verified.py'],
 ['scripts/baseline/benchmarks/tools/verify_virtual_artifacts_r26_verified.py']]
for args in steps:
 require(runtime_sources()==sources,'Source changed at batch boundary')
 subprocess.run([str(VENV/'bin/python'),'-B',*args],check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Final source changed')
