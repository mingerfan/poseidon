import subprocess,time
from hecate_python_env import WORK,VENV
from benchmark_graph import require
from semantic_benchmark_execution import runtime_sources
sources=runtime_sources();start=time.monotonic()
steps=[
 ['scripts/baseline/benchmarks/tools/verify_downsample_units_r25.py'],
 ['scripts/baseline/benchmarks/tools/run_upstream_downsample_batch.py','--output',str(WORK/'results/upstream-ds-r25-native')],
 ['scripts/baseline/benchmarks/tools/verify_downsample_regression_r25.py'],
 ['scripts/baseline/benchmarks/tools/verify_downsample_artifacts_r25.py']]
for args in steps:
 require(runtime_sources()==sources,'Source changed at batch boundary')
 subprocess.run([str(VENV/'bin/python'),'-B',*args],check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Final source changed')
