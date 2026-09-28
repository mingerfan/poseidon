import subprocess,time
from hecate_python_env import WORK,VENV
from benchmark_graph import require
from semantic_benchmark_execution import runtime_sources
sources=runtime_sources();start=time.monotonic()
steps=[
 ['-m','unittest','test_upstream_fused','test_upstream_spatial_mapped','-v'],
 ['scripts/baseline/benchmarks/tools/run_upstream_fused_batch.py','--output',str(WORK/'results/upstream-fused-r24-final-native')],
 ['scripts/baseline/benchmarks/tools/verify_fused_regression_r24_final.py'],
 ['scripts/baseline/benchmarks/tools/verify_fused_artifacts_r24_final.py']]
for args in steps:
 require(runtime_sources()==sources,'Source changed at batch boundary')
 subprocess.run([str(VENV/'bin/python'),'-B',*args],check=True,timeout=max(1,1800-(time.monotonic()-start)))
require(runtime_sources()==sources,'Final source changed')
