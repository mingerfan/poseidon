from pathlib import Path
import sys,shlex,unittest,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 import run as live;live.no_active()
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
import qualification_scope as q
modules=["test_capacity_weighted_support","test_capacity_nested_masks","test_capacity_neutral_factors","test_capacity_normalization_lowering","test_source_layernorm_calibration","test_source_module_calibration","test_source_module_service","test_source_approximation_preparation","test_source_fx_observer","test_source_calibration","test_source_preparation_service"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in modules)
with(HERE/"controls-weighted.log").open("x")as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
record=dict(format="poseidon-weighted-support-controls-v1",run=result.testsRun,
 passed=result.testsRun-len(result.errors)-len(result.failures)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 tested_modules=modules,api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=q.digest(record);q.write(HERE/"controls-weighted.json",record)
print(json.dumps(record));raise SystemExit(0 if result.wasSuccessful()and not result.skipped else 1)
