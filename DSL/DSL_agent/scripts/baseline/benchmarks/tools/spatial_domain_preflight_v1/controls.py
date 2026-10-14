"""Affected domain/execution/identity controls in the existing Nix Python."""
import hashlib,json,shlex,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(ROOT/"scripts"),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
from process_resource_limits import limits
limits()
names=("test_execution_input_domain_spatial","test_execution_input_domain","test_operator_spatial_graph",
 "test_operator_arithmetic_graph","test_program_execution","test_application_component","test_staged_application_reuse")
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
def ids(value):
    if isinstance(value,unittest.TestSuite):return[i for item in value for i in ids(item)]
    return[value.id()]
test_ids=ids(suite);assert len(test_ids)==len(set(test_ids))
started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(format="poseidon-spatial-domain-controls-v1",status="passed"if result.wasSuccessful()else"failed",
 tests=result.testsRun,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,test_ids=test_ids,
 source_hashes={name:hashlib.sha256((BASE/(name+".py")).read_bytes()).hexdigest()for name in names},
 checker_sha256=hashlib.sha256((BASE/"execution_input_domain.py").read_bytes()).hexdigest(),
 fixture_sha256=hashlib.sha256((HERE/"model-cases.json").read_bytes()).hexdigest(),
 source_capture_calls=0,compiler_calls=0,encrypted_calls=0,provider_calls=0,
 scope="Real domain/reference evaluation plus injected application packing/runtime controls; no FHE")
from benchmark_graph import digest
report["binding"]=digest(report)
with(HERE/"controls.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps({k:report[k]for k in("binding","status","tests","passed","failed","errors","skipped","seconds")}),flush=True)
raise SystemExit(not result.wasSuccessful())
