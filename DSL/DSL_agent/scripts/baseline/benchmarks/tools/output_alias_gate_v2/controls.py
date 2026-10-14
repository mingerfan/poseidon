"""Pinned component controls; no Provider or native execution."""
import sys,shlex,json,unittest,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE/"benchmarks/tools/static_broadcast_live_v1"),str(BASE)]
import run as live
if "--inside"not in sys.argv:
    live.no_active()
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
        [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from component_contract import reconstruct_request
old=BASE/"benchmarks/tools/public_mask_intake_v1/prepared-requests.json"
index=json.loads(old.read_text())
legacy=[]
for item in index["cases"]:
    path=Path(item["request_file"]);request=json.loads(path.read_text())
    assert reconstruct_request(request)==request
    legacy.append(request["request_id"])
modules=["test_unified_output_binding","test_poseidon_runtime_backend","test_application_diagnostics",
 "test_binding_feedback","test_static_work_feedback","test_compiled_execution_precheck",
 "test_unified_construction_guidance","test_cli_feedback_policy","test_validation_adapter"]
start=time.time();suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(format="poseidon-output-binding-controls-v1",status="passed"if result.wasSuccessful()else"failed",
 tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 modules=modules,legacy_v13_request_ids=legacy,compiler_calls=0,encrypted_calls=0,http_attempts=0,
 seconds=time.time()-start)
(HERE/"controls.json").write_text(json.dumps(record,indent=2))
raise SystemExit(0 if result.wasSuccessful()else 1)
