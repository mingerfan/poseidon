"""Capability and application regressions in the existing fixed Python environment."""
import sys, json, shlex, resource, unittest, hashlib, time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside" not in sys.argv:
    from hecate_python_env import enter_nix,VENV
    command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
names=("test_application_source_versions","test_application_capabilities",
       "test_pytorch_input_binding","test_application_component",
       "test_application_diagnostics","test_staged_application_reuse")
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(format="poseidon-source-capabilities-controls-v1",tests=result.testsRun,
 passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,status="passed"if result.wasSuccessful()else"failed",
 test_source_hashes={name:hashlib.sha256((BASE/(name+".py")).read_bytes()).hexdigest()for name in names},
 api_calls=0,source_capture_calls=0,compiler_calls=0,encrypted_calls=0,
 scope="Capability declarations and injected Fake Backend/Provider; no model capture or real execution.")
with(HERE/"controls-first.json").open("x")as f:json.dump(report,f,sort_keys=True,indent=2)
print(json.dumps(report,ensure_ascii=False),flush=True)
raise SystemExit(not result.wasSuccessful())
