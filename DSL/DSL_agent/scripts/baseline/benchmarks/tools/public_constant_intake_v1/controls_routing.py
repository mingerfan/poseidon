"""One pinned Nix environment for public-transform and prior-policy controls."""
import sys,json,shlex,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
    from hecate_python_env import VENV,enter_nix
    command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
names=["test_pytorch_source_contract","test_pytorch_model_intake","test_pytorch_source_package","test_pytorch_source_package_intake","test_pytorch_input_binding","test_pytorch_static_broadcast","test_application_capabilities","test_application_model_source_context"]
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful())
target=Path(__file__).with_name("controls-routing-initial.json")
assert not target.exists();target.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
raise SystemExit(0 if result.wasSuccessful()else 1)
