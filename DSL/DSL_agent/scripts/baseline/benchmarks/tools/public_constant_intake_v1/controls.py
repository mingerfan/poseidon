"""No API/backend execution. Capture/control tests in the pinned environment."""
import sys,json,time,shlex,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside",*sys.argv[1:]])
 raise SystemExit(enter_nix(cmd,seconds=300,environment="poseidon-cpu"))
names=["test_public_constants","test_pytorch_public_constant_capture",
       "test_public_tensor_transforms","test_pytorch_public_transform_capture",
       "test_pytorch_public_dtype_cast","test_pytorch_input_binding",
       "test_application_capabilities"]
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
out=dict(format="poseidon-public-constant-controls-v1",tests=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 seconds=time.monotonic()-started,api_calls=0,compiler_calls=0,encrypted_calls=0)
path=HERE/("controls"+sys.argv[sys.argv.index("--suffix")+1]+".json"if "--suffix"in sys.argv else"controls-initial.json")
with path.open("x")as f:json.dump(out,f,indent=2)
print(json.dumps(out))
raise SystemExit(0 if out["status"]=="passed"else 1)
