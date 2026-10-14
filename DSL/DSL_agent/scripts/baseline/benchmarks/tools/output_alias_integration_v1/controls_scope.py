import sys,shlex,json,unittest,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2]
sys.path[:0]=[str(HERE),str(BASE)]
if "--inside"not in sys.argv:
 from hecate_python_env import VENV,enter_nix
 cmd='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(cmd,seconds=120,environment="poseidon-cpu"))
started=time.monotonic()
suite=unittest.defaultTestLoader.loadTestsFromName("test_qualification_scope")
result=unittest.TextTestRunner(verbosity=1).run(suite)
record=dict(format="poseidon-output-alias-execution-controls-v1",status="passed"if result.wasSuccessful()and not result.skipped else"failed",
 tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,compiler_calls=0,encrypted_calls=0,api_calls=0)
with(HERE/"controls-scope.json").open("x")as f:json.dump(record,f,indent=2)
print(json.dumps(record));raise SystemExit(0 if record["status"]=="passed"else 1)
