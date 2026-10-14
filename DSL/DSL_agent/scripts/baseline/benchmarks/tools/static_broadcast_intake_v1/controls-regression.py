import sys,shlex,resource,json,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
if "--inside"not in sys.argv:
    from hecate_python_env import enter_nix,VENV
    cmd='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
    raise SystemExit(enter_nix(cmd,seconds=240,environment="poseidon-cpu"))
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(150,150))
loader=unittest.TestLoader()
suite=unittest.TestSuite(loader.loadTestsFromName(name)for name in ("test_pytorch_static_broadcast","test_pytorch_static_capture","test_pytorch_spatial_source","test_pytorch_spatial_cli","test_pytorch_input_binding","test_pytorch_closed_public_arithmetic"))
result=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),backend_modified=False,api_calls=0,compiler_calls=0,encrypted_calls=0)
with (HERE/"controls-regression.json").open("x")as f:json.dump(report,f,indent=2)
raise SystemExit(not result.wasSuccessful())
