from pathlib import Path
import sys,shlex,unittest,time,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=180,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
before=PoseidonBackend().identity_manifest();started=time.monotonic()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n)for n in ("test_source_calibration","test_task_source_calibration")])
with (HERE/"controls-final.log").open("w")as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
report=dict(format="poseidon-source-calibration-controls-v1",status="passed"if result.wasSuccessful()else"failed",
 tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 seconds=time.monotonic()-started,sdk=before["manifest_sha256"],sdk_unchanged=PoseidonBackend().identity_manifest()==before,
 api_calls=0,compiler_calls=0,encrypted_calls=0,
 log_sha256=q.sha(HERE/"controls-final.log"),source_files={str(B/n):q.sha(B/n)for n in
 ("source_calibration.py","source_calibration_contract.py","source_calibration_observer.py","task_source_calibration.py","test_source_calibration.py","test_task_source_calibration.py")})
report["binding"]=q.digest(report);q.write(HERE/"controls-final.json",report)
print(json.dumps(report),flush=True)
raise SystemExit(0 if result.wasSuccessful()and report["sdk_unchanged"]else 1)
