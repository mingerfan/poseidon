from pathlib import Path
import sys,shlex,time,json,unittest
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=240,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
sdk=PoseidonBackend().identity_manifest();start=time.monotonic()
names=["test_source_fx_observer","test_source_fx_calibration_protocol","test_source_calibration","test_task_source_calibration","test_task_calibration_dataset"]
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in names)
files=["source_fx_observer.py","source_calibration.py","source_calibration_contract.py",
 "source_calibration_observer.py","task_source_calibration.py","task_calibration_dataset.py","test_source_fx_observer.py","test_source_fx_calibration_protocol.py"]
frozen={str(B/n):q.sha(B/n)for n in files}
with (HERE/"controls-dtype.log").open("x")as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
report=dict(format="poseidon-fx-calibration-controls-v1",tests=result.testsRun,
 failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 status="passed"if result.wasSuccessful()and not result.skipped else "failed",
 source_files=frozen,sdk=sdk["manifest_sha256"],sdk_unchanged=PoseidonBackend().identity_manifest()==sdk,
 seconds=time.monotonic()-start,api_calls=0,compiler_calls=0,encrypted_calls=0,log_sha256=q.sha(HERE/"controls-dtype.log"))
report["binding"]=q.digest(report);q.write(HERE/"controls-dtype.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="source_files"}),flush=True)
if not result.wasSuccessful():
 for _,error in (result.failures+result.errors)[:4]:print(error[-2500:],flush=True)
raise SystemExit(0 if report["status"]=="passed"else 1)
