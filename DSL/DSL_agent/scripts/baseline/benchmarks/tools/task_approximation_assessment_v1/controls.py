from pathlib import Path
import sys,shlex,time,json,unittest
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
before=PoseidonBackend().identity_manifest()
modules=["test_task_approximation_assessment","test_application_metric_boundary",
         "test_task_numerical_assessment","test_task_native_assessment","test_task_dataset_assessment"]
def tests(value):
 if isinstance(value,unittest.TestSuite):
  for child in value:yield from tests(child)
 else:yield value
unique={};duplicates=0
for name in modules:
 for test in tests(unittest.defaultTestLoader.loadTestsFromName(name)):
  if test.id() in unique:duplicates+=1
  else:unique[test.id()]=test
started=time.monotonic()
with (HERE/"controls.log").open("w") as log:
 result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite(unique.values()))
report=dict(format="poseidon-reference-only-assessment-controls-v1",
 status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 duplicate_test_ids_removed=duplicates,seconds=time.monotonic()-started,
 sdk_unchanged=PoseidonBackend().identity_manifest()==before,sdk=before["manifest_sha256"],
 failures_by_id=[test.id() for test,_ in result.failures+result.errors],
 log_sha256=q.sha(HERE/"controls.log"),api_calls=0,compiler_calls=0,encrypted_calls=0,
 source_files={str(B/name):q.sha(B/name) for name in ("task_approximation_assessment.py","test_task_approximation_assessment.py")})
report["binding"]=q.digest(report);q.write(HERE/"controls.json",report)
print(json.dumps(report),flush=True)
raise SystemExit(0 if result.wasSuccessful() and report["sdk_unchanged"] else 1)
