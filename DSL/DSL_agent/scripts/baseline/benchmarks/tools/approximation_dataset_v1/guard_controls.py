from pathlib import Path
import sys,shlex,unittest,time,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=420,environment="poseidon-cpu"))
from application_backend import PoseidonBackend
identity=PoseidonBackend().identity_manifest()
modules=["test_approximation_dataset_assessment","test_application_approximation_assessment"]
def flatten(suite):
 if isinstance(suite,unittest.TestSuite):
  for item in suite:yield from flatten(item)
 else:yield suite
unique={};duplicates=0
for module in modules:
 for test in flatten(unittest.defaultTestLoader.loadTestsFromName(module)):
  if test.id()in unique:duplicates+=1
  else:unique[test.id()]=test
started=time.monotonic()
with (HERE/"guard-controls.log").open("w")as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(unique.values()))
report=dict(format="poseidon-approximation-dataset-controls-v1",
 status="passed"if result.wasSuccessful()else "failed",tests=result.testsRun,
 failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 duplicate_test_ids_removed=duplicates,seconds=time.monotonic()-started,
 sdk=identity["manifest_sha256"],sdk_unchanged=PoseidonBackend().identity_manifest()==identity,
 synthetic_rows_are_not_cifar_evidence=True,
 failed_ids=[test.id()for test,_ in result.failures+result.errors],
 api_calls=0,compiler_calls=0,encrypted_calls=0,native_reference_calls=0,
 log_sha256=q.sha(HERE/"guard-controls.log"),
 source_files={str(B/name):q.sha(B/name)for name in ("approximation_dataset_assessment.py",
  "application_approximation_assessment.py","test_approximation_dataset_assessment.py",
  "test_application_approximation_assessment.py")})
report["binding"]=q.digest(report);q.write(HERE/"guard-controls.json",report)
print(json.dumps(report),flush=True)
raise SystemExit(0 if result.wasSuccessful() and report["sdk_unchanged"]else 1)
