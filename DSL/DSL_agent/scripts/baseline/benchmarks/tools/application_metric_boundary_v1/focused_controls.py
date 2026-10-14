from pathlib import Path
import sys,shlex,json,time,unittest
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='PYTHONPATH=scripts/baseline LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
modules=["test_application_metric_boundary","test_task_numerical_assessment","test_task_native_assessment","test_task_dataset_assessment"]
def flatten(suite):
 for item in suite:
  if isinstance(item,unittest.TestSuite):yield from flatten(item)
  else:yield item
seen=set();suite=unittest.TestSuite();duplicates=0
for name in modules:
 for test in flatten(unittest.defaultTestLoader.loadTestsFromName(name)):
  if test.id()in seen:duplicates+=1;continue
  seen.add(test.id());suite.addTest(test)
started=time.monotonic()
with(HERE/"focused-controls.log").open("x")as stream:
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
record=dict(format="poseidon-application-metric-focused-controls-v1",status="passed"if result.wasSuccessful()else"failed",
 tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
 duplicate_test_ids_removed=duplicates,modules=modules,seconds=time.monotonic()-started,
 log_sha256=q.sha(HERE/"focused-controls.log"),api_calls=0,compiler_calls=0,encrypted_calls=0,
 prior_broad_controls_timeout_retained=True)
record["binding"]=q.digest(record);q.write(HERE/"focused-controls.json",record)
print(json.dumps(record))
raise SystemExit(0 if result.wasSuccessful()else 1)
