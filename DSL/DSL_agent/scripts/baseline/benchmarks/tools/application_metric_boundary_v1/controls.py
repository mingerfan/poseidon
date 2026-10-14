from pathlib import Path
import sys,shlex,subprocess,time,re,json
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"static_broadcast_live_v1"),str(T/"tensor_contraction_live_v1")]
import run as live,qualification_scope as q
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 from application_backend import PoseidonBackend
 q.write(HERE/"identity-before-controls.json",PoseidonBackend().identity_manifest())
 command='PYTHONPATH=scripts/baseline LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=300,environment="poseidon-cpu"))
modules=["test_application_metric_boundary","test_task_numerical_assessment","test_task_native_assessment",
 "test_task_dataset_assessment","test_task_dataset_evaluation","test_application_assessment",
 "test_assessment_array_dataset","test_assessment_time_budget","test_source_precision_reference"]
start=time.monotonic()
with(HERE/"controls.log").open("xb")as stream:
 child_result=subprocess.run([sys.executable,"-B","-m","unittest",*modules,"-v"],stdout=stream,stderr=subprocess.STDOUT,timeout=240)
text=(HERE/"controls.log").read_text();match=re.search(r"Ran (\d+) tests",text)
record=dict(format="poseidon-application-metric-existing-controller-regression-v1",
 status="passed"if child_result.returncode==0 else"failed",returncode=child_result.returncode,tests=int(match.group(1))if match else None,
 modules=modules,seconds=time.monotonic()-start,log_sha256=q.sha(HERE/"controls.log"),api_calls=0,compiler_calls=0,encrypted_calls=0)
record["binding"]=q.digest(record);q.write(HERE/"controls.json",record)
print(json.dumps(record));print(text[-3800:])
raise SystemExit(child_result.returncode)
