"""Bind current calibration runner to unchanged true-Agent source tasks."""
from pathlib import Path
import sys,shlex,json,time,shutil
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=600,environment="poseidon-cpu"))
import numpy as np
from task_source_calibration import TaskSourceCalibration
from source_calibration import implementation
from source_calibration_contract import FORMAT
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_backend import PoseidonBackend
controls=q.read(HERE/"controls-r5.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==24,"current_controls")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
sdk=PoseidonBackend().identity_manifest();q.require(sdk["manifest_sha256"]==prior["sdk"],"existing_program_identity")
OUT=q.RESULTS/"agent-task-source-calibration-r1623"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-task-calibration-scope-v1",authorization="human_explicit_current_plan_test_approval",
 cases=[dict(id="matrix",argument=0),dict(id="scalar_intermediate",argument=1)],
 sdk=sdk["manifest_sha256"],implementation=implementation(),controls_binding=controls["binding"],
 max_native_workers=2,budget=dict(wall_seconds=600,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 source_files={str(B/"task_source_calibration.py"):q.sha(B/"task_source_calibration.py"),
  str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())},
 api_calls=0,compiler_calls=0,encrypted_calls=0,network_changed=False,backend_changed=False)
scope["binding"]=q.digest(scope);q.write(HERE/"task-scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(directory):
 return {str(p.relative_to(directory)):q.sha(p)for p in directory.rglob("*")if p.is_file()and p.name!=".env"}
rows=[]
for case in scope["cases"]:
 name=case["id"];folder=OUT/name;folder.mkdir(mode=0o700)
 old=Path(prior["campaign"])/name;task=q.read(old/"task-id.json")["task_id"];store=old/"tasks"
 previous_hashes=hashes(store/task);app=application(WorkerContext.capture(store))
 outcome=app.result(task);request=app._checked_task_metadata(task)["request"]
 rng=np.random.default_rng(2026101419)
 samples=[{entry["name"]:rng.uniform(-.6,.6,entry["shape"]).tolist()for entry in request["model"]["inputs"]}for _ in range(3)]
 selection=dict(format=FORMAT,purpose="calibration",selectors=[dict(id="task_inputs",module="",
  argument=case["argument"],projection=dict(kind="tensor"),domain=[-.1,.1])])
 frame=TaskSourceCalibration(app).observe(task,samples,folder/"observation",
  expected_package_binding=outcome["package"]["binding"],selection=selection,
  expected_selection_sha256=q.digest(selection),input_dtypes={i["name"]:"float64"for i in request["model"]["inputs"]},
  trust_local_code=True)
 q.write(folder/"frame.json",frame)
 q.require(frame["status"]=="observed"and frame["calibration_observed"] is True,"application_source_observation")
 q.bound(frame);report=frame["calibration_report"];q.bound(report)
 name_input=request["model"]["inputs"][case["argument"]]["name"]
 expected=np.concatenate([np.asarray(s[name_input],dtype=np.float64).reshape(-1)for s in samples])
 observed=report["observations"]["task_inputs"]
 q.require(observed["values"]==expected.size and observed["minimum"]==float(expected.min())and
  observed["maximum"]==float(expected.max()),"actual_bound_input_observation")
 q.require(frame["package_validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True)and
  frame["encrypted_execution"] is False and frame["reference_modified"] is False,"grade_not_replaced")
 q.require(app.result(task)==outcome and hashes(store/task)==previous_hashes,"original_true_agent_program_unchanged")
 q.require(implementation()==scope["implementation"]and PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
 row=dict(case=name,status="passed",samples=3,observed_values=observed["values"],domain_status=observed["domain_status"],
  original_program_unchanged=True,frame_binding=frame["binding"],report_binding=report["binding"],
  new_native_worker=1,api_calls=0,compiler_calls=0,encrypted_calls=0)
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(format="poseidon-task-source-calibration-acceptance-v1",passed=2,failed=0,skipped=0,cases=rows,
 samples=6,new_native_workers=2,api_calls=0,compiler_calls=0,encrypted_calls=0,sdk=scope["sdk"],
 scope_binding=scope["binding"],bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"task-acceptance.json",report);q.write(OUT/"report.json",report)
