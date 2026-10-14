"""Rebind unchanged native observations to current collector; no worker execution."""
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
from task_calibration_dataset import TaskCalibrationDataset,identity
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_backend import PoseidonBackend
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
old=q.read(HERE/"acceptance.json");q.bound(old)
controls=q.read(HERE/"controls-final.json");q.bound(controls)
q.require(controls["tests"]==47 and controls["status"]=="passed","controls")
O=q.RESULTS/"agent-task-calibration-dataset-r1627"
OUT=q.RESULTS/"agent-calibration-dataset-current-audit-r1628"
q.require(not OUT.exists(),"fresh_campaign")
def hashes(root):
 return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
before=hashes(O)
q.require(live.measure(q.RESULTS)+64*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+64*1024**2,"storage")
OUT.mkdir(mode=0o700)
frozen=identity();sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==old["sdk"]==controls["sdk"],"sdk")
scope=dict(format="poseidon-calibration-current-audit-scope-v1",collector_identity=frozen,
 original_acceptance_binding=old["binding"],controls_binding=controls["binding"],
 authorization="human_current_plan_all_tests_approved",api_calls=0,compiler_calls=0,
 encrypted_calls=0,native_workers=0,reused_native_observations=6,
 budget=dict(wall_seconds=600,new_bytes=64*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 source_files={str(B/"task_calibration_dataset.py"):q.sha(B/"task_calibration_dataset.py"),
  str(B/"test_task_calibration_dataset.py"):q.sha(B/"test_task_calibration_dataset.py"),
  str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
scope["binding"]=q.digest(scope);q.write(HERE/"audit-scope.json",scope);q.write(OUT/"scope.json",scope)
start=time.monotonic();rows=[]
for entry in old["cases"]:
 name=entry["case"];folder=O/name
 plan=q.read(folder/"plan.json");q.bound(plan)
 source=Path(prior["campaign"])/name;task=q.read(source/"task-id.json")["task_id"]
 store=source/"tasks";task_before=hashes(store/task)
 app=application(WorkerContext.capture(store))
 args=dict(expected_dataset_binding=plan["dataset_binding"],calibration_declaration=plan["declaration"],
  expected_declaration_sha256=plan["declaration_sha256"],expected_package_binding=plan["anchor"]["bundle_binding"],
  selection=plan["selection"],expected_selection_sha256=plan["selection_sha256"],
  input_dtypes=plan["input_dtypes"],part_samples=16,trust_local_code=True)
 collector=TaskCalibrationDataset(app,task,folder/"dataset/dataset.json",**args)
 current=OUT/name;current.mkdir(mode=0o700);q.write(current/"plan.json",collector.plan)
 for index,part_id in enumerate(collector.parts):
  frame=q.read(folder/("frame_"+str(index)+".json"));q.bound(frame)
  collector.record(part_id,frame,expected_binding=frame["binding"])
 summary=collector.summary();q.bound(summary);q.write(current/"summary.json",summary)
 previous=q.read(folder/"summary.json");q.bound(previous)
 q.require(summary["complete"]and summary["observations"]==previous["observations"]and
  summary["observed_samples"]==17 and summary["package_validation"]==previous["package_validation"],
  "current_collector_equivalent_statistics")
 q.require(hashes(store/task)==task_before and identity()==frozen and
  PoseidonBackend().identity_manifest()==sdk,"unchanged_program_and_sdk")
 rows.append(dict(case=name,status="passed",samples=17,
  old_summary_binding=previous["binding"],current_summary_binding=summary["binding"],
  reused_observation_reports=2,new_native_workers=0))
 print(json.dumps(rows[-1]),flush=True)
 q.require(time.monotonic()-start<600,"wall_budget")
q.require(hashes(O)==before,"historical_native_evidence_unchanged")
report=dict(format="poseidon-calibration-current-audit-v1",status="passed",
 scope_binding=scope["binding"],original_acceptance_binding=old["binding"],
 controls_binding=controls["binding"],collector_identity=frozen,sdk=old["sdk"],
 passed=3,failed=0,skipped=0,samples=51,reused_native_observations=6,
 new_native_workers=0,api_calls=0,compiler_calls=0,encrypted_calls=0,
 original_native_evidence_unchanged=True,seconds=time.monotonic()-start,cases=rows)
report["binding"]=q.digest(report);q.write(HERE/"audit.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
