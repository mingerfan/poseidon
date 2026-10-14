"""Explicit bounded calibration-only acceptance over historical true-Agent tasks."""
from pathlib import Path
import sys,shlex,json,time,shutil,hashlib
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=900,environment="poseidon-cpu"))
import numpy as np
from task_calibration_dataset import TaskCalibrationDataset,DECLARATION,identity
from task_source_calibration import TaskSourceCalibration
from source_calibration_contract import FORMAT
from assessment_array_dataset import export_dataset
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_backend import PoseidonBackend
controls=q.read(HERE/"controls-r3.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==44,"controls")
prior=q.read(T/"tensor_contraction_live_v1/live-plan.json");q.bound(prior)
sdk=PoseidonBackend().identity_manifest()
q.require(sdk["manifest_sha256"]==prior["sdk"]==controls["sdk"],"existing_sdk")
OUT=q.RESULTS/"agent-task-calibration-dataset-r1627"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+128*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+128*1024**2,"storage")
OUT.mkdir(mode=0o700)
scope=dict(format="poseidon-calibration-dataset-scope-v1",
 authorization="human_current_plan_all_tests_approved_and_results_48GiB",
 cases=[dict(id="dot_scalar",argument=0,dtype="float64"),
        dict(id="matrix",argument=0,dtype="float32"),
        dict(id="scalar_intermediate",argument=1,dtype="float64")],
 samples_per_case=17,part_samples=16,max_native_workers=6,
 max_api_calls=0,max_compiler_calls=0,max_encrypted_calls=0,
 collector_identity=identity(),sdk=sdk["manifest_sha256"],controls_binding=controls["binding"],
 budget=dict(wall_seconds=900,new_bytes=128*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3,
             native_memory_bytes=4*1024**3,native_cpu_seconds=150),
 source_files={str(B/"task_calibration_dataset.py"):q.sha(B/"task_calibration_dataset.py"),
               str(Path(__file__).resolve()):q.sha(Path(__file__).resolve())})
scope["binding"]=q.digest(scope)
q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(directory):
 return {str(p.relative_to(directory)):q.sha(p)for p in directory.rglob("*")
         if p.is_file()and p.name!=".env"}
start=time.monotonic();rows=[]
for case in scope["cases"]:
 name=case["id"];folder=OUT/name;folder.mkdir(mode=0o700)
 old=Path(prior["campaign"])/name;task=q.read(old/"task-id.json")["task_id"]
 store=old/"tasks";old_hashes=hashes(store/task)
 app=application(WorkerContext.capture(store))
 outcome=app.result(task);meta=app._checked_task_metadata(task);request=meta["request"]
 rng=np.random.default_rng(2026101421)
 arrays={item["name"]:rng.uniform(-.6,.6,[17]+item["shape"]).astype(case["dtype"])
         for item in request["model"]["inputs"]}
 manifest=export_dataset(request,arrays,folder/"dataset")
 declared=dict(format=DECLARATION,role="calibration",dataset_binding=manifest["binding"],
               source=dict(name="fixed-seed-component-acceptance",revision="2026101421"))
 selection=dict(format=FORMAT,purpose="calibration",selectors=[dict(id="selected_input",module="",
                  argument=case["argument"],projection=dict(kind="tensor"),domain=[-.1,.1])])
 args=dict(expected_dataset_binding=manifest["binding"],calibration_declaration=declared,
           expected_declaration_sha256=q.digest(declared),expected_package_binding=outcome["package"]["binding"],
           selection=selection,expected_selection_sha256=q.digest(selection),
           input_dtypes={i["name"]:case["dtype"]for i in request["model"]["inputs"]},
           part_samples=16,trust_local_code=True)
 collector=TaskCalibrationDataset(app,task,folder/"dataset/dataset.json",**args)
 q.write(folder/"plan.json",collector.plan)
 q.require(not collector.summary()["complete"],"empty_not_passed")
 batches=collector.batches()
 for index,batch in enumerate(batches):
  q.require(time.monotonic()-start<scope["budget"]["wall_seconds"],"wall_budget")
  q.require(identity()==scope["collector_identity"]and
            PoseidonBackend().identity_manifest()==sdk,"frozen_identity")
  frame=TaskSourceCalibration(app).observe(task,batch["samples"],folder/("observation_"+str(index)),
       expected_package_binding=args["expected_package_binding"],selection=selection,
       expected_selection_sha256=q.digest(selection),input_dtypes=args["input_dtypes"],trust_local_code=True)
  q.write(folder/("frame_"+str(index)+".json"),frame)
  q.require(frame.get("status")=="observed","source_calibration_observed")
  collector.record(batch["id"],frame,expected_binding=frame["binding"])
  if index==0:
   partial=collector.summary();q.write(folder/"partial.json",partial)
   q.require(not partial["complete"]and partial["observed_samples"]==16,"partial_scope")
   cache=collector.receipts;q.write(folder/"cached-receipts.json",cache)
   collector=TaskCalibrationDataset(app,task,folder/"dataset/dataset.json",cached_receipts=cache,**args)
   q.require(collector.summary()==partial,"receipt_resume_without_reexecution")
 summary=collector.summary();q.write(folder/"summary.json",summary)
 q.require(summary["complete"]and summary["observed_samples"]==17 and summary["completed_parts"]==2,"complete")
 target=request["model"]["inputs"][case["argument"]]["name"]
 values=arrays[target].astype(np.float64).reshape(-1)
 stats=summary["observations"]["selected_input"]
 q.require(stats["values"]==values.size and stats["minimum"]==float(values.min())and
  stats["maximum"]==float(values.max())and
  abs(stats["mean"]-float(values.mean()))<=1e-12+1e-12*abs(float(values.mean()))and
  abs(stats["std_population"]-float(values.std()))<=1e-12+1e-12*abs(float(values.std())),"independent_statistics")
 q.require(app.result(task)==outcome and hashes(store/task)==old_hashes,"historical_true_agent_program_unchanged")
 q.require(summary["package_validation"]==dict(level="numerical",encrypted_execution=True,numerically_validated=True)and
  summary["encrypted_execution"]is False and summary["application_accuracy_certified"]is False,"grade_scope")
 row=dict(case=name,status="passed",samples=17,parts=2,native_workers=2,dtype=case["dtype"],
          observed_values=int(values.size),domain_status=stats["domain_status"],summary_binding=summary["binding"],
          first_receipt_reused=True,historical_task_unchanged=True)
 rows.append(row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=scope["budget"]["new_bytes"],"batch_space")
report=dict(format="poseidon-calibration-dataset-acceptance-v1",scope_binding=scope["binding"],
            controls_binding=controls["binding"],collector_identity=identity(),sdk=scope["sdk"],
            passed=3,failed=0,skipped=0,samples=51,native_workers=6,
            api_calls=0,compiler_calls=0,encrypted_calls=0,cases=rows,seconds=time.monotonic()-start,
            bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
