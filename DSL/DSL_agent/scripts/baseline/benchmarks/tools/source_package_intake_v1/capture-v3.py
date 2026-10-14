"""Bounded source-only qualification; public cases remain immutable."""
import hashlib,json,os,resource,shlex,shutil,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
sys.path[:0]=[str(BASE),str(BASE/"benchmarks/tools/static_broadcast_live_v1")]
import run as live
from benchmark_graph import digest
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2)
def verify(plan):
 assert plan["binding"]==digest({key:value for key,value in plan.items()if key!="binding"})
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan["capture_tool_sha256"]
 for rel,expected in plan["sdk_sources"].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
 for case in plan["cases"]:
  for rel,expected in case["files"].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
 return plan
plan=verify(read(HERE/"capture-plan-v3.json"));budget=plan["budget"];run=Path(plan["campaign"])
if "--inside"not in sys.argv:
 live.no_active()
 assert not run.exists()
 assert live.measure(Path(plan["results_root"]))+budget["new_artifact_bytes"]<=budget["global_result_bytes"]
 assert shutil.disk_usage(run.parent).free>=budget["disk_reserve_bytes"]
 run.mkdir(mode=0o700)
 outer=dict(plan_binding=plan["binding"],owner=live.process_identity(os.getpid()),
            started=time.time(),deadline=time.time()+budget["batch_wall_seconds"])
 write(run/"owner.json",outer)
 from hecate_python_env import enter_nix,VENV
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
  [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 outer_tree=live.OwnedTree().start()
 try:
  code=enter_nix(command,seconds=budget["batch_wall_seconds"],environment="poseidon-cpu")
 finally:
  background=outer_tree.finish()
 assert not background,"nix_background_survived"
 raise SystemExit(code)
resource.setrlimit(resource.RLIMIT_AS,(budget["native_memory_bytes"],budget["native_memory_bytes"]))
resource.setrlimit(resource.RLIMIT_CPU,(budget["native_cpu_seconds"],budget["native_cpu_seconds"]))
assert tuple(sys.version_info[:3])==(3,10,14)
# Preserve the existing child sandbox limit: its preexec_fn sets a 16 MiB hard
# file limit. A smaller inherited hard limit prevents the child from starting.
# Aggregate case/campaign/global quotas remain enforced by the owned watchdog.
assert budget["sandbox_per_file_bytes"] == 16 * 1024**2
from pytorch_ordered_intake import run as capture
OwnedTree=live.OwnedTree
owner=read(run/"owner.json")
rows=[]
for case in plan["cases"]:
 verify(plan)
 assert time.time()<owner["deadline"]
 assert live.measure(run)+budget["per_case_bytes"]<=budget["new_artifact_bytes"]
 assert live.measure(Path(plan["results_root"]))+budget["per_case_bytes"]<=budget["global_result_bytes"]
 assert shutil.disk_usage(run).free>=budget["disk_reserve_bytes"]
 target=run/case["name"]
 result={}
 stop=None
 tree=OwnedTree().start()
 started=time.monotonic()
 # The watchdog only terminates descendants owned by this source-capture task.
 import threading
 finished=threading.Event()
 def monitor():
  global stop
  while not finished.wait(0.2):
   if time.time()>=owner["deadline"]or time.monotonic()-started>=budget["per_capture_seconds"]:
    stop="time_budget"
   elif target.exists()and live.measure(target)>budget["per_case_bytes"]:
    stop="case_artifact_budget"
   elif live.measure(run)>budget["new_artifact_bytes"]or live.measure(Path(plan["results_root"]))>budget["global_result_bytes"]:
    stop="global_artifact_budget"
   if stop:
    tree.finish()
    break
 watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
 try:
  status=capture(case["manifest"],target,resource_profile="general-tensors-v1",trusted=True)
  reference=read(target/"reference-report.json")
  finite=reference["finite_reference"]
  assert reference["reference_status"]=="passed"and reference["probes"]==16
  assert reference["encrypted_execution"]is False and reference["paid_calls"]==0
  result=dict(status="passed",capture=status,groups=16,values=finite["compared_values"],
   source_format=reference["capture_metadata"]["public_value_expressions"]["source_intake_version"],
   max_reference_ratio=finite["max_reference_tolerance_ratio"],
   max_normalization_ratio=finite["max_normalization_tolerance_ratio"])
 except Exception as error:
  result=dict(status="failed",error_type=type(error).__name__,reason=str(error))
 finally:
  finished.set();watcher.join(timeout=35)
  assert not watcher.is_alive(),"budget_observer_still_running"
  background=tree.finish()
 result.update(case=case["name"],seconds=time.monotonic()-started,
  bytes=live.measure(target)if target.exists()else 0,budget_stop=stop,background_survived=bool(background))
 if stop or background or result["bytes"]>budget["per_case_bytes"]:
  result.update(status="failed",failure_layer="resource"if stop or result["bytes"]>budget["per_case_bytes"]else"environment")
 rows.append(result)
 write(run/(case["name"]+"-status.json"),result)
 print(json.dumps({key:result[key]for key in("case","status","seconds","bytes","budget_stop")}),flush=True)
 if stop or background or result["status"]!="passed":break
report=dict(format="poseidon-source-package-capture-results-v1",plan_binding=plan["binding"],
 sdk=plan["sdk"],cases=rows,planned=3,passed=sum(row["status"]=="passed"for row in rows),
 failed=sum(row["status"]!="passed"for row in rows),not_run=3-len(rows),skipped=0,
 capture_attempts=len(rows),successful_source_captures=sum(row["status"]=="passed"for row in rows),
 source_capture_calls=len(rows),provider_calls=0,compiler_calls=0,encrypted_calls=0,
 original_models_or_references_modified=False,results_bytes=live.measure(run),
 global_result_bytes=live.measure(Path(plan["results_root"])))
report["binding"]=digest(report);write(run/"report.json",report)
print(json.dumps({key:report[key]for key in("passed","failed","not_run","source_capture_calls","results_bytes","global_result_bytes")}),flush=True)
raise SystemExit(not(report["passed"]==3 and report["failed"]==report["not_run"]==0))
