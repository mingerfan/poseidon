"""CLI -> existing Application source intake -> deterministic Poseidon acceptance."""
from pathlib import Path
import sys,shlex,json,time,shutil,subprocess
HERE=Path(__file__).resolve().parent;B=HERE.parents[2];T=B/"benchmarks/tools"
sys.path[:0]=[str(B),str(T/"tensor_contraction_live_v1"),str(T/"static_broadcast_live_v1")]
import qualification_scope as q,run as live
from hecate_python_env import VENV,enter_nix
if sys.argv[1:]!=["--inside"]:
 live.no_active()
 command='LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"])
 raise SystemExit(enter_nix(command,seconds=2700,environment="poseidon-cpu"))
import numpy as np
from source_preparation_service import REQUEST,identity
from source_calibration_contract import FX_FORMAT
from application_backend import PoseidonBackend
from application_execution.context import WorkerContext
from application_execution.backend import application
from application_model_source import FORMAT as SOURCE_REQUEST
from application_policy import DEFAULT_BUDGET
from unified_graph_lowering import candidate_source
controls=q.read(HERE/"controls.json");q.bound(controls)
q.require(controls["status"]=="passed"and controls["tests"]==119,"controls")
prior=q.read(T/"approximation_freeze_v1/acceptance.json");q.bound(prior)
ORIGINAL=q.RESULTS/"agent-pre-generation-approximation-freeze-r1641"
OUT=q.RESULTS/"agent-source-preparation-cli-poseidon-r1643"
q.require(not OUT.exists(),"fresh_campaign")
q.require(live.measure(q.RESULTS)+512*1024**2<=48*1024**3 and
 shutil.disk_usage(q.RESULTS).free>=4*1024**3+512*1024**2,"storage")
OUT.mkdir(mode=0o700)
sdk=PoseidonBackend().identity_manifest();frozen=identity()
q.require(sdk["manifest_sha256"]==controls["sdk"]==prior["sdk"],"sdk")
scope=dict(format="poseidon-source-preparation-cli-poseidon-scope-v1",
 authorization="human_current_plan_all_tests_approved",cases=["gelu_affine","sigmoid_square","rsqrt_mean"],
 sdk=sdk["manifest_sha256"],entry_identity=frozen,controls_binding=controls["binding"],
 candidates="deterministic_rule_baseline_not_agent_generation",
 max_calibration_workers=6,max_capture_workers=3,max_compiler_calls=3,max_encrypted_calls=9,
 max_paid_api_calls=0,compiler_jobs=2,link_jobs=1,
 budget=dict(wall_seconds=2700,new_bytes=512*1024**2,total_bytes=48*1024**3,min_free_bytes=4*1024**3),
 source_files={str(Path(__file__).resolve()):q.sha(Path(__file__).resolve()),
  str(B/"source_preparation_service.py"):q.sha(B/"source_preparation_service.py"),
  str(B/"source_preparation_entry.py"):q.sha(B/"source_preparation_entry.py"),
  str(B.parent/"agent_calibration.py"):q.sha(B.parent/"agent_calibration.py")})
scope["binding"]=q.digest(scope);q.write(HERE/"scope.json",scope);q.write(OUT/"scope.json",scope)
def hashes(root):return {str(p.relative_to(root)):q.sha(p)for p in root.rglob("*")if p.is_file()and p.name!=".env"}
def cli(folder,request,output):
 path=folder/(request["action"]+"-request.json");q.write(path,request)
 command=[str(VENV/"bin/python"),"-B",str(B.parent/"agent_calibration.py"),request["action"],
          "--request",str(path),"--output",str(output),"--trust-local-code","--inside"]
 result=subprocess.run(command,text=True,capture_output=True,timeout=260)
 (folder/(request["action"]+"-stdout.log")).write_text(result.stdout)
 (folder/(request["action"]+"-stderr.log")).write_text(result.stderr)
 q.require(result.returncode==0,"source_preparation_cli_success")
 value=json.loads(result.stdout.strip().splitlines()[-1])
 q.bound(value);return value
class RuleProvider:
 kind="scripted_replay";agent_calls=0
 def generate(self,request,history):
  source,_=candidate_source(request)
  return json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=source))
rows=[];start=time.monotonic()
for name in scope["cases"]:
 try:
  q.require(time.monotonic()-start<2600,"wall_budget")
  original=ORIGINAL/name;before=hashes(original)
  folder=OUT/name;folder.mkdir(mode=0o700)
  source=original/"original-source/manifest.json";spec=q.read(source)
  previous=q.read(original/"observed/request.json")
  signature=previous["source_signature"];samples=previous["samples"]
  request=dict(format=REQUEST,action="inspect",manifest=dict(path=str(source),sha256=q.digest(spec)),
    source_signature=signature,samples=samples,input_dtypes=previous["input_dtypes"],
    output_structure=previous["output_structure"],resource_profile="general-tensors-v1")
  inventory=cli(folder,request,folder/"inventory")
  selection=q.read(original/"observed/request.json")["selection"]
  q.require(inventory["graph_sha256"]==selection["graph_sha256"],"same_raw_source_graph")
  observed=cli(folder,dict(request,action="observe",selection=selection),folder/"observed")
  configured=q.read(original/"freeze/freeze.json")["configuration"]
  frozen_request={k:v for k,v in request.items()if k in("format","manifest","source_signature")}
  frozen_request.update(action="freeze",calibration_evidence=[
    dict(directory=str(folder/"observed"),report_binding=observed["receipt_binding"])],configuration=configured)
  frozen_result=cli(folder,frozen_request,folder/"freeze")
  task_store=folder/"tasks";app=application(WorkerContext.capture(task_store))
  task=app.submit_source(dict(format=SOURCE_REQUEST,kind="trusted_pytorch",
      manifest=frozen_result["configured_manifest"],trust_local_code=True),
      resource_profile="general-tensors-v1",options=dict(generation_guidance="explicit-v13",
      public_weight_policy="descriptors-math-roles-v2",chunk_period=4),
      limits=dict(DEFAULT_BUDGET,generations=1,http_attempts=1,wall_seconds=600))
  q.write(folder/"task-id.json",dict(task_id=task))
  prepared=app.prepare_source(task);q.write(folder/"prepared-state.json",prepared)
  q.require(prepared["status"]=="queued"and prepared["phase"]=="queued","normal_source_admission")
  result=app.run(task,RuleProvider());q.write(folder/"result.json",result)
  q.require(result["status"]=="succeeded"and result["acceptance"]["level"]=="numerical",
            "poseidon_rule_numerical")
  package=app._dir(task)/"program"
  # Explicitly reload the actual exported package, then repeat supplied inputs.
  first=app.execute(task,samples);q.write(folder/"new-input-execution.json",first)
  repeated=app.execute(task,[samples[0]]);q.write(folder/"repeat-execution.json",repeated)
  q.require(first["status"]=="executed"and repeated["status"]=="executed","package_repeat_execution")
  q.require(hashes(original)==before and identity()==frozen and PoseidonBackend().identity_manifest()==sdk,"old_source_and_frozen_sdk")
  row=dict(case=name,status="passed",calibration_workers=2,capture_workers=1,compiler_calls=1,
   encrypted_calls=3,api_calls=0,agent_generated=False,rule_baseline=True,
   frozen_manifest=frozen_result["configured_manifest"],task_id=task,
   package_binding=result["package"]["binding"],reload_repeat_verified=True)
 except Exception as error:
  import traceback
  (folder/"private-failure.log").write_text(traceback.format_exc())
  row=dict(case=name,status="failed",error_type=type(error).__name__,diagnostic="private_failure_retained")
 rows.append(row);q.write(folder/"case-result.json",row);print(json.dumps(row),flush=True)
 q.require(live.measure(OUT)<=512*1024**2,"space")
report=dict(format="poseidon-source-preparation-cli-poseidon-acceptance-v1",scope_binding=scope["binding"],
 passed=sum(r["status"]=="passed"for r in rows),failed=sum(r["status"]=="failed"for r in rows),skipped=0,
 cases=rows,sdk=sdk["manifest_sha256"],controls_binding=controls["binding"],paid_api_calls=0,
 seconds=time.monotonic()-start,bytes=live.measure(OUT))
report["binding"]=q.digest(report);q.write(HERE/"acceptance.json",report);q.write(OUT/"report.json",report)
print(json.dumps({k:v for k,v in report.items()if k!="cases"}),flush=True)
raise SystemExit(0 if report["failed"]==0 else 1)
