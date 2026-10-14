"""Rebind only the already proposed 15+60 live cases. Preparation is offline."""
import hashlib,json,os,sys,time,subprocess,shutil,traceback,importlib.util
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import canonical,digest,require
from pytorch_model_intake import run as capture
from semantic_benchmark_execution import runtime_sources
from candidate_bundle import load_model_provenance,check_provenance
from deepseek_provider import public_request
from component_contract import prepare_task
from hecate_python_env import VENV
work=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
out=work/"agent-pending-live-preparation-r1084"
require(not out.exists(),"Fresh offline live-preparation directory")
require(os.environ.get("IN_NIX_SHELL")=="pure" and Path(sys.prefix)==VENV,"Pinned offline preparation")
out.mkdir(mode=0o700);(out/"imports").mkdir();(out/"plans").mkdir()
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
approved=load(ROOT/"docs/baseline/agent-common-pytorch-offline-proposal-r1081.json")
sources=approved["source_hashes"]
tool=ROOT/"scripts/baseline/benchmarks/tools/run_application_matrix.py"
spec=importlib.util.spec_from_file_location("existing_application_evaluation",tool)
evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
started=time.monotonic();rows=[];parents={}
def boundary():
 require(time.monotonic()-started<600,"Offline preparation time budget")
 require(runtime_sources()==sources,"Frozen production source identity")
 require(shutil.disk_usage(work).free>=4*1024**3,"Disk reserve")
 require(int(subprocess.check_output(["du","-sb",str(out)],text=True,timeout=20).split()[0])<=256*1024**2,"Preparation artifact bound")
 require(int(subprocess.check_output(["du","-sb",str(work)],text=True,timeout=30).split()[0])<=32*1024**3,"Global artifact bound")
def save(status):
 report=dict(format="poseidon-approved-pending-live-preparation-v1",status=status,planned=76,rows=rows,
  prepared=sum(r["status"]=="prepared" for r in rows),blocked=sum(r["status"]=="blocked" for r in rows),
  not_run=76-len(rows),source_hashes=sources,parents=parents,seconds=time.monotonic()-started,
  api_calls=0,backend_changes=0,model_changes=0,reference_changes=0,tolerance=dict(atol=1e-5,rtol=1e-4))
 (out/"report.json").write_bytes(canonical(report))
groups={}
save("running")
try:
 for group,parent_name in (("functional15","agent-pytorch-functional-live-proposal-r675.json"),
                           ("matrix_instance60","agent-pytorch-general-input-live-proposal-r848.json")):
  parent_path=ROOT/"docs/baseline"/parent_name;parent=load(parent_path)
  parents[group]=dict(path=str(parent_path),sha256=sha(parent_path),
    old_binding=parent.get("binding",parent.get("proposal_sha256")),
    cases=len(parent["cases"]),generation=parent.get("generation",parent.get("provider")),
    original_limits=parent.get("resources",parent.get("limits")))
  groups[group]=[]
  for item in parent["cases"]:
   boundary();name=item["id"];old=Path(item.get("imported_directory",item.get("imported")))
   manifest=Path(item.get("manifest",old/"trusted-python/manifest.json"))
   original_model=load(old/"model.json");original_high=load(old/"original.json")
   row=dict(id=name,group=group,status="preparing",original_import=str(old),manifest=str(manifest),
      original_model_sha256=digest(original_model),original_high_sha256=digest(original_high))
   rows.append(row);save("running")
   fresh=out/"imports"/name
   try:
    result=capture(manifest,fresh,resource_profile=original_model.get("resource_profile","legacy-v1"),trusted=True)
    model=load(fresh/"model.json");high=load(fresh/"original.json")
    require(model==original_model,"Original approved normalized model changed")
    require(high==original_high,"Original approved high-level model changed")
    provenance=load_model_provenance(fresh,model);check_provenance(provenance,model)
    options=item["options"];request=prepare_task(model,options)
    require(request["request_id"]==item["request_id"],"Original approved request changed")
    context=public_request(request)
    expected=item.get("public_context_sha256",item.get("public_request_sha256"))
    require(digest(context)==expected,"Original approved outbound content changed")
    frozen=dict(id=name,imported_directory=str(fresh),options=options)
    record=evaluation.snapshot_case(frozen)[0]
    row.update(status="prepared",current_import=str(fresh),request_id=request["request_id"],
       public_context_sha256=digest(context),public_context_bytes=len(canonical(context)),
       capture=result,current_snapshot=record)
    groups[group].append(frozen)
   except (ValueError,TypeError,KeyError,IndexError,AssertionError) as exc:
    row.update(status="blocked",layer="input_preparation",error_type=type(exc).__name__,error=str(exc)[:800])
   save("running");print(json.dumps({k:row[k] for k in ("id","status")}),flush=True)
 parent_path=ROOT/"docs/baseline/poseidon-attention-feedback-live-proposal-r536.json";parent=load(parent_path)
 item=parent["cases"][0];name=item["id"];group="attention1"
 parents[group]=dict(path=str(parent_path),sha256=sha(parent_path),old_binding=parent["binding"],cases=1,
     generation=parent["provider"],original_limits=parent["limits"])
 groups[group]=[];row=dict(id=name,group=group,status="preparing",manifest=str(ROOT/item["case_file"]))
 rows.append(row);save("running")
 try:
  boundary();model_path=ROOT/item["case_file"];model=load(model_path)
  require(sha(model_path)==item["case_file_sha256"] and digest(model)==item["model_sha256"],"Original Attention model unchanged")
  options=dict(compiler_configuration="poseidon-cpu-eva-w45-v1",generation_guidance="explicit-v10")
  request=prepare_task(model,options)
  old_parent=load(ROOT/"docs/baseline/poseidon-whole-models-live-proposal-r368.json")
  old_case=next(c for c in old_parent["cases"] if c["id"]==name)
  old_request=load(Path(old_case["prepare_evidence"])/"request.json")
  require(request["request_id"]==item["request_id"] and request==old_request,"Original Attention mathematical request unchanged")
  from benchmark_math import evaluate as math_ref
  from benchmark_torch import evaluate as torch_ref
  from benchmark_graph import samples
  import numpy as np
  for inputs in samples(model,16):
   a,b=math_ref(model,inputs),torch_ref(model,inputs)
   for key in a:np.testing.assert_allclose(a[key],b[key],atol=2e-12,rtol=2e-12)
  context=public_request(request)
  record=dict(id=name,model_file=str(model_path),model_sha256=digest(model),model_file_sha256=sha(model_path),
     request_id=request["request_id"],public_context_sha256=digest(context),options=options)
  row.update(status="prepared",current_snapshot=record,public_context_sha256=digest(context),reference_groups=16)
  groups[group].append(record)
 except (ValueError,TypeError,KeyError,IndexError,AssertionError) as exc:
  row.update(status="blocked",layer="input_preparation",error_type=type(exc).__name__,error=str(exc)[:800])
 save("running");print(json.dumps({k:row[k] for k in ("id","status")}),flush=True)
 for group,items in groups.items():
  for shard,start in enumerate(range(0,len(items),48),1):
   boundary()
   if group=="attention1":
    from application_backend import PoseidonBackend
    plan=dict(format="poseidon-approved-attention-evaluation-v1",cases=items[start:start+48],
       source_hashes=sources,backend_identity=PoseidonBackend().identity(),
       provider=evaluation.PROVIDER,limits=evaluation.BOUNDS,max_generations=4,max_http_attempts=16)
    plan["binding"]=digest(plan)
   else:plan=evaluation.freeze(items[start:start+48])
   path=out/"plans"/(group+"-"+str(shard)+".json");path.write_bytes(canonical(plan))
 boundary()
 summary=dict(format="poseidon-current-pending-live-approval-v1",
  approval_text="批准当前计划内积累的所有待批准执行",approval_source="Direct human request",
  groups={k:dict(parent=parents[k],prepared_cases=[i["id"] for i in v],
     shared_batch_seconds=43200,max_generations=4*parents[k]["cases"],max_http_attempts=16*parents[k]["cases"],
     plans=[dict(path=str(p),binding=load(p)["binding"],sha256=sha(p)) for p in sorted((out/"plans").glob(k+"-*.json"))])
     for k,v in groups.items()},
  source_hashes=sources,case_changes=0,public_request_changes=0,api_calls_so_far=0,
  superseded_proposals=["agent-pytorch-state-live-proposal-r635.json","agent-pytorch-matrix-live-proposal-r816.json"],
  provider_documentation=["https://api-docs.deepseek.com/api/create-chat-completion/","https://api-docs.deepseek.com/guides/thinking_mode/"],
  tool_hashes={str(p):sha(p) for folder in (ROOT/"scripts/baseline/benchmarks/tools/approved_pending_live_prepare_v2", ROOT/"scripts/baseline/benchmarks/tools/approved_pending_execution_v1") for p in folder.glob("*.py")},
  no_monetary_ceiling=True,timeout_retries_may_duplicate_billing=True,
  excluded_outbound=["Python source","rule DSL answers","private test inputs","reference outputs","credentials","secret keys"])
 summary["binding"]=digest(summary);(out/"approval.json").write_bytes(canonical(summary))
 with (out/"output-delivery-read-only.log").open("x") as log:
  child=subprocess.run([sys.executable,"-B",str(ROOT/"scripts/baseline/benchmarks/tools/approved_pending_execution_v1/output_check.py")],stdout=log,stderr=subprocess.STDOUT,timeout=55)
 require(child.returncode==0,"Current read-only package delivery check")
 save("complete");print(json.dumps(dict(status="complete",prepared=sum(len(x) for x in groups.values()),blocked=sum(r["status"]=="blocked" for r in rows),paid_calls=0,approval_binding=summary["binding"])),flush=True)
except BaseException:
 save("stopped");raise
