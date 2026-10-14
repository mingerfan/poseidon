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
 report=dict(format="poseidon-approved-pending-live-preparation-v1",status=status,planned=75,rows=rows,
  prepared=sum(r["status"]=="prepared" for r in rows),blocked=sum(r["status"]=="blocked" for r in rows),
  not_run=75-len(rows),source_hashes=sources,parents=parents,seconds=time.monotonic()-started,
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
 for group,items in groups.items():
  for shard,start in enumerate(range(0,len(items),48),1):
   boundary()
   plan=evaluation.freeze(items[start:start+48])
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
  no_monetary_ceiling=True,timeout_retries_may_duplicate_billing=True,
  excluded_outbound=["Python source","rule DSL answers","private test inputs","reference outputs","credentials","secret keys"])
 summary["binding"]=digest(summary);(out/"approval.json").write_bytes(canonical(summary))
 save("complete");print(json.dumps(dict(status="complete",prepared=sum(len(x) for x in groups.values()),blocked=sum(r["status"]=="blocked" for r in rows),paid_calls=0,approval_binding=summary["binding"])),flush=True)
except BaseException:
 save("stopped");raise
