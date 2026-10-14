"""Freeze source-normalization requests in the existing sandbox. No paid calls."""
import copy,json,sys,time,subprocess,shlex,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
INVENTORY=HERE.parent/"pending_source_live_inventory_v1/inventory.json"
sys.path.insert(0,str(BASE))
from hecate_python_env import VENV,enter_nix,WORK
def write(path,value):
 with Path(path).open("x")as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")
def main():
 if "--inside"not in sys.argv:
  import run
  run.no_active()
  command='PYTHONPATH='+shlex.quote(str(BASE))+' LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(
    [str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]+(["--resume-remaining"]if "--resume-remaining"in sys.argv else[]))
  return enter_nix(command,seconds=210 if "--resume-remaining"in sys.argv else 600,keep_env=(),environment="poseidon-cpu")
 import scope
 from scope import CASES,PROVIDER,BOUNDS,OPTIONS,PROFILE,PROPOSAL,digest,read,sha,require
 from application_execution.context import WorkerContext
 from application_execution.backend import application
 from application_backend import PoseidonBackend
 from deepseek_provider import public_request
 from candidate_bundle import check_provenance
 from component_contract import reconstruct_request
 from pytorch_ordered_contract import source_identity
 from application_execution.context_identity import authority as source_authority
 require(not PROPOSAL.exists(),"fresh_source_proposal")
 inventory=read(INVENTORY)
 require(inventory["binding"]==digest({k:v for k,v in inventory.items()if k!="binding"}),"source_inventory_hash")
 require(inventory["source_contexts"]==51 and inventory["distinct_public_requests"]==26,"source_inventory_denominator")
 require(tuple(g["representative_context"].replace("/","__")for g in inventory["request_groups"])==CASES,"source_inventory_cases")
 for path,want in inventory["source_files"].items():require(sha(path)==want,"source_inventory_frozen_files")
 sdk=PoseidonBackend().identity_manifest()
 active_previous=read(ROOT/"docs/baseline/agent-spatial-source-live-proposal-r1407.json")
 require(sdk==active_previous["source_stage_identity"],"previous_verified_sdk")
 previous_terminal=read(WORK/"results/agent-spatial-source-live-v1/report.json")
 require(previous_terminal["status"]in("complete","stopped")and previous_terminal["calls_resolved"],"previous_campaign_terminal")
 import run
 run.no_active()
 recovery=None;recovered={}
 if "--resume-remaining"in sys.argv:
  from capture_checkpoint import validate,validate_extra,load_metadata
  recovery=read(HERE/"capture-checkpoint-first.json")
  recovered=validate(recovery,inventory,sdk,here=HERE,results=WORK/"results")
  extra=read(HERE/"capture-checkpoint-extra.json")
  recovered.update(validate_extra(extra,inventory,sdk,here=HERE,results=WORK/"results"))
 baseline_bytes=run.measure(WORK/"results");started=time.monotonic()
 wall_seconds=180 if recovery else 570
 new_bytes=64*1024**2 if recovery else 512*1024**2
 def boundary(*,check_identity=True):
  require(time.monotonic()-started<wall_seconds,"source_preparation_time_ceiling")
  total=run.measure(WORK/"results")
  require(total<=BOUNDS["global_result_bytes"]and total-baseline_bytes<=new_bytes and
   shutil.disk_usage(WORK).free>=BOUNDS["minimum_free_bytes"],"source_preparation_resource_ceiling")
  if check_identity:require(PoseidonBackend().identity_manifest()==sdk,"source_preparation_sdk_identity")
 boundary()
 request_root=HERE/"requests"
 if recovery:
  require({p.name for p in request_root.iterdir()}=={v["source_folder"].rsplit("/",1)[1]for v in recovered.values()},
   "fresh_remaining_source_contexts")
 else:request_root.mkdir()
 receipts=WORK/"results"/("agent-normalized-source-live-prepare-r1409-"+str(time.time_ns()));receipts.mkdir()
 sources=[];start=time.monotonic()
 options=dict(OPTIONS,capacity_packing_version="capacity-public-tensors-v9")
 for item in inventory["contexts"]:
  boundary(check_identity=item["context"]not in recovered)
  context=item["context"];folder=request_root/context.replace("/","__")
  previous=recovered.get(context)
  if previous:
   source=folder/"source";manifest=source/"manifest.json";m=read(manifest)
   task=previous["task_id"];state=read(Path(previous["task_directory"])/"state.json")
  else:
   folder.mkdir();source=folder/"source";source.mkdir();old=Path(item["source_manifest"]);m=copy.deepcopy(read(old))
   require(m["format"]==item["source_format"],"exact_source_contract")
   for name,want in m["files"].items():
    original=old.parent/name;require(sha(original)==want,"exact_original_source")
    dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(original.read_bytes())
   for info in m.get("weights",{}).values():
    original=old.parent/info["file"];require(sha(original)==info["sha256"],"exact_original_weight")
    dest=source/info["file"];dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():dest.write_bytes(original.read_bytes())
   manifest=source/"manifest.json";write(manifest,m)
   app=application(WorkerContext.capture(receipts/context.replace("/","__")/"tasks"))
   source_request=dict(format="poseidon-model-source-request-v1",kind="trusted_pytorch",manifest=str(manifest),trust_local_code=True)
   task=app.submit_source(source_request,resource_profile=PROFILE,options=options,
    limits=dict(generations=4,http_attempts=16,wall_seconds=14400,api_timeout_seconds=1200,max_tokens=384000))
   state=app.prepare_source(task)
   state=read(app._dir(task)/"state.json")
  old=Path(item["source_manifest"])
  require(m==read(old)and m["format"]==item["source_format"],"exact_original_manifest")
  for name,want in m["files"].items():
   require(sha(old.parent/name)==sha(source/name)==want,"exact_original_source")
  for info in m.get("weights",{}).values():
   require(sha(old.parent/info["file"])==sha(source/info["file"])==info["sha256"],"exact_original_weight")
  require(state["status"]=="queued"and state["intake"]["reference_verified"]is True,"source_reference_qualification")
  require(state["generations"]==state["http_attempts"]==0 and state["ledger"]==[],"unpaid_source_preparation")
  if previous:
   from model_source_provenance import build as source_origin
   meta=load_metadata(previous,sdk,source_origin)
  else:meta=app._checked_task_metadata(task)
  q=meta["request"]
  require(meta["identity"]==sdk and reconstruct_request(q)==q,"source_current_identity")
  require(q==read(item["historical_public_request"]),"historical_mathematics_weights_and_request_preserved")
  check_provenance(meta["model_provenance"],q["model"])
  full=folder/"request.json";public=folder/"public-request.json"
  if previous:require(read(full)==q,"recovered_exact_public_request")
  else:write(full,q)
  payload=public_request(q)
  require(not any(k in payload for k in("hecate_source","reference_outputs","test_inputs","model_provenance","api_key")),"public_generation_context")
  if previous:require(read(public)==payload,"recovered_public_payload")
  else:write(public,payload)
  qualified=Path(previous["task_directory"])if previous else app._dir(task)
  entry=dict(context=context,source_manifest=str(manifest.relative_to(ROOT)),source_manifest_sha256=sha(manifest),
   source_format=m["format"],input_signature=m["inputs"],output_names=m["outputs"],dtype_policy=m["dtype_policy"],
   source_files={str(source/n):h for n,h in m["files"].items()},
   declared_weight_files={str(source/v["file"]):v["sha256"]for v in m.get("weights",{}).values()},
   request_id=q["request_id"],model_sha256=digest(q["model"]),public_request_sha256=digest(q),
   request_file=str(full),request_file_sha256=sha(full),public_request_file=str(public),public_payload_sha256=sha(public),
   historical_source_manifest=str(old),historical_public_request=item["historical_public_request"],
   qualification=dict(current_source_capture="passed",current_independent_reference="passed",
    compilations=0,encrypted_calls=0,source_task_directory=str(qualified),source_task_metadata=str(qualified/"task.json"),
    source_task_metadata_sha256=sha(qualified/"task.json"),not_current_agent_generation=True))
  boundary(check_identity=previous is None)
  sources.append(entry)
  print(json.dumps(dict(source_context=context,status="checkpoint_revalidated"if previous else"captured_reference_verified",provider_calls=0)),flush=True)
 boundary()
 source_map={s["context"]:s for s in sources};cases=[];groups=[]
 for group in inventory["request_groups"]:
  base=source_map[group["representative_context"]];name=group["representative_context"].replace("/","__")
  for context in group["contexts"]:
   require(source_map[context]["public_request_sha256"]==base["public_request_sha256"],"exact_public_request_group")
  c=copy.deepcopy(base);c.pop("context")
  c["public_request_sha256"]=sha(c["public_request_file"])
  c.update(id=name,resource_profile=PROFILE,generation_guidance="explicit-v13",slot_period=4,validation_level="numerical",
   track="free_generation",max_repairs=3,max_generations=4,max_http_attempts=16,approximation_accuracy_certified=False)
  cases.append(c)
  groups.append(dict(representative_case=name,representative_context=group["representative_context"],contexts=group["contexts"],
   request_id=base["request_id"],public_request_sha256=base["public_request_sha256"]))
 require(len(cases)==26 and len(sources)==51,"frozen_denominator")
 source_impl=dict(source_identity(),**source_authority(ROOT)["sources"])
 proposal=dict(format="poseidon-normalized-source-live-proposal-v1",status="frozen_plan",
  source_root=str(ROOT),head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True,timeout=10).strip(),
  branch="lhy-agent-dsl",provider=PROVIDER,bounds=BOUNDS,numeric_tolerance=dict(atol=1e-5,rtol=1e-4),
  monetary_cap=None,retries_may_duplicate_billing=True,source_stage_identity=sdk,
  source_stage_counts={s:len(sdk[s]["sources"])for s in("generation","compilation","execution","validation")},
  source_capture_implementation=source_impl,cases=cases,source_contexts=sources,request_groups=groups,
  qualification_records={str(f):sha(f)for f in(INVENTORY,ROOT/"docs/baseline/agent-public-dtype-offline-results-r1380.json",
    ROOT/"docs/baseline/agent-public-expression-offline-results-r1370.json",
    ROOT/"docs/baseline/agent-public-expression-output-results-r1373.json")},
  source_replay_requirements=dict(remaining_source_contexts=25,generation_calls=0,
   origin="unchanged real-Agent source from corresponding representative request",cache_source_identity_unchanged=True,
   separate_bounded_offline_proposal_required=True),
  paid_calls=0,http_attempts=0,old_paid_campaigns_closed=True,no_retry_of_historical_unknown_calls=True,
  external_content=["public model/decomposed graph","public weights or descriptors","layout","DSL rules","bounded diagnostics"],
  excluded_content=["rule candidate/answer","private test inputs","reference outputs","credentials","private keys"],
  protocol_evidence=dict(local_config="generation_protocol.py",checked_date="2026-10-12",
   chat_completion="https://api-docs.deepseek.com/api/create-chat-completion/",
   documented_models=["deepseek-flash","deepseek-v4-pro"],documented_max_tokens=393216,
   requested_max_tokens=384000,reasoning_effort="high",stream=True,
   verification="Official documentation rechecked before this frozen plan"),
  network_changes=False,dependency_installation=False,backend_changes=False,parameter_changes=False,
  preparation_result=str(receipts),preparation_seconds=time.monotonic()-start,
  preparation_checkpoint=None if recovery is None else dict(path=str(HERE/"capture-checkpoint-first.json"),
   file_sha256=sha(HERE/"capture-checkpoint-first.json"),binding=recovery["binding"],revalidated_contexts=50,
   extra_checkpoint_path=str(HERE/"capture-checkpoint-extra.json"),extra_checkpoint_sha256=sha(HERE/"capture-checkpoint-extra.json"),
   extra_checkpoint_binding=extra["binding"],new_source_captures=1,original_stop="source_preparation_time_ceiling",continuation_bounds=recovery["continuation"]))
 proposal["binding"]=digest(proposal);write(PROPOSAL,proposal)
 text=(HERE/"scope.py").read_text().replace("FROZEN_BINDING",proposal["binding"]).replace("FROZEN_SHA",sha(PROPOSAL))
 (HERE/"scope.py").write_text(text)
 import importlib;scope=importlib.reload(scope)
 manifest=dict(format="poseidon-normalized-source-live-runner-v1",proposal_binding=proposal["binding"],
  proposal_sha256=sha(PROPOSAL),tool_hashes=scope.tools());manifest["binding"]=digest(manifest);write(scope.MANIFEST,manifest)
 require(PoseidonBackend().identity_manifest()==sdk,"stable_frozen_sdk")
 scope.verify()
 print(json.dumps(dict(proposal=str(PROPOSAL),binding=proposal["binding"],generation_requests=26,source_contexts=51,
  sdk_identity=sdk["manifest_sha256"],provider_calls=0,compiler_calls=0,encrypted_calls=0,
  seconds=time.monotonic()-start)),flush=True)
 return 0
if __name__=="__main__":sys.exit(main())
