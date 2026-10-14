"""Read-only inventory of source contexts missing dedicated real-Agent evaluation.

This tool groups identical public requests. It does not generate, run candidates,
or convert historical rule results into Agent evidence.
"""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,code):
 if not value:raise ValueError(code)
def safe_child(root,name):
 require(type(name)is str and name and not Path(name).is_absolute(),"source_relative_file")
 require(".."not in Path(name).parts,"source_parent_file")
 path=root/name
 require(not path.is_symlink()and path.is_file()and path.resolve().is_relative_to(root.resolve()),"source_regular_file")
 return path
def make_inventory():
 specs=(
  ("public_expression","agent-public-expression-offline-proposal-r1369.json",
   "poseidon-trusted-pytorch-v23",18),
  ("public_dtype","agent-public-dtype-offline-proposal-r1376.json",
   "poseidon-trusted-pytorch-v24",33))
 contexts=[];groups={};observations={};request_ids={}
 for category,filename,source_format,count in specs:
  proposal_path=ROOT/"docs/baseline"/filename
  plan=read(proposal_path)
  require(plan["binding"]==digest({k:v for k,v in plan.items()if k!="binding"}),"historical_proposal_hash")
  observations[str(proposal_path)]=sha(proposal_path);accepted=0
  for old in plan["cases"]:
   if category=="public_dtype"and old.get("source_contract")!=source_format:continue
   manifests=[Path(old["manifest"])]if "manifest"in old else [
     Path(name)for name in old["files"]if Path(name).name=="manifest.json"]
   require(len(manifests)==1,"source_manifest_count")
   manifest_path=manifests[0]
   require(str(manifest_path)in old["files"]and sha(manifest_path)==old["files"][str(manifest_path)],"historical_manifest_hash")
   source=read(manifest_path);require(source["format"]==source_format,"source_contract")
   source_hashes={str(manifest_path):sha(manifest_path)}
   for name,want in source["files"].items():
    path=safe_child(manifest_path.parent,name)
    require(sha(path)==want,"source_file_hash");source_hashes[str(path)]=want
   for descriptor in source.get("weights",{}).values():
    require(type(descriptor)is dict,"source_weight_descriptor")
    path=safe_child(manifest_path.parent,descriptor["file"])
    require(sha(path)==descriptor["sha256"]and path.stat().st_size==descriptor["bytes"],"source_weight_hash")
    source_hashes[str(path)]=sha(path)
   request_path=Path(old["request"])
   require(sha(request_path)==old["files"][str(request_path)],"historical_public_request_hash")
   request=read(request_path)
   require(request["request_id"]==old["request_id"],"historical_request_id")
   public_sha=digest(request)
   if request["request_id"]in request_ids:
    require(request_ids[request["request_id"]]==public_sha,"request_id_collision")
   request_ids[request["request_id"]]=public_sha
   entry=dict(context=category+"/"+old["case"],category=category,
    source_format=source_format,source_manifest=str(manifest_path),source_hashes=source_hashes,
    historical_public_request=str(request_path),public_request_sha256=public_sha,
    request_id=request["request_id"],old_offline_proposal_binding=plan["binding"],
    source_acceptance="historical_offline_verified",
    current_source_acceptance="not_run",
    new_agent_generation="not_run",
    current_poseidon_validation="not_run")
   contexts.append(entry)
   groups.setdefault(public_sha,dict(request_id=request["request_id"],
     public_request_sha256=public_sha,representative_context=entry["context"],contexts=[]))["contexts"].append(entry["context"])
   observations.update(source_hashes);observations[str(request_path)]=sha(request_path)
   accepted+=1
  require(accepted==count,"source_context_denominator")
 result=dict(format="poseidon-pending-source-test-inventory-v1",status="inventory_only",
  development_checkout=str(ROOT),contexts=contexts,request_groups=list(groups.values()),
  source_contexts=len(contexts),distinct_public_requests=len(groups),
  grouping="exact canonical public request including model weights configuration and rules",
  historical_sources_preserved=True,new_provider_calls=0,new_compiler_calls=0,new_encrypted_calls=0,
  current_source_freeze=False,execution_queued=False,
  next_steps=["current batch terminal audit","current source recapture and independent reference",
              "freeze current request and implementation identities","bounded real-Agent group evaluation",
              "bind each original source context to its independently checked generated package"],
  source_files=observations,authority="current-plan-test-authorization-2026-10-11.json",
  limits=dict(shard_maximum_requests=48,api_concurrency=1,encrypted_concurrency=1,
   compile_jobs=2,link_jobs=1,per_generation_timeout=1200,max_tokens=384000,
   generations_per_request=4,http_attempts_per_request=16,global_result_bytes=32*1024**3,
   minimum_free_bytes=4*1024**3),
  excludes=["candidate source answers","private test inputs","reference output arrays","credentials",
            "new backend changes","new dependencies","parameter changes","budget expansion"])
 result["binding"]=digest(result)
 return result
if __name__=="__main__":
 value=make_inventory();path=HERE/"inventory.json"
 with path.open("x")as stream:stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({k:value[k]for k in ("status","source_contexts","distinct_public_requests","binding")}))
