"""Explicit immutable source campaign; --plan never authorizes or calls the API."""
import argparse,hashlib,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
PROPOSAL=ROOT/"docs/baseline/agent-source-expansions-live-proposal-r1349.json"
CASE_DATA=HERE.parent/"source_scope_refresh_v1/cases.json"
CASE_ROWS=json.loads(CASE_DATA.read_text());CASES=tuple(c["id"]for c in CASE_ROWS)
PROVIDER=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True,timeout_seconds=1200,max_tokens=384000,transport_retries=3)
BUDGET=dict(max_generations=144,max_http_attempts=576,max_repairs=3,
 api_concurrency=1,native_concurrency=1,compile_jobs=2,link_jobs=1,batch_wall_seconds=86400,case_wall_seconds=14400,
 new_batch_bytes=4*1024**3,per_case_bytes=4*1024**3,global_result_bytes=32*1024**3,reserve_bytes=4*1024**3,
 native_address_bytes=4*1024**3,native_cpu_seconds=150,currency_limit=None)
def require(ok,code):
 if not ok:raise ValueError(code)
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def rooted(p,root):
 p=pathlib.Path(p)
 require(p.is_absolute()and p==p.resolve()and p.is_relative_to(root)and p.is_file()and not any(v.is_symlink()for v in(p,*p.parents)),"frozen_file_path")
 return p
def fields(value):
 require(type(value)is dict and value.get("format")=="poseidon-source-expansions-live-proposal-v1","proposal_format")
 require(value.get("binding")==digest({k:v for k,v in value.items()if k!="binding"}),"proposal_binding")
 require(value.get("approved")is False and value.get("executed")is False,"planning_is_not_authorization")
 require(value.get("provider")==PROVIDER and value.get("budget")==BUDGET,"reviewed_budget")
 require(value.get("previous_unknown_http_retried")is False,"unknown_call_not_retried")
 cases=value.get("cases");require(type(cases)is list and tuple(c.get("id")for c in cases)==CASES and len(CASES)==36,"reviewed_cases")
 require(len({c.get("request_id")for c in cases})==36,"unique_requests")
 for c,expected in zip(cases,CASE_ROWS):
  require(all(c.get(k)==expected[k]for k in ["id","group","source_manifest","source_manifest_sha256","source_format"]),"original_source_case")
  require(c.get("validation_level")=="numerical"and c.get("max_generations")==4 and c.get("max_http_attempts")==16 and c.get("max_repairs")==3,"case_contract")
  require(type(c.get("public_context_bytes"))is int and 0<c["public_context_bytes"]<=131072,"context_budget")
 require(value.get("provider_calls")==value.get("compiler_calls")==value.get("encrypted_calls")==0,"data_freeze_only")
 return value
def verify(value,*,root=ROOT,base=BASE,check_tools=True):
 fields(value);root=pathlib.Path(root);base=pathlib.Path(base)
 require(value["development_checkout"]==str(root)and value["branch"]=="lhy-agent-dsl"and value["platform"]=="aarch64-linux","checkout_scope")
 require(sha(rooted(CASE_DATA,root))==value["case_inventory_sha256"],"case_inventory_changed")
 identity=value["source_stage_identity"]
 require(identity["manifest_sha256"]==value["identity_sha256"],"identity_reference")
 for stage in ["generation","compilation","execution","validation"]:
  for name,h in identity[stage]["sources"].items():require(sha(rooted(base/name,root))==h,"source_identity_changed")
  for name,h in identity[stage]["configuration_files"].items():require(sha(rooted(root/name,root))==h,"configuration_changed")
 for c in value["cases"]:
  require(sha(rooted(c["source_manifest"],root))==c["source_manifest_sha256"],"source_manifest_changed")
  for name,h in {**c["source_files"],**c["weight_files"]}.items():require(sha(rooted(root/name,root))==h,"source_or_weight_changed")
  for name in ["full_request_file","public_request_file"]:require(sha(rooted(c[name],root))==c[name+"_sha256"],"frozen_request_changed")
  sys.path.insert(0,str(BASE))
  from unified_request_storage import restore_public_constants,restore_bindings
  from public_weight_context import project
  full=read(c["full_request_file"]);public=read(c["public_request_file"])
  restore_public_constants(full);restore_bindings(public["constant_origins"]["capacity_tensor_packing"])
  require(project(full)==public and full["request_id"]==public["request_id"]==c["request_id"],"public_request_identity")
  require(digest({k:v for k,v in public.items()if k!="request_id"})==c["request_id"],"public_request_binding")
  require(pathlib.Path(c["public_request_file"]).stat().st_size==c["public_context_bytes"],"context_size")
 if check_tools:
  for name,h in value["preparation_tools"].items():require(sha(rooted(root/name,root))==h,"preparation_tool_changed")
 return dict(format="poseidon-source-expansions-live-plan-v1",proposal_binding=value["binding"],approved=False,executed=False,
 cases=list(CASES),provider=PROVIDER,budget=BUDGET,identity_sha256=value["identity_sha256"],
 provider_calls=0,status="prepared_waiting_for_separate_approval")
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",action="store_true",required=True);parser.parse_args()
 print(json.dumps(verify(read(PROPOSAL)),sort_keys=True))
if __name__=="__main__":main()
