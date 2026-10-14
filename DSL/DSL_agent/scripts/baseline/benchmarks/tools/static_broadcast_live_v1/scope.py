"""Frozen source-normalization scope; standard library only and no Provider imports."""
import copy,hashlib,json,subprocess,fcntl
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5]);BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux");RESULTS=WORK/"results"
RUN=RESULTS/"agent-static-broadcast-live-v1"
PROPOSAL=ROOT/"docs/baseline/agent-static-broadcast-live-proposal-r1428.json"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=ROOT/"docs/baseline/agent-static-broadcast-live-authorization-r1428.json"
PROPOSAL_BINDING="cecd9ba0273880891ae02bbe5cf29c67755d5b656f5cdfd25cb16df59817580a";PROPOSAL_SHA="997420b7679ac663816eb19dd296bd82bf5db060516adbad0226f6ce23d2266e"
TOOL_NAMES=("scope.py","run.py","package_checks.py","test_scope.py")
CASES=('expand_as__arithmetic', 'expand_as__transpose_alias', 'expand_as__scalar_linear')
OPTIONS=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
PROFILE="general-tensors-v1"
PROVIDER={'name': 'deepseek', 'model': 'deepseek-flash', 'reasoning_effort': 'high', 'max_tokens': 384000, 'timeout_seconds': 1200, 'transport_retries': 3, 'stream': True}
BOUNDS={'api_concurrency': 1, 'batch_wall_seconds': 43200, 'case_wall_seconds': 14400, 'compiler_jobs': 2, 'encrypted_concurrency': 1, 'global_result_bytes': 34359738368, 'link_jobs': 1, 'max_generations': 12, 'max_http_attempts': 48, 'minimum_free_bytes': 4294967296, 'native_address_bytes': 4294967296, 'native_cpu_seconds': 150, 'new_batch_bytes': 100663296, 'max_compiler_calls': 12, 'max_encrypted_calls': 18}
BUDGET={'api_concurrency': 1, 'batch_wall_seconds': 43200, 'case_wall_seconds': 14400, 'compile_jobs': 2, 'currency_limit': None, 'global_result_bytes': 34359738368, 'link_jobs': 1, 'max_generations': 12, 'max_http_attempts': 48, 'native_concurrency': 1, 'per_case_bytes': 4294967296, 'new_batch_bytes': 100663296, 'reserve_bytes': 4294967296}
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def decode(raw):
 def pairs(items):
  out={}
  for k,v in items:require(k not in out,"duplicate_json_field");out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError("nonfinite_json")))
def read(p):return decode(Path(p).read_bytes())
def check_binding(v):
 require(type(v)is dict and v.get("binding")==digest({k:x for k,x in v.items()if k!="binding"}),"self_binding")
def regular(path):
 p=Path(path)
 require(p.is_absolute()and p==p.resolve()and p.is_file()and not any(x.is_symlink()for x in(p,*p.parents)),"regular_frozen_path")
 require(p.is_relative_to(ROOT)or p.is_relative_to(RESULTS),"frozen_path_scope")
 return p
@contextmanager
def owned_lock(path):
 with Path(path).open("a+")as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield f
def tools():return {str((HERE/n).relative_to(ROOT)):sha(regular(HERE/n))for n in TOOL_NAMES}
def check_manifest(m=None):
 m=read(regular(MANIFEST))if m is None else m
 require(type(m)is dict and set(m)=={"format","proposal_binding","proposal_sha256","tool_hashes","binding"},"runner_manifest_fields")
 check_binding(m)
 require(m["format"]=="poseidon-static-broadcast-live-runner-v1"and m["proposal_binding"]==PROPOSAL_BINDING and
  m["proposal_sha256"]==PROPOSAL_SHA and m["tool_hashes"]==tools(),"runner_manifest_binding")
 return m
def validate(p):
 check_binding(p)
 require(p["binding"]==PROPOSAL_BINDING and p["format"]=="poseidon-static-broadcast-live-proposal-v1"and
  p["status"]=="frozen_plan","original_proposal_binding")
 require(canonical(p["provider"])==canonical(PROVIDER)and canonical(p["bounds"])==canonical(BOUNDS)and
  all(type(v)is int for v in p["bounds"].values()),"reviewed_provider_or_budget")
 require(p["monetary_cap"]is None and p["retries_may_duplicate_billing"]is True,"billing_disclosure")
 require(tuple(c["id"]for c in p["cases"])==CASES,"reviewed_cases")
 require(p["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4)and
  p["source_stage_identity"]["format"]=="poseidon-application-identities-v1"and
  p["source_stage_identity"]["backend"]["format"]=="poseidon-backend-identity-v2","poseidon_backend_required")
 require(len(p["source_contexts"])==6 and len({a["context"]for a in p["source_contexts"]})==6,"source_context_count")
 require(len({c["request_id"]for c in p["cases"]})==3,"generation_request_count")
 require(len(p["request_groups"])==3 and
  [g["representative_case"]for g in p["request_groups"]]==list(CASES),"source_group_order")
 mapped=[]
 for g in p["request_groups"]:
  require(g["representative_context"]in g["contexts"],"source_representative_membership")
  require(g["request_id"]==next(c["request_id"]for c in p["cases"]if c["id"]==g["representative_case"]),"source_group_request")
  for context in g["contexts"]:
   entry=next((a for a in p["source_contexts"]if a["context"]==context),None)
   require(entry is not None and entry["request_id"]==g["request_id"]and
    entry["public_request_sha256"]==g["public_request_sha256"],"source_group_binding")
  mapped.extend(g["contexts"])
 require(len(mapped)==len(set(mapped))==6 and set(mapped)=={a["context"]for a in p["source_contexts"]},"source_group_coverage")
 for c in p["cases"]:
  require(c["source_format"]in("poseidon-trusted-pytorch-v28",)and c["generation_guidance"]=="explicit-v13"and
   c["validation_level"]=="numerical"and c["track"]=="free_generation"and c["slot_period"]==4 and
   c["max_generations"]==4 and c["max_http_attempts"]==16 and c["max_repairs"]==3 and
   c["resource_profile"]==PROFILE and c["approximation_accuracy_certified"]is False,"reviewed_validation")
 return p
def verify(value=None):
 require(sha(regular(PROPOSAL))==PROPOSAL_SHA,"original_proposal_bytes")
 p=read(PROPOSAL)if value is None else value
 require(canonical(p)==canonical(read(PROPOSAL)),"exact_original_proposal");validate(p);m=check_manifest()
 inventory={}
 def freeze(path,want):
  path=regular(path);require(sha(path)==want,"frozen_source_changed");inventory[str(path)]=want
 for stage in("generation","compilation","execution","validation"):
  sources=p["source_stage_identity"][stage]["sources"]
  require(type(sources)is dict and len(sources)==p["source_stage_counts"][stage],"frozen_stage_source_inventory")
  for rel,want in sources.items():
   require(type(rel)is str and not Path(rel).is_absolute()and ".."not in Path(rel).parts,"frozen_source_scope");freeze(BASE/rel,want)
 for rel,want in p["source_capture_implementation"].items():freeze(BASE/rel,want)
 for rel,want in p["qualification_records"].items():freeze(rel,want)
 for c in p["cases"]:
  freeze(ROOT/c["source_manifest"],c["source_manifest_sha256"])
  manifest=read(ROOT/c["source_manifest"])
  require(manifest["format"]==c["source_format"]and manifest["inputs"]==c["input_signature"]and
   manifest["outputs"]==c["output_names"]and manifest["dtype_policy"]==c["dtype_policy"],"source_manifest_metadata")
  for table in("source_files","declared_weight_files"):
   for path,want in c[table].items():freeze(path,want)
  freeze(c["request_file"],c["request_file_sha256"]);freeze(c["public_request_file"],c["public_request_sha256"])
  q=read(c["request_file"]);require(q["request_id"]==c["request_id"]and digest(q["model"])==c["model_sha256"],"exact_frozen_request_id")
 for a in p["source_contexts"]:
  freeze(ROOT/a["source_manifest"],a["source_manifest_sha256"])
  manifest=read(ROOT/a["source_manifest"])
  require(manifest["format"]==a["source_format"]and manifest["inputs"]==a["input_signature"]and
   manifest["outputs"]==a["output_names"]and manifest["dtype_policy"]==a["dtype_policy"],"source_alias_manifest_metadata")
  for table in("source_files","declared_weight_files"):
   for path,want in a[table].items():freeze(path,want)
  freeze(a["request_file"],a["request_file_sha256"])
  freeze(a["qualification"]["source_task_metadata"],a["qualification"]["source_task_metadata_sha256"])
  q=read(a["request_file"]);require(q["request_id"]==a["request_id"]and digest(q)==a["public_request_sha256"],"source_context_request")
 for path,want in inventory.items():require(sha(regular(path))==want,"file_changed_during_read")
 require(tools()==m["tool_hashes"],"runner_tools_changed_during_read")
 for cmd,want in((["git","branch","--show-current"],"lhy-agent-dsl"),(["git","rev-parse","HEAD"],p["head"])):
  require(subprocess.check_output(cmd,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
 return execution_view(p,m)
def execution_view(p,m):
 v=copy.deepcopy(p);v.update(budget=copy.deepcopy(BUDGET),application_stage_identity=copy.deepcopy(p["source_stage_identity"]),
 tool_hashes=copy.deepcopy(m["tool_hashes"]))
 for c in v["cases"]:c.update(full_request_file=c["request_file"],
  current_validation_sources=copy.deepcopy(p["source_stage_identity"]["validation"]["sources"]))
 return v
def checked_authorization(v,p,proposal_sha):
 fields={"format","approved","authorization","binding","bounds","cases","human_message","proposal_binding","proposal_sha256","provider","question_item_id"}
 require(type(v)is dict and set(v)==fields,"human_authorization_fields");check_binding(v)
 require(v["format"]=="poseidon-public-arithmetic-human-authorization-v1"and v["approved"]is True and
  v["authorization"]=="human_explicit_current_plan_test_approval"and
  (v["proposal_binding"],v["proposal_sha256"])==(PROPOSAL_BINDING,PROPOSAL_SHA)and
  (p["binding"],proposal_sha)==(PROPOSAL_BINDING,PROPOSAL_SHA)and
  canonical(v["bounds"])==canonical(BOUNDS)and canonical(v["provider"])==canonical(PROVIDER)and
  v["cases"]==list(CASES),"human_authorization_scope")
 require(type(v["human_message"])is str and 1<=len(v["human_message"].encode())<=4096 and
  type(v["question_item_id"])is str and 1<=len(v["question_item_id"])<=512,"human_authorization_record")
 return v
def authority(receipt,proposal,proposal_sha):
 fields={"format","approved","proposal_binding","proposal_sha256","runner_manifest_sha256","tool_hashes","authorization","authorization_record","authorization_sha256"}
 require(type(receipt)is dict and set(receipt)==fields,"approval_record_fields")
 require(receipt["format"]=="poseidon-static-broadcast-live-approval-v1"and receipt["approved"]is True and
  receipt["proposal_binding"]==PROPOSAL_BINDING==proposal["binding"]and receipt["proposal_sha256"]==PROPOSAL_SHA==proposal_sha and
  receipt["runner_manifest_sha256"]==sha(regular(MANIFEST))and receipt["tool_hashes"]==tools()==proposal["tool_hashes"]and
  receipt["authorization"]=="human_explicit_current_plan_test_approval","approval_required")
 require(receipt["authorization_record"]==str(AUTHORIZATION)and receipt["authorization_sha256"]==sha(regular(AUTHORIZATION)),"human_authorization_bytes")
 original=read(regular(PROPOSAL));validate(original)
 require(canonical(proposal)==canonical(execution_view(original,check_manifest())),"execution_view_changed")
 checked_authorization(read(AUTHORIZATION),original,proposal_sha)
def totals(states):
 g=h=0;resolved=True
 for s in states:
  require(type(s["generations"])is int and type(s["http_attempts"])is int and 0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_call_ceiling")
  reserved=[e for e in s["ledger"]if e.get("event")=="http_reserved"]
  require([e["index"]for e in reserved]==list(range(s["http_attempts"])),"reservation_ledger")
  checkpoints=[e for e in s["ledger"]if e.get("event")=="provider_checkpoint"];outcomes=checkpoints[-1].get("outcomes",[])if checkpoints else []
  resolved=resolved and len(outcomes)==s["http_attempts"]and all(o.get("index")==i and o.get("status")in("response_received","failed")for i,o in enumerate(outcomes))
  g+=s["generations"];h+=s["http_attempts"]
 require(g<=BOUNDS["max_generations"] and h<=BOUNDS["max_http_attempts"],"campaign_call_ceiling")
 return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def terminal(rows,states):
 require(len(rows)<=len(CASES)and len({r["case"]for r in rows})==len(rows)and all(r["case"]in CASES for r in rows),"result_case_scope")
 data=totals(states);passed=sum(r["status"]=="passed"for r in rows)
 return dict(status="complete"if len(rows)==len(CASES)and data["calls_resolved"]else"stopped",passed=passed,failed=len(rows)-passed,not_run=len(CASES)-len(rows),**data)
