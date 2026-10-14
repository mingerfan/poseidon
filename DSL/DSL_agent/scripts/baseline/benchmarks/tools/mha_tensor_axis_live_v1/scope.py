"""Frozen r1333 v12 scope and exact derived execution view; no provider imports."""
import copy,hashlib,json,math,subprocess,fcntl
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
RUN=RESULTS/"agent-mha-tensor-axis-live-v1"
PROPOSAL=ROOT/"docs/baseline/agent-mha-tensor-axis-live-proposal-r1333.json"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=ROOT/"docs/baseline/agent-mha-tensor-axis-live-authorization-r1333.json"
PREVIOUS=ROOT/"docs/baseline/agent-mha-feedback-live-results-r1331.json"
TOOL_NAMES=("scope.py","run.py","test_scope.py","test_run.py","package_checks.py")

REVIEWED_PROPOSAL=ROOT/'docs/baseline/agent-mha-feedback-live-proposal-r1330.json'
REVIEWED_BINDING='c9932a659a0ac6a0c34ed4038b067b1ee7129930df0a5e581b0240d1f312c9e5'
REVIEWED_SHA='6c911ceccea0a42b75daac78a07135d478e773dd081dcc8022c5c48b80fbd84b'

PROPOSAL_BINDING='475391a7ef31efef047900e89fe39db9bbc70d63ea63a900e1e88c0f82ae2d1f'
PROPOSAL_SHA='c6735908b974db4b802323f2a65dc4c922ca71f7eb5c4a686000c56dd0fe4825'
PREVIOUS_SHA='fd0dbe56726bc68728c5eb1c0e2a37f2cf72f720f5b1f63d7c886c81c50ec186'
CASES=('self_post_linear_residual__declared_shallow_v1', 'linear_into_separate_cross_none__declared_shallow_v1')
CASE_IDS=('5a4136d4ccdab5dd1ee478fe7b4fe71f727ec7a5f5386ace41e1c488c10185b2', '8d51d457fab5f907b287daaa7a62a933e1156cdba3de340d099de5d811a6c15a')
OPTIONS=dict(generation_guidance="explicit-v12",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
PROFILE="general-tensors-v1"
PROVIDER={'max_tokens': 384000, 'model': 'deepseek-flash', 'name': 'deepseek', 'reasoning_effort': 'high', 'stream': True, 'timeout_seconds': 1200, 'transport_retries': 3}
BOUNDS={'api_concurrency': 1, 'batch_wall_seconds': 28800, 'case_wall_seconds': 14400, 'compiler_jobs': 2, 'encrypted_concurrency': 1, 'global_result_bytes': 34359738368, 'link_jobs': 1, 'max_generations': 8, 'max_http_attempts': 32, 'minimum_free_bytes': 4294967296, 'native_address_bytes': 4294967296, 'native_cpu_seconds': 150, 'new_batch_bytes': 4294967296}
BUDGET={'api_concurrency': 1, 'batch_wall_seconds': 28800, 'case_wall_seconds': 14400, 'compile_jobs': 2, 'currency_limit': None, 'global_result_bytes': 34359738368, 'link_jobs': 1, 'max_generations': 8, 'max_http_attempts': 32, 'native_concurrency': 1, 'per_case_bytes': 4294967296, 'new_batch_bytes': 4294967296, 'reserve_bytes': 4294967296}
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024**2),b""):h.update(block)
 return h.hexdigest()
def decode(raw):
 def pairs(items):
  out={}
  for k,v in items:
   require(k not in out,"duplicate_json_field");out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite_json")))
def read(path):return decode(Path(path).read_bytes())
def check_binding(v):
 require(type(v) is dict and v.get("binding")==digest({k:x for k,x in v.items() if k!="binding"}),"self_binding")
def regular(path):
 p=Path(path)
 require(p.is_absolute() and p==p.resolve() and p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),"regular_frozen_path")
 require(p.is_relative_to(ROOT) or p.is_relative_to(RESULTS),"frozen_path_scope")
 return p
@contextmanager
def owned_lock(path):
 with Path(path).open("a+") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield f
def tools():return {str((HERE/n).relative_to(ROOT)):sha(regular(HERE/n)) for n in TOOL_NAMES}
def check_manifest(m=None):
 m=read(regular(MANIFEST)) if m is None else m
 require(type(m) is dict and set(m)=={"format","proposal_binding","proposal_sha256","tool_hashes","binding"},"runner_manifest_fields");check_binding(m)
 require(m["format"]=="poseidon-mha-tensor-axis-live-runner-v1" and m["proposal_binding"]==PROPOSAL_BINDING and m["proposal_sha256"]==PROPOSAL_SHA and m["tool_hashes"]==tools(),"runner_manifest_binding")
 return m
def validate(p):
 require(type(p) is dict,"proposal_fields");check_binding(p)
 require(p["binding"]==PROPOSAL_BINDING and p["format"]=="poseidon-mha-tensor-axis-live-proposal-v1" and p["status"]=="awaiting_user_approval","original_proposal_binding")
 require(p["provider"]==PROVIDER and p["bounds"]==BOUNDS and all(type(v) is int for v in p["bounds"].values()),"reviewed_provider_or_budget")
 require(p["monetary_cap"] is None and p["retries_may_duplicate_billing"] is True,"billing_disclosure")
 require(tuple(c["id"] for c in p["cases"])==CASES and tuple(c["request_id"] for c in p["cases"])==CASE_IDS,"reviewed_cases")
 require(p["source_stage_identity"]["backend"]["format"]=="poseidon-backend-identity-v2" and p["source_stage_identity"]["format"]=="poseidon-application-identities-v1","poseidon_backend_required")
 require(p["numeric_tolerance"]==dict(atol=1e-5,rtol=1e-4) and all(c["validation_level"]=="numerical" and c["track"]=="free_generation" and c["slot_period"]==4 and c["max_generations"]==4 and c["max_http_attempts"]==16 and c["max_repairs"]==3 and c["approximation_accuracy_certified"] is False for c in p["cases"]),"reviewed_validation")
 return p
def verify(value=None):
 require(sha(regular(PROPOSAL))==PROPOSAL_SHA,"original_proposal_bytes")
 p=read(PROPOSAL) if value is None else value
 require(canonical(p)==canonical(read(PROPOSAL)),"exact_original_proposal");validate(p);m=check_manifest()
 corrected_review_equivalence(p)
 inventory={}
 def freeze(path,want):
  path=regular(path);require(sha(path)==want,"frozen_source_changed");inventory[str(path)]=want
 for stage in ("generation","compilation","execution","validation"):
  sources=p["source_stage_identity"][stage]["sources"]
  require(type(sources) is dict and len(sources)=={"generation":628,"compilation":622,"execution":622,"validation":622}[stage],"frozen_stage_source_inventory")
  for rel,want in sources.items():
   require(type(rel) is str and not Path(rel).is_absolute() and ".." not in Path(rel).parts,"frozen_source_scope");freeze(BASE/rel,want)
 for rel,want in p["source_capture_implementation"].items():freeze(BASE/rel,want)
 for path,want in p["source_implementation_files"].items():freeze(ROOT/path,want)
 for c in p["cases"]:
  require(c["source_format"]=="poseidon-trusted-pytorch-v16" and c["generation_guidance"]=="explicit-v12" and c["resource_profile"]==PROFILE,"public_generation_boundary")
  mp=regular(c["source_manifest"]);freeze(mp,c["source_manifest_sha256"])
  for table in ("source_files","declared_weight_files"):
   for path,want in c[table].items():freeze(path,want)
  freeze(c["request_file"],c["request_file_sha256"])
  require(read(c["request_file"])["request_id"]==c["request_id"],"exact_frozen_request_id")
 for path,want in inventory.items():require(sha(regular(path))==want,"file_changed_during_read")
 require(tools()==m["tool_hashes"],"runner_tools_changed_during_read")
 for cmd,want in ((["git","branch","--show-current"],"lhy-agent-dsl"),(["git","rev-parse","HEAD"],"7588e48093631eaf8385506e3db058e7ca3cfe7e")):
  require(subprocess.check_output(cmd,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
 require(sha(regular(PREVIOUS))==PREVIOUS_SHA,"closed_prior_evidence_changed")
 previous=read(PREVIOUS)
 require(previous["status"]=="complete" and previous["generations"]==5 and
  previous["http_attempts"]==7 and previous["calls_resolved"] is True and
  p["old_quota_closed"] is True,"unresolved_prior_scope")
 return execution_view(p,m)
def execution_view(p,m):
 # Derived fields are execution conveniences, never approval authority.
 view=copy.deepcopy(p)
 view.update(budget=copy.deepcopy(BUDGET),application_stage_identity=copy.deepcopy(p["source_stage_identity"]),
             tool_hashes=copy.deepcopy(m["tool_hashes"]))
 for c in view["cases"]:c.update(full_request_file=c["request_file"],
  current_validation_sources=copy.deepcopy(p["source_stage_identity"]["validation"]["sources"]))
 return view
def authority(receipt,proposal,proposal_sha):
 require(type(receipt) is dict and set(receipt)=={"format","approved","proposal_binding","proposal_sha256","runner_manifest_sha256","tool_hashes","authorization","authorization_record","authorization_sha256"},"approval_record_fields")
 require(receipt["format"]=="poseidon-mha-tensor-axis-live-approval-v1" and receipt["approved"] is True and receipt["proposal_binding"]==PROPOSAL_BINDING==proposal["binding"] and receipt["proposal_sha256"]==PROPOSAL_SHA==proposal_sha and receipt["runner_manifest_sha256"]==sha(regular(MANIFEST)) and receipt["tool_hashes"]==tools()==proposal["tool_hashes"] and receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
 require(receipt["authorization_record"]==str(AUTHORIZATION) and
  receipt["authorization_sha256"]==sha(regular(AUTHORIZATION)),"human_authorization_bytes")
 original=read(regular(PROPOSAL));validate(original)
 require(canonical(proposal)==canonical(execution_view(original,check_manifest())),"execution_view_changed")
 checked_authorization(read(AUTHORIZATION),original,proposal_sha)
def corrected_review_equivalence(proposal):
 require(sha(regular(REVIEWED_PROPOSAL))==REVIEWED_SHA==proposal["prior_proposal_sha256"],"reviewed_proposal_bytes")
 reviewed=read(REVIEWED_PROPOSAL);check_binding(reviewed)
 require(reviewed["binding"]==REVIEWED_BINDING,"reviewed_proposal_binding")
 require(proposal["bounds"]==reviewed["bounds"] and proposal["provider"]==reviewed["provider"] and
  proposal["numeric_tolerance"]==reviewed["numeric_tolerance"],"rebound_expands_reviewed_scope")
 require(set(proposal["source_capture_implementation"])==set(reviewed["source_capture_implementation"]) and
  set(proposal["source_implementation_files"])==set(reviewed["source_implementation_files"]),"rebound_source_inventory")
 changed={"generation_guidance","request_file","request_file_sha256","request_id",
  "prior_request_id","semantic_binding_unchanged","public_context_bytes"}
 strip=lambda v:{k:x for k,x in v.items() if k not in changed}
 require(len(proposal["cases"])==len(reviewed["cases"]),"rebound_case_scope")
 for old,current in zip(reviewed["cases"],proposal["cases"]):
  require(canonical(strip(old))==canonical(strip(current)) and current["prior_request_id"]==old["request_id"] and
   current["semantic_binding_unchanged"] is True and current["generation_guidance"]=="explicit-v12","rebound_changes_mathematical_model")
 for stage in ("generation","compilation","execution","validation"):
  require(proposal["source_stage_identity"][stage]["backend"]==reviewed["source_stage_identity"][stage]["backend"],"rebound_backend_change")
 require(sha(regular(PREVIOUS))==PREVIOUS_SHA==proposal["prior_live_report_sha256"],"closed_prior_evidence_changed")
 prior=read(PREVIOUS);check_binding(prior)
 require(prior["calls_resolved"] is True and prior["status"]=="complete" and prior["planned"]==2 and
  prior["passed"]==0 and prior["generations"]==5 and prior["http_attempts"]==7 and proposal["old_quota_closed"] is True,"unresolved_prior_scope")
 require(sha(regular(prior["batch_report"]))==prior["batch_report_sha256"],"closed_prior_batch_changed")
 offline=regular(proposal["offline_report"])
 require(sha(offline)==proposal["offline_report_sha256"],"qualified_offline_report_changed")
 receipt=read(offline);check_binding(receipt)
 require(receipt["status"]=="offline_passed" and receipt["version"]=="explicit-v12" and
  receipt["identity_sha256"]==proposal["source_stage_identity"]["manifest_sha256"] and
  receipt["numeric_tolerance"]==proposal["numeric_tolerance"] and receipt["native"]["passed"]==1 and
  receipt["native"]["failed"]==0 and receipt["protected_sources_changed"]==0,"qualified_offline_identity")
 require(sha(regular(receipt["native_report"]))==receipt["native_report_sha256"],"qualified_native_evidence_changed")
 return reviewed
def checked_authorization(value,proposal,proposal_sha):
 fields={"format","approved","authorization","binding","bounds","cases","human_message",
         "proposal_binding","proposal_sha256","provider","question_item_id"}
 require(type(value) is dict and set(value)==fields,"human_authorization_fields")
 check_binding(value)
 require(value["format"]=="poseidon-mha-tensor-axis-human-authorization-v1" and
  value["approved"] is True and value["authorization"]=="human_explicit_case_budget_approval" and
  proposal["binding"]==PROPOSAL_BINDING and proposal_sha==PROPOSAL_SHA and
  (value["proposal_binding"],value["proposal_sha256"])==(PROPOSAL_BINDING,PROPOSAL_SHA) and
  canonical(value["bounds"])==canonical(BOUNDS) and canonical(value["provider"])==canonical(PROVIDER) and
  value["cases"]==list(CASES),"human_authorization_scope")
 corrected_review_equivalence(proposal)
 require(type(value["human_message"]) is str and 1<=len(value["human_message"].encode())<=4096 and
  type(value["question_item_id"]) is str and 1<=len(value["question_item_id"])<=512,
  "human_authorization_record")
 return value
def totals(states):
 g=h=0;resolved=True
 for s in states:
  require(type(s["generations"]) is int and type(s["http_attempts"]) is int and 0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_call_ceiling")
  reserved=[e for e in s["ledger"] if e.get("event")=="http_reserved"]
  require([e["index"] for e in reserved]==list(range(s["http_attempts"])),"reservation_ledger")
  checkpoints=[e for e in s["ledger"] if e.get("event")=="provider_checkpoint"];outcomes=checkpoints[-1].get("outcomes",[]) if checkpoints else []
  resolved=resolved and len(outcomes)==s["http_attempts"] and all(o.get("index")==i and o.get("status") in ("response_received","failed") for i,o in enumerate(outcomes))
  g+=s["generations"];h+=s["http_attempts"]
 require(g<=8 and h<=32,"campaign_call_ceiling");return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def terminal(rows,states):
 require(len(rows)<=len(CASES) and len({r["case"] for r in rows})==len(rows) and all(r["case"] in CASES for r in rows),"result_case_scope")
 data=totals(states);passed=sum(r["status"]=="passed" for r in rows)
 return dict(status="complete" if len(rows)==len(CASES) and data["calls_resolved"] else "stopped",passed=passed,failed=len(rows)-passed,not_run=len(CASES)-len(rows),**data)
