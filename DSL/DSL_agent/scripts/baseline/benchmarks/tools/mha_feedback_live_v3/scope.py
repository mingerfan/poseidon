"""Frozen r1316 two-case MHA feedback scope. No provider or credential imports."""
import copy,hashlib,json,math,subprocess,fcntl
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
RUN=RESULTS/"agent-mha-feedback-live-v3"
PROPOSAL=ROOT/"docs/baseline/agent-mha-feedback-live-proposal-r1317.json"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=ROOT/"docs/baseline/agent-mha-feedback-live-authorization-r1316.json"
PREVIOUS=RESULTS/"agent-mha-source-live-v2/report.json"
TOOL_NAMES=("scope.py","run.py","test_scope.py","test_run.py","package_checks.py")

REVIEWED_PROPOSAL=ROOT/'docs/baseline/agent-mha-feedback-live-proposal-r1316.json'
REVIEWED_BINDING='f35dd50e53380d7d2b42fb8548bd09e2696aaa8d9d546a1853735e8c3ab9ef5e'
REVIEWED_SHA='0d3e0f26e4f4877aa97a28b8a8341b083d9460812a505c933f1c09c56779b93b'
CORRECTION_PATHS=('candidate_check_diagnostic.py', 'decorated_functions.py', 'unified_graph_contract.py')

PROPOSAL_BINDING='509391f78f2bc75e6f5fb749e9c048ce9ea049b9aa0e51f5d7c62521b8fd4d69'
PROPOSAL_SHA='6c627aa4e73d3062de0e4bcab4850d947156142dc82977c32247aef99ced4110'
PREVIOUS_SHA='e6aa6379a3d2ec282e44a675a198ee6d181c381327401b703c90ae70c745d100'
CASES=('self_post_linear_residual__declared_shallow_v1', 'linear_into_separate_cross_none__declared_shallow_v1')
CASE_IDS=('147f85356e26b914e94f6c346243b3e9ddf1c42b2dfc387375481392850973af', '5eebc4fccd6b1f0ece4bd65989daed863475c982ffbf3acaf53fd349c4eb9a54')
OPTIONS=dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
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
 require(m["format"]=="poseidon-mha-feedback-live-runner-v1" and m["proposal_binding"]==PROPOSAL_BINDING and m["proposal_sha256"]==PROPOSAL_SHA and m["tool_hashes"]==tools(),"runner_manifest_binding")
 return m
def validate(p):
 require(type(p) is dict,"proposal_fields");check_binding(p)
 require(p["binding"]==PROPOSAL_BINDING and p["format"]=="poseidon-mha-v16-feedback-live-proposal-v1" and p["status"]=="awaiting_user_approval","original_proposal_binding")
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
  require(type(sources) is dict and len(sources)==691,"frozen_stage_source_inventory")
  for rel,want in sources.items():
   require(type(rel) is str and not Path(rel).is_absolute() and ".." not in Path(rel).parts,"frozen_source_scope");freeze(BASE/rel,want)
 for rel,want in p["source_capture_implementation"].items():freeze(BASE/rel,want)
 for path,want in p["source_implementation_files"].items():freeze(ROOT/path,want)
 for c in p["cases"]:
  require(c["source_format"]=="poseidon-trusted-pytorch-v16" and c["generation_guidance"]=="explicit-v11" and c["resource_profile"]==PROFILE,"public_generation_boundary")
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
 require(previous["status"] in ("stopped","complete") and previous["generations"]==8 and
  previous["http_attempts"]==8 and previous["calls_resolved"] is True and
  p["prior_r1313_quota_closed"] is True,"unresolved_prior_scope")
 view=copy.deepcopy(p)
 view.update(budget=BUDGET,application_stage_identity=p["source_stage_identity"],tool_hashes=m["tool_hashes"])
 for c in view["cases"]:c.update(full_request_file=c["request_file"],current_validation_sources=p["source_stage_identity"]["validation"]["sources"])
 return view
def authority(receipt,proposal,proposal_sha):
 require(type(receipt) is dict and set(receipt)=={"format","approved","proposal_binding","proposal_sha256","runner_manifest_sha256","tool_hashes","authorization","authorization_record","authorization_sha256"},"approval_record_fields")
 require(receipt["format"]=="poseidon-mha-feedback-live-approval-v1" and receipt["approved"] is True and receipt["proposal_binding"]==PROPOSAL_BINDING==proposal["binding"] and receipt["proposal_sha256"]==PROPOSAL_SHA==proposal_sha and receipt["runner_manifest_sha256"]==sha(regular(MANIFEST)) and receipt["tool_hashes"]==tools()==proposal["tool_hashes"] and receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
 require(receipt["authorization_record"]==str(AUTHORIZATION) and
  receipt["authorization_sha256"]==sha(regular(AUTHORIZATION)),"human_authorization_bytes")
 checked_authorization(read(AUTHORIZATION),proposal,proposal_sha)
def corrected_review_equivalence(proposal):
 require(sha(regular(REVIEWED_PROPOSAL))==REVIEWED_SHA,"reviewed_proposal_bytes")
 reviewed=read(REVIEWED_PROPOSAL);check_binding(reviewed)
 require(reviewed["binding"]==REVIEWED_BINDING,"reviewed_proposal_binding")
 strip=lambda v:{k:x for k,x in v.items() if k not in ("binding","source_capture_implementation")}
 require(canonical(strip(reviewed))==canonical(strip(proposal)),"correction_expands_reviewed_scope")
 before=reviewed["source_capture_implementation"];after=proposal["source_capture_implementation"]
 require(set(before)==set(after) and tuple(k for k in before if before[k]!=after[k])==CORRECTION_PATHS,
         "correction_paths")
 for rel in CORRECTION_PATHS:
  require(all(proposal["source_stage_identity"][stage]["sources"][rel]==after[rel]
       for stage in ("generation","compilation","execution","validation")),"correction_stage_agreement")
 return reviewed
def checked_authorization(value,proposal,proposal_sha):
 fields={"format","approved","authorization","binding","bounds","cases","human_message",
         "proposal_binding","proposal_sha256","provider","question_item_id"}
 require(type(value) is dict and set(value)==fields,"human_authorization_fields")
 check_binding(value)
 require(value["format"]=="poseidon-mha-feedback-human-authorization-v1" and
  value["approved"] is True and value["authorization"]=="human_explicit_case_budget_approval" and
  proposal["binding"]==PROPOSAL_BINDING and proposal_sha==PROPOSAL_SHA and
  (value["proposal_binding"],value["proposal_sha256"]) in (
      (PROPOSAL_BINDING,PROPOSAL_SHA),(REVIEWED_BINDING,REVIEWED_SHA)) and
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
