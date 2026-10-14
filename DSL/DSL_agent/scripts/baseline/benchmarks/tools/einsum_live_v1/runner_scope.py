"""Execution view and strict bounded authority for immutable r1341, with no API imports."""
import copy,fcntl,hashlib,json,math,subprocess,sys
from contextlib import contextmanager
from pathlib import Path
import scope as preparation
ROOT=preparation.ROOT;BASE=preparation.BASE;HERE=preparation.HERE
PROPOSAL=preparation.PROPOSAL;CASES=preparation.CASES;PROVIDER=preparation.PROVIDER;BUDGET=preparation.BUDGET
PROFILE="general-tensors-v1"
OPTIONS=dict(generation_guidance="explicit-v13",public_weight_policy="descriptors-math-roles-v2",chunk_period=4)
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux");RESULTS=WORK/"results"
RUN=RESULTS/"agent-einsum-live-v1"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=ROOT/"docs/baseline/agent-einsum-live-authorization-r1341.json"
PROPOSAL_BINDING="f6a8b322dbcbf083e8ce7e914940d0d5ca4f60cfb114a8b59c0fe5eaf287083d"
PROPOSAL_SHA="b70cf2d9d94fc6dc07c94beca32cda515f0dab1674424a8e97a11dc173b5f484"
TOOL_NAMES=("scope.py","prepare.py","runner_scope.py","run.py","package_checks.py",
            "test_scope.py","test_runner_scope.py","test_run.py","runtime_controls.py")
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb")as f:
  for block in iter(lambda:f.read(1024**2),b""):h.update(block)
 return h.hexdigest()
def decode(raw):
 def pairs(items):
  out={}
  for k,v in items:
   require(k not in out,"duplicate_json_field");out[k]=v
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError("nonfinite_json")))
def read(path):return decode(Path(path).read_bytes())
def request_read(path):
 value=read(path)
 sys.path.insert(0,str(BASE))
 from unified_request_storage import restore_public_constants
 restore_public_constants(value)
 return value
def check_binding(v):
 require(type(v)is dict and v.get("binding")==digest({k:x for k,x in v.items()if k!="binding"}),"self_binding")
def regular(path):
 p=Path(path)
 require(p.is_absolute()and p==p.resolve()and p.is_file()and not any(x.is_symlink()for x in(p,*p.parents)),"regular_frozen_path")
 require(p.is_relative_to(ROOT)or p.is_relative_to(RESULTS),"frozen_path_scope")
 return p
@contextmanager
def owned_lock(path):
 with Path(path).open("a+")as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield f
def tools():return{str((HERE/n).relative_to(ROOT)):sha(regular(HERE/n))for n in TOOL_NAMES}
def check_manifest(value=None):
 value=read(regular(MANIFEST))if value is None else value
 require(type(value)is dict and set(value)=={"format","proposal_binding","proposal_sha256","tool_hashes","binding"},"runner_manifest_fields")
 check_binding(value)
 require(value["format"]=="poseidon-einsum-live-runner-v1"and
  value["proposal_binding"]==PROPOSAL_BINDING and value["proposal_sha256"]==PROPOSAL_SHA and
  canonical(value["tool_hashes"])==canonical(tools()),"runner_manifest_binding")
 return value
def validate(proposal):
 preparation.fields(proposal)
 require(proposal["binding"]==PROPOSAL_BINDING and canonical(proposal["provider"])==canonical(PROVIDER)and
  canonical(proposal["budget"])==canonical(BUDGET),"reviewed_provider_or_budget")
 require(all(c["options"]==OPTIONS and c["resource_profile"]==PROFILE and c["new_live_task_required"]is True and
  c["baseline_task_is_closed"]is True for c in proposal["cases"]),"public_generation_boundary")
 require(proposal["source_stage_identity"]["backend"]["format"]=="poseidon-backend-identity-v2","poseidon_backend_required")
 return proposal
def execution_view(proposal,manifest):
 view=copy.deepcopy(proposal);view["application_stage_identity"]=copy.deepcopy(proposal["source_stage_identity"])
 view["tool_hashes"]=copy.deepcopy(manifest["tool_hashes"])
 for case in view["cases"]:
  case["current_validation_sources"]=copy.deepcopy(proposal["source_stage_identity"]["validation"]["sources"])
 return view
def verify(value=None):
 require(sha(regular(PROPOSAL))==PROPOSAL_SHA,"original_proposal_bytes")
 original=read(PROPOSAL)
 if value is not None:require(canonical(value)==canonical(original),"exact_original_proposal")
 validate(original);preparation.verify(original);manifest=check_manifest()
 require(tuple(len(original["source_stage_identity"][stage]["sources"])for stage in
  ("generation","compilation","execution","validation"))==(631,625,625,625),"frozen_stage_source_inventory")
 require(sha(regular(original["baseline_report"]))==original["baseline_report_sha256"],"offline_evidence_changed")
 prior=read(original["baseline_report"]);check_binding(prior)
 require(prior["status"]=="passed"and prior["passed"]==8 and prior["failed"]==0 and
  prior["identity_sha256"]==original["identity_sha256"]and
  prior["tolerance"]==dict(atol=1e-5,rtol=1e-4),"offline_numerical_evidence")
 for command,want in ((["git","branch","--show-current"],"lhy-agent-dsl"),
                      (["git","rev-parse","HEAD"],"7588e48093631eaf8385506e3db058e7ca3cfe7e")):
  require(subprocess.check_output(command,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
 return execution_view(original,manifest)
def checked_authorization(value,proposal,proposal_sha):
 keys={"format","approved","authorization","binding","budget","cases","human_message",
       "proposal_binding","proposal_sha256","provider","question_item_id"}
 require(type(value)is dict and set(value)==keys,"human_authorization_fields");check_binding(value)
 require(value["format"]=="poseidon-einsum-human-authorization-v1"and value["approved"]is True and
  value["authorization"]=="human_explicit_case_budget_approval"and
  proposal["binding"]==PROPOSAL_BINDING and proposal_sha==PROPOSAL_SHA and
  (value["proposal_binding"],value["proposal_sha256"])==(PROPOSAL_BINDING,PROPOSAL_SHA)and
  canonical(value["budget"])==canonical(BUDGET)and canonical(value["provider"])==canonical(PROVIDER)and
  value["cases"]==list(CASES),"human_authorization_scope")
 require(type(value["human_message"])is str and 1<=len(value["human_message"].encode())<=4096 and
  type(value["question_item_id"])is str and 1<=len(value["question_item_id"])<=512,"human_authorization_record")
 return value
def authority(receipt,proposal,proposal_sha):
 fields={"format","approved","proposal_binding","proposal_sha256","runner_manifest_sha256","tool_hashes",
         "authorization","authorization_record","authorization_sha256"}
 require(type(receipt)is dict and set(receipt)==fields,"approval_record_fields")
 require(receipt["format"]=="poseidon-einsum-live-approval-v1"and receipt["approved"]is True and
  receipt["proposal_binding"]==PROPOSAL_BINDING==proposal["binding"]and
  receipt["proposal_sha256"]==PROPOSAL_SHA==proposal_sha and
  receipt["runner_manifest_sha256"]==sha(regular(MANIFEST))and
  receipt["tool_hashes"]==tools()==proposal["tool_hashes"]and
  receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
 require(receipt["authorization_record"]==str(AUTHORIZATION)and
  receipt["authorization_sha256"]==sha(regular(AUTHORIZATION)),"human_authorization_bytes")
 original=read(regular(PROPOSAL));validate(original)
 require(canonical(proposal)==canonical(execution_view(original,check_manifest())),"execution_view_changed")
 checked_authorization(read(AUTHORIZATION),original,proposal_sha)
def totals(states):
 generations=http=0;resolved=True
 for state in states:
  require(type(state["generations"])is int and type(state["http_attempts"])is int and
   0<=state["generations"]<=4 and 0<=state["http_attempts"]<=16,"case_call_ceiling")
  reserved=[e for e in state["ledger"]if e.get("event")=="http_reserved"]
  require([e["index"]for e in reserved]==list(range(state["http_attempts"])),"reservation_ledger")
  checkpoints=[e for e in state["ledger"]if e.get("event")=="provider_checkpoint"]
  outcomes=checkpoints[-1].get("outcomes",[])if checkpoints else[]
  resolved=resolved and len(outcomes)==state["http_attempts"]and all(o.get("index")==i and
   o.get("status")in("response_received","failed")for i,o in enumerate(outcomes))
  generations+=state["generations"];http+=state["http_attempts"]
 require(generations<=BUDGET["max_generations"]and http<=BUDGET["max_http_attempts"],"campaign_call_ceiling")
 return dict(generations=generations,http_attempts=http,calls_resolved=resolved)
def terminal(rows,states):
 require(len(rows)<=len(CASES)and len({r["case"]for r in rows})==len(rows)and
  all(r["case"]in CASES and r["status"]in("passed","failed")for r in rows),"result_case_scope")
 data=totals(states);passed=sum(r["status"]=="passed"for r in rows)
 return dict(status="complete"if len(rows)==len(CASES)and data["calls_resolved"]else"stopped",
  passed=passed,failed=len(rows)-passed,not_run=len(CASES)-len(rows),**data)
