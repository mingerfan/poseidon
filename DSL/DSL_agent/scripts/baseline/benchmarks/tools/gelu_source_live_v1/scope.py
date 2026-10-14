"""Frozen r1306 GELU scope. Only standard-library file/authority checks."""
import copy,hashlib,json,math,subprocess,fcntl
from contextlib import contextmanager
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
BASE=ROOT/"scripts/baseline"
WORK=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux")
RESULTS=WORK/"results"
RUN=RESULTS/"agent-gelu-source-live-v1"
PROPOSAL=ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json"
MANIFEST=HERE/"runner-manifest.json"
AUTHORIZATION=ROOT/"docs/baseline/agent-gelu-live-authorization-r1306.json"
PROPOSAL_BINDING="4f4af52bb39c48169abee43e335e81792891460447032130bf1fd91e31dc4e5e"
PROPOSAL_SHA="7dc2a3c0b1d7c8c8a6cbfe6f584297f058b74833ac70602cc92836df9089e273"
AUTHORIZATION_SHA="f33cc3d800d9a291c2da9e13261eb3f088b9c92fdd96130a2f26716013a556d5"
CASES=("tanh_module","tanh_functional","tanh_input_keyword")
CASE_IDS=("08207c445f7965168b87736d0f046830d630e9bf354adb26a2628926f0fa21ad","3dbe1a88bfb2a0500fccd52c7304cbf0300b4635f0ed30251a2a76491d068d2b","917106345a4005cbef28fb9ca2580dab7f69e6b0e61ad7949f52574e084aea2d")
OPTIONS=dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-v1")
PROFILE="general-tensors-v1"
PROVIDER=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",stream=True)
LIMITS=dict(generations=12,http_attempts=48,repairs_per_case=3,transport_retries_per_generation=3,api_timeout_seconds=1200,max_tokens=384000,wall_seconds=43200,case_wall_seconds=14400,api_concurrency=1,native_concurrency=1,compile_jobs=2,link_jobs=1,case_artifact_bytes=4*1024**3,global_artifact_bytes=32*1024**3,minimum_free_bytes=4*1024**3)
BUDGET=dict(api_concurrency=1,batch_wall_seconds=43200,case_wall_seconds=14400,compile_jobs=2,currency_limit=None,global_result_bytes=32*1024**3,link_jobs=1,max_generations=12,max_http_attempts=48,native_concurrency=1,per_case_bytes=4*1024**3,reserve_bytes=4*1024**3)
TOOL_NAMES=("scope.py","run.py","test_scope.py","test_run.py","package_checks.py")
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
 require(m["format"]=="poseidon-gelu-source-live-runner-v1" and m["proposal_binding"]==PROPOSAL_BINDING and m["proposal_sha256"]==PROPOSAL_SHA and m["tool_hashes"]==tools(),"runner_manifest_binding")
 return m
def validate(p):
 require(type(p) is dict,"proposal_fields");check_binding(p)
 require(p["binding"]==PROPOSAL_BINDING and p["format"]=="poseidon-gelu-live-proposal-v1" and p["status"]=="awaiting_human_approval","original_proposal_binding")
 require(p["provider"]==PROVIDER and p["limits"]==LIMITS and all(type(v) is int for v in p["limits"].values()),"reviewed_provider_or_budget")
 require(p["explicit_currency_cap"] is None and p["retries_may_repeat_billing"] is True,"billing_disclosure")
 require(tuple(c["id"] for c in p["cases"])==CASES and tuple(c["request_id"] for c in p["cases"])==CASE_IDS,"reviewed_cases")
 require(p["backend_identity"]["backend"]=="poseidon_cpu" and p["source_stage_identity"]["format"]=="poseidon-application-identities-v1","poseidon_backend_required")
 require(p["validation"]==dict(level="numerical",backend="poseidon_cpu",qualification_groups=4,new_input_groups=3,package_reload=True,repeat_execution=True,cache_reuse=True,atol=1e-5,rtol=1e-4,source_variant_mathematics="tanh",accuracy_against_original_application="not certified"),"reviewed_validation")
 return p
def verify(value=None):
 require(sha(regular(PROPOSAL))==PROPOSAL_SHA,"original_proposal_bytes")
 p=read(PROPOSAL) if value is None else value
 require(canonical(p)==canonical(read(PROPOSAL)),"exact_original_proposal");validate(p);m=check_manifest()
 inventory={}
 def freeze(path,want):
  path=regular(path);require(sha(path)==want,"frozen_source_changed");inventory[str(path)]=want
 for stage in ("generation","compilation","validation"):
  sources=p["source_stage_identity"][stage]["sources"]
  require(type(sources) is dict and len(sources)==684,"frozen_stage_source_inventory")
  for rel,want in sources.items():
   require(type(rel) is str and not Path(rel).is_absolute() and ".." not in Path(rel).parts,"frozen_source_scope");freeze(BASE/rel,want)
 for c in p["cases"]:
  require(c["source_format"]=="poseidon-trusted-pytorch-v15" and c["raw_source_sent_to_provider"] is False and c["private_reference_sent"] is False and c["deterministic_dsl_answer_sent"] is False,"public_generation_boundary")
  require(c["source_manifest"]==f"scripts/baseline/cases/trusted-pytorch-gelu-v15/{c['id']}.json","reviewed_source_manifest")
  mp=ROOT/c["source_manifest"];freeze(mp,c["source_manifest_sha256"])
  for rel,want in c["source_files"].items():
   require(not Path(rel).is_absolute() and ".." not in Path(rel).parts,"fixture_file_scope");freeze(mp.parent/rel,want)
  for pathkey,hashkey in (("full_request_file","full_request_sha256"),("generation_context_file","generation_context_sha256")):
   rel=c[pathkey];require(not Path(rel).is_absolute() and ".." not in Path(rel).parts,"request_file_scope");freeze(ROOT/rel,c[hashkey])
   require(read(ROOT/rel)["request_id"]==c["request_id"],"exact_frozen_request_id")
 for key,want in inventory.items():require(sha(regular(key))==want,"file_changed_during_read")
 require(tools()==m["tool_hashes"],"runner_tools_changed_during_read")
 for cmd,want in ((["git","branch","--show-current"],"lhy-agent-dsl"),(["git","rev-parse","HEAD"],"7588e48093631eaf8385506e3db058e7ca3cfe7e")):
  require(subprocess.check_output(cmd,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
 view=copy.deepcopy(p)
 view.update(budget=BUDGET,application_backend_identity=p["backend_identity"],application_stage_identity=p["source_stage_identity"],tool_hashes=m["tool_hashes"])
 for c in view["cases"]:c.update(request_file=c["full_request_file"],current_validation_sources=p["source_stage_identity"]["validation"]["sources"])
 return view
def authority(receipt,proposal,proposal_sha):
 require(type(receipt) is dict and set(receipt)=={"format","approved","proposal_binding","proposal_sha256","runner_manifest_sha256","tool_hashes","authorization","authorization_record","authorization_sha256"},"approval_record_fields")
 require(receipt["format"]=="poseidon-gelu-source-live-approval-v1" and receipt["approved"] is True and receipt["proposal_binding"]==PROPOSAL_BINDING==proposal["binding"] and receipt["proposal_sha256"]==PROPOSAL_SHA==proposal_sha and receipt["runner_manifest_sha256"]==sha(regular(MANIFEST)) and receipt["tool_hashes"]==tools()==proposal["tool_hashes"] and receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
 require(receipt["authorization_record"]==str(AUTHORIZATION) and receipt["authorization_sha256"]==AUTHORIZATION_SHA==sha(regular(AUTHORIZATION)),"original_human_authorization")
def totals(states):
 g=h=0;resolved=True
 for s in states:
  require(type(s["generations"]) is int and type(s["http_attempts"]) is int and 0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_call_ceiling")
  reserved=[e for e in s["ledger"] if e.get("event")=="http_reserved"]
  require([e["index"] for e in reserved]==list(range(s["http_attempts"])),"reservation_ledger")
  checkpoints=[e for e in s["ledger"] if e.get("event")=="provider_checkpoint"];outcomes=checkpoints[-1].get("outcomes",[]) if checkpoints else []
  resolved=resolved and len(outcomes)==s["http_attempts"] and all(o.get("index")==i and o.get("status") in ("response_received","failed") for i,o in enumerate(outcomes))
  g+=s["generations"];h+=s["http_attempts"]
 require(g<=12 and h<=48,"campaign_call_ceiling");return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def terminal(rows,states):
 require(len(rows)<=3 and len({r["case"] for r in rows})==len(rows) and all(r["case"] in CASES for r in rows),"result_case_scope")
 data=totals(states);passed=sum(r["status"]=="passed" for r in rows)
 return dict(status="complete" if len(rows)==3 and data["calls_resolved"] else "stopped",passed=passed,failed=len(rows)-passed,not_run=3-len(rows),**data)
