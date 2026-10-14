"""Finite reviewed application-source campaign; no Provider or credentials."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
PROPOSAL=ROOT/"docs/baseline/agent-v12-live-proposal-r1232.json"
CASES=("linear","norm","split_weight")
PROVIDER=dict(name="deepseek",model="deepseek-flash",reasoning_effort="high",
 max_tokens=384000,max_repairs=3,provider_retries=3,stream=True,timeout_seconds=1200)
BUDGET=dict(api_concurrency=1,batch_wall_seconds=43200,case_wall_seconds=14400,
 compile_jobs=2,currency_limit=None,global_result_bytes=32*1024**3,link_jobs=1,
 max_generations=12,max_http_attempts=48,native_concurrency=1,
 per_case_bytes=4*1024**3,reserve_bytes=4*1024**3)
def require(ok,code):
 if not ok:raise ValueError(code)
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
 def pairs(items):
  result={}
  for k,v in items:
   require(k not in result,"duplicate_json_field");result[k]=v
  return result
 return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
  parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite_json")))
def validate(proposal):
 require(proposal["binding"]==digest({k:v for k,v in proposal.items() if k!="binding"}),"proposal_binding")
 require(proposal["approved"] is False,"proposal_is_not_authorization")
 require(proposal["source_root"]==str(ROOT) and proposal["branch"]=="lhy-agent-dsl","source_checkout")
 require(proposal["provider"]==PROVIDER and proposal["budget"]==BUDGET,"reviewed_provider_or_budget")
 require(tuple(c["id"] for c in proposal["cases"])==CASES,"reviewed_cases")
 for c in proposal["cases"]:
  require(c["source_format"]=="poseidon-trusted-pytorch-v12" and
   c["free_generation"] is True and c["directed_construction"] is False,"reviewed_generation_mode")
  require(c["request_id"]==read(ROOT/c["request_file"])["request_id"],"case_request_identity")
 return proposal
def verify(proposal):
 validate(proposal)
 expected=dict(proposal["source_hashes"])
 for c in proposal["cases"]:
  expected.update(c["source_files"])
  expected[c["request_file"]]=c["request_file_sha256"]
  p=Path(c["rule_qualification_evidence"])
  require(p.is_file() and not p.is_symlink() and sha(p)==c["rule_qualification_evidence_sha256"],
   "qualification_receipt_changed")
  qualified=read(p);package=Path(qualified["package"])
  metadata=read(package.parent/"task.json");reference=c["reference_binding"]
  require(sha(package/"model-provenance.json")==reference["model_provenance_sha256"] and
   digest(metadata["identity"]["validation"])==reference["qualification_validation_identity_sha256"],
   "qualification_reference_binding")
  for name,want in reference["validation_sources"].items():
   relative="scripts/baseline/"+name
   require(relative not in expected or expected[relative]==want,"reference_sources_binding")
   expected[relative]=want
 for relative,want in expected.items():
  p=ROOT/relative
  require(p.resolve().is_relative_to(ROOT) and p.is_file() and not p.is_symlink()
   and sha(p)==want,"frozen_source_changed")
 for args,want in ((["git","branch","--show-current"],proposal["branch"]),
                   (["git","rev-parse","HEAD"],proposal["head"])):
  require(subprocess.check_output(args,cwd=ROOT,text=True,timeout=10).strip()==want,"checkout_changed")
 return proposal
def authority(receipt,proposal,proposal_sha):
 require(set(receipt)=={"format","approved","proposal_binding","proposal_sha256","authorization"},
  "approval_record_fields")
 require(receipt["format"]=="poseidon-user-campaign-approval-v1" and receipt["approved"] is True
  and receipt["proposal_binding"]==proposal["binding"] and receipt["proposal_sha256"]==proposal_sha
  and receipt["authorization"]=="human_explicit_case_budget_approval","approval_required")
def totals(states):
 g=h=0;resolved=True
 for s in states:
  require(type(s["generations"]) is int and type(s["http_attempts"]) is int
   and 0<=s["generations"]<=4 and 0<=s["http_attempts"]<=16,"case_call_ceiling")
  reserved=[e for e in s["ledger"] if e.get("event")=="http_reserved"]
  require([e["index"] for e in reserved]==list(range(s["http_attempts"])),"reservation_ledger")
  checkpoints=[e for e in s["ledger"] if e.get("event")=="provider_checkpoint"]
  outcomes=checkpoints[-1].get("outcomes",[]) if checkpoints else []
  resolved=resolved and len(outcomes)==s["http_attempts"] and all(
   o.get("index")==i and o.get("status") in ("response_received","failed")
   for i,o in enumerate(outcomes))
  g+=s["generations"];h+=s["http_attempts"]
 require(g<=12 and h<=48,"campaign_call_ceiling")
 return dict(generations=g,http_attempts=h,calls_resolved=resolved)
def terminal(rows,states):
 account=totals(states)
 status="complete" if len(rows)==3 and all(r["status"] in ("passed","failed") for r in rows) else "stopped"
 if not account["calls_resolved"]:status="stopped"
 return dict(status=status,passed=sum(r["status"]=="passed" for r in rows),
  failed=sum(r["status"]=="failed" for r in rows),not_run=3-len(rows),**account)
