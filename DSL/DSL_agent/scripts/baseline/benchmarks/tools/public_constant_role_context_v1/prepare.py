"""Offline draft fixture preparation. Never qualifies, executes, or creates a Provider."""
import hashlib,json
from pathlib import Path
import public_math_context as m
HERE=Path(__file__).resolve().parent
ROOT=(Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
PROPOSAL=ROOT/"docs/baseline/agent-gelu-live-proposal-r1306.json"
PROPOSAL_SHA="7dc2a3c0b1d7c8c8a6cbfe6f584297f058b74833ac70602cc92836df9089e273"
OUT=HERE/"prepared-v2"
def main():
 m.require(not OUT.exists(),"fresh_draft_preparation")
 m.require(hashlib.sha256(PROPOSAL.read_bytes()).hexdigest()==PROPOSAL_SHA,"historical_proposal_bytes")
 prior=json.loads(PROPOSAL.read_bytes())
 m.require(prior["binding"]=="4f4af52bb39c48169abee43e335e81792891460447032130bf1fd91e31dc4e5e","historical_proposal_binding")
 OUT.mkdir();rows=[]
 for case in prior["cases"]:
  p=ROOT/case["full_request_file"]
  m.require(hashlib.sha256(p.read_bytes()).hexdigest()==case["full_request_sha256"],"historical_fixture_bytes")
  q=json.loads(p.read_bytes());before=m.canonical(q)
  new=m.upgrade(q);context=m.project(new)
  m.require(before==m.canonical(q),"historical_request_immutable")
  request_path=OUT/(case["id"]+"-request.json");context_path=OUT/(case["id"]+"-context.json")
  request_path.write_bytes(m.canonical(new));context_path.write_bytes(m.canonical(context))
  rows.append(dict(case=case["id"],source_request=str(p.relative_to(ROOT)),source_sha256=case["full_request_sha256"],
   historical_request_id=q["request_id"],new_draft_request_id=new["request_id"],policy=m.POLICY,
   request_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
   context_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),context_bytes=context_path.stat().st_size,
   mathematical_roles_binding=context["public_weight_binding"]["math_roles"]["binding"],
   mathematical_elements=context["public_weight_binding"]["math_roles"]["math_elements"],
   production_applied=False,compiler_validated=False,encrypted_execution_validated=False,new_agent_generated=False))
 report=dict(format="poseidon-public-math-role-draft-preparation-v1",status="prepared",cases=rows,
  provider_calls=0,http_attempts=0,compiler_calls=0,encrypted_executions=0,installs=0,downloads=0,
  source_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
  historical_proposal_sha256=PROPOSAL_SHA,historical_records_rewritten=False)
 report["binding"]=m.digest(report);(OUT/"report.json").write_bytes(m.canonical(report))
 print(json.dumps(dict(status="draft_prepared",cases=len(rows),binding=report["binding"],provider_calls=0,
  new_draft_request_ids=[r["new_draft_request_id"] for r in rows]),sort_keys=True))
if __name__=="__main__":main()
