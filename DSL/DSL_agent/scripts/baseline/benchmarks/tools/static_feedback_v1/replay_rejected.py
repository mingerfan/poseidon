"""Read-only authoritative recheck of original Agent responses. No compiler/API."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path("scripts/baseline").resolve()))
from component_contract import validate_program
from application_diagnostics import attach
from benchmark_graph import require
proposal=json.loads(Path("docs/baseline/agent-mha-v16-live-proposal-r1313.json").read_text())
old=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-source-live-v2")
rows=[]
for case in proposal["cases"]:
    path=Path(case["request_file"])
    require(hashlib.sha256(path.read_bytes()).hexdigest()==case["request_file_sha256"],"original_request_bytes")
    request=json.loads(path.read_text())
    for response in sorted((old/case["id"]).glob("tasks/*/response-*.json")):
        candidate=json.loads(json.loads(response.read_text())["raw"])
        result=validate_program(candidate,request)
        row=dict(case=case["id"],response=response.name,status=result["status"],
                 candidate_source_sha256=hashlib.sha256(candidate["hecate_source"].encode()).hexdigest(),
                 request_id=request["request_id"],compiler_calls=0,encrypted_calls=0,provider_calls=0)
        if result["status"]=="rejected":
            checked=attach(dict(status="failed",failure=result["failure"]),candidate=candidate,request=request)
            row.update(diagnostic=checked["failure"],public_feedback=checked["public_feedback"])
        rows.append(row)
require(len(rows)==7 and sum(r["status"]=="rejected" for r in rows)==5,"original_response_count")
reasons=[r.get("diagnostic",{}).get("reason_code") for r in rows]
require(reasons.count("expanded_work_budget")==3 and reasons.count("native_signature_mismatch")==2,"unchanged_rejection_layers")
out=dict(format="poseidon-original-mha-static-feedback-replay-v1",rows=rows,provider_calls=0,
         compiler_calls=0,encrypted_calls=0,original_responses_modified=False)
Path(__file__).with_name("original-response-recheck.json").write_text(json.dumps(out,sort_keys=True,indent=2))
print(json.dumps(dict(responses=len(rows),rejected=5,static_only_accepted=2,provider_calls=0,
                     work_costs=[r["diagnostic"]["context"]["observed"] for r in rows
                                 if r.get("diagnostic",{}).get("reason_code")=="expanded_work_budget"])),flush=True)
