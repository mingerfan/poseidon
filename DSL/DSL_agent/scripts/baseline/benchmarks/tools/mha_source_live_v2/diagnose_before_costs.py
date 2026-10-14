from pathlib import Path
import json,sys
sys.path.insert(0,str(Path("/home/lhohy/Code/Poseidon/scripts/baseline")))
from candidate_contract import validate_candidate
P=json.loads(Path("/home/lhohy/Code/Poseidon/docs/baseline/agent-mha-v16-live-proposal-r1313.json").read_text())
D=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/agent-mha-source-live-v2")
out=[]
for case in P["cases"]:
 request=json.loads(Path(case["request_file"]).read_text())
 for p in sorted((D/case["id"]).glob("tasks/*/response-*.json")):
  row=dict(case=case["id"],response=p.name,read_only=True,compiler_calls=0,encrypted_calls=0,provider_calls=0)
  try:
   response=json.loads(json.loads(p.read_text())["raw"])
   result=validate_candidate(response,request)
   row.update(status="accepted",operator_counts=result.get("operator_counts"))
  except Exception as error:
   row.update(status="rejected",error_type=type(error).__name__,diagnostic=str(error)[:500],location=getattr(error,"location",None),reason=getattr(error,"reason",None))
  out.append(row)
print(json.dumps(out),flush=True)
