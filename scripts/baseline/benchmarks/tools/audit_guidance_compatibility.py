"""Replay frozen r97 static checks, without inference, compilation or response edits."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest,canonical
from unified_graph_contract import prepare,validate_candidate,validate_request,static_repair_hint,EXPLICIT_GUIDANCE
from semantic_benchmark_execution import runtime_sources
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results/stage2-guidance-r99")
baseline=json.loads((R/"before.json").read_text())
assert baseline["binding"]==digest({k:v for k,v in baseline.items() if k!="binding"})
rows=[]
for row in baseline["rows"]:
 request=json.loads(Path(row["request"]).read_text())
 assert request["request_id"]==row["request_id"]
 kw=dict(construction_profile=request.get("construction_profile"),
         constant_policy=request["constant_origins"].get("policy"),
         helper_profile=request.get("upstream_helpers",{}).get("profile"),
         helper_exercise=request.get("upstream_exercise",{}).get("required_helpers"))
 def make(**extra):
  return prepare(request["model"],request["compiler_profile_sha256"],request.get("compiler_configuration"),
                 request.get("construction_exercise",{}).get("id"),**kw,**extra)
 assert canonical(make())==canonical(request),row["id"]
 guided=make(generation_guidance=EXPLICIT_GUIDANCE)
 validate_request(guided)
 out=dict(id=row["id"],old_request_unchanged=True,guided_request_id=guided["request_id"],attempts=[])
 for item in row["attempts"]:
  raw=Path(item["response"]).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==item["response_sha256"]
  candidate=json.loads(raw)
  verdicts=[]
  for r in (request,guided):
   c=dict(candidate,request_id=r["request_id"])
   try:
    val=validate_candidate(c,r)
    # Only this documented witness field binds the newly opted-in request.
    # Verify its identity before comparing semantics with the old request.
    if r is guided and "upstream_exercise" in val:
     assert val["upstream_exercise"]["request_id"]==guided["request_id"]
     val=copy.deepcopy(val)
     val["upstream_exercise"]["request_id"]=request["request_id"]
    result={"accepted":True,"metadata_sha256":digest(val)}
   except Exception as error:
    result={"accepted":False,"error_type":type(error).__name__,"diagnostic":str(error)}
   verdicts.append(result)
  expected={k:v for k,v in item.items() if k not in ("response","response_sha256")}
  assert verdicts==[expected,expected],(row["id"],item["response"])
  hint=static_repair_hint(candidate["hecate_source"],guided) if not expected["accepted"] else ""
  out["attempts"].append(dict(response=item["response"],response_sha256=item["response_sha256"],
                             verdict_unchanged=True,static_accepted=expected["accepted"],hint=hint))
 rows.append(out)
sources=runtime_sources();changed=sorted(k for k in set(sources)|set(baseline["sources"]) if sources.get(k)!=baseline["sources"].get(k))
assert changed==["scripts/baseline/run_candidate.py","scripts/baseline/test_unified_generation_guidance.py","scripts/baseline/unified_graph_contract.py"],changed
report=dict(format="poseidon-guidance-compatibility-v1",before_binding=baseline["binding"],
            sources=sources,source_digest=digest(sources),changed=changed,rows=rows,
            requests_unchanged=len(rows),responses_unchanged=sum(len(r["attempts"]) for r in rows),
            actionable_hints=sum(bool(a["hint"]) for r in rows for a in r["attempts"]),
            paid_calls=0,compiled=False,encrypted_execution=False)
report["binding"]=digest(report)
with (R/"compatibility.json").open("x") as f:json.dump(report,f,indent=2);f.write("\n")
print(json.dumps({k:report[k] for k in ("binding","source_digest","requests_unchanged","responses_unchanged","actionable_hints")}))
