"""Offline request/verdict compatibility across r97 and r100; no provider."""
import copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/"scripts/baseline"))
from benchmark_graph import digest,canonical
from unified_graph_contract import prepare,validate_candidate,validate_request
from semantic_benchmark_execution import runtime_sources
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
out=R/"stage2-composite-guidance-r104"
before=json.loads((out/"before.json").read_text())
assert before["binding"]==digest({k:v for k,v in before.items() if k!="binding"})
rows=[]
for batch in ("stage2-agent-pilot-r97-live","stage2-guidance-retest-r100-live"):
 for item in json.loads((R/batch/"report.json").read_text())["rows"]:
  folder=Path(item["evidence"]);request=json.loads((folder/"request.json").read_text())
  kw=dict(construction_profile=request.get("construction_profile"),
          constant_policy=request["constant_origins"].get("policy"),
          helper_profile=request.get("upstream_helpers",{}).get("profile"),
          helper_exercise=request.get("upstream_exercise",{}).get("required_helpers"))
  def make(version):
   return prepare(request["model"],request["compiler_profile_sha256"],request.get("compiler_configuration"),
                  request.get("construction_exercise",{}).get("id"),generation_guidance=version,**kw)
  assert canonical(make(request.get("generation_guidance",{}).get("version")))==canonical(request)
  guided=make("explicit-v2");validate_request(guided)
  checks=[]
  for path in sorted(folder.glob("attempt-*/response.txt")):
   raw=path.read_bytes();candidate=json.loads(raw);verdicts=[]
   for r in (request,guided):
    try:
     value=validate_candidate(dict(candidate,request_id=r["request_id"]),r)
     if r is guided and "upstream_exercise" in value:
      assert value["upstream_exercise"]["request_id"]==guided["request_id"]
      value=copy.deepcopy(value);value["upstream_exercise"]["request_id"]=request["request_id"]
     verdict=dict(accepted=True,metadata_sha256=digest(value))
    except (ValueError,TypeError) as e:verdict=dict(accepted=False,diagnostic=str(e),error_type=type(e).__name__)
    verdicts.append(verdict)
   assert verdicts[0]==verdicts[1],(batch,item["id"],str(path),verdicts)
   checks.append(dict(response=str(path),response_sha256=hashlib.sha256(raw).hexdigest(),verdict=verdicts[0]))
  rows.append(dict(batch=batch,id=item["id"],old_request_id=request["request_id"],v2_request_id=guided["request_id"],checks=checks))
sources=runtime_sources()
changed=sorted(k for k in set(sources)|set(before["sources"]) if sources.get(k)!=before["sources"].get(k))
assert changed==["scripts/baseline/run_candidate.py","scripts/baseline/test_unified_generation_guidance.py","scripts/baseline/unified_graph_contract.py"],changed
body=dict(format="poseidon-composite-guidance-compatibility-v1",before_binding=before["binding"],sources=sources,
          source_digest=digest(sources),changed=changed,rows=rows,requests_unchanged=len(rows),
          response_verdicts_equal=sum(len(r["checks"]) for r in rows),paid_calls=0,compiled=False,encrypted_execution=False,
          scope="Static old-to-v2 compatibility only; no historical execution rebound")
body["binding"]=digest(body)
with (out/"compatibility.json").open("x") as f:json.dump(body,f,indent=2);f.write("\n")
print(json.dumps({k:body[k] for k in ("binding","source_digest","requests_unchanged","response_verdicts_equal")}))
