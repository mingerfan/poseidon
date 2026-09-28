"""Three scoped scripted composite replays; no API or new Agent success."""
import json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from unified_graph_contract import prepare,validate_candidate
from unified_graph_lowering import lower
from unified_public_exercises import golden_variant
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
out=R/"stage2-composite-guidance-r104"
binding=json.loads((out/"compatibility.json").read_text())
plan=json.loads((ROOT/"docs/baseline/stage2-guidance-retest-r100.json").read_text())
old=next(c for c in plan["cases"] if c["id"]=="construct_000_0")
r=old["request"];model_path=out/"offline-model.json"
if model_path.exists():assert json.loads(model_path.read_text())==r["model"]
else:model_path.write_text(json.dumps(r["model"])+"\n")
progress=out/"offline-progress.json"
prior=json.loads(progress.read_text()) if progress.exists() else []
assert len({x["id"] for x in prior})==len(prior)
assert {x["id"] for x in prior}<=set(("arithmetic","mutation","view_copy"))
rows=[]
for feature in ("arithmetic","mutation","view_copy"):
 assert digest(runtime_sources())==binding["source_digest"]
 exercise="unified-public-composite-array-"+feature
 request=prepare(r["model"],r["compiler_profile_sha256"],r["compiler_configuration"],exercise,
                 construction_profile=r["construction_profile"],constant_policy=r["constant_origins"].get("policy"),
                 generation_guidance="explicit-v2")
 source=golden_variant(lower(request),exercise,request)
 validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
 responses=[json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=s))
            for s in (source.replace("def golden(","def main(",1),source)]
 replay=out/("offline-"+feature+"-responses.json")
 retained=next((x for x in prior if x["id"]==feature),None)
 if retained is not None:
  folder=Path(retained["evidence"]);report=json.loads((folder/"report.json").read_text())
  assert json.loads((folder/"request.json").read_text())==request
  assert json.loads(replay.read_text())==responses
  assert (folder/"attempt-01/candidate.py").read_text()==source
  assert report["agent_calls"]==0 and report["status"]==retained["status"]!="running"
  if report["status"]=="passed":retained["audit"]=verify_candidate(folder)
  rows.append(retained);print(feature,"retained; not rerun",flush=True);continue
 with replay.open("x") as f:json.dump(responses,f)
 command=[sys.executable,"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(model_path),
          "--compiler-configuration",r["compiler_configuration"]["name"],"--unified-profile","public-v1",
          "--unified-exercise",exercise,"--unified-guidance","explicit-v2","--replay",str(replay),"--max-repairs","1"]
 log=out/("offline-"+feature+".log");start=time.monotonic()
 with log.open("x") as f:proc=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 text=log.read_text();match=re.search(r"Candidate evidence: (.+)",text)
 if not match:raise RuntimeError(text)
 folder=Path(match.group(1));report=json.loads((folder/"report.json").read_text())
 assert report["agent_calls"]==0
 row=dict(id=feature,evidence=str(folder),exit_code=proc.returncode,seconds=time.monotonic()-start,status=report["status"])
 rows.append(row);(out/"offline-progress.json").write_text(json.dumps(rows,indent=2)+"\n")
 if proc.returncode:
  print(feature,"failed; retaining independent cases",flush=True)
  continue
 assert json.loads((folder/"request.json").read_text())==request
 feedback=json.loads((folder/"attempt-00/feedback.json").read_text())
 for item in request["generation_guidance"]["construction_requirements"]:
  assert item["instruction"] in feedback["diagnostic"]
 row["audit"]=verify_candidate(folder)
 print(feature,"scripted real FHE replay passed",flush=True)
assert digest(runtime_sources())==binding["source_digest"]
body=dict(format="poseidon-composite-guidance-offline-v1",source_digest=binding["source_digest"],
          compatibility_binding=binding["binding"],rows=rows,passed=sum(x["status"]=="passed" for x in rows),failed=sum(x["status"]!="passed" for x in rows),skipped=0,
          paid_calls=0,new_agent_generation=False,stage2_complete=False)
body["binding"]=digest(body)
with (out/"offline-execution.json").open("x") as f:json.dump(body,f,indent=2);f.write("\n")
print(json.dumps({k:body[k] for k in ("binding","passed","paid_calls")}))
