"""Three bounded, serial offline replays. No provider or paid generation."""
import hashlib,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/"scripts/baseline"
sys.path.insert(0,str(BASE))
from benchmark_graph import digest
from unified_graph_contract import prepare,EXPLICIT_GUIDANCE
from unified_graph_lowering import lower
from compiler_configuration import PROFILE_SHA256,configuration
from semantic_benchmark_execution import runtime_sources
from audit_unified_candidate import verify_candidate
R=Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
out=R/"stage2-guidance-r99"
binding=json.loads((out/"compatibility.json").read_text())
assert digest(runtime_sources())==binding["source_digest"]
specs=[]
model=json.loads((BASE/"cases/unified-two-input-two-output.json").read_text())
for profile in (None,"hecate-unified-public-v1"):
 r=prepare(model,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),
           construction_profile=profile,generation_guidance=EXPLICIT_GUIDANCE)
 args=["--case",str(BASE/"cases/unified-two-input-two-output.json"),"--compiler-configuration","seal-cpu-eva-w45-v1"]
 if profile:args+=["--unified-profile","public-v1"]
 specs.append(("public" if profile else "native",r,lower(r),args))
old=R/"agent-deepseek-twrg_yso";previous=json.loads((old/"request.json").read_text())
r=prepare(previous["model"],PROFILE_SHA256,previous["compiler_configuration"],
          helper_profile=previous["upstream_helpers"]["profile"],
          helper_exercise=previous["upstream_exercise"]["required_helpers"],
          generation_guidance=EXPLICIT_GUIDANCE)
source=(old/"attempt-00/candidate.py").read_text()
specs.append(("helper_replay",r,source,["--case",str(old/"model.json"),
 "--compiler-configuration",previous["compiler_configuration"]["name"],
 "--unified-helpers",previous["upstream_helpers"]["profile"],
 "--unified-helper-exercise","HE_SiLU"]))
rows=[]
for name,request,source,args in specs:
 assert digest(runtime_sources())==binding["source_digest"]
 replay=out/(name+"-scripted-responses.json")
 bad=source.replace("def golden(","def main(",1)
 assert bad!=source
 responses=[json.dumps(dict(schema=1,request_id=request["request_id"],hecate_source=s)) for s in (bad,source)]
 with replay.open("x") as f:json.dump(responses,f)
 command=[sys.executable,"-B",str(BASE/"run_candidate.py"),"--inside",*args,
          "--unified-guidance","explicit-v1","--replay",str(replay),"--max-repairs","1"]
 started=time.monotonic()
 with (out/(name+"-execution.log")).open("x") as f:
  p=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 text=(out/(name+"-execution.log")).read_text()
 match=re.search(r"Candidate evidence: (.+)",text)
 if not match:raise RuntimeError("Missing evidence directory: "+name)
 evidence=Path(match.group(1));report=json.loads((evidence/"report.json").read_text())
 assert report["agent_calls"]==0
 row=dict(id=name,command=command,seconds=time.monotonic()-started,evidence=str(evidence),
          exit_code=p.returncode,status=report["status"])
 rows.append(row)
 with (out/"execution-progress.json").open("w") as f:json.dump(rows,f,indent=2)
 if p.returncode:
  print(text);raise RuntimeError("Offline candidate failure; evidence preserved: "+name)
 assert json.loads((evidence/"request.json").read_text())==request
 feedback=json.loads((evidence/"attempt-00/feedback.json").read_text())
 assert feedback["layer"]=="static_check" and "named golden" in feedback["diagnostic"]
 assert "Contract clarification:" in feedback["diagnostic"]
 row["audit"]=verify_candidate(evidence)
 row["feedback_verified"]=True
 print(name,"passed (scripted replay, not new Agent generation)",flush=True)
# Verify the new flag cannot silently affect legacy schemas.
args=[sys.executable,"-B",str(BASE/"run_candidate.py"),"--inside","--case",
      str(BASE/"cases/linear-example.json"),"--unified-guidance","explicit-v1","--prepare"]
with (out/"legacy-rejection.log").open("x") as f:
 p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=60)
assert p.returncode!=0
assert "Generation guidance requires unified graph" in (out/"legacy-rejection.log").read_text()
assert digest(runtime_sources())==binding["source_digest"]
body=dict(format="poseidon-guidance-offline-r99-v1",source_digest=binding["source_digest"],
          compatibility_binding=binding["binding"],rows=rows,passed=len(rows),failed=0,skipped=0,
          legacy_guidance_rejection=True,paid_calls=0,new_agent_generation=False,
          stage2_complete=False,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
body["binding"]=digest(body)
with (out/"offline-execution.json").open("x") as f:json.dump(body,f,indent=2);f.write("\n")
print(json.dumps({k:body[k] for k in ("binding","passed","failed","skipped","paid_calls")}))
