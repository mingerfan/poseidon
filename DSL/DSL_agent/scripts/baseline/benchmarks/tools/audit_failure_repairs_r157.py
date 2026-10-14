"""One repaired fixture, plus read-only failure reclassification and capacity checks."""
import argparse,json,os,shlex,sys,subprocess,time
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from benchmark_graph import digest
from benchmark_runner import dump
from campaign_live_state import sealed
from stage2_agent_pilot_plan import sha
from run_stage2_guidance_retest import evidence_path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True,type=Path);p.add_argument("--inside",action="store_true");a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_contract import prepare
 from candidate_contract import strict_json,validate_candidate,request_rotations
 from seal_artifact_gate import inspect_artifacts
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output required")
 a.output.mkdir();before=runtime_sources()
 prior=RESULTS/"stage2-failure-repairs-r156-v2/report.json";old=json.loads(prior.read_text())
 assert old["source_hashes"]==before
 for f,h in old["parents"].items():assert sha(Path(f))==h
 assert old["tests"]==dict(run=51,failed=0,errors=0,skipped=0)
 baseline=RESULTS/"stage2-targeted-retry-result-r154.json";d=json.loads(baseline.read_text());rows=[]
 for row in d["rows"]:
  if row["terminal_failure_layer"]!="static_check":continue
  folder=Path(row["evidence"]);assert sha(folder/"report.json")==row["report_sha256"]
  request=strict_json((folder/"request.json").read_text());report=strict_json((folder/"report.json").read_text())
  attempt=report["attempts"][-1]["index"];response=folder/f"attempt-{attempt:02d}/response.txt"
  c=strict_json(response.read_text())
  try:
   validate_candidate(c,request);status="static_now_passes";diagnostic=None
  except (ValueError,TypeError,KeyError,IndexError) as e:status="still_static_rejected";diagnostic=str(e)
  rows.append(dict(id=row["id"],status=status,diagnostic=diagnostic,prior_diagnostic=report["attempts"][-1].get("diagnostic"),
    response=str(response),response_sha256=sha(response),request_sha256=sha(folder/"request.json"),
    actual_compilation=False,encrypted_execution=False))
 assert len(rows)==89,len(rows)
 dump(a.output/"static-review.json",sealed(dict(rows=rows,parent=str(baseline),parent_sha256=sha(baseline),source_hashes=before,new_paid_calls=0)))
 print("static",len(rows),sum(x["status"]=="static_now_passes" for x in rows),flush=True)
 # Re-check historical capacity artifacts under the improved current diagnostic.
 pipeline=RESULTS/"stage2-targeted-pipeline-diagnosis-r153.json";pipe=json.loads(pipeline.read_text())
 for f,h in pipe["parents"].items():assert sha(Path(f))==h
 gates=[]
 for row in pipe["rows"]:
  for item in row["artifact_metadata"]:
   hevm=Path(item["file"]);assert sha(hevm)==item["sha256"]
   request=json.loads((hevm.parents[2]/"request.json").read_text());cst=hevm.parent/"_hecate_golden.cst"
   try:
    inspect_artifacts(hevm.read_bytes(),cst.read_bytes(),rotation_steps=request_rotations(request),
       expected_inputs=len(request["layout"]["inputs"])+1,execution_abi=request["layout"]["execution_abi"],
       input_period=request["layout"]["input_slot_period"])
   except ValueError as e:
    assert "Invalid level/scale: output" in str(e),str(e)
    gates.append(dict(file=str(hevm),sha256=sha(hevm),cst_sha256=sha(cst),rejected=True,diagnostic=str(e)))
   else:raise ValueError("Invalid scale artifact unexpectedly accepted")
 transparent=RESULTS/"agent-deepseek-ho8ulryh";request=json.loads((transparent/"request.json").read_text())
 out=transparent/"attempt-03/output";hevm=out/"lowered._hecate_golden.hevm";cst=out/"_hecate_golden.cst"
 try:
  inspect_artifacts(hevm.read_bytes(),cst.read_bytes(),rotation_steps=request_rotations(request),expected_inputs=5,
   execution_abi=request["layout"]["execution_abi"],input_period=4)
 except ValueError as e:
  assert "Transparent ciphertext risk" in str(e),str(e);transparent_diagnostic=str(e)
 else:raise ValueError("Transparent original not rejected")
 dump(a.output/"artifact-review.json",sealed(dict(capacity_parent_sha256=sha(pipeline),verified_parent_files=len(pipe["parents"]),
  scale_rejections=gates,transparent_diagnostic=transparent_diagnostic,transparent_artifact_sha256=sha(hevm),
  new_compilation=False,new_encrypted_execution=False)))
 # Only the fixture whose public vector shape changed is executed again.
 request=json.loads((RESULTS/"agent-deepseek-xpl3ywtu/request.json").read_text());exercise=request["construction_exercise"]["id"]
 r=prepare(request["model"],request["compiler_profile_sha256"],request["compiler_configuration"],exercise,
   construction_profile=request["construction_profile"],constant_policy=request["constant_origins"].get("policy"),generation_guidance="explicit-v5")
 fixture=BASE/"benchmarks/fixtures/failure-repairs-r156/construct_088_0.py";fh=sha(fixture);source=fixture.read_text()
 validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
 work=a.output/"construct_088_0";work.mkdir();dump(work/"model.json",r["model"])
 dump(work/"responses.json",[json.dumps(dict(schema=1,request_id=r["request_id"],hecate_source=source))])
 command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
  "--unified-guidance","explicit-v5","--compiler-configuration","seal-cpu-eva-w45-v1","--unified-exercise",exercise,
  "--unified-profile","public-v1","--replay",str(work/"responses.json"),"--max-repairs","0"]
 with (work/"run.log").open("x") as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 folder=evidence_path(work/"run.log",RESULTS);report=json.loads((folder/"report.json").read_text());read_request(folder/"request.json",r)
 assert report["agent_calls"]==0 and runtime_sources()==before and sha(fixture)==fh
 row=dict(id="construct_088_0",status=report["status"],evidence=str(folder),report_sha256=sha(folder/"report.json"),
  fixture_sha256=fh,manual_repair=True,agent_generation=False,failure_layer=report["attempts"][-1].get("failure_layer"),
  diagnostic=report["attempts"][-1].get("diagnostic"))
 if not code and report["status"]=="passed":
  row["audit"]=verify_candidate(folder);row["max_absolute_error"]=report["attempts"][-1]["comparison"]["max_absolute_error"]
 result=sealed(dict(source_hashes=before,prior_validation=str(prior),prior_validation_sha256=sha(prior),
  reused_tests=old["tests"],rows=[row],new_paid_calls=0,static_rechecked=len(rows),
  static_now_passes=sum(x["status"]=="static_now_passes" for x in rows),remaining_static=sum(x["status"]!="static_now_passes" for x in rows),
  parents={str(a.output/n):sha(a.output/n) for n in ("static-review.json","artifact-review.json")}))
 dump(a.output/"report.json",result);print(json.dumps(row),flush=True)
 return code
if __name__=="__main__":raise SystemExit(main())
