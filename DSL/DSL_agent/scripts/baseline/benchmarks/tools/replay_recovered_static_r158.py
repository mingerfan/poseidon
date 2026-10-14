"""Replay only newly static-admitted original responses without edits or paid calls."""
import argparse,json,os,sys,shlex,subprocess
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
from benchmark_runner import dump
from run_stage2_guidance_retest import evidence_path
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--inside",action="store_true");a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 before=runtime_sources();parent=RESULTS/"stage2-failure-repairs-r157-v3/static-review.json";review=json.loads(parent.read_text())
 assert review["source_hashes"]==before
 selected=[r for r in review["rows"] if r["status"]=="static_now_passes"]
 assert [r["id"] for r in selected]==["construct_079_0"]
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output")
 a.output.mkdir();rows=[]
 for case in selected:
  response=Path(case["response"]);assert sha(response)==case["response_sha256"]
  oldroot=response.parents[1];request=json.loads((oldroot/"request.json").read_text());assert sha(oldroot/"request.json")==case["request_sha256"]
  work=a.output/case["id"];work.mkdir();dump(work/"model.json",request["model"])
  dump(work/"responses.json",[response.read_text()])
  cmd=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
       "--unified-profile","public-v1","--unified-exercise",request["construction_exercise"]["id"],"--unified-guidance","explicit-v4",
       "--compiler-configuration","seal-cpu-eva-w45-v1","--replay",str(work/"responses.json"),"--max-repairs","0"]
  with (work/"run.log").open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
  folder=evidence_path(work/"run.log",RESULTS);report=json.loads((folder/"report.json").read_text());read_request(folder/"request.json",request)
  assert report["agent_calls"]==0
  row=dict(id=case["id"],status=report["status"],evidence=str(folder),report_sha256=sha(folder/"report.json"),
      original_response_sha256=sha(response),request_unchanged=True,source_unchanged=True,new_agent_generation=False,
      failure_layer=report["attempts"][-1].get("failure_layer"),diagnostic=report["attempts"][-1].get("diagnostic"))
  if not code and report["status"]=="passed":
   audit=verify_candidate(folder);row["audit"]=audit;row["max_absolute_error"]=audit["comparison"]["max_absolute_error"]
  rows.append(row)
 assert runtime_sources()==before
 result=sealed(dict(source_hashes=before,parent_sha256=sha(parent),rows=rows,new_paid_calls=0,new_agent_generation=False))
 dump(a.output/"report.json",result);print(json.dumps({k:v for k,v in rows[0].items() if k!="audit"}))
 return 0 if all(r["status"]=="passed" for r in rows) else 1
if __name__=="__main__":raise SystemExit(main())
