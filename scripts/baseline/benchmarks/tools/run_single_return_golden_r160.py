"""Scripted scalar-return/object-unary E2E; no model API."""
import argparse,json,sys,shlex,subprocess
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
 from benchmark_suite import Builder
 from compiler_configuration import PROFILE_SHA256,configuration
 from unified_graph_contract import prepare,validate_candidate
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 before=runtime_sources();frozen=json.loads((RESULTS/"stage2-noncompiler-before-r159.json").read_text())
 for f,h in frozen.items():assert sha(Path(f))==h
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh output")
 a.output.mkdir();b=Builder([(2,)]);model=b.finish(b.node("square",["input0"]))
 exercise="unified-public-operator-USub"
 r=prepare(model,PROFILE_SHA256,configuration("seal-cpu-eva-w45-v1"),exercise,
   construction_profile="hecate-unified-public-v1",generation_guidance="explicit-v5")
 fixture=BASE/"benchmarks/fixtures/typed-return-r160/single-expr.py"
 source=fixture.read_text();response=dict(schema=1,request_id=r["request_id"],hecate_source=source)
 validate_candidate(response,r);dump(a.output/"model.json",model);dump(a.output/"responses.json",[json.dumps(response)])
 cmd=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(a.output/"model.json"),
      "--unified-profile","public-v1","--unified-exercise",exercise,"--unified-guidance","explicit-v5",
      "--compiler-configuration","seal-cpu-eva-w45-v1","--replay",str(a.output/"responses.json"),"--max-repairs","0"]
 with (a.output/"run.log").open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300).returncode
 folder=evidence_path(a.output/"run.log",RESULTS);report=json.loads((folder/"report.json").read_text());read_request(folder/"request.json",r)
 assert report["agent_calls"]==0 and runtime_sources()==before
 for f,h in frozen.items():assert sha(Path(f))==h
 result=dict(source_hashes=before,fixture_sha256=sha(fixture),status=report["status"],evidence=str(folder),
  report_sha256=sha(folder/"report.json"),manual_golden=True,new_agent_generation=False,new_paid_calls=0,compiler_files_unchanged=len(frozen))
 if code==0 and report["status"]=="passed":result["audit"]=verify_candidate(folder)
 dump(a.output/"report.json",sealed(result));print(json.dumps({k:v for k,v in result.items() if k not in ("source_hashes","audit")}))
 return code
if __name__=="__main__":raise SystemExit(main())
