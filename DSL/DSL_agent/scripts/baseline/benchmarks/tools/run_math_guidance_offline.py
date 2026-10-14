"""Replay two retained programs under new requests; explicitly not Agent generation."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha,check_binding
from campaign_live_state import sealed
from benchmark_runner import strict_file,dump
from run_stage2_guidance_retest import evidence_path
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned Python")
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 plan=strict_file(a.plan,8*1024**2);check_binding(plan)
 if plan["source_hashes"]!=runtime_sources():raise ValueError("Runtime drift")
 if a.output.exists():raise ValueError("Fresh replay output")
 old_audit_path=RESULTS/"stage2-agent-campaign-r111-final-audit/report.json"
 old=strict_file(old_audit_path,16*1024**2);check_binding(old)
 history={r["id"]:r for r in old["rows"]};specs={r["id"]:r for r in plan["cases"]}
 a.output.mkdir();rows=[];parents={str(a.plan):sha(a.plan),str(old_audit_path):sha(old_audit_path)}
 for task_id in ("free_bench_boundary_0030","free_bench_boundary_0044"):
  previous=history[task_id];spec=specs[task_id]
  if previous["status"]!="passed" or previous["model_sha256"]!=spec["model_sha256"]:raise ValueError("Wrong historical model")
  folder=Path(previous["evidence"]);old_report=strict_file(folder/"report.json",8*1024**2)
  if sha(folder/"report.json")!=previous["report_sha256"]:raise ValueError("Old report changed")
  successes=[x for x in old_report["attempts"] if x["status"]=="passed"]
  if len(successes)!=1:raise ValueError("Historical successful source identity")
  source_file=folder/("attempt-%02d"%successes[0]["index"])/"candidate.py"
  source=source_file.read_text();parents[str(source_file)]=sha(source_file)
  work=a.output/task_id;work.mkdir();dump(work/"model.json",spec["model"])
  response=json.dumps(dict(schema=1,request_id=spec["request_id"],hecate_source=source))
  dump(work/"responses.json",[response])
  command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
   "--unified-guidance","explicit-v3","--compiler-configuration",spec["compiler_configuration"],
   "--replay",str(work/"responses.json"),"--max-repairs","0"]
  start=time.monotonic()
  with (work/"run.log").open("x") as stream:code=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=240).returncode
  result_folder=evidence_path(work/"run.log",RESULTS)
  if result_folder is None:raise ValueError("Missing replay evidence")
  report=strict_file(result_folder/"report.json",8*1024**2)
  if report["agent_calls"]!=0:raise ValueError("Replay called provider")
  read_request(result_folder/"request.json",spec["request"])
  row=dict(id=task_id,evidence=str(result_folder),seconds=time.monotonic()-start,exit_code=code,status=report["status"],scripted=True)
  if code==0 and report["status"]=="passed":row["audit"]=verify_candidate(result_folder)
  rows.append(row);dump(a.output/"progress.json",rows)
  if code:raise ValueError("Scripted numerical/compile failure; keep original evidence")
 if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime drift")
 result=sealed(dict(format="poseidon-math-guidance-offline-v1",plan_binding=plan["binding"],parents=parents,rows=rows,
  passed=2,failed=0,skipped=0,new_paid_calls=0,new_agent_generation=False,new_encrypted_executions=2,
  runner_sha256=sha(Path(__file__)),source_hashes=runtime_sources(),stage2_complete=False))
 dump(a.output/"report.json",result)
 print(json.dumps({k:result[k] for k in ("binding","passed","failed","skipped","new_paid_calls","new_agent_generation","new_encrypted_executions")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
