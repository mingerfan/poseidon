"""No-API repair feasibility; immutable old Agent evidence is never relabelled."""
import argparse, hashlib, json, shlex, sys, subprocess, time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument("--inside",action="store_true");a.add_argument("--execute",action="store_true");args=a.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not args.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]+(["--execute"] if args.execute else [])),seconds=3600)
 from unified_graph_lowering import candidate_source
 from unified_graph_contract import validate_request,validate_candidate
 from semantic_benchmark_execution import runtime_sources
 from workspace_paths import RESULTS
 out=RESULTS/"stage2-remaining-r174"
 if not args.execute:
  out.mkdir()
  before=json.loads((RESULTS/"validation-adapter-r167/before.json").read_text())
  assert all(sha(p)==h for p,h in before["compiler"].items())
  report=json.loads((ROOT/"docs/baseline/stage2-agent-repair-result-r173.json").read_text())
  snapshot={"compiler":before["compiler"],"sources":runtime_sources(),"git_status":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),"parents":{}}
  rows=[]
  for r in report["rows"]:
   if r["status"]=="passed":continue
   reqpath=Path(r["evidence"])/"request.json";q=json.loads(reqpath.read_text());validate_request(q)
   snapshot["parents"][str(reqpath)]=sha(reqpath)
   for p in Path(r["evidence"]).glob("attempt-*/response.txt"):snapshot["parents"][str(p)]=sha(p)
   if r["terminal_failure_layer"]!="static_check":continue
   entry={"id":r["id"],"request":q,"original_evidence":r["evidence"],"agent_status":"failed","manual_fixture":True}
   try:
    s,meta=candidate_source(q)
    c={"schema":1,"request_id":q["request_id"],"hecate_source":s};validate_candidate(c,q)
    entry.update(status="prepared",candidate=c,strategy=meta)
   except Exception as exc:entry.update(status="preparation_failed",diagnostic=str(exc))
   rows.append(entry)
  (out/"before.json").write_text(json.dumps(snapshot,indent=2))
  (out/"prepared.json").write_text(json.dumps(rows,indent=2))
  print(json.dumps([{"id":r["id"],"status":r["status"],"diagnostic":r.get("diagnostic")} for r in rows]),flush=True)
  return 0
 from component_contract import request_options, runner_options
 from unified_graph_contract import prepare
 from audit_unified_candidate import verify_candidate
 from run_stage2_guidance_retest import evidence_path,stop_child
 import os,shutil
 target=out/"execution-v2";target.mkdir()
 before=json.loads((out/"before.json").read_text())
 frozen=runtime_sources()
 def guard():
  if frozen!=runtime_sources():raise ValueError("Runtime changed")
  if not all(sha(p)==h for p,h in before["compiler"].items()):raise ValueError("Compiler changed")
  if not all(sha(p)==h for p,h in before["parents"].items()):raise ValueError("Historical evidence changed")
  if shutil.disk_usage(RESULTS).free<4*1024**3:raise ValueError("Disk reserve")
  kib=int(subprocess.check_output(["du","-sk",str(RESULTS)],text=True).split()[0])
  if kib>=32*1024**2:raise ValueError("Total results space budget")
  kib=int(subprocess.check_output(["du","-sk",str(target)],text=True).split()[0])
  if kib>=4*1024**2:raise ValueError("Batch space budget")
 guard()
 (target/"sources.json").write_text(json.dumps(frozen,indent=2))
 result={"format":"poseidon-offline-remaining-r174","planned":50,"paid_calls":0,"new_agent_successes":0,
  "evidence_kind":"deterministic_repair_not_agent","source_hashes":frozen,"compiler_files_unchanged":len(before["compiler"]),"rows":[]}
 started=time.monotonic()
 try:
  for n,item in enumerate(json.loads((out/"prepared.json").read_text())):
   guard()
   if time.monotonic()-started>3300:raise TimeoutError("Batch wall budget")
   record={"id":item["id"],"old_agent_status":"failed","shard":n//48,"status":"not_run"}
   result["rows"].append(record)
   if item["status"]!="prepared":
    record.update(status="preparation_failed",diagnostic=item["diagnostic"]);continue
   old=item["request"];options=request_options(old);options.pop("compiler_configuration");options["generation_guidance"]="explicit-v6"
   request=prepare(old["model"],old["compiler_profile_sha256"],old.get("compiler_configuration"),
    constant_policy=old["constant_origins"].get("policy"),**options)
   ignore={"request_id","generation_guidance"}
   assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in request.items() if k not in ignore}
   candidate=dict(item["candidate"],request_id=request["request_id"])
   validate_candidate(candidate,request)
   folder=target/item["id"];folder.mkdir()
   (folder/"model.json").write_text(json.dumps(request["model"]))
   (folder/"job.json").write_text(json.dumps({"request":request,"candidate":candidate,"manual_fixture":True},indent=2))
   (folder/"responses.json").write_text(json.dumps([json.dumps(candidate)]))
   cmd=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(folder/"model.json"),
    "--replay",str(folder/"responses.json"),"--max-repairs","0",*runner_options(request)]
   child=None
   try:
    with (folder/"run.log").open("x") as log:
     child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     child.wait(timeout=min(300,max(1,3300-(time.monotonic()-started))))
    ev=evidence_path(folder/"run.log",RESULTS)
    if ev is None:raise ValueError("Missing execution evidence")
    report=json.loads((ev/"report.json").read_text())
    assert report["agent_calls"]==0
    assert json.loads((ev/"request.json").read_text())==request
    record.update(status=report["status"],evidence=str(ev),report_sha256=sha(ev/"report.json"))
    if report["status"]=="passed":
     audit=verify_candidate(ev);(folder/"audit.json").write_text(json.dumps(audit,indent=2))
     record.update(max_absolute_error=audit["comparison"]["max_absolute_error"],audit_sha256=sha(folder/"audit.json"))
    else:
     last=report["attempts"][-1] if report.get("attempts") else report
     record.update(failure_layer=last.get("failure_layer"),diagnostic=last.get("diagnostic"))
     if record["failure_layer"]=="dsl_trace":
      log=(ev/("attempt-%02d"%last["index"])/"trace.log").read_text()
      if any(v in log for v in ("Changed unified generation guidance","ModuleNotFoundError","Permission denied")):
       record["failure_layer"]="integrity_or_environment"
       raise ValueError("Trace infrastructure failure; stop batch")
     if record["failure_layer"] in ("integrity","environment"):raise ValueError("Integrity/environment gate")
   except subprocess.TimeoutExpired:
    record.update(status="timeout",failure_layer="budget")
    raise
   finally:
    if child is not None:stop_child(child)
   result["seconds"]=time.monotonic()-started
   (target/"report.json").write_text(json.dumps(result,indent=2))
   print(json.dumps(record),flush=True)
  guard();result["completed"]=True
 except BaseException as exc:
  result.update(completed=False,stop_reason=type(exc).__name__+": "+str(exc));raise
 finally:
  result["seconds"]=time.monotonic()-started
  (target/"report.json").write_text(json.dumps(result,indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
