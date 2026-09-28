"""Original 19 tasks, new opt-in public guidance; manual feasibility only."""
import json,sys,shlex,hashlib,subprocess,shutil,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=3600)
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from component_contract import request_options
 from unified_graph_contract import prepare,validate_candidate
 from unified_graph_lowering import lower,candidate_source
 from component_backend import qualify
 from benchmark_graph import digest
 out=RESULTS/"stage2-noncompiler-r177/execution";out.mkdir()
 before=json.loads((out.parent/"before.json").read_text());frozen=runtime_sources()
 parents={};oldcases=[]
 path=ROOT/"docs/baseline/stage2-agent-repair-result-r176.json";parents[str(path)]=sha(path)
 for row in json.loads(path.read_text())["rows"]:
  if row["status"]=="passed":continue
  qp=Path(row["evidence"])/"request.json";parents[str(qp)]=sha(qp)
  oldcases.append((row["id"],json.loads(qp.read_text()),"directed_construction"))
 index=json.loads((ROOT/"docs/baseline/compiler-blocked-models-r159/index.json").read_text())
 for row in index["rows"]:
  if row["model_id"] not in {"bench_helper_%04d"%i for i in range(112,120)}:continue
  p=Path(row["origin_shard"]);parents[str(p)]=sha(p)
  task=next(v for v in json.loads(p.read_text())["cases"] if v["id"]==row["task_id"])
  oldcases.append((row["task_id"],task["request"],"free_generation"))
 assert len(oldcases)==19
 prepared=[];started=time.monotonic()
 def guard():
  assert runtime_sources()==frozen,"Runtime changed"
  assert all(sha(p)==v for p,v in before["compiler"].items()),"Compiler changed"
  assert all(sha(p)==v for p,v in parents.items()),"Historical evidence changed"
  assert time.monotonic()-started<3300,"Batch time budget"
  assert shutil.disk_usage(RESULTS).free>4*1024**3,"Disk reserve"
  assert int(subprocess.check_output(["du","-sk",str(RESULTS)]).split()[0])<32*1024**2,"Global results budget"
 for task_id,old,track in oldcases:
  options=request_options(old);options.pop("compiler_configuration");options["generation_guidance"]="explicit-v7"
  q=prepare(old["model"],old["compiler_profile_sha256"],old.get("compiler_configuration"),
      constant_policy=old["constant_origins"].get("policy"),**options)
  ignore={"request_id","generation_guidance"}
  assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in q.items() if k not in ignore}
  source=lower(q,packed_prefix=True,balanced_chebyshev=True) if track=="free_generation" else candidate_source(q)[0]
  candidate=dict(schema=1,request_id=q["request_id"],hecate_source=source)
  validate_candidate(candidate,q)
  folder=out/task_id;folder.mkdir()
  job=dict(request=q,candidate=candidate,manual_fixture=True)
  (folder/"job.json").write_text(json.dumps(job,indent=2))
  prepared.append(dict(id=task_id,track=track,request=q,old_request_id=old["request_id"],
      job_sha256=sha(folder/"job.json")))
 (out/"prepared.json").write_text(json.dumps(prepared,indent=2))
 report=dict(planned=19,source_hashes=frozen,parents=parents,paid_calls=0,new_agent_successes=0,
     compiler_files_unchanged=93,rows=[])
 try:
  for row in prepared:
   guard();job=json.loads((out/row["id"]/"job.json").read_text())
   result=qualify(job["request"],job["candidate"])
   entry=dict(id=row["id"],track=row["track"],request_id=row["request"]["request_id"],
       old_request_id=row["old_request_id"],result=result,
       report_sha256=sha(Path(result["evidence"])/"report.json"))
   report["rows"].append(entry)
   print(json.dumps(dict(id=row["id"],status=result["status"],error=result.get("max_absolute_error"),
                         failure=result.get("failure"))),flush=True)
   if result.get("failure",{}).get("layer") in ("integrity","environment"):
    raise ValueError("Integrity/environment stop")
   (out/"report.json").write_text(json.dumps(report,indent=2))
  guard();report["completed"]=True
 finally:
  report["seconds"]=time.monotonic()-started
  report["binding"]=digest({k:v for k,v in report.items() if k!="binding"})
  (out/"report.json").write_text(json.dumps(report,indent=2))
 return 0
if __name__=="__main__":raise SystemExit(main())
