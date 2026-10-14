"""Two independent no-API replays with observed simultaneous native slot ownership."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
from stage2_agent_campaign import BASE,ROOT
from stage2_agent_pilot_plan import sha,check_binding
from benchmark_runner import strict_file,dump
from campaign_live_state import sealed
from run_stage2_guidance_retest import evidence_path,stop_child
from parallel_campaign_queue import available_mib

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true");a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned Python")
 from workspace_paths import RESULTS,WORK
 from semantic_benchmark_execution import runtime_sources
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 planpath=RESULTS/"stage2-math-pilot-prepare-r117-ready/plan.json"
 auditpath=RESULTS/"stage2-math-pilot-r117-audit/report.json"
 plan=strict_file(planpath,8*1024**2);check_binding(plan)
 previous=strict_file(auditpath,16*1024**2);check_binding(previous)
 sources=runtime_sources()
 if plan["source_hashes"]!=sources:raise ValueError("Source drift")
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh probe directory")
 specs={s["id"]:s for s in plan["cases"]};hist={s["id"]:s for s in previous["rows"]}
 parents={str(planpath):sha(planpath),str(auditpath):sha(auditpath)}
 a.output.mkdir();children=[];rows=[];overlap=False;samples=0;minimum=available_mib();start=time.monotonic()
 try:
  for key in ("free_bench_boundary_0030","free_bench_boundary_0044"):
   old=hist[key];spec=specs[key];folder=Path(old["evidence"])
   if old["status"]!="passed" or sha(folder/"report.json")!=old["report_sha256"]:raise ValueError("Historical evidence")
   report=strict_file(folder/"report.json",8*1024**2)
   passed=[x for x in report["attempts"] if x["status"]=="passed"]
   if len(passed)!=1:raise ValueError("Source ambiguity")
   source=folder/("attempt-%02d"%passed[0]["index"])/"candidate.py"
   parents[str(source)]=sha(source);parents[str(folder/"report.json")]=sha(folder/"report.json")
   work=a.output/key;work.mkdir();dump(work/"model.json",spec["model"])
   dump(work/"responses.json",[json.dumps(dict(schema=1,request_id=spec["request_id"],hecate_source=source.read_text()))])
   command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
    "--unified-guidance","explicit-v3","--compiler-configuration",spec["compiler_configuration"],
    "--replay",str(work/"responses.json"),"--max-repairs","0"]
   with (work/"run.log").open("x") as stream:
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   children.append((key,child,work,spec))
  while any(child.poll() is None for _,child,_,_ in children):
   if time.monotonic()-start>240:raise TimeoutError("Concurrent replay budget")
   minimum=min(minimum,available_mib());samples+=1
   if minimum<1024:raise ValueError("Concurrent memory reserve")
   inodes={p.stat().st_ino for p in (WORK/"cache/agent-native-slots").glob("slot-*.lock")}
   owned=set()
   for line in Path("/proc/locks").read_text().splitlines():
    parts=line.split()
    if len(parts)>=6 and parts[1]=="FLOCK" and parts[4].isdigit() and int(parts[4]) in {c.pid for _,c,_,_ in children}:
     if int(parts[5].rsplit(":",1)[-1]) in inodes:owned.add(int(parts[4]))
   overlap=overlap or len(owned)==2
   time.sleep(.02)
  for key,child,work,spec in children:
   folder=evidence_path(work/"run.log",RESULTS)
   if folder is None:raise ValueError("Missing evidence")
   report=strict_file(folder/"report.json",8*1024**2)
   if child.returncode or report["status"]!="passed" or report["agent_calls"]!=0:raise ValueError("Offline concurrent replay failed")
   read_request(folder/"request.json",spec["request"])
   audit=verify_candidate(folder);dump(work/"audit.json",audit)
   parents[str(work/"audit.json")]=sha(work/"audit.json");parents[str(folder/"report.json")]=sha(folder/"report.json")
   rows.append(dict(id=key,evidence=str(folder),native_resources=report["native_resources"],status="passed"))
  if not overlap:raise ValueError("Two actual simultaneous native slots not observed")
  if runtime_sources()!=sources:raise ValueError("Source changed")
  for name,h in parents.items():
   if sha(Path(name))!=h:raise ValueError("Evidence changed")
  report=sealed(dict(format="poseidon-concurrent-native-probe-v1",source_hashes=sources,parents=parents,
   runner_sha256=sha(Path(__file__)),rows=rows,passed=2,failed=0,skipped=0,
   two_native_slots_observed=overlap,min_available_mib=minimum,samples=samples,seconds=time.monotonic()-start,
   new_paid_calls=0,new_agent_generation=False,new_encrypted_executions=2,stage2_complete=False))
  dump(a.output/"report.json",report)
  print(json.dumps({k:report[k] for k in ("binding","passed","two_native_slots_observed","min_available_mib","seconds","new_paid_calls")}))
 finally:
  for _,child,_,_ in children:stop_child(child)
 return 0
if __name__=="__main__":raise SystemExit(main())
