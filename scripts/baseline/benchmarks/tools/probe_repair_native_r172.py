"""No-API native concurrency check using two unchanged, already qualified sources."""
import argparse,json,os,shlex,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from stage2_agent_pilot_plan import sha
from campaign_live_state import sealed
from benchmark_runner import dump
from run_stage2_guidance_retest import evidence_path,stop_child
from parallel_campaign_queue import available_mib
def main():
 p=argparse.ArgumentParser();p.add_argument("--inside",action="store_true");a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=300)
 from workspace_paths import RESULTS,WORK
 from semantic_benchmark_execution import runtime_sources
 from component_contract import runner_options
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 output=RESULTS/"stage2-agent-repair-native-r172";output.mkdir()
 sources=runtime_sources();parents={};children=[];rows=[];overlap=False
 minimum=available_mib();started=time.monotonic()
 try:
  for name in ("construct_117_0","construct_190_0"):
   prior=RESULTS/"validation-adapter-r167/acceptance-final"/name
   job=json.loads((prior/"job.json").read_text())
   assert json.loads((prior/"result.json").read_text())["status"]=="passed"
   parents[str(prior/"job.json")]=sha(prior/"job.json")
   request=job["request"];work=output/name;work.mkdir()
   dump(work/"model.json",request["model"]);dump(work/"responses.json",[json.dumps(job["candidate"])])
   command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
    "--replay",str(work/"responses.json"),"--max-repairs","0",*runner_options(request)]
   with (work/"run.log").open("x") as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   children.append((name,child,work,request))
  while any(c.poll() is None for _,c,_,_ in children):
   if time.monotonic()-started>240:raise TimeoutError("Native probe timeout")
   minimum=min(minimum,available_mib())
   if minimum<1024:raise ValueError("Native memory reserve")
   inodes={p.stat().st_ino for p in (WORK/"cache/agent-native-slots").glob("slot-*.lock")}
   owned=set()
   for line in Path("/proc/locks").read_text().splitlines():
    fields=line.split()
    if len(fields)>=6 and fields[1]=="FLOCK" and fields[4].isdigit() and int(fields[4]) in {c.pid for _,c,_,_ in children}:
     if int(fields[5].rsplit(":",1)[-1]) in inodes:owned.add(int(fields[4]))
   overlap=overlap or len(owned)==2
   time.sleep(.02)
  for name,child,work,request in children:
   folder=evidence_path(work/"run.log",RESULTS)
   if folder is None or child.returncode:raise ValueError("Probe execution failed")
   report=json.loads((folder/"report.json").read_text())
   assert report["status"]=="passed" and report["agent_calls"]==0
   read_request(folder/"request.json",request)
   audit=verify_candidate(folder);dump(work/"audit.json",audit)
   parents[str(work/"audit.json")]=sha(work/"audit.json")
   parents[str(folder/"report.json")]=sha(folder/"report.json")
   rows.append(dict(id=name,status="passed",evidence=str(folder)))
  assert overlap and sources==runtime_sources()
  report=sealed(dict(format="poseidon-repair-native-probe-r172",passed=2,rows=rows,
   two_native_slots_observed=True,min_available_mib=minimum,seconds=time.monotonic()-started,
   source_hashes=sources,parents=parents,runner_sha256=sha(Path(__file__)),new_paid_calls=0))
  dump(output/"report.json",report);print(json.dumps({k:v for k,v in report.items() if k not in ("parents","source_hashes","rows")}))
 finally:
  for _,child,_,_ in children:stop_child(child)
 return 0
if __name__=="__main__":raise SystemExit(main())
