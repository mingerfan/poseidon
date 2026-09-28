"""Bounded v4 offline regression in one existing Nix environment; never calls API."""
import argparse,json,os,shlex,subprocess,sys,time,unittest
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from benchmark_graph import digest
from benchmark_runner import dump
from campaign_live_state import sealed
from stage2_agent_pilot_plan import sha
from run_stage2_guidance_retest import evidence_path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true")
 a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=900)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned environment required")
 from workspace_paths import RESULTS
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_contract import prepare,validate_candidate
 from unified_graph_lowering import lower
 from audit_unified_candidate import verify_candidate
 from campaign_request_files import read_request
 if a.output.exists() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result directory required")
 a.output.mkdir();before=runtime_sources()
 plan=sealed(dict(format="poseidon-directed-guidance-offline-plan-v1",source_hashes=before,
  source_sha256=digest(before),runner_sha256=sha(Path(__file__)),cases=["construct_003_0","construct_012_2"],
  max_seconds=900,case_timeout_seconds=300,max_repairs=0,new_paid_calls=0,scripted=True))
 dump(a.output/"plan.json",plan)
 prior=RESULTS/"stage2-directed-guidance-r146"
 oldplan=json.loads((prior/"plan.json").read_text())
 if oldplan["source_hashes"]!=before:raise ValueError("Preflight source changed")
 testinfo=json.loads((prior/"unittest.json").read_text())
 compat=json.loads((prior/"compatibility.json").read_text())
 if testinfo!={"run":31,"failed":0,"errors":0,"skipped":0}:raise ValueError("Preflight tests")
 if compat["source_hashes"]!=before or compat["old_request_identical_count"]!=2030 or compat["new_statuses"]!={"prepared_not_run":2030}:raise ValueError("Compatibility proof")
 for name in ["unittest.log","unittest.json","compatibility.json","compatibility.log"]:
  content=(prior/name).read_bytes()
  with (a.output/name).open("xb") as f:f.write(content)
 print("Retained 31 tests and 2030 compatibility checks verified; not rerun",flush=True)
 index=json.loads((RESULTS/"stage2-agent-campaign-v7-r130/index.json").read_text());specs={}
 for ref in index["shards"]:
  path=RESULTS/"stage2-agent-campaign-v7-r130"/ref["file"]
  if sha(path)!=ref["sha256"]:raise ValueError("Frozen shard changed")
  for s in json.loads(path.read_text())["cases"]:
   if s["id"] in plan["cases"]:specs[s["id"]]=s
 rows=[]
 for ident in plan["cases"]:
  spec=specs[ident];old=spec["request"];exercise=spec["exercise"]
  r=prepare(spec["model"],old["compiler_profile_sha256"],old["compiler_configuration"],exercise,
   construction_profile=old.get("construction_profile"),constant_policy=old["constant_origins"].get("policy"),
   generation_guidance="explicit-v4")
  if old.get("construction_profile"):
   from unified_public_exercises import golden_variant
   source=golden_variant(lower(r),exercise,r)
  else:
   from unified_graph_exercises import golden_variant
   source=golden_variant(lower(r),exercise)
  validate_candidate(dict(schema=1,request_id=r["request_id"],hecate_source=source),r)
  work=a.output/ident;work.mkdir()
  dump(work/"model.json",spec["model"])
  dump(work/"responses.json",[json.dumps(dict(schema=1,request_id=r["request_id"],hecate_source=source))])
  command=[str(VENV/"bin/python"),"-B",str(BASE/"run_candidate.py"),"--inside","--case",str(work/"model.json"),
   "--unified-guidance","explicit-v4","--compiler-configuration",spec["compiler_configuration"],
   "--unified-exercise",exercise,"--replay",str(work/"responses.json"),"--max-repairs","0"]
  if old.get("construction_profile"):command+=["--unified-profile","public-v1"]
  start=time.monotonic()
  with (work/"run.log").open("x") as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
  folder=evidence_path(work/"run.log",RESULTS)
  if folder is None:raise ValueError("Missing candidate evidence")
  report=json.loads((folder/"report.json").read_text())
  if report["agent_calls"]!=0:raise ValueError("Unexpected API call")
  read_request(folder/"request.json",r)
  row=dict(id=ident,evidence=str(folder),report_sha256=sha(folder/"report.json"),status=report["status"],exit_code=code,
   seconds=time.monotonic()-start,scripted=True,agent_generation=False)
  if code==0 and report["status"]=="passed":row["audit"]=verify_candidate(folder)
  rows.append(row);dump(a.output/"progress.json",rows)
  print(ident,row["status"],flush=True)
 if runtime_sources()!=before:raise ValueError("Source drift")
 result=sealed(dict(format="poseidon-directed-guidance-offline-v1",plan_binding=plan["binding"],
  source_hashes=before,source_sha256=digest(before),tests=testinfo,rows=rows,passed=sum(x["status"]=="passed" for x in rows),
  failed=sum(x["status"]!="passed" for x in rows),skipped=0,new_paid_calls=0,new_agent_generation=False,
  stage2_complete=False,parents={str(a.output/n):sha(a.output/n) for n in ["plan.json","unittest.log","unittest.json","compatibility.json","compatibility.log"]}))
 dump(a.output/"report.json",result)
 print(json.dumps({k:result[k] for k in ["binding","passed","failed","skipped","new_paid_calls"]}),flush=True)
 return 0 if not result["failed"] else 1
if __name__=="__main__":raise SystemExit(main())
