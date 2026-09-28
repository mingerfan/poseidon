"""Prepare all remaining requests and qualify two known fixes; no paid calls."""
import sys,shlex,json,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path[:0]=[str(BASE),str(Path(__file__).parent)]
def main():
 from hecate_python_env import enter_nix,VENV,WORK
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=600)
 from component_contract import reconstruct_request,request_options
 from component_backend import qualify
 from unified_graph_contract import prepare
 from semantic_benchmark_execution import runtime_sources
 from deepseek_provider import public_request
 from benchmark_graph import digest
 from stage2_agent_pilot_plan import sha
 out=WORK/"results/seal-gate-r186";sources=runtime_sources()
 previous={}
 indexpath=WORK/"results/stage2-agent-campaign-v7-r130/index.json"
 idx=json.loads(indexpath.read_text())
 for ref in idx["shards"]:
  p=indexpath.parent/ref["file"];assert sha(p)==ref["sha256"]
  previous.update({r["id"]:r for r in json.loads(p.read_text())["cases"]})
 for v in ("r175","r178"):
  p=WORK/("results/stage2-agent-repair-"+v+"/plan.json")
  previous.update({s["id"]:s for r in json.loads(p.read_text())["shards"] for s in r["plan"]["cases"]})
 tracker=json.loads((ROOT/"docs/baseline/stage2-noncompiler-closure-r180.json").read_text())
 selected=[r for r in tracker["rows"] if r["agent_status"]!="passed"];assert len(selected)==28
 cases=[]
 for row in selected:
  s=dict(previous[row["id"]]);old=s["request"];opts=request_options(old)
  opts.pop("compiler_configuration");opts["generation_guidance"]="explicit-v8"
  q=prepare(old["model"],old["compiler_profile_sha256"],old.get("compiler_configuration"),
      constant_policy=old["constant_origins"].get("policy"),**opts)
  assert reconstruct_request(q)==q and public_request(q)==q
  ignore={"request_id","generation_guidance"}
  assert {k:v for k,v in old.items() if k not in ignore}=={k:v for k,v in q.items() if k not in ignore}
  args=list(s["candidate_arguments"]);args[args.index("--unified-guidance")+1]="explicit-v8"
  repairs=3 if row["id"] in ("construct_088_1","free_bench_helper_0113") else 1
  args[args.index("--max-repairs")+1]=str(repairs)
  s.update(request=q,request_id=q["request_id"],candidate_arguments=args,original_request_id=old["request_id"],
    max_repairs=repairs,manual_repair_source_sent=False,new_generation=True)
  cases.append(s)
 assert sum(s["max_repairs"]+1 for s in cases)==60
 (out/"prepared.json").write_text(json.dumps(dict(cases=cases,source_hashes=sources),indent=2))
 manual=[]
 for name,rel in [("construct_088_1","followup-construct-088-1"),("free_bench_helper_0113","followup-helper-0113")]:
  q=next(s["request"] for s in cases if s["id"]==name)
  job=json.loads((WORK/"results/stage2-noncompiler-r177"/rel/"job.json").read_text())
  candidate=dict(job["candidate"],request_id=q["request_id"])
  r=qualify(q,candidate);manual.append(dict(id=name,result=r))
  assert r["status"]=="passed",r
 assert runtime_sources()==sources
 report=dict(cases=28,source_hashes=sources,manual=manual,paid_calls=0)
 report["binding"]=digest(report)
 (out/"prelive-qualification.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(dict(cases=28,manual_passed=2,paid_calls=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
