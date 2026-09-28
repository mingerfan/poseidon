"""Read-only historical request reconstruction and new metadata preparation."""
import json,sys,shlex
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
def main():
 from hecate_python_env import enter_nix,VENV
 if "--inside" not in sys.argv:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),"--inside"]),seconds=300)
 from workspace_paths import RESULTS
 from component_contract import reconstruct_request,request_options
 from unified_graph_contract import prepare,validate_request
 from benchmark_graph import digest
 from stage2_agent_pilot_plan import sha,check_binding
 from semantic_benchmark_execution import runtime_sources
 sources=runtime_sources();indexpath=RESULTS/"stage2-agent-campaign-v7-r130/index.json"
 index=json.loads(indexpath.read_text());check_binding(index);count=0;parents={str(indexpath):sha(indexpath)}
 for ref in index["shards"]:
  p=indexpath.parent/ref["file"];assert sha(p)==ref["sha256"];parents[str(p)]=sha(p)
  for row in json.loads(p.read_text())["cases"]:
   q=row["request"];assert reconstruct_request(q)==q;count+=1
 reportpath=ROOT/"docs/baseline/stage2-agent-repair-result-r173.json"
 report=json.loads(reportpath.read_text());parents[str(reportpath)]=sha(reportpath);new=[]
 for row in report["rows"]:
  p=Path(row["evidence"])/"request.json";q=json.loads(p.read_text());assert reconstruct_request(q)==q
  parents[str(p)]=sha(p)
  if row["status"]=="passed":continue
  opts=request_options(q);opts.pop("compiler_configuration");opts["generation_guidance"]="explicit-v6"
  n=prepare(q["model"],q["compiler_profile_sha256"],q.get("compiler_configuration"),constant_policy=q["constant_origins"].get("policy"),**opts)
  validate_request(json.loads(json.dumps(n,sort_keys=True)))
  ignore={"request_id","generation_guidance"}
  assert {k:v for k,v in q.items() if k not in ignore}=={k:v for k,v in n.items() if k not in ignore}
  assert reconstruct_request(n)==n
  new.append(dict(id=row["id"],old_request_id=q["request_id"],new_request_id=n["request_id"],request=n))
 assert sources==runtime_sources()
 value=dict(old_campaign_exact=count,r172_exact=len(report["rows"]),v6_prepared=len(new),paid_calls=0,
  source_hashes=sources,parents=parents,requests=new)
 value["binding"]=digest(value)
 output=RESULTS/"stage2-remaining-r174/compatibility.json"
 with output.open("x") as f:json.dump(value,f,indent=2)
 print(json.dumps({k:value[k] for k in ("old_campaign_exact","r172_exact","v6_prepared","paid_calls","binding")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
