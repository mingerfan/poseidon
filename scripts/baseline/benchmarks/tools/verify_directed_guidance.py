"""Offline exact old-request compatibility and v4 directed-guidance preparation. No provider."""
import argparse,json,os,shlex,sys
from pathlib import Path
from stage2_agent_campaign import ROOT,BASE
from stage2_agent_pilot_plan import sha,check_binding
from campaign_live_state import sealed
from benchmark_graph import digest
from benchmark_runner import strict_file,dump

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--index",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument("--inside",action="store_true",help=argparse.SUPPRESS);a=p.parse_args()
 from hecate_python_env import VENV,enter_nix
 if not a.inside:
  return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
   shlex.join([str(VENV/"bin/python"),"-B",str(Path(__file__).resolve()),*sys.argv[1:],"--inside"]),seconds=600)
 if os.environ.get("IN_NIX_SHELL")!="pure" or Path(sys.prefix)!=VENV:raise ValueError("Pinned Python")
 from semantic_benchmark_execution import runtime_sources
 from unified_graph_contract import prepare,validate_request
 from compiler_configuration import PROFILE_SHA256,configuration
 if a.output.exists():raise ValueError("Fresh report")
 before=runtime_sources();index=strict_file(a.index,16*1024**2);check_binding(index)
 parents={str(a.index):sha(a.index)};rows=[]
 for ref in index["shards"]:
  path=a.index.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(path)!=ref["sha256"]:raise ValueError("Frozen original shard changed")
  plan=strict_file(path,8*1024**2);check_binding(plan);parents[str(path)]=sha(path)
  for spec in plan["cases"]:
   public=spec.get("construction_profile")=="hecate-unified-public-v1"
   kw=dict(construction_profile=spec["construction_profile"] if public else None,
    helper_profile=spec.get("helper_profile"),helper_exercise=spec.get("required_helpers"),chunk_period=spec.get("chunk_period"))
   def build(version):
    return prepare(spec["model"],PROFILE_SHA256,configuration(spec["compiler_configuration"]),spec.get("exercise"),generation_guidance=version,**kw)
   old=build("explicit-v3");validate_request(old)
   if old!=spec["request"]:raise ValueError("Old request bytes/identity changed: "+spec["id"])
   row=dict(id=spec["id"],old_request_identical=True,old_request_id=old["request_id"])
   try:new=build("explicit-v4");validate_request(new)
   except ValueError as error:
    row.update(new_status="request_preparation_blocked",diagnostic=str(error),agent_unsupported=False)
   else:
    fixed=lambda r:{k:v for k,v in r.items() if k not in ("generation_guidance","request_id")}
    # Compact-policy changes, if needed to keep the existing 128 KiB cap, are explicit.
    changed=sorted(k for k in set(fixed(old))|set(fixed(new)) if fixed(old).get(k)!=fixed(new).get(k))
    if set(changed)-{"public_constants","constant_origins"}:raise ValueError("Unexpected model/layout/config change")
    if old["model"]!=new["model"] or old["layout"]!=new["layout"]:raise ValueError("Mathematical/ABI change")
    row.update(new_status="prepared_not_run",new_request_id=new["request_id"],constant_policy_changed=bool(changed))
   rows.append(row)
 if runtime_sources()!=before:raise ValueError("Runtime drift during verification")
 if len(rows)!=2030 or len({r["id"] for r in rows})!=2030:raise ValueError("Task denominator")
 for name,hsh in parents.items():
  if sha(Path(name))!=hsh:raise ValueError("Parent changed")
 from collections import Counter
 result=sealed(dict(format="poseidon-directed-guidance-compatibility-v1",runner_sha256=sha(Path(__file__)),
  parents=parents,source_hashes=before,source_sha256=digest(before),rows=rows,
  old_request_identical_count=len(rows),new_statuses=dict(Counter(r["new_status"] for r in rows)),
  new_paid_calls=0,new_encrypted_executions=0,stage2_complete=False))
 dump(a.output,result)
 print(json.dumps({k:result[k] for k in ("binding","source_sha256","old_request_identical_count","new_statuses","new_paid_calls")}))
 return 0
if __name__=="__main__":raise SystemExit(main())
