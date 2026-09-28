"""Bind request-file bookkeeping and narrow candidate rejection handling; no requests regenerated."""
import argparse,json,sys
from pathlib import Path
from stage2_agent_campaign import ROOT
from stage2_agent_pilot_plan import sha,check_binding
from campaign_live_state import sealed
from benchmark_runner import strict_file,dump
from semantic_benchmark_execution import runtime_sources

def upgrade(plan):
 check_binding(plan)
 if runtime_sources()!=plan["source_hashes"]:raise ValueError("Runtime changed")
 for name,hsh in plan["proposal_files"].items():
  if sha(ROOT/name)!=hsh:raise ValueError("Original proposal changed")
 if "recovery" in plan:raise ValueError("Use audited checkpoint recovery for an active shard")
 result=dict(plan);result.pop("binding")
 result["proposal_files"]=dict(plan["proposal_files"])
 for name in ("run_stage2_agent_campaign_v3.py","audit_stage2_campaign_shard_v2.py","campaign_request_files.py","campaign_candidate_failures.py","upgrade_campaign_plans_v3.py"):
  p=Path(__file__).with_name(name);result["proposal_files"][str(p.relative_to(ROOT))]=sha(p)
 result["previous_plan_binding"]=plan["binding"]
 return sealed(result)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--index",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 from workspace_paths import RESULTS
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct result output")
 index=strict_file(a.index,16*1024**2);check_binding(index)
 refs=[]
 for ref in index["shards"]:
  f=a.index.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(f)!=ref["sha256"]:raise ValueError("Original shard changed")
  old=strict_file(f,8*1024**2);new=upgrade(old)
  if old["cases"]!=new["cases"] or old["limits"]!=new["limits"]:raise ValueError("Request or budget changed")
  refs.append((ref["file"],new,sha(f)))
 a.output.mkdir()
 for name,plan,_ in refs:dump(a.output/name,plan)
 manifest=sealed(dict(format="poseidon-campaign-runner-upgrade-v1",original_index=str(a.index),original_index_sha256=sha(a.index),
  shards=[dict(file=name,sha256=sha(a.output/name),binding=plan["binding"],original_sha256=hsh) for name,plan,hsh in refs],
  requests_regenerated=0,paid_calls=0,limits_changed=False,scope="Bookkeeping implementation only; claims still prevent repeated evaluations"))
 dump(a.output/"index.json",manifest)
 print(json.dumps(dict(binding=manifest["binding"],shards=len(refs),paid_calls=0,requests_regenerated=0)))
 return 0
if __name__=="__main__":raise SystemExit(main())
