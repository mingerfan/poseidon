"""Freeze scheduling-only variants of v4 plans. Models, requests and identities unchanged."""
import argparse,json
from pathlib import Path
from stage2_agent_campaign import ROOT
from benchmark_runner import strict_file,dump
from benchmark_graph import digest
from stage2_agent_pilot_plan import sha,check_binding
from semantic_benchmark_execution import runtime_sources
from campaign_live_state import sealed

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--index",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 from workspace_paths import RESULTS
 if a.output.exists() or a.output.is_symlink() or a.output.resolve().parent!=RESULTS.resolve():raise ValueError("Fresh direct output")
 index=strict_file(a.index,16*1024**2);check_binding(index);sources=runtime_sources()
 if index["runtime_source_sha256"]!=digest(sources):raise ValueError("Runtime drift")
 a.output.mkdir();refs=[]
 for ref in index["shards"]:
  old=a.index.parent/ref["file"]
  if Path(ref["file"]).name!=ref["file"] or sha(old)!=ref["sha256"]:raise ValueError("Old shard changed")
  plan=strict_file(old,8*1024**2);check_binding(plan)
  if plan["source_hashes"]!=sources:raise ValueError("Shard runtime")
  plan.pop("binding");plan["proposal_files"]=dict(plan["proposal_files"])
  for n in ("run_stage2_agent_campaign_v6.py","parallel_campaign_queue_v2.py","upgrade_campaign_parallel_v2.py"):
   q=Path(__file__).with_name(n);plan["proposal_files"][str(q.relative_to(ROOT))]=sha(q)
  plan["shared_scheduling"]={"api_workers":10,"native_workers":2,"version":1}
  plan["scheduling_parent"]={"file":str(old),"sha256":sha(old)}
  plan=sealed(plan);dest=a.output/ref["file"];dump(dest,plan)
  refs.append(dict(ref,sha256=sha(dest),binding=plan["binding"]))
 index.pop("binding");index["shards"]=refs
 index["scheduling_parent"]={"file":str(a.index),"sha256":sha(a.index)}
 index["shared_scheduling"]={"api_workers":10,"native_workers":2,"version":1}
 index=sealed(index);dump(a.output/"index.json",index)
 print(json.dumps(dict(binding=index["binding"],tasks=index["planned"],shards=len(refs),new_paid_calls=0,requests_unchanged=True)))
if __name__=="__main__":main()
