"""Read-only frozen task/context integrity audit; never awards execution passes."""
import argparse,json
from collections import Counter
from pathlib import Path
from stage2_agent_campaign import ROOT,GROUPS
from benchmark_graph import signature,digest
from benchmark_runner import strict_file,dump
from stage2_agent_pilot_plan import sha,check_binding
from semantic_benchmark_execution import runtime_sources
from campaign_live_state import sealed

def contexts(index,specs):
    parts=index["semantic_partitions"]
    if len(parts)!=401 or len({p["id"] for p in parts})!=401:raise ValueError("Partition denominator")
    if Counter(s["group"] for s in specs.values())!=Counter(GROUPS):raise ValueError("Task denominator")
    if len(specs)!=2030:raise ValueError("Task identity denominator")
    actual={key:signature(s["model"],True) for key,s in specs.items()}
    blocked={p["id"]:p["backend_blocker"] for p in parts if p["current_status"]=="backend_blocked"}
    if blocked!={"helper.HE_MaxPad":"bootstrap_backend","helper.HE_Max":"bootstrap_backend","helper.HE_ReLU":"bootstrap_backend"}:
        raise ValueError("Backend blocker classification changed")
    rows=[]
    for p in parts:
        if p["current_status"]=="backend_blocked" or p["layer"] in ("rejection","compiler"):continue
        if p["required_distinct_contexts"]!=3:raise ValueError("Three-context requirement changed")
        tops=set();ids=set()
        for c in p["contexts"]:
            key=c["task_id"]
            if key not in specs:raise ValueError("Unmapped context")
            if c["topology"]!=actual[key]:raise ValueError("Context topology mismatch")
            tops.add(actual[key]);ids.add(key)
        if len(tops)<3:raise ValueError("Insufficient distinct topology contexts")
        rows.append(dict(id=p["id"],task_count=len(ids),distinct_topologies=len(tops),execution_pass_claim=False))
    if len(rows)!=377:raise ValueError("Executable partition denominator")
    return rows,blocked

def build(index_path):
    index=strict_file(index_path,16*1024**2);check_binding(index)
    sources=runtime_sources()
    if index["runtime_source_sha256"]!=digest(sources):raise ValueError("Runtime drift")
    parents={str(index_path):sha(index_path)};specs={}
    for ref in index["shards"]:
        if Path(ref["file"]).name!=ref["file"]:raise ValueError("Unsafe shard path")
        p=index_path.parent/ref["file"]
        if p.is_symlink() or sha(p)!=ref["sha256"]:raise ValueError("Shard changed")
        shard=strict_file(p,8*1024**2);check_binding(shard)
        if shard["binding"]!=ref["binding"] or shard["source_hashes"]!=sources:raise ValueError("Shard identity")
        parents[str(p)]=sha(p)
        for s in shard["cases"]:
            if s["id"] in specs:raise ValueError("Duplicate task")
            if digest(s["model"])!=s["model_sha256"]:raise ValueError("Model changed")
            specs[s["id"]]=s
    rows,blocked=contexts(index,specs)
    if runtime_sources()!=sources:raise ValueError("Runtime changed during read")
    for p,h in parents.items():
        if sha(Path(p))!=h:raise ValueError("Input changed during read")
    return sealed(dict(format="poseidon-campaign-context-plan-audit-v1",parents=parents,
        runner_sha256=sha(Path(__file__)),runtime_source_sha256=digest(sources),
        campaign_binding=index["binding"],planned_tasks=len(specs),semantic_partitions=401,
        executable_context_partitions=len(rows),rows=rows,backend_blocked=blocked,
        compiler_partitions_separate=5,static_controls_separate=16,
        classification_only=True,actual_compile=False,actual_ciphertext_execution=False,
        paid_calls=0,execution_passes_awarded=0,stage2_complete=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--index",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error("Preserve prior audit")
    r=build(a.index);dump(a.output,r)
    print(json.dumps({k:r[k] for k in ("binding","planned_tasks","semantic_partitions","executable_context_partitions","backend_blocked","execution_passes_awarded")}))
