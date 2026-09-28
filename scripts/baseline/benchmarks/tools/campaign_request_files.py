"""Bound retained request JSON by its frozen expected serialization, then compare."""
import json
from pathlib import Path
from stage2_agent_campaign import ROOT
from stage2_agent_pilot_plan import check_binding,sha
from benchmark_runner import strict_file

def read_request(path, expected):
    # The planner already limits the entire shard to 8 MiB. Keep that hard cap
    # and allow only the exact frozen request, not an arbitrary larger payload.
    size=len((json.dumps(expected,indent=2,sort_keys=True,allow_nan=False)+"\n").encode())
    if size>8*1024**2:raise ValueError("Frozen request exceeds shard boundary")
    actual=strict_file(path,size)
    if actual!=expected:raise ValueError("Executed request changed")
    return actual

def verify_recovery(plan):
    recovery=plan.get("recovery")
    if recovery is None:return
    path=Path(recovery["record"])
    if path.is_symlink() or sha(path)!=recovery["sha256"]:raise ValueError("Recovery record changed")
    record=strict_file(path,8*1024**2);check_binding(record)
    if record["source_hashes"]!=plan["source_hashes"]:raise ValueError("Recovery runtime changed")
    for name,hsh in record["parents"].items():
        p=Path(name)
        if p.is_symlink() or sha(p)!=hsh:raise ValueError("Recovery parent changed")
    return record
