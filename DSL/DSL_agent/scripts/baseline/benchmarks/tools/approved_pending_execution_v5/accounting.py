"""Closed-ledger selection of untouched cases; no provider or runtime calls."""
import copy, math
TERMINAL={"succeeded","budget_exhausted","service_error","validation_failed","unsupported","cancelled","timed_out","failed"}
def select_unstarted(cases,rows):
    if type(cases) is not list or type(rows) is not list: raise ValueError("Approved case table")
    by_id={x["case"]["id"]:x["case"] for x in cases}
    if len(by_id)!=len(cases):raise ValueError("Duplicate approved context")
    seen=set()
    for row in rows:
        name=row.get("id")
        if name not in by_id or name in seen:raise ValueError("Unknown or repeated original case")
        seen.add(name)
        result=row.get("result",{})
        if row.get("status") not in ("passed","failed") or result.get("phase")!="complete" or result.get("status") not in TERMINAL:
            raise ValueError("Original task not terminal")
        if row.get("request_id")!=by_id[name]["request_id"]:raise ValueError("Original request changed")
        for key,bound in (("generations",4),("http_attempts",16)):
            value=result.get(key)
            if type(value) is not int or not 0<=value<=bound:raise ValueError("Original call accounting")
    return copy.deepcopy([x for x in cases if x["case"]["id"] not in seen])
def closed_calls(state):
    if state.get("phase")!="complete" or state.get("status") not in TERMINAL:return False
    n=state.get("http_attempts")
    if type(n) is not int or not 0<=n<=16:return False
    if n==0:return True
    checkpoints=[x for x in state.get("ledger",[]) if x.get("event")=="provider_checkpoint"]
    if not checkpoints:return False
    last=checkpoints[-1];calls=last.get("outcomes")
    return last.get("attempts")==n and type(calls) is list and len(calls)==n and sorted(x.get("index") for x in calls)==list(range(n)) and all(x.get("status") in ("response_received","failed") for x in calls)
def deadline(original,now):
    if original is None:return None
    if type(original) not in (int,float) or not math.isfinite(original) or original<=now:raise ValueError("Original group expired or invalid")
    return original
def merge_closed(prior,added,approved,remaining):
    by_id={x["case"]["id"]:x["case"]["request_id"] for x in approved}
    if len(by_id)!=len(approved):raise ValueError("Duplicate original approval")
    combined=copy.deepcopy(prior+added);seen=set()
    for item in combined:
        if item.get("id") not in by_id or item["id"] in seen or item.get("request_id")!=by_id[item["id"]]:
            raise ValueError("Closed identity or duplicate changed")
        if item.get("status") not in ("passed","failed") or item.get("calls_resolved") is not True or item.get("not_retried") is not True:
            raise ValueError("Closed outcome unresolved")
        for key,bound in (("generations",4),("http_attempts",16)):
            if type(item.get(key)) is not int or not 0<=item[key]<=bound:raise ValueError("Closed per-case quota")
        seen.add(item["id"])
    pending=[x["case"]["id"] for x in remaining]
    if len(set(pending))!=len(pending) or seen&set(pending) or seen|set(pending)!=set(by_id):
        raise ValueError("Approval coverage or replay")
    if any(x["case"]["request_id"]!=by_id[x["case"]["id"]] for x in remaining):
        raise ValueError("Pending original request changed")
    return combined
