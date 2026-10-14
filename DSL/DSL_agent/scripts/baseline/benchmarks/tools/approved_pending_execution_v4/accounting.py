"""Pure selection of unstarted contexts from an already approved batch."""
import copy,math
TERMINAL={"succeeded","budget_exhausted","service_error","validation_failed","unsupported","cancelled","timed_out","failed"}
def select_unstarted(cases,rows):
    if type(cases) is not list or type(rows) is not list:
        raise ValueError("Approved case table")
    ids=[x["case"]["id"] for x in cases]
    if len(set(ids))!=len(ids):raise ValueError("Duplicate approved context")
    seen=set()
    for row in rows:
        name=row.get("id")
        if name not in ids or name in seen:raise ValueError("Unknown or repeated original case")
        seen.add(name)
        if row.get("status") not in ("passed","failed"):
            raise ValueError("Original case not terminal")
        result=row.get("result",{})
        if result.get("phase")!="complete" or result.get("status") not in TERMINAL:
            raise ValueError("Original task not terminal")
        source=next(x["case"] for x in cases if x["case"]["id"]==name)
        if row.get("request_id")!=source["request_id"]:raise ValueError("Original request changed")
        for key,bound in (("generations",4),("http_attempts",16)):
            v=result.get(key)
            if type(v) is not int or not 0<=v<=bound:raise ValueError("Original call accounting")
    return copy.deepcopy([x for x in cases if x["case"]["id"] not in seen])
def closed_calls(state):
    if state.get("phase")!="complete" or state.get("status") not in TERMINAL:return False
    n=state.get("http_attempts")
    if type(n) is not int or not 0<=n<=16:return False
    if n==0:return True
    checkpoints=[x for x in state.get("ledger",[]) if x.get("event")=="provider_checkpoint"]
    if not checkpoints:return False
    last=checkpoints[-1]
    calls=last.get("outcomes")
    return (last.get("attempts")==n and type(calls) is list and len(calls)==n and
            sorted(x.get("index") for x in calls)==list(range(n)) and
            all(x.get("status") in ("response_received","failed") for x in calls))
def deadline(original,now):
    if original is None:return None
    if type(original) not in (int,float) or not math.isfinite(original):
        raise ValueError("Original deadline")
    if original<=now:raise ValueError("Original group expired")
    return original
