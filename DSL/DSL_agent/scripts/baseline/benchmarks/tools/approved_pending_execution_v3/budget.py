"""Pure accounting for the originally approved pending live cases."""
import math
def remaining(generations,http_attempts):
    if type(generations) is not int or type(http_attempts) is not int or not 0<=generations<=4 or not 0<=http_attempts<=16:
        raise ValueError("Original per-case call accounting")
    return {"generations":4-generations,"http_attempts":16-http_attempts}
def limits(deadline,now,used_generations=0,used_http=0):
    if type(deadline) not in (int,float) or type(now) not in (int,float) or not math.isfinite(deadline) or not math.isfinite(now):
        raise ValueError("Original shared deadline")
    calls=remaining(used_generations,used_http)
    seconds=math.floor(deadline-now)
    if seconds<1 or calls["generations"]<1 or calls["http_attempts"]<1:return None
    return dict(**calls,wall_seconds=min(14400,seconds),api_timeout_seconds=1200,max_tokens=384000)
def exact_certificate_stop(state):
    if state.get("status")!="service_error" or state.get("generations")!=1 or state.get("http_attempts")!=1 or state.get("package") is not None:return False
    ledger=state.get("ledger")
    if type(ledger) is not list or not ledger:return False
    outcome=ledger[-1]
    return outcome.get("event")=="provider_checkpoint" and outcome.get("outcomes")==[{"index":0,"status":"failed","error":"transport_certificate_failed"}]
