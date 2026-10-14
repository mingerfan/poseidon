"""Bounded host transport observations, separate from generation provenance."""
import re
from benchmark_graph import canonical,digest,require
from deepseek_http_worker import safe_diagnostics
from deepseek_provider import SAFE_FAILURE_CODES

FORMAT="poseidon-application-transport-v1"
MAX_CALLS=16
MAX_BYTES=32768
TRUST="host_collected_transport_counters_not_acceptance_or_billing_proof"

def snapshot(metrics):
    """Keep allowlisted transport facts only; never response bodies or environment."""
    require(type(metrics) is dict and metrics.get("provider")=="deepseek_api",
            "Transport receipt provider")
    calls=metrics.get("calls")
    require(type(calls) is list and len(calls)<=MAX_CALLS,"Transport receipt call bound")
    rows=[]
    for call in calls:
        require(type(call) is dict,"Transport receipt call fields")
        row={}
        for key,maximum in (("index",15),("generation_index",3),("retry_index",3)):
            v=call.get(key)
            if type(v) is int and 0<=v<=maximum:row[key]=v
        if call.get("status") in ("in_flight","failed","response_received"):
            row["status"]=call["status"]
        error=call.get("error")
        if type(error) is str and (error in SAFE_FAILURE_CODES or
            error in ("transport_failed","provider_response_or_transport_failed") or
            re.fullmatch(r"http_status_[1-5][0-9]{2}",error)):
            row["error"]=error
        for key in ("retryable","retry_scheduled"):
            if type(call.get(key)) is bool:row[key]=call[key]
        if type(call.get("retry_delay_seconds")) is int and call["retry_delay_seconds"] in (5,15,30):
            row["retry_delay_seconds"]=call["retry_delay_seconds"]
        row["transport_diagnostics"]=safe_diagnostics(call.get("transport_diagnostics"))
        rows.append(row)
    result=dict(format=FORMAT,trust=TRUST,provider="deepseek_api",calls=rows)
    for key in ("request_attempts","generation_attempts","transport_retries"):
        v=metrics.get(key)
        maximum=4 if key=="generation_attempts"else MAX_CALLS
        if type(v) is int and 0<=v<=maximum:result[key]=v
    result["binding"]=digest(result)
    require(len(canonical(result))<=MAX_BYTES,"Transport receipt byte bound")
    return result
