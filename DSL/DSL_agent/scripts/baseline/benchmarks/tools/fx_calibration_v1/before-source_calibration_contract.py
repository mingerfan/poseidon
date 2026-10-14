"""Closed, hash-bound module observation selection; no fitting or reference edits."""
import math,re
from model_output_contract import snapshot,OutputError
from benchmark_graph import digest
FORMAT="poseidon-source-calibration-selection-v1"
NAME=re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
PATH=re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*|\d+)(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*")
MAX_SELECTORS=32
MAX_EVENTS_PER_SAMPLE=256
MAX_VALUES_PER_SAMPLE=4*1024**2
CODES={"calibration_input_rejected","calibration_selector_unavailable",
 "calibration_projection_rejected","calibration_observation_budget_exhausted",
 "calibration_nonfinite_observation","calibration_observer_changed_forward",
 "calibration_observation_nondeterministic","calibration_binding_changed",
 "calibration_source_failed","calibration_environment_unavailable","calibration_timeout",
 "calibration_worker_protocol_error"}
class CalibrationError(ValueError):
    def __init__(self,code):
        if code not in CODES:raise ValueError("Unknown calibration diagnostic")
        self.code=code;super().__init__(code)
def require(ok,code="calibration_input_rejected"):
    if not ok:raise CalibrationError(code)
def checked(value):
    try:value=snapshot(value,limit=131072)
    except OutputError:raise CalibrationError("calibration_input_rejected")from None
    require(type(value)is dict and set(value)=={"format","purpose","selectors"} and
        value["format"]==FORMAT and value["purpose"]=="calibration" and
        type(value["selectors"])is list and 1<=len(value["selectors"])<=MAX_SELECTORS)
    ids=set();targets=set()
    for item in value["selectors"]:
        require(type(item)is dict and set(item)=={"id","module","argument","projection","domain"} and
            type(item["id"])is str and NAME.fullmatch(item["id"])and item["id"]not in ids and
            type(item["module"])is str and (item["module"]==""or PATH.fullmatch(item["module"]))and
            type(item["argument"])in (int,str) and (
             type(item["argument"])is int and 0<=item["argument"]<=3 or
             type(item["argument"])is str and NAME.fullmatch(item["argument"])))
        projection=item["projection"]
        require(type(projection)is dict and type(projection.get("kind"))is str)
        if projection["kind"]in ("tensor","layer_norm_variance"):
            require(set(projection)=={"kind"})
        else:
            require(projection["kind"]=="rms_mean_square" and set(projection)=={"kind","epsilon"} and
                type(projection["epsilon"])in (int,float) and
                math.isfinite(projection["epsilon"])and 0<=projection["epsilon"]<=1)
        domain=item["domain"]
        require(domain is None or type(domain)is list and len(domain)==2 and
            all(type(x)in (int,float)and math.isfinite(x)for x in domain)and
            -1024<=domain[0]<domain[1]<=1024)
        target=digest({k:v for k,v in item.items()if k!="id"})
        require(target not in targets)
        targets.add(target);ids.add(item["id"])
    return value
