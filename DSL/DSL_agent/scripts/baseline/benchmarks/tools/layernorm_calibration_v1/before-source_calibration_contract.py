"""Closed, hash-bound module observation selection; no fitting or reference edits."""
import math,re
from model_output_contract import snapshot,OutputError
from benchmark_graph import digest
FORMAT="poseidon-source-calibration-selection-v1"
FX_FORMAT="poseidon-source-fx-calibration-selection-v1"
FX_MODULE_FORMAT="poseidon-source-fx-calibration-selection-v2"
FX_FORMATS=(FX_FORMAT,FX_MODULE_FORMAT)
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
 "calibration_worker_protocol_error","calibration_fx_trace_rejected","calibration_fx_graph_changed"}
class CalibrationError(ValueError):
    def __init__(self,code):
        if code not in CODES:raise ValueError("Unknown calibration diagnostic")
        self.code=code;super().__init__(code)
def require(ok,code="calibration_input_rejected"):
    if not ok:raise CalibrationError(code)
def checked(value):
    try:value=snapshot(value,limit=131072)
    except OutputError:raise CalibrationError("calibration_input_rejected")from None
    if type(value)is dict and value.get("format")in FX_FORMATS:
        return checked_fx(value)
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


def checked_fx(value):
    from model_output_contract import is_hash
    require(type(value) is dict and set(value) == {
        "format", "purpose", "mode", "graph_sha256", "selectors"} and
        value["format"] in FX_FORMATS and value["purpose"] == "calibration" and
        value["mode"] in ("describe", "observe") and type(value["selectors"]) is list)
    if value["mode"] == "describe":
        require(value["graph_sha256"] is None and value["selectors"] == [])
        return value
    require(is_hash(value["graph_sha256"]) and 1 <= len(value["selectors"]) <= MAX_SELECTORS)
    ids, targets = set(), set()
    for item in value["selectors"]:
        require(type(item) is dict and set(item) == {
            "id", "node", "op", "target", "point", "argument", "projection", "domain"})
        require(type(item["id"]) is str and NAME.fullmatch(item["id"]) and item["id"] not in ids)
        require(type(item["node"]) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,95}", item["node"]))
        require(item["op"] in (("call_function","call_method","call_module")if value["format"]==FX_MODULE_FORMAT else ("call_function","call_method"))
                and type(item["target"])is str)
        require((len(item["target"])<=256 and PATH.fullmatch(item["target"]))
                if item["op"]=="call_module" else
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,191}",item["target"]))
        require(item["projection"] == {"kind": "tensor"} and item["point"] in ("input", "output"))
        arg = item["argument"]
        require((item["point"] == "output" and arg is None) or
                (item["point"] == "input" and
                 ((type(arg) is int and 0 <= arg <= 7) or
                  (type(arg) is str and NAME.fullmatch(arg)))))
        domain = item["domain"]
        require(domain is None or type(domain) is list and len(domain) == 2 and
                all(type(x) in (int, float) and math.isfinite(x) for x in domain) and
                -1024 <= domain[0] < domain[1] <= 1024)
        key = digest({k:v for k,v in item.items() if k!="id"})
        require(key not in targets)
        ids.add(item["id"]);targets.add(key)
    return value
