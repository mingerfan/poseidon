"""Deterministic, bounded layout preparation over existing Agent contracts.

This service selects an immutable request layout before generation. It does not
emit DSL, execute models or certify runtime memory and numerical correctness.
"""
import copy
import hashlib
import math
import signal
import threading
from pathlib import Path
from benchmark_graph import (FORMAT as GRAPH_FORMAT,CAPACITY_FORMAT,canonical,digest,
                             graph_resource_profile,require,validate)
from component_contract import OPTION_KEYS
from application_capabilities import assess_request
from application_policy import BACKEND,prepare as prepare_application
from compiler_configuration import CPU_CONFIGURATIONS
from public_weight_context import descriptor
from unified_capacity_layout import WIDE_PERIODS,EXTENDED_PROFILES,physical_limits
from unified_chunk_layout import PERIODS

FORMAT="poseidon-layout-plan-v1"
POLICY="bound-layouts-v1"
DEFAULT_LIMITS=dict(wall_seconds=120,candidate_limit=8)
MAX_LIMITS=dict(wall_seconds=180,candidate_limit=8)
PINNED_CONSTRUCTS=("construction_profile","construction","helper_profile",
                   "helper_exercise","capability_composition")
ROOT=Path(__file__).resolve().parents[3]

class _Deadline(Exception):pass

def _limits(value):
    value=copy.deepcopy(DEFAULT_LIMITS if value is None else value)
    require(type(value) is dict and set(value)==set(DEFAULT_LIMITS),"Planning limit fields")
    require(all(type(v) is int and 1<=v<=MAX_LIMITS[k] for k,v in value.items()),
            "Planning limits")
    return value

def source_authority():
    from application_identity import source_closure,POLICY as SOURCE_POLICY
    root=ROOT/"scripts/baseline"
    closure=source_closure(root,("application_planning/layout.py","application_component.py"))
    return dict(policy=SOURCE_POLICY,files=closure["sources"],
                sha256=digest(closure))

def _finish(value):
    result=copy.deepcopy(value)
    result["binding"]=digest(result)
    return result

def _record():
    return dict(format=FORMAT,policy=POLICY,status="planning",failure=None,
        model_sha256=None,source_authority=None,backend=BACKEND,
        limits={},base_options={},candidate_specs=[],candidates=[],selection=None,
        acceptance=dict(compiled=False,encrypted_execution=False,numerically_validated=False))

def _failure(value,code,layer,*,status="rejected"):
    value.update(status=status,failure=dict(code=code,layer=layer,location=None,context={}))
    return _finish(value)

def candidate_options(model,options,limit):
    """Retain explicit user bindings; explore only registered periods and packing."""
    profile=graph_resource_profile(model)
    checked=validate(model)
    current=copy.deepcopy(options)
    result=[current]
    if "chunk_period" in options or any(options.get(k) is not None for k in PINNED_CONSTRUCTS):
        return result
    capacity=model["format"]==CAPACITY_FORMAT
    allowed=list(PERIODS)+list(WIDE_PERIODS if capacity and profile in EXTENDED_PROFILES else ())
    _,inputs,outputs=physical_limits(profile) if capacity else (None,4,16)
    input_sizes=[math.prod(s["shape"]) for s in model["inputs"]]
    output_sizes=[math.prod(checked["shapes"][s["value"]]) for s in model["outputs"]]
    feasible=[p for p in allowed if sum(math.ceil(n/p) for n in input_sizes)<=inputs
              and sum(math.ceil(n/p) for n in output_sizes)<=outputs]
    if not feasible:return result
    lower=feasible[0]
    target=next((p for p in feasible if p>=max(input_sizes+output_sizes)),feasible[-1])
    chosen=sorted({lower,target,feasible[min(1,len(feasible)-1)]})
    for period in chosen:
        item={**copy.deepcopy(options),"chunk_period":period}
        if item not in result:result.append(item)
    if capacity and "capacity_packing_version" not in options:
        from unified_capacity_constants import VERSION,CHANNEL_VERSION
        for period in (lower,target):
            if period in WIDE_PERIODS:continue
            for version in (VERSION,CHANNEL_VERSION):
                item={**copy.deepcopy(options),"chunk_period":period,
                      "capacity_packing_version":version}
                if item not in result:result.append(item)
    # The candidate bound limits preparation work, not model expressibility.
    return result[:limit]

def request_resources(request):
    assessment=assess_request(request)
    require(assessment["admission"]=="accepted_pending_validation","Bound request assessment")
    layout=request["layout"]
    shapes=[descriptor(value)["shape"] for value in request["public_constants"].values()]
    entries=len(shapes)
    elements=sum(math.prod(shape) for shape in shapes)
    objects=len(layout["inputs"])+layout["output_ciphertexts"]
    resources=copy.deepcopy(assessment["resources"])
    resources.update(
        public_registry_entries=dict(value=entries,certainty="exact",basis="bound numeric registry"),
        public_registry_elements=dict(value=elements,certainty="exact",basis="bound registry shapes"),
        logical_public_array_bytes=dict(value=elements*8,certainty="exact",
            basis="float64 logical payload; excludes CKKS encoding, caches and allocation"),
        declared_io_ciphertexts=dict(value=objects,certainty="exact",
            basis="physical declarations; output aliases can share actual runtime storage"))
    # Unknown quantities remain unknown, irrespective of a layout being admitted.
    require(resources["peak_memory_bytes"]["certainty"]=="unknown" and
            resources["execution_seconds"]["certainty"]=="unknown","Precompile scope")
    return resources,assessment["stages"]

def _score(resources):
    return [resources[k]["value"] for k in
            ("declared_io_ciphertexts","public_registry_entries","public_registry_elements","slot_period")]

def plan_model(model,*,options=None,backend=BACKEND,limits=None):
    value=_record()
    try:
        bounded=_limits(limits);value["limits"]=bounded
    except (ValueError,TypeError):
        return _failure(value,"planning_limits_rejected","budget")
    if threading.current_thread() is not threading.main_thread() or signal.getitimer(signal.ITIMER_REAL)!=(0.,0.):
        return _failure(value,"planning_worker_busy","environment",status="service_error")
    previous=signal.getsignal(signal.SIGALRM)
    def deadline(*_):raise _Deadline()
    try:
        signal.signal(signal.SIGALRM,deadline)
        signal.setitimer(signal.ITIMER_REAL,bounded["wall_seconds"])
        value["source_authority"]=source_authority()
        try:
            require(backend==BACKEND,"Planner backend")
            require(type(options) in (dict,type(None)),"Planner options")
            options=copy.deepcopy(options or {})
            require(set(options)<=OPTION_KEYS,"Planner option fields")
            require(options.get("compiler_configuration","poseidon-cpu-eva-w45-v1") in CPU_CONFIGURATIONS,
                    "Planner compiler")
            require(type(model) is dict and model.get("format") in (GRAPH_FORMAT,CAPACITY_FORMAT),
                    "Normalized core graph required")
            model=copy.deepcopy(model);validate(model)
            value["model_sha256"]=digest(model);value["base_options"]=options
            value["candidate_specs"]=candidate_options(model,options,bounded["candidate_limit"])
        except (ValueError,TypeError,KeyError,IndexError,OverflowError,RecursionError):
            return _failure(value,"planning_input_rejected","input")
        unique={}
        for index,item in enumerate(value["candidate_specs"]):
            row=dict(index=index,options=copy.deepcopy(item),status="preparing",failure=None,
                     request_id=None,resources=None,stages=None,score=None)
            value["candidates"].append(row)
            try:
                request=prepare_application(model,BACKEND,"numerical",item)[0]
                resources,stages=request_resources(request)
                row.update(status="admitted",request_id=request["request_id"],
                           resources=resources,stages=stages,score=_score(resources))
                if row["request_id"] in unique:
                    row.update(status="duplicate",same_as=unique[row["request_id"]])
                else:unique[row["request_id"]]=index
            except (ValueError,TypeError,KeyError,IndexError,OverflowError,RecursionError):
                row.update(status="rejected",failure=dict(code="layout_binding_rejected",layer="layout",
                                                          location=None,context={}))
        admitted=[row for row in value["candidates"] if row["status"]=="admitted"]
        if not admitted:return _failure(value,"no_admitted_layout","layout")
        selected=min(admitted,key=lambda row:(row["score"],row["index"]))
        value.update(status="prepared",selection=dict(index=selected["index"],
            request_id=selected["request_id"],options=copy.deepcopy(selected["options"]),
            score=selected["score"],ranking="declared IO objects, public entries, public elements, period"))
        require(value["source_authority"]==source_authority(),"Planning dependency drift")
        return _finish(value)
    except _Deadline:
        return _failure(value,"planning_budget_exhausted","budget",status="budget_exhausted")
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)

def check_plan(value,model):
    """Check a host-produced decision and its selected immutable request.

    The hash is an integrity binding, not a signature proving who created it.
    Unselected costs are explanatory; executable choices are reconstructed.
    """
    require(type(value) is dict and value.get("format")==FORMAT and value.get("policy")==POLICY,
            "Layout plan format")
    require(value.get("binding")==digest({k:v for k,v in value.items() if k!="binding"}),"Layout plan integrity")
    require(value["status"]=="prepared" and value["failure"] is None and
            value["model_sha256"]==digest(model),"Prepared layout/model binding")
    require(value["source_authority"]==source_authority(),"Layout plan source identity")
    bounded=_limits(value["limits"])
    expected=candidate_options(model,value["base_options"],bounded["candidate_limit"])
    require(expected==value["candidate_specs"],"Layout candidate inventory")
    selection=value["selection"]
    require(type(selection) is dict and type(selection.get("index")) is int
            and 0<=selection["index"]<len(value["candidates"]),"Layout selection index")
    row=value["candidates"][selection["index"]]
    require(row["status"]=="admitted" and row["options"]==selection["options"]==
            expected[selection["index"]] and row["request_id"]==selection["request_id"],"Selected layout binding")
    request=prepare_application(model,BACKEND,"numerical",selection["options"])[0]
    resources,stages=request_resources(request)
    require(request["request_id"]==selection["request_id"] and row["resources"]==resources
            and row["stages"]==stages and row["score"]==selection["score"]==_score(resources),
            "Selected layout resource binding")
    admitted=[r for r in value["candidates"] if r["status"]=="admitted"]
    require(min(admitted,key=lambda r:(r["score"],r["index"]))["index"]==selection["index"],
            "Layout ranking")
    return request

def submit_planned(app,model,planned,**requirements):
    """Use the existing Application budget, status, repair and package workflow."""
    require("options" not in requirements,"Planning owns explicit selected options")
    request=check_plan(planned,model)
    task=app.submit(model,options=copy.deepcopy(planned["selection"]["options"]),**requirements)
    return dict(task_id=task,request_id=request["request_id"],
                layout_plan_binding=planned["binding"],layout_policy=POLICY)
