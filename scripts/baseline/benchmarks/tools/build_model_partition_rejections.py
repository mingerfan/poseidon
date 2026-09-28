"""Generate concrete model-gate counterexamples for mathematical partitions.

Not part of the 1200 positive models and not evidence of ciphertext execution.
The original source suite and validator are bound by hashes. Candidate/trace
counterexamples for construction partitions are separate.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE))
from benchmark_graph import validate,digest,signature
from build_model_semantic_contexts import read_suite
from benchmark_semantics import features

def node_for(model,op):
    return next((i for i,n in enumerate(model["nodes"]) if n["op"]==op),None)

def invalid_attribute(node,key):
    old=node["attrs"][key]
    if key in ("axis","dim0","dim1"):return 99
    if key in ("step","eps","groups"):return 0
    if key in ("start","stop"):return "not_an_integer"
    if key=="shape":return [-2]
    if key=="dims":return []
    if key=="axes":return [0,0]
    if key in ("keepdims","count_include_pad"):return "not_a_boolean"
    if key=="basis":return "unsupported_basis"
    if key=="exponent":return 3
    if key in ("stride","dilation","kernel"):return [0]*len(old)
    if key=="padding":return [-1]*len(old)
    if key=="sections":return [0]
    raise ValueError("Unreviewed attribute "+key)

def counterexample(model,feature):
    validate(model)
    if feature not in features(model):raise ValueError("Positive control lacks feature")
    negative=copy.deepcopy(model)
    parts=feature.split(".")
    if parts[0]=="input":
        if parts[1]=="count":path=["inputs"];negative["inputs"]=[]
        else:
            i=next(i for i,x in enumerate(negative["inputs"])
                   if parts[1]!="rank" or len(x["shape"])==int(parts[2]))
            path=["inputs",i,"shape"]
            negative["inputs"][i]["shape"]=[] if parts[1]=="rank" else [257]
    elif parts[0]=="output":
        path=["outputs"];negative["outputs"]=[]
    elif parts[0]=="graph":
        if feature!="graph.shared_value":raise ValueError("Unreviewed graph partition")
        counts=Counter(ref for n in negative["nodes"] for ref in set(n["inputs"])
                       if ref not in negative["constants"])
        ref=next(ref for ref,c in counts.items() if c>1)
        i=next(i for i,n in enumerate(negative["nodes"]) if ref in n["inputs"])
        j=negative["nodes"][i]["inputs"].index(ref)
        names={x["name"] for x in negative["inputs"]}|set(negative["constants"])
        names|={v for n in negative["nodes"] for v in n["outputs"]}
        missing="missing_context_value"
        while missing in names:missing+="x"
        negative["nodes"][i]["inputs"][j]=missing;path=["nodes",i,"inputs",j]
    elif parts[0] in ("op","parameter","overload"):
        op=parts[1];i=node_for(negative,op)
        if i is None:raise ValueError("Missing target operation")
        node=negative["nodes"][i]
        if parts[0]=="parameter":
            key=parts[2];node["attrs"][key]=invalid_attribute(node,key)
            path=["nodes",i,"attrs",key]
        elif parts[0]=="overload":
            # Exact encrypted/public combination gate: public value in first position.
            if negative["constants"]:public=next(iter(negative["constants"]))
            else:
                public="context_public"
                negative["constants"][public]=0.25
            node["inputs"][0]=public;path=["nodes",i,"inputs",0]
        elif op in ("add","subtract","multiply"):
            node["inputs"]=node["inputs"][:1];path=["nodes",i,"inputs"]
        elif op in ("negate","square","flatten"):
            node["inputs"].append(node["inputs"][0]);path=["nodes",i,"inputs"]
        elif op=="linear":
            node["inputs"][1]=negative["inputs"][0]["name"];path=["nodes",i,"inputs",1]
        else:
            key={"power":"exponent","reshape":"shape","transpose":"dim0","permute":"dims",
                 "rotate":"step","concat":"axis","stack":"axis","slice":"step",
                 "split":"sections","sum":"axes","mean":"axes","batch_norm":"eps",
                 "polynomial":"basis","conv1d":"groups","conv2d":"groups",
                 "avg_pool1d":"kernel","avg_pool2d":"kernel"}[op]
            node["attrs"][key]=invalid_attribute(node,key);path=["nodes",i,"attrs",key]
    else:raise ValueError("Unreviewed partition "+feature)
    try:validate(negative)
    except ValueError as error:reason=str(error)
    else:raise ValueError("Counterexample unexpectedly accepted: "+feature)
    return dict(requirement=feature,positive_model_id=model["id"],
                positive_model_sha256=digest(model),positive_control_accepted=True,
                negative_model=negative,negative_model_sha256=digest(negative),
                mutation_path=path,observed_rejection=reason,
                failure_layer="model_graph_validation",
                evidence_scope="Illegal type/shape/resource/operand or attribute gate; not a numerical equivalence counterexample",
                encrypted_execution=False,counts_as_positive_model=False)

def build(rows,ledger):
    by={r["model"]["id"]:r for r in rows}
    results=[]
    for req in ledger["requirements"]:
        if req["layer"]!="model":continue
        parent=min((by[n] for n in req["models"]),key=lambda r:(len(r["model"]["nodes"]),r["model"]["id"]))
        result=counterexample(parent["model"],req["id"])
        result["id"]="model_reject_"+digest(req["id"])[:16]
        result["task_sha256"]=digest(result)
        results.append(result)
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error("Preserve existing evidence")
    rows,index,ledger=read_suite(a.suite);tasks=build(rows,ledger)
    payload=dict(schema=1,kind="mathematical_partition_rejection_examples",
                 authoritative_ledger=False,original_model_count=1200,
                 planned=len(tasks),positive_controls_passed=len(tasks),expected_rejections_passed=len(tasks),
                 failed=0,skipped=0,encrypted_execution=False,paid_calls=0,tasks=tasks,
                 sources={str(p.relative_to(BASE.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__).resolve(),BASE/"benchmark_graph.py",BASE/"benchmark_semantics.py"]},
                 suite_index_sha256=hashlib.sha256((a.suite/"index.json").read_bytes()).hexdigest(),
                 coverage_sha256=index["coverage_sha256"])
    payload["bundle_sha256"]=digest(payload)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in payload.items() if k not in ("tasks","sources")}))
    return 0
if __name__=="__main__":raise SystemExit(main())
