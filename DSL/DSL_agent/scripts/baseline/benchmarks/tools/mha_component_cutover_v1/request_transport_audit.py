"""Audit transport representation against immutable original request files."""
import json,sys,hashlib
from pathlib import Path
BASE=Path(__file__).resolve().parents[3];sys.path.insert(0,str(BASE))
from hecate_python_env import WORK
from component_contract import prepare_task
from candidate_bundle import load_model_provenance
from candidate_contract import request_file_bytes
from pytorch_model_intake import preparation_options
from benchmark_graph import canonical,digest
NAMES=("self_post_linear_residual__declared_shallow_v1","linear_into_separate_cross_none__declared_shallow_v1","boolean_padding_residual__declared_shallow_v1")
def differences(a,b,path="",result=None):
    if result is None:result=[]
    if len(result)>=12:return result
    if type(a)!=type(b):result.append(dict(path=path,reason="type",current=type(a).__name__,prior=type(b).__name__));return result
    if isinstance(a,dict):
        if set(a)!=set(b):result.append(dict(path=path,added=sorted(set(a)-set(b)),missing=sorted(set(b)-set(a))))
        for k in a:
            if k in b:differences(a[k],b[k],path+"/"+k,result)
    elif isinstance(a,list):
        if len(a)!=len(b):result.append(dict(path=path,current_length=len(a),prior_length=len(b)))
        for i,(x,y) in enumerate(zip(a,b)):differences(x,y,path+"/"+str(i),result)
    elif a!=b:result.append(dict(path=path,reason="value",current=str(a)[:80],prior=str(b)[:80]))
    return result
def main():
    rows=[]
    for n in NAMES:
        cap=WORK/"results/agent-mha-layout-component-backend-v1/capture"/n
        m=json.loads((cap/"model.json").read_bytes());p=load_model_provenance(cap,m,max_bytes=4*1024**2)
        options=preparation_options(p,m,dict(generation_guidance="explicit-v11",public_weight_policy="descriptors-math-roles-v2",chunk_period=4))
        q=prepare_task(m,options)
        old=json.loads((WORK/"results/agent-mha-ordered-backend-v1"/n/"request.json").read_bytes())
        stored=json.loads(request_file_bytes(q))
        rows.append(dict(id=n,same_request_id=q["request_id"]==old["request_id"],
            same_model=q["model"]==old["model"],same_layout=q["layout"]==old["layout"],
            same_compiler=q["compiler_configuration"]==old["compiler_configuration"],
            same_serialized_request=canonical(stored)==canonical(old),
            differences=differences(q,old),serialized_differences=differences(stored,old)))
    report=dict(format="poseidon-request-transport-comparison-v1",cases=rows,paid_api_calls=0,compiler_calls=0,encrypted_calls=0)
    report["binding"]=digest(report)
    (Path(__file__).parent/"request-transport-audit.json").write_bytes(canonical(report))
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
