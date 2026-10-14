"""Closed public arithmetic source controls; no compiler, FHE or provider calls."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from benchmark_graph import digest,require
from hecate_python_env import VENV,WORK
from application_identity import current_identity_manifest
from pytorch_weight_package import export_state,binding
from pytorch_public_expression_flow import _arrays
from pytorch_public_expression_capture import prepare,check_record,wrapped_metadata,make_reference
from pytorch_model_intake import run as isolated_run,package_format
from pytorch_ordered_contract import capture_binding,encode_weight_files
from candidate_bundle import check_provenance
from operator_reference import evaluate
from benchmarks.tools.public_arithmetic_expression_v1.models import Model,input_shape

PROFILE="general-tensors-v1"
def write(path,value):
    with path.open("x")as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
def expected(operation,context,x,state):
    s=state
    if operation=="add":
        if context==0:return (x-s["mean"])/(s["std"]+.01)
        if context==1:return x*(s["a"]+s["b"])
        return x+s["a"]+.25*s["b"]
    if operation=="subtract":
        if context==0:return x*(1.0-s["mask"].astype("<f8"))
        if context==1:return x*(s["a"]-s["b"])
        return x*(s["a"]-.5*s["b"]).T
    if operation=="multiply":
        if context==0:return x*s["a"]*s["b"]
        if context==1:return x*(.5*s["a"])[None,:]
        # Independent scalar summation, not F.linear or DSL lowering.
        w=s["a"]*s["b"]
        return np.asarray([[sum(float(x[row,j])*float(w[i,j])for j in range(3))
                            for i in range(2)]for row in range(x.shape[0])])
    if operation=="divide":
        if context==0:return x/s["std"]
        if context==1:return x*s["a"]/s["std"]
        return x*s["a"]/(s["std"]+.01)
    if context==0:return -x*s["a"]
    if context==1:return x-s["a"][None,:]
    w=-s["a"].T
    return np.asarray([[sum(float(x[row,j])*float(w[i,j])for j in range(3))
                        for i in range(2)]for row in range(x.shape[0])])
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned source control environment")
source=(HERE/"models.py").read_bytes();sdk=current_identity_manifest()
fixtures=HERE/"fixtures";fixtures.mkdir()
result_root=WORK/"results"/"agent-public-arithmetic-source-r1384";result_root.mkdir()
start=time.monotonic();rows=[]
for operation in("add","subtract","multiply","divide","negate"):
    for context in range(3):
        name=f"{operation}_c{context}";folder=fixtures/name;folder.mkdir();model_root=folder/"source";model_root.mkdir()
        module=Model(operation,context).double().eval()
        state=_arrays(module);declarations=export_state(module,model_root,resource_profile=PROFILE)
        (model_root/"model.py").write_bytes(source)
        spec=dict(format="poseidon-trusted-pytorch-v25",files={"model.py":hashlib.sha256(source).hexdigest()},
                  entry="model.build",factory=dict(args=[operation,context],kwargs={}),
                  inputs=[dict(name="x",shape=input_shape(operation,context))],outputs=["y"],
                  approximations=dict(format="poseidon-pytorch-approximations-v1",defaults={},modules=[],nodes=[]),
                  dtype_policy="strict-float64-v1",weights=declarations)
        write(model_root/"manifest.json",spec)
        direct=folder/"direct";direct.mkdir()
        capture=prepare(module,spec,direct,expected_weight_binding=binding(state,declarations),resource_profile=PROFILE)
        check_record(capture,spec,state,expected_record_binding=capture["binding"])
        metadata=wrapped_metadata(capture)
        report=make_reference(capture,spec,metadata,resource_profile=PROFILE,expected_weight_arrays=state)
        model,decomposition=capture_binding(capture["high_graph"],spec,resource_profile=PROFILE,
                         capture_metadata=metadata,expected_weight_binding=binding(state,declarations),
                         expected_weight_arrays=state)
        raw_files={v["file"]:(model_root/v["file"]).read_bytes()for v in declarations.values()}
        provenance=dict(original=capture["high_graph"],decomposition=decomposition,reference_report=report,
            trusted_pytorch=dict(format=package_format(spec),manifest=spec,files={"model.py":source.decode()},
                weight_files=encode_weight_files(spec,raw_files,resource_profile=PROFILE)))
        check_provenance(provenance,model)
        shape=input_shape(operation,context);count=math.prod(shape);rng=np.random.default_rng(731)
        probes=[np.zeros(shape),np.linspace(-.75,.875,count).reshape(shape),
                rng.uniform(-.9,.9,size=shape),np.resize(np.asarray([-.99,.99,-.5,.5,-.125,.125]),count).reshape(shape)]
        maximum=0.0;values=0
        for x in probes:
            actual=evaluate(capture["high_graph"],dict(x=x))["y"]
            wanted=expected(operation,context,x,state);error=float(np.max(np.abs(actual-wanted)))
            maximum=max(maximum,error);values+=wanted.size
            require(np.all(np.abs(actual-wanted)<=1e-12+1e-12*np.abs(wanted)),"Independent public arithmetic math")
        write(folder/"model.json",model);write(folder/"provenance.json",provenance);write(folder/"capture.json",capture)
        output=result_root/name
        result=isolated_run(model_root/"manifest.json",output,resource_profile=PROFILE,trusted=True)
        require(result["status"]=="captured"and result["reference_verified"]and not result["encrypted_execution"]and
                not result["agent_generated"],"Isolated capture scope")
        require(json.loads((output/"model.json").read_text())==model,"Isolated/direct model equality")
        isolated_provenance=dict(original=json.loads((output/"original.json").read_text()),
             decomposition=json.loads((output/"decomposition.json").read_text()),
             reference_report=json.loads((output/"reference-report.json").read_text()),
             trusted_pytorch=provenance["trusted_pytorch"])
        check_provenance(isolated_provenance,model)
        write(output/"provenance.json",isolated_provenance)
        flow=capture["public_expression_flow"]
        rows.append(dict(case=name,operation=operation,context=context,manifest=str(model_root/"manifest.json"),
            model=str(folder/"model.json"),provenance=str(folder/"provenance.json"),isolated_output=str(output),
            model_sha256=digest(model),manifest_sha256=digest(spec),capture_binding=capture["binding"],
            source_groups=capture["original_reference"]["groups"],source_values=capture["original_reference"]["compared_values"],
            independent_groups=4,independent_values=values,independent_max_absolute_error=maximum,
            public_entries=len(flow["entries"]),isolated_status="passed"))
        write(result_root/(name+"-checked.json"),rows[-1]);print(name,"source+isolated passed",flush=True)
require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"Source SDK unchanged")
data=dict(format="poseidon-closed-public-arithmetic-source-controls-v1",source_contract="poseidon-trusted-pytorch-v25",
    sdk_identity=sdk["manifest_sha256"],cases=rows,contexts=len(rows),
    source_reference_groups=sum(v["source_groups"]for v in rows),source_reference_values=sum(v["source_values"]for v in rows),
    independent_groups=sum(v["independent_groups"]for v in rows),independent_values=sum(v["independent_values"]for v in rows),
    independent_max_absolute_error=max(v["independent_max_absolute_error"]for v in rows),isolated_passed=len(rows),
    isolated_failed=0,seconds=time.monotonic()-start,provider_calls=0,compiler_calls=0,encrypted_calls=0)
data["binding"]=digest(data);write(HERE/"source-controls.json",data);write(result_root/"source-controls.json",data)
print(json.dumps({k:v for k,v in data.items()if k!="cases"}),flush=True)
