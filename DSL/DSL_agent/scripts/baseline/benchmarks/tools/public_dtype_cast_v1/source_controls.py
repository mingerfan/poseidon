"""Small public-cast source contexts, independent math, and isolated capture."""
import hashlib,json,math,os,struct,sys,time,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];sys.path.insert(0,str(BASE))
from benchmark_graph import digest,require
from pytorch_weight_package import binding
from pytorch_public_expression_capture import prepare,check_record,wrapped_metadata,_arrays
from pytorch_model_intake import run as isolated_run,package_format
from candidate_bundle import check_provenance
from operator_reference import evaluate
from operator_arithmetic_graph import decompose
from test_pytorch_public_dtype_cast import PublicModule,specification,DTYPES
from application_identity import current_identity_manifest
from hecate_python_env import VENV
require(os.environ.get("IN_NIX_SHELL")=="pure"and Path(sys.prefix)==VENV,"Pinned source controls")
MODEL_SOURCE='''import torch
class Model(torch.nn.Module):
    def __init__(self,dtype,target,context=0,roundtrip=False):
        super().__init__()
        types={"<f4":torch.float32,"<f8":torch.float64,"<i4":torch.int32,"<i8":torch.int64,"|b1":torch.bool}
        self.register_buffer("public",torch.zeros((2,3),dtype=types[dtype]))
        self.target=target;self.context=context;self.roundtrip=roundtrip
    def forward(self,x):
        w=self.public
        if self.target=="<f4":w=w.float()
        else:w=w.to(dtype=torch.float64)
        if self.roundtrip:w=w.double()
        if self.context==1:return x+w.transpose(0,1)
        y=x*w
        if self.context==2:return {"primary":y+x,"saved":[x,None]}
        return y
def build(dtype,target,context=0,roundtrip=False):
    return Model(dtype,target,context,roundtrip)
'''
def write(path,value):
    with path.open("x")as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
sdk=current_identity_manifest()
fixtures=HERE/"fixtures-r2";fixtures.mkdir()
controls=[];qualified=[];start=time.monotonic()
cases=[(f"{short}_to_{target_name}_c{context}",dtype,target,context,False)
       for dtype,short in DTYPES for target,target_name in(("<f4","f32"),("<f8","f64"))for context in range(3)]
cases +=[(f"f64_roundtrip_c{context}","<f8","<f4",context,True)for context in range(3)]
def f32(value):return float(struct.unpack("<f",struct.pack("<f",float(value)))[0])
for name,dtype,target,context,roundtrip in cases:
    directory=fixtures/name;directory.mkdir();module=PublicModule(dtype,target,context,roundtrip)
    weights=directory/"weights";weights.mkdir()
    spec=specification(module,folder=weights);spec["files"]={"model.py":hashlib.sha256(MODEL_SOURCE.encode()).hexdigest()}
    spec["factory"]=dict(args=[dtype,target,context],kwargs=dict(roundtrip=roundtrip))
    (weights/"model.py").write_text(MODEL_SOURCE);write(weights/"manifest.json",spec)
    output=directory/"direct";output.mkdir()
    arrays=_arrays(module);record=prepare(module,spec,output,expected_weight_binding=binding(arrays,spec["weights"]))
    check_record(record,spec,arrays,expected_record_binding=record["binding"])
    metadata=wrapped_metadata(record)
    raw_files={d["file"]:(weights/d["file"]).read_bytes()for d in spec["weights"].values()}
    from pytorch_ordered_contract import encode_weight_files
    report=__import__("pytorch_public_expression_capture").make_reference(record,spec,metadata,
            resource_profile="general-tensors-v1",expected_weight_arrays=arrays)
    original=record["high_graph"]
    from pytorch_ordered_contract import capture_binding
    model,lowering=capture_binding(original,spec,resource_profile="general-tensors-v1",
        capture_metadata=metadata,expected_weight_binding=binding(arrays,spec["weights"]),expected_weight_arrays=arrays)
    require(model==record["lowered_graph"],"Captured model and typed provenance binding")
    provenance=dict(original=original,decomposition=lowering,reference_report=report,trusted_pytorch=dict(
        format=package_format(spec),manifest=spec,files={"model.py":MODEL_SOURCE},
        weight_files=encode_weight_files(spec,raw_files,resource_profile="general-tensors-v1")))
    check_provenance(provenance,model)
    shape=(3,2)if context==1 else(2,3)
    rng=np.random.default_rng(731)
    probes=[np.zeros(shape),np.linspace(-.75,.875,6).reshape(shape),rng.uniform(-.9,.9,size=shape),
            np.array([-.99,.99,-.5,.5,-.125,.125]).reshape(shape)]
    original_values=arrays["public"].astype("<f8")
    coeff=np.array([f32(x)if target=="<f4"else float(x)for x in original_values.reshape(-1)]).reshape(2,3)
    maximum=0.0
    for x in probes:
        expected=x+coeff.T if context==1 else x*coeff+x if context==2 else x*coeff
        actual=evaluate(original,dict(x=x))
        error=float(np.max(np.abs(actual["y"]-expected)));maximum=max(maximum,error)
        require(np.all(np.abs(actual["y"]-expected)<=1e-12+1e-12*np.abs(expected)),"Independent public cast math")
        if context==2:require(np.array_equal(actual["saved"],x),"Independent named saved input")
    write(directory/"record.json",record);write(directory/"model.json",model);write(directory/"provenance.json",provenance)
    selected=(name in {"f32_to_f64_c0","f64_to_f32_c1","i32_to_f32_c0","i64_to_f64_c1","bool_to_f32_c2","f64_roundtrip_c2"})
    qualified.append(dict(case=name,dtype=dtype,target_dtype=target,context=context,roundtrip=roundtrip,
        manifest_path=str(weights/"manifest.json"),model=str(directory/"model.json"),provenance=str(directory/"provenance.json"),
        capture_binding=record["binding"],source_reference_groups=record["original_reference"]["groups"],
        source_reference_values=record["original_reference"]["compared_values"],
        independent_groups=4,independent_max_absolute_error=maximum,isolated_selected=selected))
write(HERE/"source-contexts-r2.json",qualified)
isolated=[]
for row in qualified:
    if not row["isolated_selected"]:continue
    output=HERE/"isolated-r2"/row["case"]
    result=isolated_run(row["manifest_path"],output,resource_profile="general-tensors-v1",trusted=True)
    require(result["status"]=="captured"and result["reference_verified"]and
            not result["encrypted_execution"]and not result["agent_generated"],"Isolated model capture scope")
    isolated_model=json.loads((output/"model.json").read_text())
    require(isolated_model==json.loads(Path(row["model"]).read_text()),"Isolated and direct model equality")
    isolated_spec=json.loads(Path(row["manifest_path"]).read_text());weight_root=Path(row["manifest_path"]).parent
    isolated_provenance=dict(original=json.loads((output/"original.json").read_text()),
        decomposition=json.loads((output/"decomposition.json").read_text()),
        reference_report=json.loads((output/"reference-report.json").read_text()),
        trusted_pytorch=dict(format=package_format(isolated_spec),manifest=isolated_spec,
            files={"model.py":MODEL_SOURCE},weight_files=encode_weight_files(isolated_spec,
              {w["file"]:(weight_root/w["file"]).read_bytes()for w in isolated_spec["weights"].values()},
              resource_profile="general-tensors-v1")))
    check_provenance(isolated_provenance,isolated_model)
    write(output/"provenance.json",isolated_provenance)
    isolated.append(dict(case=row["case"],status="passed",output=str(output)))
require(current_identity_manifest()["manifest_sha256"]==sdk["manifest_sha256"],"Source SDK unchanged during controls")
data=dict(format="poseidon-public-dtype-cast-source-controls-v1",source_contract="poseidon-trusted-pytorch-v24",
    sdk_identity=sdk["manifest_sha256"],contexts=len(qualified),groups=10,roundtrip_contexts=3,
    source_reference_groups=sum(r["source_reference_groups"]for r in qualified),
    source_reference_values=sum(r["source_reference_values"]for r in qualified),
    independent_groups=sum(r["independent_groups"]for r in qualified),
    independent_max_absolute_error=max(r["independent_max_absolute_error"]for r in qualified),
    isolated_capture_passed=len(isolated),isolated_capture_failed=0,isolated=isolated,
    seconds=time.monotonic()-start,provider_calls=0,compiler_calls=0,encrypted_calls=0)
data["binding"]=digest(data);write(HERE/"source-controls.json",data)
print(json.dumps(data))
