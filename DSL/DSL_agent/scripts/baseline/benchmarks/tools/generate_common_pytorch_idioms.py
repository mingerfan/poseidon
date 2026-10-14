"""Deterministic common PyTorch source idioms; no model import or backend run."""
from pathlib import Path
import hashlib,json
import numpy as np
from benchmark_graph import canonical,digest,require

FAMILIES={
 "mlp":dict(styles=["sequential","loop","functional"],inputs=[dict(name="x",shape=[1,4])]),
 "cnn":dict(styles=["module","mixed","functional"],inputs=[dict(name="x",shape=[1,2,3,3])]),
 "residual":dict(styles=["operator","functional","method"],inputs=[dict(name="x",shape=[1,4])]),
 "normalization":dict(styles=["module","functional","manual"],inputs=[dict(name="x",shape=[1,4])]),
 "attention":dict(styles=["torch","functional","fused"],inputs=[dict(name=n,shape=[1,1,2,2]) for n in ("q","k","v")]),
 "dropout":dict(styles=["module","functional","spatial"],inputs=[dict(name="x",shape=[1,2,2,2])]),
}
def states(family):
    if family=="mlp":return {"layers.0.weight":(3,4),"layers.0.bias":(3,),"layers.2.weight":(2,3),"layers.2.bias":(2,)}
    if family=="cnn":return {"depth.weight":(2,1,3,3),"depth.bias":(2,),"norm.weight":(2,),"norm.bias":(2,),
        "norm.running_mean":(2,),"norm.running_var":(2,),"norm.num_batches_tracked":(),"project.weight":(3,2),"project.bias":(3,)}
    if family=="residual":return {"project.weight":(4,4),"project.bias":(4,)}
    if family=="normalization":return {"norm.weight":(4,),"norm.bias":(4,)}
    return {}
def run():
    import sys
    from hecate_python_env import ROOT,VENV
    require(Path(sys.prefix)==VENV,"Pinned NumPy environment")
    source=ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-idioms-v1/models.py"
    raw=source.read_bytes()
    output=ROOT/"scripts/baseline/cases/trusted-pytorch-common-idioms-v1"
    require(not output.exists(),"Preserve prior fixture corpus");output.mkdir()
    rows=[]
    for family,spec in FAMILIES.items():
        for style in spec["styles"]:
            name=family+"_"+style;case=output/name;case.mkdir();(case/"models.py").write_bytes(raw);weights={}
            for i,(key,shape) in enumerate(states(family).items()):
                if key.endswith("num_batches_tracked"):
                    value=np.asarray(0,dtype="<i8")
                else:
                    size=int(np.prod(shape));value=(np.arange(size,dtype=np.float64)*0.011-0.07+i*0.004).reshape(shape).astype("<f8")
                    if key.endswith("running_var"):value=np.abs(value)+1.3
                    elif family=="normalization" and key=="norm.weight":value=value+1.
                filename="state_"+str(i).zfill(4)+".npy";p=case/filename;np.save(p,value,allow_pickle=False);data=p.read_bytes()
                weights[key]=dict(file=filename,bytes=len(data),shape=list(shape),dtype=value.dtype.str,
                                  sha256=hashlib.sha256(data).hexdigest())
            defaults={}
            if family=="normalization":
                defaults["rsqrt"]=dict(method="newton",domain=[0.5,5.],iterations=1)
            elif family=="attention":
                defaults=dict(exp=dict(method="taylor",domain=[-8.,8.],degree=2),
                    reciprocal=dict(method="goldschmidt",domain=[0.5,100.],iterations=1))
            manifest=dict(format="poseidon-trusted-pytorch-v10",
                entry="models.make_model",factory=dict(args=[],kwargs=dict(family=family,style=style)),
                files={"models.py":hashlib.sha256(raw).hexdigest()},inputs=spec["inputs"],outputs=["result"],
                dtype_policy="strict-float64-v1",weights=weights,
                approximations=dict(format="poseidon-pytorch-approximations-v1",defaults=defaults,modules=[],nodes=[]))
            (case/"manifest.json").write_bytes(canonical(manifest))
            rows.append(dict(id=name,family=family,spelling=style,manifest=str(case/"manifest.json"),
                semantic_peer_group=family,expected_acceptance="audited_not_presumed",
                files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir()}))
    index=dict(format="poseidon-pytorch-common-idioms-v1",rows=rows,source_contexts=len(rows),semantic_peer_groups=len(FAMILIES),
        heldout=False,frozen_benchmark_modified=False,new_mathematical_model_count_claimed=False,
        compiler_calls=0,encrypted_execution=False,paid_calls=0)
    index["binding"]=digest(index);(output/"index.json").write_bytes(canonical(index))
    print(json.dumps({k:index[k] for k in ["binding","source_contexts","semantic_peer_groups"]}))
    return 0
if __name__=="__main__":raise SystemExit(run())
