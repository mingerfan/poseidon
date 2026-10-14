"""Freeze V11 source contexts separately; retain the V10 source/weight corpus."""
import copy,hashlib,json,shutil
from pathlib import Path
import numpy as np
from benchmark_graph import canonical,digest,require
from hecate_python_env import ROOT,VENV
from pytorch_model_intake import COMMON_FORMAT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def state_shapes(family,context,mask_kind):
    if family=="public_alpha":return {} if context==2 else {"bias":(3,)}
    result={}
    for name in ("q","k","v"):result.update({name+".weight":(4,4),name+".bias":(4,)})
    length=3 if context==1 else 2
    if mask_kind in ("boolean","additive"):result["mask"]=(length,length)
    return result
def run():
    import sys
    require(Path(sys.prefix)==VENV,"Pinned NumPy environment")
    old=ROOT/"scripts/baseline/cases/trusted-pytorch-common-idioms-v1"
    index=json.loads((old/"index.json").read_text())
    require(index["binding"]==digest({k:v for k,v in index.items() if k!="binding"}),"Legacy corpus binding")
    output=ROOT/"scripts/baseline/cases/trusted-pytorch-common-v11"
    require(not output.exists(),"Preserve prior case corpus");output.mkdir()
    rows=[]
    def row(case,**fields):
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(case.iterdir()) if p.is_file()}
        rows.append(dict(id=case.name,manifest=str(case/"manifest.json"),files=files,**fields))
    for entry in index["rows"]:
        source=old/entry["id"]
        for name,expected in entry["files"].items():require(sha(ROOT/name)==expected,"Legacy source content changed")
        case=output/entry["id"];case.mkdir()
        for p in source.iterdir():
            require(p.is_file() and not p.is_symlink(),"Legacy regular source")
            if p.name!="manifest.json":shutil.copyfile(p,case/p.name)
        original=json.loads((source/"manifest.json").read_text());updated=copy.deepcopy(original);updated["format"]=COMMON_FORMAT
        (case/"manifest.json").write_bytes(canonical(updated))
        row(case,family=entry["family"],context=entry["spelling"],semantic_peer_group=entry["semantic_peer_group"],
            source_version="v11",original_manifest=str(source/"manifest.json"),
            original_manifest_sha256=sha(source/"manifest.json"),model_source_and_weights_unchanged=True)
    raw=(ROOT/"scripts/baseline/benchmarks/tools/fixtures/common-pytorch-v11/models.py").read_bytes()
    new_specs=[("projected_attention",context,mask) for mask in ("none","boolean","additive","causal") for context in range(3)]
    new_specs += [("public_alpha",context,"none") for context in range(3)]
    for family,context,mask_kind in new_specs:
        name=family+"_"+mask_kind+"_"+str(context);case=output/name;case.mkdir();(case/"models.py").write_bytes(raw)
        rng=np.random.default_rng(711+context);weights={}
        for i,(key,shape) in enumerate(state_shapes(family,context,mask_kind).items()):
            if key=="mask" and mask_kind=="boolean":value=np.tril(np.ones(shape,dtype=bool))
            elif key=="mask":value=(np.arange(np.prod(shape),dtype=np.float64).reshape(shape)*.025-.125).astype("<f8")
            else:value=rng.uniform(-.12,.12,size=shape).astype("<f8")
            filename="state_"+str(i).zfill(4)+".npy";p=case/filename;np.save(p,value,allow_pickle=False)
            weights[key]=dict(file=filename,shape=list(shape),dtype=value.dtype.str,bytes=p.stat().st_size,sha256=sha(p))
        length=3 if context==1 else 2
        inputs=[dict(name="x",shape=[1,length,4])] if family=="projected_attention" else [dict(name="x",shape=[2,3])]
        if family=="projected_attention" and context==2:inputs.append(dict(name="other",shape=[1,length,4]))
        outputs=["context","skip"] if family=="projected_attention" and context==2 else ["result"]
        defaults={} if family=="public_alpha" else dict(
            exp=dict(method="taylor",domain=[-8.,8.],degree=2),
            reciprocal=dict(method="goldschmidt",domain=[.5,100.],iterations=1))
        manifest=dict(format=COMMON_FORMAT,entry="models.make_model",
            factory=dict(args=[],kwargs=dict(family=family,context=context,mask_kind=mask_kind)),
            files={"models.py":hashlib.sha256(raw).hexdigest()},inputs=inputs,outputs=outputs,
            dtype_policy="strict-float64-v1",weights=weights,
            approximations=dict(format="poseidon-pytorch-approximations-v1",defaults=defaults,modules=[],nodes=[]))
        (case/"manifest.json").write_bytes(canonical(manifest))
        row(case,family=family,context=context,mask_kind=mask_kind,
            semantic_peer_group=family+"_"+mask_kind+"_"+str(context),
            source_version="v11",non_repeated_weight_seed=711+context,approximation_accuracy_certified=False)
    record=dict(format="poseidon-common-pytorch-v11-contexts",rows=rows,source_contexts=len(rows),
        original_v10_source_contexts=18,new_contexts=len(new_specs),source_version=COMMON_FORMAT,
        compiler_calls=0,encrypted_execution=False,paid_calls=0,frozen_benchmark_modified=False,
        new_mathematical_model_count_claimed=False)
    record["binding"]=digest(record);(output/"index.json").write_bytes(canonical(record))
    print(json.dumps(dict(binding=record["binding"],source_contexts=len(rows),output=str(output))))
    return 0
if __name__=="__main__":raise SystemExit(run())
