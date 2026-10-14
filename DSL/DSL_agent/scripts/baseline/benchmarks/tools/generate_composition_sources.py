"""Reproduce the composition source packages with complete non-pickle state."""
import importlib.util,json,hashlib
from pathlib import Path
from pytorch_weight_package import export_state

MODELS=Path(__file__).resolve().parents[2]/"cases/trusted-pytorch-composition-v1/models.py"

def generate(output):
    output=Path(output)
    if output.exists():raise ValueError("Fresh fixture output required")
    output.mkdir(parents=True)
    raw=MODELS.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    spec=importlib.util.spec_from_file_location("composition_sources",MODELS)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    (output/"models.py").write_bytes(raw)
    rows=[]
    for family,contexts in fixture.SHAPES.items():
        for n,dims in enumerate(contexts):
            folder=output/(family+"_"+str(n))
            module=fixture.make_model(family,n)
            weights=export_state(module,folder,resource_profile="general-tensors-v1")
            (folder/"models.py").write_bytes(raw)
            manifest=dict(format="poseidon-trusted-pytorch-v7",files={"models.py":sha},
                entry="models.make_model",factory=dict(args=[family,n],kwargs={}),
                inputs=fixture.source_specs(family,n),outputs=fixture.source_outputs(n),
                approximations=fixture.source_approximations(family),dtype_policy="strict-float64-v1",weights=weights)
            (folder/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
            rows.append(dict(id=folder.name,partition=family,
                context=["tensor_output","tuple_output","dict_output"][n],
                manifest=str((folder/"manifest.json").relative_to(output)),input_shape=dims,source_origin="trusted_pytorch",
                approximation="declared approximation; original error reported separately" if fixture.source_approximations(family)["defaults"] else "not_required"))
    catalog=dict(format="poseidon-pytorch-composition-contexts-v1",cases=rows,models_source_sha256=sha,
        old_frozen_benchmark_changed=False,paid_calls=0,backend="poseidon_cpu",
        partitions=len(fixture.SHAPES),planned=len(rows))
    (output/"catalog.json").write_text(json.dumps(catalog,indent=2,sort_keys=True)+"\n")
    return catalog

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=generate(args.output)
    print(json.dumps(dict(contexts=result["planned"],partitions=result["partitions"],paid_calls=0)))
