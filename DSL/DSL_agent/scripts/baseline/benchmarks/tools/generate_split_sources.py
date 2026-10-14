"""Reproduce static split manifests and complete public NPY state."""
import json,hashlib,importlib.util
from pathlib import Path
from pytorch_weight_package import export_state
MODELS=Path(__file__).resolve().parents[2]/"cases/trusted-pytorch-split-v1/models.py"

def generate(output):
 output=Path(output)
 if output.exists():raise ValueError("Fresh fixture output required")
 output.mkdir(parents=True)
 raw=MODELS.read_bytes();source=hashlib.sha256(raw).hexdigest()
 spec=importlib.util.spec_from_file_location("owned_split_sources",MODELS)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 (output/"models.py").write_bytes(raw);rows=[]
 for family in module.SHAPES:
  for context in range(3):
   folder=output/(family+"_"+str(context))
   weights=export_state(module.make_model(family,context),folder,resource_profile="general-tensors-v1")
   (folder/"models.py").write_bytes(raw)
   value=dict(format="poseidon-trusted-pytorch-v7",files={"models.py":source},
    entry="models.make_model",factory=dict(args=[family,context],kwargs={}),
    inputs=module.source_specs(family,context),outputs=module.source_outputs(family,context),
    approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}),
    dtype_policy="strict-float64-v1",weights=weights)
   (folder/"manifest.json").write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
   rows.append(dict(id=folder.name,partition=family,context=context,
    manifest=str((folder/"manifest.json").relative_to(output)),input_shape=value["inputs"][0]["shape"],
    output_style="direct_tuple" if family=="tuple_output" else ["tensor","tuple","dict"][context]))
 catalog=dict(format="poseidon-pytorch-split-contexts-v1",partitions=6,planned=18,cases=rows,
              models_source_sha256=source,source_origin="ordinary_pytorch_fixed_public_weights",paid_calls=0)
 (output/"catalog.json").write_text(json.dumps(catalog,sort_keys=True,indent=2)+"\n")
 return catalog
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
 args=p.parse_args();result=generate(args.output);print(json.dumps(dict(contexts=result["planned"],paid_calls=0)))
