"""Reproduce paired operator graphs and explicitly trusted PyTorch sources."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from benchmark_graph import broadcast, digest
from pytorch_weight_package import export_state

MODELS = Path(__file__).resolve().parents[2] / "cases/matrix-entry-equivalence-v1/models.py"
SHAPES = {
    "matrix_matrix": [([2,3],[3,2]), ([3,2],[2,1]), ([1,4],[4,2])],
    "batched_cipher": [([2,2,2],[2,2,1]), ([1,3,2],[1,2,2]), ([2,1,3],[2,3,2])],
    "batch_broadcast": [([2,2,2],[2,1]), ([1,2,3],[2,3,1]), ([2,1,2],[2,2])],
    "higher_rank_cipher": [
        ([1,2,2,2],[1,1,2,1]),
        ([1]*5+[2,2,2],[1]*6+[2,1]),
        ([1]*29+[2,2,2],[1]*30+[2,1]),
    ],
    "vector_left": [([3],[3,2]), ([2],[2,2,1]), ([4],[1,4,3])],
    "vector_right": [([2,3],[3]), ([2,2,3],[3]), ([1,1,2,4],[4])],
    "public_right": [([2,3],[3,2]), ([1,2,3],[2,3,1]), ([2],[2,3])],
    "public_left": [([2,3],[3,2]), ([1,2,3],[2,3,1]), ([3],[3,2])],
}

def output_shape(first, second):
    left = [1,*first] if len(first)==1 else list(first)
    right = [*second,1] if len(second)==1 else list(second)
    if left[-1] != right[-2]:
        raise ValueError("Matrix contraction mismatch")
    batch = list(broadcast(tuple(left[:-2]), tuple(right[:-2])))
    return batch + ([] if len(first)==1 else [left[-2]]) + ([] if len(second)==1 else [right[-1]])

def generate(output):
    import torch
    output = Path(output)
    if output.exists():
        raise ValueError("Fresh fixture output required")
    output.mkdir(parents=True)
    raw = MODELS.read_bytes()
    source = hashlib.sha256(raw).hexdigest()
    spec = importlib.util.spec_from_file_location("owned_matrix_entry_sources", MODELS)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    (output/"models.py").write_bytes(raw)
    rows = []
    for group, contexts in SHAPES.items():
        for context, (first, second) in enumerate(contexts):
            case = group+"_"+str(context)
            side = "right" if group=="public_right" else "left" if group=="public_left" else None
            public_shape = first if side=="left" else second if side=="right" else None
            module = fixture.make_model(side, public_shape)
            if side is not None:
                # Non-repeating exact dyadic weights; no random forward/state changes.
                count = math.prod(public_shape)
                weight = torch.tensor([((i*7+context*3)%31-15)/32 for i in range(count)],
                                      dtype=torch.float64).reshape(public_shape)
                with torch.no_grad():
                    module.weight.copy_(weight)
            folder = output/case
            weights = export_state(module, folder, resource_profile="general-tensors-v1")
            (folder/"models.py").write_bytes(raw)
            inputs = ([] if side=="left" else [dict(name="left",shape=first)]) + (
                      [] if side=="right" else [dict(name="right",shape=second)])
            constants = {"public_weight":module.weight.detach().numpy().tolist()} if side else {}
            refs = ["public_weight" if side=="left" else "left",
                    "public_weight" if side=="right" else "right"]
            high = dict(format="poseidon-operator-graph-v1", id=case,
                inputs=inputs, constants=constants,
                nodes=[dict(id="product",op="matmul",inputs=refs,attrs={},outputs=["result"])],
                outputs=[dict(name="out",value="result")])
            shape = output_shape(first,second)
            if sum(math.prod(i["shape"]) for i in inputs)>32 or math.prod(shape)>16:
                raise ValueError("Small-context budget")
            manifest = dict(format="poseidon-trusted-pytorch-v7",files={"models.py":source},
                entry="models.make_model",factory=dict(args=[side,public_shape],kwargs={}),
                inputs=inputs,outputs=["out"],
                approximations=dict(format="poseidon-pytorch-approximations-v1",nodes=[],modules=[],defaults={}),
                dtype_policy="strict-float64-v1",weights=weights)
            for name, data in (("manifest.json",manifest),("operator-model.json",high)):
                (folder/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
            rows.append(dict(id=case,partition=group,context=context,public_side=side,
                first_shape=first,second_shape=second,output_shape=shape,
                manifest=str((folder/"manifest.json").relative_to(output)),
                operator_model=str((folder/"operator-model.json").relative_to(output)),
                operator_model_sha256=digest(high)))
    result = dict(format="poseidon-matrix-input-contexts-v1",cases=rows,partitions=len(SHAPES),
                  planned=len(rows),models_source_sha256=source,
                  old_frozen_benchmark_changed=False,paid_calls=0,
                  purpose="Same static mathematical operation across JSON and trusted PyTorch inputs")
    (output/"catalog.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = generate(args.output)
    print(json.dumps(dict(contexts=result["planned"],paid_calls=0)))
