"""Trusted independent reference preparation for unified candidate entry."""
import json
import math
from pathlib import Path
import numpy as np
from benchmark_graph import samples, validate
from benchmark_math import evaluate as mathematical
from benchmark_torch import evaluate as torch_reference, Model
from unified_graph_contract import layout


def prepare_case(graph,folder,chunk_period=None):
    check=validate(graph);plan=layout(graph,chunk_period);period=plan["input_slot_period"]
    inputs=[];references=[];logical={}
    for probe in samples(graph,4):
        expected=mathematical(graph,probe);observed=torch_reference(graph,probe)
        for name in expected:
            np.testing.assert_allclose(observed[name],expected[name],atol=1e-12,rtol=1e-12)
        vectors=[]
        for binding in plan["inputs"]:
            flat=probe[binding["name"]].reshape(-1)
            start=binding.get("offset",0);count=binding.get("elements",flat.size)
            vectors.append(np.pad(flat[start:start+count],(0,period-count)))
        inputs.append(np.stack(vectors))
        references.append(np.concatenate([expected[o["name"]].reshape(-1) for o in graph["outputs"]]))
    packed=np.stack(inputs)
    if len(plan["inputs"])==1:packed=packed[:,0,:]
    np.savez(folder/"arrays.npz",inputs=packed,reference=np.stack(references))
    np.savez(folder/"weights.npz",**{k:np.asarray(v,dtype=np.float64) for k,v in graph["constants"].items()})
    (folder/"logical-bindings.json").write_text(json.dumps(plan,indent=2))
    return Model(graph).eval()
