"""Independent original-model reference for saved one/multiple-input arrays."""
from model_graph import evaluate_reference,input_specs


def reference_batch(model, inputs):
    specs=input_specs(model)
    result=[]
    for row in inputs:
        supplied=(row.reshape(specs[0]["shape"]).tolist() if model["schema"]==2 else
                  {s["name"]:row[i].reshape(s["shape"]).tolist() for i,s in enumerate(specs)})
        result.append(evaluate_reference(model,supplied))
    return result
