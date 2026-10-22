"""Static source ABI for calibration before model graph construction."""
import math
from benchmark_graph import canonical
from model_output_contract import snapshot
from model_resources import resolve
from pytorch_source_contract import validate_output_tree
from pytorch_ordered_contract import allows_none_output
from source_calibration_contract import require

FORMAT="poseidon-source-calibration-signature-v1"

def checked_samples(manifest, signature, samples, input_dtypes, tree, resource_profile):
    import numpy as np
    limits=resolve(resource_profile)
    signature=snapshot(signature,limit=131072)
    require(type(signature)is dict and set(signature)=={"format","outputs"}and
            signature["format"]==FORMAT and type(signature["outputs"])is list and
            1<=len(signature["outputs"])<=limits.outputs)
    names=[row["name"]for row in signature["outputs"]if type(row)is dict and "name"in row]
    require(names==manifest["outputs"]and len(names)==len(signature["outputs"])and len(set(names))==len(names))
    shapes={}
    for row in signature["outputs"]:
        require(type(row)is dict and set(row)=={"name","shape"})
        dims=row["shape"]
        require(type(dims)is list and 1<=len(dims)<=limits.rank and
                all(type(n)is int and 1<=n<=limits.axis_size for n in dims))
        shapes[row["name"]]=dims
    require(sum(math.prod(s)for s in shapes.values())<=limits.output_elements)
    entries=manifest["inputs"]
    require(type(entries)is list and 1<=len(entries)<=limits.inputs)
    require(sum(math.prod(i["shape"])for i in entries)<=limits.input_elements)
    for item in entries:
        dims=item["shape"]
        require(type(dims)is list and len(dims)<=limits.rank and
                all(type(n)is int and 1<=n<=limits.axis_size for n in dims))
    dtypes=snapshot(input_dtypes,limit=131072)
    require(type(dtypes)is dict and set(dtypes)=={i["name"]for i in entries}and
            all(type(v)is str and v in("float32","float64")for v in dtypes.values()))
    tree=snapshot(tree,limit=131072)
    validate_output_tree(tree,names,allow_scalars=True,allow_none=allows_none_output(manifest))
    require(type(samples)is list and 1<=len(samples)<=16)
    wire=[]
    def leaves(value,dims):
        if dims:
            require(type(value)is list and len(value)==dims[0])
            for child in value:leaves(child,dims[1:])
        else:
            require(type(value)in(int,float)and math.isfinite(value))
            if type(value)is int:require(-(2**53)<=value<=2**53)
    for sample in samples:
        require(type(sample)is dict and set(sample)==set(dtypes))
        result={}
        for item in entries:
            name=item["name"];value=sample[name]
            if type(value)is list:leaves(value,item["shape"])
            else:require(type(value)is np.ndarray or not item["shape"]and type(value)in(int,float))
            array=np.asarray(value)
            require(array.dtype.kind in"fiu"and list(array.shape)==item["shape"]and np.isfinite(array).all())
            if array.dtype.kind in"iu":require(all(-(2**53)<=int(v)<=2**53 for v in array.flat))
            array=array.astype(np.float64)
            require(np.isfinite(array).all())
            result[name]=array.tolist()
        wire.append(result)
    require(len(canonical(wire))<=min(4*1024**2,limits.json_bytes))
    return wire,dtypes,tree,shapes
