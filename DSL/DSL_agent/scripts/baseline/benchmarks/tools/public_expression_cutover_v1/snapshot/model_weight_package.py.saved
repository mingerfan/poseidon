"""Hashed, non-pickle public NPY weights for offline model admission."""
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from benchmark_graph import require, NAME, canonical, digest, shape
from model_resources import resolve, identity

FORMAT="poseidon-model-package-v1"

def regular_bytes(path,limit):
    # Read exactly the opened inode, without following a final symlink.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and 0<info.st_size<=limit,"Weight file size/type")
        with os.fdopen(fd,"rb",closefd=False) as stream:data=stream.read(limit+1)
        require(len(data)==info.st_size,"Weight changed during read")
        return data
    finally:os.close(fd)

def load(path):
    import numpy as np
    path=Path(path)
    from public_weight_context import parse_data
    spec=parse_data(regular_bytes(path,2*1024**2).decode("utf-8"))
    require(type(spec) is dict and set(spec)=={"format","resource_profile","model","weights"}
            and spec["format"]==FORMAT,"Model package fields")
    limits=resolve(spec["resource_profile"]);weights=spec["weights"];model=copy.deepcopy(spec["model"])
    require(type(model) is dict and type(model.get("constants")) is dict,"Package model constants")
    require(type(weights) is dict and len(weights)+len(model["constants"])<=limits.constants,"Weight count")
    total=sum(math.prod(shape(v,resource_profile=limits.name)) for v in model["constants"].values())
    root=path.parent.resolve();records={}
    for name,item in weights.items():
        require(type(name) is str and NAME.fullmatch(name) and name not in model["constants"],"Duplicate/invalid weight")
        require(type(item) is dict and set(item)=={"file","sha256","bytes","shape","dtype"},"Weight declaration")
        filename=item["file"]
        require(type(filename) is str and re.fullmatch(r"[A-Za-z0-9_-]+\.npy",filename),"Flat weight filename")
        dims=item["shape"]
        require(type(dims) is list and 0<=len(dims)<=limits.rank and
                all(type(n) is int and n>0 for n in dims),"Weight shape")
        count=math.prod(dims);total+=count
        require(count<=limits.constant_elements and total<=limits.total_constant_elements,"Weight element budget")
        require(item["dtype"]=="<f8" and type(item["bytes"]) is int and
                0<item["bytes"]<=8*count+65536,"Weight encoding/bytes")
        require(type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",item["sha256"]),"Weight hash")
        raw=regular_bytes(root/filename,item["bytes"])
        require(len(raw)==item["bytes"] and hashlib.sha256(raw).hexdigest()==item["sha256"],"Weight size/hash mismatch")
        # Parse the small header before allocating based on its shape.
        stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)),"NPY version")
        header=(np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)
        actual,fortran,dtype=header(stream,max_header_size=65536)
        require(list(actual)==dims and dtype.str=="<f8" and not fortran and
                len(raw)-stream.tell()==count*8,"NPY header/payload mismatch")
        array=np.frombuffer(raw,dtype=dtype,count=count,offset=stream.tell()).reshape(actual)
        require(np.isfinite(array).all() and (np.abs(array)<=1024).all(),"Weight values")
        model["constants"][name]=array.tolist();records[name]=dict(item)
    require(total<=limits.total_constant_elements and len(canonical(model))<=limits.json_bytes,"Expanded package budget")
    binding=dict(format=FORMAT,resource_profile=identity(limits.name),weights=records,
                 manifest_sha256=digest(spec),model_sha256=digest(model))
    binding["binding_sha256"]=digest(binding)
    return model,limits.name,binding


def check_recorded_binding(binding,model,resource_profile):
    """Validate retained loader records against the frozen expanded model.

    The import loader checks NPY bytes. This inert record check does not reread
    external files or claim that those files are included in the DSL package.
    """
    require(type(binding) is dict and set(binding)=={"format","resource_profile","weights",
            "manifest_sha256","model_sha256","binding_sha256"} and binding["format"]==FORMAT,
            "Recorded weight package fields")
    require(binding["resource_profile"]==identity(resource_profile) and
            binding["model_sha256"]==digest(model) and
            binding["binding_sha256"]==digest({k:v for k,v in binding.items() if k!="binding_sha256"}),
            "Recorded weight package binding")
    require(type(binding["manifest_sha256"]) is str and re.fullmatch("[0-9a-f]{64}",binding["manifest_sha256"]),
            "Recorded weight manifest hash")
    limits=resolve(resource_profile)
    require(type(binding["weights"]) is dict and len(binding["weights"])<=limits.constants,"Recorded weight count")
    for name,item in binding["weights"].items():
        require(type(name) is str and NAME.fullmatch(name) and name in model["constants"] and
                type(item) is dict and set(item)=={"file","sha256","bytes","shape","dtype"} and
                type(item["file"]) is str and re.fullmatch("[A-Za-z0-9_-]+\.npy",item["file"]) and
                type(item["sha256"]) is str and re.fullmatch("[0-9a-f]{64}",item["sha256"]),
                "Recorded weight file fields")
        dims=list(shape(model["constants"][name],resource_profile=resource_profile));count=math.prod(dims)
        require(type(item["shape"]) is list and all(type(n) is int for n in item["shape"]) and
                item["shape"]==dims and item["dtype"]=="<f8" and type(item["bytes"]) is int and
                0<item["bytes"]<=8*count+65536,"Recorded weight shape/dtype")
    return copy.deepcopy(binding)
