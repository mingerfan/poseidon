"""Bounded non-pickle state_dict files for explicitly trusted PyTorch capture."""
import copy
import hashlib
import io
import math
import re
from pathlib import Path
from benchmark_graph import canonical, digest, require
from model_resources import resolve
from model_weight_package import regular_bytes

FORMAT = "poseidon-pytorch-state-weights-v1"
DTYPES = {"<f4":4, "<f8":8, "<i4":4, "<i8":8, "|b1":1}

def declarations(value, *, resource_profile="legacy-v1"):
    limits=resolve(resource_profile)
    require(type(value) is dict and len(value)<=limits.constants,"State weight inventory")
    total=0; files={}
    for name,item in value.items():
        require(type(name) is str and 0<len(name)<=256 and
                all(ord(c)>=32 and ord(c)!=127 for c in name),"State weight name")
        require(type(item) is dict and set(item)=={"file","sha256","bytes","shape","dtype"},"State weight declaration")
        require(type(item["file"]) is str and re.fullmatch(r"[A-Za-z0-9_-]+\.npy",item["file"]),"Flat NPY weight filename")
        require(type(item["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",item["sha256"]),"State weight hash")
        dims=item["shape"]
        require(type(dims) is list and len(dims)<=limits.rank and
                all(type(n) is int and n>0 for n in dims),"State weight shape")
        count=math.prod(dims); total+=count
        require(count<=limits.constant_elements and total<=limits.total_constant_elements,"State weight element budget")
        require(type(item["dtype"]) is str and item["dtype"] in DTYPES,"State weight dtype")
        require(type(item["bytes"]) is int and 0<item["bytes"]<=count*DTYPES[item["dtype"]]+65536,
                "State weight byte budget")
        prior=files.get(item["file"])
        require(prior is None or prior==item,"Conflicting shared weight filename")
        files[item["file"]]=item
    return copy.deepcopy(value)

def decode(raw,item):
    import numpy as np
    require(type(raw) is bytes and len(raw)==item["bytes"] and
            hashlib.sha256(raw).hexdigest()==item["sha256"],"State weight size/hash mismatch")
    stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
    require(version in ((1,0),(2,0)),"State NPY version")
    header=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
    dims,fortran,dtype=header(stream,max_header_size=65536)
    require(list(dims)==item["shape"] and not fortran and dtype.str==item["dtype"] and
            len(raw)-stream.tell()==math.prod(dims)*dtype.itemsize,"State NPY header/payload")
    value=np.frombuffer(raw,dtype=dtype,offset=stream.tell()).reshape(dims)
    if dtype.kind=="f":
        require(np.isfinite(value).all() and (np.abs(value)<=1024).all(),"State floating values")
    elif dtype.kind=="i":
        require(((value>=-(2**53)) & (value<=2**53)).all(),"State integer precision")
    return value

def read(root,records,*,resource_profile="legacy-v1"):
    records=declarations(records,resource_profile=resource_profile)
    arrays={};raw_files={}
    for name,item in records.items():
        raw=regular_bytes(Path(root)/item["file"],item["bytes"])
        arrays[name]=decode(raw,item);raw_files[item["file"]]=raw
    return arrays,raw_files

def binding(arrays,records):
    import numpy as np
    require(set(arrays)==set(records),"State weight binding inventory")
    return dict(format=FORMAT,declarations_sha256=digest(records),
                normalized_float64_sha256={name:hashlib.sha256(np.asarray(value,dtype="<f8").tobytes(order="C")).hexdigest()
                                           for name,value in arrays.items()})

def check_binding(value,records):
    require(type(value) is dict and set(value)=={"format","declarations_sha256","normalized_float64_sha256"} and
            value["format"]==FORMAT and value["declarations_sha256"]==digest(records),"State weight binding")
    hashes=value["normalized_float64_sha256"]
    require(type(hashes) is dict and set(hashes)==set(records) and all(
            type(v) is str and re.fullmatch(r"[0-9a-f]{64}",v) for v in hashes.values()),"State weight value hashes")
    return value

def load_state(module,arrays,records):
    """Verify exact source dtype and contents before any explicit float normalization."""
    import torch
    import numpy as np
    require(isinstance(module,torch.nn.Module),"State loading requires nn.Module")
    state=module.state_dict()
    require(set(state)==set(arrays)==set(records),"Complete state_dict names required")
    tensors={}
    for name,value in arrays.items():
        current=state[name]
        require(isinstance(current,torch.Tensor) and current.device.type=="cpu" and
                current.layout==torch.strided,"CPU tensor state required")
        require(list(current.shape)==records[name]["shape"],"State/module shape mismatch")
        require(current.detach().numpy().dtype.str==records[name]["dtype"],"State/module dtype mismatch")
        tensors[name]=torch.from_numpy(value.copy())
    module.load_state_dict(tensors,strict=True)
    loaded=module.state_dict()
    require(set(loaded)==set(tensors),"State names changed after loading")
    for name,value in arrays.items():
        actual=loaded[name].detach().cpu().numpy()
        require(actual.dtype.str==records[name]["dtype"] and list(actual.shape)==records[name]["shape"] and
                actual.tobytes(order="C")==value.tobytes(order="C"),"State load changed tensor values")
    return binding(arrays,records)

def export_state(module, directory, *, resource_profile="legacy-v1"):
    """Write a complete CPU state_dict as deterministic NPY files; return records.

    The caller supplies an already constructed, explicitly trusted nn.Module.
    No module import, checkpoint unpickling or implicit dtype conversion occurs.
    Existing files are never overwritten.
    """
    import numpy as np
    import torch
    require(isinstance(module, torch.nn.Module), "State export requires nn.Module")
    state = module.state_dict()
    limits = resolve(resource_profile)
    require(len(state)<=limits.constants, "State weight inventory")
    total = 0
    for name, tensor in state.items():
        require(type(name) is str and isinstance(tensor, torch.Tensor), "Tensor state name")
        total += tensor.numel()
        require(tensor.ndim<=limits.rank and all(n>0 for n in tensor.shape)
                and tensor.numel()<=limits.constant_elements
                and total<=limits.total_constant_elements, "State weight element budget")
    records, payloads = {}, {}
    for index, name in enumerate(sorted(state)):
        tensor = state[name]
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
                and tensor.layout == torch.strided, "CPU tensor state required")
        require(tensor.dtype in (torch.float32, torch.float64, torch.int32,
                                 torch.int64, torch.bool), "State export dtype")
        value = tensor.detach().numpy()
        # ascontiguousarray changes scalar shape; preserve rank zero explicitly.
        value = np.array(value, copy=True, order="C")
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        raw = stream.getvalue()
        filename = f"state_{index:04d}.npy"
        item = dict(file=filename, sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw), shape=list(value.shape), dtype=value.dtype.str)
        records[name] = item
        payloads[filename] = raw
    declarations(records, resource_profile=resource_profile)
    for item in records.values():
        decode(payloads[item["file"]], item)
    root = Path(directory)
    require(not root.is_symlink(), "State export directory symlink")
    if root.exists():
        require(root.is_dir() and not any(root.iterdir()), "Fresh state export directory")
    else:
        root.mkdir(parents=True)
    for filename, raw in payloads.items():
        with (root / filename).open("xb") as stream:
            stream.write(raw)
    return records
