"""Versioned logical admission budgets. These do not certify a runtime layout."""
from dataclasses import asdict, dataclass
from types import MappingProxyType
import hashlib
import json

@dataclass(frozen=True)
class Resources:
    name: str
    input_elements: int
    output_elements: int
    tensor_elements: int
    original_nodes: int
    expanded_nodes: int
    constants: int
    constant_elements: int
    total_constant_elements: int
    json_bytes: int
    axis_size: int
    inputs: int = 4
    outputs: int = 4
    rank: int = 4

PROFILES = MappingProxyType({
    # The historical bound was per constant, not aggregate. Preserve that contract.
    "legacy-v1": Resources("legacy-v1",256,256,256,64,64,32,4096,32*4096,131072,256),
    "medium-v1": Resources("medium-v1",1024,1024,4096,256,1024,256,65536,65536,4*1024**2,4096),
    "resnet20-v1": Resources("resnet20-v1",4096,4096,65536,1024,4096,1024,1000000,1000000,32*1024**2,65536),
    "general-tensors-v1": Resources("general-tensors-v1",4096,4096,65536,1024,4096,1024,1000000,1000000,32*1024**2,65536,rank=32),
})

def resolve(name="legacy-v1"):
    if type(name) is not str or name not in PROFILES:
        raise ValueError("Unknown model resource profile")
    return PROFILES[name]

def identity(name="legacy-v1"):
    value=asdict(resolve(name))
    value["sha256"]=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return value
