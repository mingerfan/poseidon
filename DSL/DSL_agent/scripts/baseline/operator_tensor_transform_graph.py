"""Operator graph V5: bounded static tensor transforms over existing primitives."""
import copy, hashlib
from pathlib import Path
from types import MappingProxyType
from benchmark_graph import CAPACITY_FORMAT, canonical, digest, require, validate
from model_resources import resolve, identity
from model_decomposition import Operator, REGISTRY, REGISTRY_VERSION
from operator_spatial_graph import SpatialBuilder, FORMAT as PRIOR_FORMAT, SPATIAL_REGISTRY, implementation_sources as prior_sources
from static_tensor_transform_plan import plan, OPERATORS, ATTRIBUTES, POLICY

FORMAT = "poseidon-operator-graph-v5"
REGISTRY_EXTENSION = MappingProxyType({name:Operator(ATTRIBUTES[name], name, "static_tensor_transform")
                                      for name in OPERATORS})

def implementation_sources():
    result = prior_sources(); base = Path(__file__).parent
    for name in ("operator_tensor_transform_graph.py","static_tensor_transform_plan.py","operator_tensor_transform_reference.py"):
        result[name] = hashlib.sha256((base/name).read_bytes()).hexdigest()
    return result

class TensorTransformBuilder(SpatialBuilder):
    def __init__(self, model, *, resource_profile):
        require(type(model) is dict and model.get("format")==FORMAT, "Operator graph V5 version")
        require(set(model)=={"format","id","inputs","constants","nodes","outputs"} and
                len(canonical(model))<=resolve(resource_profile).json_bytes,"Operator graph V5 fields/budget")
        previous=copy.deepcopy(model); previous["format"]=PRIOR_FORMAT
        super().__init__(previous,resource_profile=resource_profile)
        self.original=copy.deepcopy(model); self.tensor_transform_entries=[]

    def call(self, op, refs, attrs=None):
        if op not in OPERATORS: return super().call(op,refs,attrs)
        require(type(refs) is list and len(refs)==1 and refs[0] in self.shapes and
                self.kinds[refs[0]]=="cipher", "static_transform_cipher_operand")
        selected=plan(op,self.shapes[refs[0]],attrs,resource_profile=self.resource_profile)
        require(len(self.graph["nodes"])+selected["expanded_nodes"]<=self.limits.expanded_nodes,
                "static_transform_expanded_capacity")
        mapping={"input":refs[0]}; start=len(self.graph["nodes"])
        for step in selected["steps"]:
            mapping[step["name"]]=super().call(step["op"],[mapping[r] for r in step["inputs"]],step["attrs"])[0]
        result=mapping[selected["result"]]
        self.tensor_transform_entries.append(dict(origin=self.origin,plan=selected,
            core_node_ids=[n["id"] for n in self.graph["nodes"][start:]]))
        return [result]

def decompose(model, *, resource_profile="legacy-v1"):
    b=TensorTransformBuilder(model,resource_profile=resource_profile); ids=set()
    for node in model["nodes"]:
        require(node["id"] not in ids,"Duplicate node id"); ids.add(node["id"]); b.origin=node["id"]
        require(type(node["inputs"]) is list and all(type(r) is str and r in b.refs for r in node["inputs"]),
                "Forward/undefined operator input")
        try: values=b.call(node["op"],[b.refs[r] for r in node["inputs"]],node["attrs"])
        except ValueError as error: raise ValueError(f"node {node['id']} ({node['op']}): {error}") from error
        require(type(node["outputs"]) is list and len(node["outputs"])==len(values),"Output arity")
        for name,value in zip(node["outputs"],values): b.refs[name]=value
    require(type(model["outputs"]) is list,"Output declarations")
    for out in model["outputs"]:
        require(type(out) is dict and set(out)=={"name","value"} and out["value"] in b.refs,"Named output")
        b.graph["outputs"].append(dict(name=out["name"],value=b.refs[out["value"]]))
    validate(b.graph,resource_profile=resource_profile)
    if resource_profile!="legacy-v1":
        b.graph.update(format=CAPACITY_FORMAT,resource_profile=resource_profile); validate(b.graph)
    sources=implementation_sources(); entries={**REGISTRY,**SPATIAL_REGISTRY,**REGISTRY_EXTENSION}
    record=dict(format="poseidon-decomposition-binding-v5",registry_version=REGISTRY_VERSION+"+spatial-v3+tensor-v1",
        original_sha256=digest(model),lowered_sha256=digest(b.graph),source_map=b.origins,
        implementation_sources=sources,implementation_sha256=digest(sources),
        required_capabilities=sorted({c for n in model["nodes"] if n["op"] in entries for c in entries[n["op"]].capabilities}),
        approximation_profiles={n["id"]:n["attrs"] for n in model["nodes"] if any(k in n["attrs"] for k in ("approximation","exp","reciprocal"))},
        normalization=dict(version=POLICY,matrix_entries=b.matrix_entries,arithmetic_entries=b.arithmetic_entries,
            spatial_entries=b.spatial_evidence,tensor_transform_entries=b.tensor_transform_entries,
            scalar_carrier_shape=[1],public_constants_sha256=digest(model["constants"])), no_dsl_answer=True)
    if resource_profile!="legacy-v1": record.update(resource_profile=identity(resource_profile),runtime_layout_validated=False)
    record["binding_sha256"]=digest(record)
    return b.graph,record

def evaluate_reference(model, inputs, ideal=False):
    from operator_tensor_transform_reference import evaluate
    return evaluate(model,inputs,ideal=ideal)
