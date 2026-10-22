"""Explicit operator graph V3 arithmetic normalization; V1/V2 remain unchanged."""
import copy
import hashlib
from pathlib import Path

from benchmark_graph import CAPACITY_FORMAT, canonical, digest, require, validate
from model_resources import identity as resource_identity, resolve as resource_limits
from model_decomposition import REGISTRY, REGISTRY_VERSION
from operator_graph import (FORMATS as PRIOR_FORMATS, FORMAT as MATRIX_FORMAT,
    MatrixBuilder, decompose as prior_decompose,
    evaluate_reference as prior_reference, preparation_options as prior_options,
    implementation_sources as prior_sources)

FORMAT="poseidon-operator-graph-v3"
SPATIAL_FORMAT="poseidon-operator-graph-v4"
TENSOR_FORMAT="poseidon-operator-graph-v5"
FORMATS=(*PRIOR_FORMATS,FORMAT,SPATIAL_FORMAT,TENSOR_FORMAT)
POLICY="operator-graph-public-arithmetic-v1"
BINDING="poseidon-decomposition-binding-v3"

def implementation_sources(format=FORMAT):
    if format==TENSOR_FORMAT:
        from operator_tensor_transform_graph import implementation_sources as tensor_sources
        return tensor_sources()
    if format==SPATIAL_FORMAT:
        from operator_spatial_graph import implementation_sources as spatial_sources
        return spatial_sources()
    if format in PRIOR_FORMATS:
        return prior_sources()
    require(format==FORMAT,"Arithmetic graph version")
    sources=prior_sources()
    sources[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return sources

class ArithmeticBuilder(MatrixBuilder):
    def __init__(self,model,*,resource_profile):
        require(type(model) is dict and model.get("format")==FORMAT,"Operator graph V3 version")
        require(set(model)=={"format","id","inputs","constants","nodes","outputs"} and
                len(canonical(model))<=resource_limits(resource_profile).json_bytes,
                "Operator graph V3 fields/budget")
        prior=copy.deepcopy(model);prior["format"]=MATRIX_FORMAT
        super().__init__(prior,resource_profile=resource_profile)
        self.original=copy.deepcopy(model)
        self.arithmetic_entries=[]

    def call(self,op,refs,attrs=None):
        if op not in ("add","subtract","multiply"):
            return super().call(op,refs,attrs)
        require(type(refs) is list and len(refs)==2 and
                all(type(r) is str and r in self.shapes for r in refs),"Arithmetic operands")
        require(attrs is None or type(attrs) is dict and not attrs,"Arithmetic attributes")
        plain=[self.kinds[r]=="plain" for r in refs]
        require(not all(plain),"Arithmetic requires an encrypted operand")
        if not plain[0]:
            return super().call(op,refs,attrs)
        start=len(self.graph["nodes"])
        if op=="subtract":
            negative=self.one("negate",[refs[1]])
            normalized=[negative,refs[0]]
            values=super().call("add",normalized,{})
            action="negate_cipher_then_add_public"
        else:
            normalized=[refs[1],refs[0]]
            values=super().call(op,normalized,{})
            action="swap_commutative_operands"
        self.arithmetic_entries.append(dict(origin=self.origin,operator=op,operands=list(refs),
            input_kinds=["plain","cipher"],public_shape=list(self.shapes[refs[0]]),
            cipher_shape=list(self.shapes[refs[1]]),action=action,normalized_operands=normalized,
            core_node_ids=[n["id"] for n in self.graph["nodes"][start:]],
            result_values=list(values)))
        return values

def decompose(model,*,resource_profile="legacy-v1"):
    if type(model)is dict and model.get("format")==TENSOR_FORMAT:
        from operator_tensor_transform_graph import decompose as tensor_decompose
        return tensor_decompose(model,resource_profile=resource_profile)
    if type(model)is dict and model.get("format")==SPATIAL_FORMAT:
        from operator_spatial_graph import decompose as spatial_decompose
        return spatial_decompose(model,resource_profile=resource_profile)
    if type(model) is dict and model.get("format") in PRIOR_FORMATS:
        return prior_decompose(model,resource_profile=resource_profile)
    builder=ArithmeticBuilder(model,resource_profile=resource_profile)
    ids=set()
    for node in model["nodes"]:
        require(node["id"] not in ids,"Duplicate node id");ids.add(node["id"])
        builder.origin=node["id"]
        require(type(node["inputs"]) is list and
                all(type(r) is str and r in builder.refs for r in node["inputs"]),
                "Forward/undefined operator input")
        try:
            values=builder.call(node["op"],[builder.refs[r] for r in node["inputs"]],node["attrs"])
        except ValueError as error:
            raise ValueError(f"node {node['id']} ({node['op']}): {error}") from error
        require(type(node["outputs"]) is list and len(node["outputs"])==len(values),"Output arity")
        for name,value in zip(node["outputs"],values):
            builder.refs[name]=value
    require(type(model["outputs"]) is list,"Output declarations")
    for output in model["outputs"]:
        require(type(output) is dict and set(output)=={"name","value"} and
                output["value"] in builder.refs,"Named output")
        builder.graph["outputs"].append(dict(name=output["name"],value=builder.refs[output["value"]]))
    validate(builder.graph,resource_profile=resource_profile)
    if resource_profile!="legacy-v1":
        builder.graph.update(format=CAPACITY_FORMAT,resource_profile=resource_profile)
        validate(builder.graph)
    sources=implementation_sources()
    binding=dict(format=BINDING,registry_version=REGISTRY_VERSION,
        implementation_sha256=digest(sources),implementation_sources=sources,
        original_sha256=digest(model),lowered_sha256=digest(builder.graph),source_map=builder.origins,
        required_capabilities=sorted({c for n in model["nodes"] if n["op"] in REGISTRY
                                      for c in REGISTRY[n["op"]].capabilities}),
        approximation_profiles={n["id"]:n["attrs"] for n in model["nodes"]
                                if any(k in n["attrs"] for k in ("approximation","exp","reciprocal"))},
        normalization=dict(version=POLICY,matrix_policy="static-tensor-matmul-v1",
            matrix_entries=builder.matrix_entries,arithmetic_entries=builder.arithmetic_entries,
            scalar_carrier_shape=[1],public_constants_sha256=digest(model["constants"])),
        no_dsl_answer=True)
    if resource_profile!="legacy-v1":
        binding["resource_profile"]=resource_identity(resource_profile)
        binding["runtime_layout_validated"]=False
    binding["binding_sha256"]=digest(binding)
    return builder.graph,binding

def preparation_options(original,lowered,options=None):
    if type(original) is dict and original.get("format")in(FORMAT,SPATIAL_FORMAT,TENSOR_FORMAT):
        require(options is None or type(options) is dict,"Component option fields")
        result=copy.deepcopy(options or {})
        if lowered.get("format")==CAPACITY_FORMAT and result.get("capacity_packing_version") is None:
            from unified_capacity_constants import INDEX_VERSION
            result["capacity_packing_version"]=INDEX_VERSION
        return result
    return prior_options(original,lowered,options)

def evaluate_reference(model,inputs,ideal=False):
    if type(model)is dict and model.get("format")==TENSOR_FORMAT:
        from operator_tensor_transform_graph import evaluate_reference as tensor_reference
        return tensor_reference(model,inputs,ideal=ideal)
    if type(model)is dict and model.get("format")==SPATIAL_FORMAT:
        from operator_spatial_graph import evaluate_reference as spatial_reference
        return spatial_reference(model,inputs,ideal=ideal)
    if type(model) is dict and model.get("format")==FORMAT:
        # Reference arithmetic is computed by the existing independent interpreter.
        # This changes only the admitted version tag; it does not reorder operands.
        compatible=copy.deepcopy(model);compatible["format"]=MATRIX_FORMAT
        return prior_reference(compatible,inputs,ideal=ideal)
    return prior_reference(model,inputs,ideal=ideal)
