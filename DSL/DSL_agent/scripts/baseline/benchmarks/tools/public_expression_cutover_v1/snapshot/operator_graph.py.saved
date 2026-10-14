"""Versioned Agent operator-graph normalization and inert provenance.

V1 delegates to the unchanged decomposition. V2 reuses the static matrix plan
already used by trusted PyTorch, then the same bounded core decomposition.
No DSL rewriting, compiler/runtime mutation, dynamic callbacks or provider calls.
"""
import copy
import hashlib
from pathlib import Path

from benchmark_graph import CAPACITY_FORMAT, canonical, digest, require, validate
from model_decomposition import Builder, REGISTRY, REGISTRY_VERSION
from model_decomposition import FORMAT as LEGACY_FORMAT, decompose as legacy_decompose
from model_resources import identity as resource_identity, resolve as resource_limits
from pytorch_matrix_plan import POLICY as MATRIX_POLICY, plan, constant_value

FORMAT = "poseidon-operator-graph-v2"
FORMATS = (LEGACY_FORMAT, FORMAT)
BINDING = "poseidon-decomposition-binding-v2"
POLICY = "operator-graph-static-matrix-v1"
SOURCE_FILES = ("operator_graph.py", "pytorch_matrix_plan.py", "model_decomposition.py",
                "model_resources.py", "benchmark_graph.py")

def implementation_sources():
    root = Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}

class MatrixBuilder(Builder):
    def __init__(self, model, *, resource_profile):
        require(type(model) is dict and model.get("format")==FORMAT,
                "Operator graph V2 version")
        limits=resource_limits(resource_profile)
        require(set(model)=={"format","id","inputs","constants","nodes","outputs"}
                and len(canonical(model))<=limits.json_bytes,"Operator graph V2 fields/budget")
        original=copy.deepcopy(model)
        legacy=copy.deepcopy(model)
        legacy["format"]=LEGACY_FORMAT
        super().__init__(legacy,resource_profile=resource_profile)
        self.original=original
        self.matrix_entries=[]

    def call(self, op, refs, attrs=None):
        if op not in ("dot","matmul"):
            return super().call(op,refs,attrs)
        attrs={} if attrs is None else attrs
        require(type(attrs) is dict and not attrs,"Matrix operator attributes")
        require(type(refs) is list and len(refs)==2 and
                all(type(r) is str and r in self.shapes for r in refs),"Matrix operands")
        plain=[self.kinds[r]=="plain" for r in refs]
        require(not all(plain),"Matrix requires an encrypted operand")
        side="left" if plain[0] else "right" if plain[1] else None
        planned=plan(list(self.shapes[refs[0]]),list(self.shapes[refs[1]]),
                     operation=op,public_side=side,resource_profile=self.resource_profile)
        mapped={"left":refs[0],"right":refs[1]}
        constant=None
        if planned["constant"] is not None:
            public=refs[0 if side=="left" else 1]
            require(public in self.graph["constants"],"Matrix immutable public operand")
            constant=self.constant(constant_value(self.graph["constants"][public],planned))
            mapped["constant"]=constant
        start=len(self.graph["nodes"])
        for index,step in enumerate(planned["steps"]):
            # A normalized rank-2/3 matmul deliberately invokes V1, avoiding
            # recursive normalization while preserving the existing core rules.
            result=Builder.call(self,step["op"],[mapped[r] for r in step["inputs"]],step["attrs"])
            require(len(result)==1,"Matrix plan single-output step")
            mapped["step_"+str(index)]=result[0]
        result=mapped["step_"+str(len(planned["steps"])-1)]
        require(list(self.shapes[result])==planned["core_shape"],"Matrix output shape binding")
        self.matrix_entries.append(dict(origin=self.origin,operands=list(refs),plan=planned,
            derived_constant=constant,core_node_ids=[n["id"] for n in self.graph["nodes"][start:]]))
        return [result]

def decompose(model, *, resource_profile="legacy-v1"):
    if type(model) is dict and model.get("format")==LEGACY_FORMAT:
        return legacy_decompose(model,resource_profile=resource_profile)
    builder=MatrixBuilder(model,resource_profile=resource_profile)
    ids=set()
    for node in model["nodes"]:
        require(node["id"] not in ids,"Duplicate node id")
        ids.add(node["id"])
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
        original_sha256=digest(model),lowered_sha256=digest(builder.graph),
        source_map=builder.origins,
        required_capabilities=sorted({c for n in model["nodes"] if n["op"] in REGISTRY
                                      for c in REGISTRY[n["op"]].capabilities}),
        approximation_profiles={n["id"]:n["attrs"] for n in model["nodes"]
                                if any(k in n["attrs"] for k in ("approximation","exp","reciprocal"))},
        normalization=dict(version=POLICY,matrix_policy=MATRIX_POLICY,entries=builder.matrix_entries,
                           scalar_carrier_shape=[1]),
        no_dsl_answer=True)
    if resource_profile!="legacy-v1":
        binding["resource_profile"]=resource_identity(resource_profile)
        binding["runtime_layout_validated"]=False
    binding["binding_sha256"]=digest(binding)
    return builder.graph,binding

def preparation_options(original, lowered, options=None):
    require(options is None or type(options) is dict,"Component option fields")
    result=copy.deepcopy(options or {})
    if (type(original) is dict and original.get("format")==FORMAT
            and lowered.get("format")==CAPACITY_FORMAT
            and result.get("capacity_packing_version") is None):
        from unified_capacity_constants import INDEX_VERSION
        result["capacity_packing_version"]=INDEX_VERSION
    return result

def evaluate_reference(model, inputs, ideal=False):
    """Reuse independent mathematical operators, with explicit scalar carriers.

    Matmul vector-vector results use a one-element logical tensor. This adapter
    applies that representation at each node, including downstream consumers;
    no matrix plan or decomposition computes expected numerical answers.
    """
    from operator_reference import evaluate
    if type(model) is dict and model.get("format")==LEGACY_FORMAT:
        return evaluate(model,inputs,ideal=ideal)
    require(type(model) is dict and model.get("format")==FORMAT,"Operator reference version")
    import numpy as np
    require(set(inputs)=={s["name"] for s in model["inputs"]},"Reference input names")
    values={name:np.asarray(value,dtype=np.float64) for name,value in model["constants"].items()}
    for spec in model["inputs"]:
        value=np.asarray(inputs[spec["name"]])
        require(value.dtype==np.float64 and list(value.shape)==spec["shape"]
                and np.isfinite(value).all(),"Reference input type/shape")
        values[spec["name"]]=value
    for node in model["nodes"]:
        refs=list(dict.fromkeys(node["inputs"]))
        local=dict(format=LEGACY_FORMAT,id=model["id"],
            inputs=[dict(name=r,shape=list(values[r].shape)) for r in refs],constants={},
            nodes=[node],outputs=[dict(name=r,value=r) for r in node["outputs"]])
        expected=evaluate(local,{r:values[r] for r in refs},ideal=ideal)
        for name,value in expected.items():
            values[name]=value.reshape(1) if node["op"]=="matmul" and value.ndim==0 else value
    return {o["name"]:values[o["value"]] for o in model["outputs"]}
