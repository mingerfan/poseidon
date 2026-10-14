"""Registered value operations composed with ordered tensor alias semantics.

Agent-owned staging module. Public submission/capture still uses its frozen
version. This extends an earlier Agent draft; no backend implementation is copied.
"""
import hashlib, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from benchmark_graph import NAME, digest, canonical, dimensions, shape, broadcast, axis
from model_resources import resolve, identity
from model_decomposition import polynomial_profile, REGISTRY
from operator_graph import FORMAT as GRAPH_FORMAT, decompose, MatrixBuilder as Builder
from benchmark_graph import OPS
import benchmark_graph as graph_contract

# These create independent tensor values. View-like calls and inference Dropout
# must use the alias paths so later mutations retain their original meaning.
VALUE_OPERATORS = frozenset((set(OPS) - {
    "flatten", "reshape", "transpose", "permute", "slice", "split"
}) | (set(REGISTRY) - {"broadcast", "dropout"}))

FORMAT = "poseidon-alias-capture-normalization-v4"
SOURCE_FORMAT = "poseidon-ordered-tensor-source-v4"
MESSAGES = {
    "alias_source_rejected": "The ordered source must use the registered static data schema.",
    "alias_call_rejected": "The call must match the registered fixed Torch signature.",
    "alias_shape_rejected": "Shapes, axes and static indices must match the tensor operation.",
    "alias_view_rejected": "A view must preserve a compatible non-overlapping stride mapping.",
    "alias_public_mutation_rejected": "Public weights and constants are immutable.",
    "alias_overlap_rejected": "This mutation has repeated destinations or partially overlapping operands.",
    "alias_approximation_rejected": "An explicit registered polynomial and input domain are required.",
    "alias_resource_rejected": "Source, storage maps and expanded graphs must fit the resource profile.",
    "alias_graph_rejected": "The normalized graph must satisfy the existing decomposition contract.",
    "alias_integrity_rejected": "The plan must match its trusted source and implementation identity.",
}
class AliasError(ValueError):
    def __init__(self, code, position):
        if code not in MESSAGES or type(position) is not int or position < -1:
            raise ValueError("Invalid trusted alias rejection")
        self.code, self.position = code, position
        super().__init__(MESSAGES[code])
    def feedback(self):
        return dict(format="poseidon-alias-composition-rejection-v3", code=self.code,
            layer="integrity" if self.code == "alias_integrity_rejected" else "input",
            location=dict(source_position=self.position), context=dict(constraint=MESSAGES[self.code]),
            retry_policy="never_automatic")
def check(value, code, position=-1):
    if not value: raise AliasError(code, position)
def plain_data(value, limits):
    count = 0
    def visit(v, depth):
        nonlocal count
        count += 1
        check(count <= 2*limits.total_constant_elements + limits.expanded_nodes*64 + 256
              and depth <= limits.rank+8, "alias_resource_rejected")
        if type(v) is dict:
            check(all(type(k) is str for k in v), "alias_source_rejected")
            return {k: visit(x, depth+1) for k,x in v.items()}
        if type(v) in (list,tuple): return [visit(x, depth+1) for x in v]
        if v is None or type(v) in (bool,int,float,str): return v
        raise AliasError("alias_source_rejected", -1)
    result = visit(value,0)
    try: encoded = canonical(result)
    except (ValueError,TypeError,OverflowError): raise AliasError("alias_source_rejected",-1)
    check(len(encoded) <= limits.json_bytes, "alias_resource_rejected")
    return result
def implementation():
    base = Path(graph_contract.__file__).resolve().parent
    files = (Path(__file__), base/"benchmark_graph.py", base/"model_resources.py",
             base/"model_decomposition.py", base/"operator_reference.py", base/"benchmark_math.py",
             base/"operator_graph.py", base/"pytorch_matrix_plan.py")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def reshape_dims(old, wanted, limits, position):
    check(type(wanted) is list and 1 <= len(wanted) <= limits.rank and
          all(type(x) is int and (x == -1 or 1 <= x <= limits.axis_size) for x in wanted)
          and wanted.count(-1) <= 1, "alias_shape_rejected", position)
    size = math.prod(old); known = math.prod(x for x in wanted if x != -1)
    check(size % known == 0, "alias_shape_rejected", position)
    result = tuple(size//known if x == -1 else x for x in wanted)
    check(math.prod(result) == size, "alias_shape_rejected", position)
    return result
def compatible_view(indices, wanted):
    candidate = indices.reshape(wanted)
    first = int(candidate.flat[0])
    predicted = np.full(wanted, first, dtype=np.int64)
    for dim,size in enumerate(wanted):
        if size == 1: continue
        coordinate = [0]*len(wanted); coordinate[dim] = 1
        stride = int(candidate[tuple(coordinate)]) - first
        view_shape = [1]*len(wanted); view_shape[dim] = size
        predicted += np.arange(size,dtype=np.int64).reshape(view_shape)*stride
    return np.array_equal(candidate,predicted)
def decode_index(spec, rank, position):
    check(type(spec) is list, "alias_shape_rejected", position)
    ellipses = sum(x.get("kind") == "ellipsis" for x in spec if type(x) is dict)
    check(ellipses <= 1, "alias_shape_rejected", position)
    consumed = sum(x.get("kind") not in ("newaxis","ellipsis") for x in spec if type(x) is dict)
    check(consumed <= rank, "alias_shape_rejected", position)
    result = []
    for item in spec:
        check(type(item) is dict and "kind" in item, "alias_shape_rejected", position)
        kind = item["kind"]
        if kind == "slice":
            check(set(item) == {"kind","start","stop","step"} and
                  all(x is None or type(x) is int for x in (item["start"],item["stop"],item["step"])) and
                  (item["step"] is None or item["step"] > 0), "alias_shape_rejected",position)
            result.append(slice(item["start"],item["stop"],item["step"]))
        elif kind == "int":
            check(set(item) == {"kind","value"} and type(item["value"]) is int, "alias_shape_rejected",position)
            result.append(item["value"])
        elif kind == "newaxis":
            check(set(item) == {"kind"}, "alias_shape_rejected",position); result.append(None)
        elif kind == "ellipsis":
            check(set(item) == {"kind"}, "alias_shape_rejected",position)
            result.extend([slice(None)]*(rank-consumed))
        else: raise AliasError("alias_shape_rejected",position)
    if not ellipses: result.extend([slice(None)]*(rank-consumed))
    return tuple(result)

@dataclass
class Alias:
    root: str
    indices: object
@dataclass
class Storage:
    value: str
    size: int
    role: str
    public: object = None
    version: int = 0

class Normalizer:
    def __init__(self, source, approximations, limits):
        self.source,self.approximations,self.limits = source,approximations,limits
        self.graph = dict(format=GRAPH_FORMAT,id="static_alias_source",inputs=source["inputs"],
                          constants={},nodes=[],outputs=[])
        self.storages,self.values,self.shapes,self.cache = {},{},{},{}
        self.used = set(); self.count = 0; self.position = -1; self.events = []
        self.live_map_elements = 0; self.cache_bytes = 0
        self.node_origins = {}; self.phase = "input_binding"
        self.public_constant_sources = {}; self.source_constant_roots = {}
    def name(self,prefix):
        self.count += 1; result = prefix+"_"+str(self.count)
        while result in self.used: self.count += 1; result = prefix+"_"+str(self.count)
        self.used.add(result); return result
    def tensor_shape(self,s):
        try: dimensions(list(s),resource_profile=self.limits.name)
        except (ValueError,TypeError,OverflowError): raise AliasError("alias_shape_rejected",self.position)
        return tuple(s)
    def emit(self,op,refs,s,attrs=None):
        s = self.tensor_shape(s)
        check(len(self.graph["nodes"]) < self.limits.original_nodes, "alias_resource_rejected",self.position)
        value = self.name("tensor")
        identifier = self.name("node")
        self.graph["nodes"].append(dict(id=identifier,op=op,inputs=refs,attrs=attrs or {},outputs=[value]))
        self.node_origins[identifier] = dict(phase=self.phase,source_position=self.position,
            source_id=self.source["nodes"][self.position]["id"] if self.position >= 0 else None)
        self.shapes[value] = s; return value
    def constant(self,value):
        try: s = shape(value,resource_profile=self.limits.name)
        except (ValueError,TypeError,OverflowError): raise AliasError("alias_source_rejected",self.position)
        check(len(self.graph["constants"]) < self.limits.constants, "alias_resource_rejected",self.position)
        name = self.name("public"); self.graph["constants"][name] = value; self.shapes[name] = s
        return name
    def alias(self,root,indices):
        indices = np.asarray(indices,dtype=np.int64)
        self.live_map_elements += indices.size
        check(self.live_map_elements <= self.limits.json_bytes//16,
              "alias_resource_rejected",self.position)
        if indices.shape:
            if self.storages[root].role=="plain":
                check(len(indices.shape)<=self.limits.rank and indices.size<=self.limits.constant_elements
                      and all(0<n<=self.limits.axis_size for n in indices.shape),
                      "alias_resource_rejected",self.position)
            else:self.tensor_shape(indices.shape)
        return Alias(root,indices)
    def allocate(self,value,s,role="cipher",public=None,indices=None):
        size = math.prod(s)
        root = self.name("storage")
        if role == "cipher":
            flat = value if self.shapes[value] == (size,) else self.emit("reshape",[value],(size,),dict(shape=[size]))
        else: flat = value
        self.storages[root] = Storage(flat,size,role,public)
        return self.alias(root,np.arange(size).reshape(s) if indices is None else indices)
    def read(self,value):
        check(type(value) is Alias, "alias_source_rejected",self.position)
        storage = self.storages[value.root]; s = tuple(value.indices.shape)
        key = (value.root,storage.version,s,value.indices.tobytes())
        if key in self.cache: return self.cache[key]
        self.cache_bytes += len(key[3])
        check(self.cache_bytes <= self.limits.json_bytes, "alias_resource_rejected", self.position)
        indices = value.indices.ravel()
        if storage.role == "plain":
            full = (s == storage.public.shape and
                    np.array_equal(indices,np.arange(storage.size)))
            if full:
                result = storage.value
            else:
                arr = storage.public.ravel()[indices].reshape(s)
                result = self.constant(arr.tolist())
                source = self.source_constant_roots.get(value.root)
                if source is not None:
                    self.public_constant_sources[result] = dict(source=source,
                        full_identity=False,source_shape=list(storage.public.shape),
                        result_shape=list(s),indices_sha256=digest(value.indices.tolist()))
        else:
            result = storage.value
            if indices.size != storage.size or not np.array_equal(indices,np.arange(storage.size)):
                result = self.emit("public_gather",[result],(len(indices),),dict(axis=0,indices=indices.tolist()))
            if self.shapes[result] != s:
                result = self.emit("reshape",[result],s,dict(shape=list(s)))
        self.cache[key] = result
        return result
    def refs(self,names):
        check(type(names) is list and all(type(n) is str and n in self.values for n in names),
              "alias_source_rejected",self.position)
        return [self.values[n] for n in names]
    def new_tensor(self,op,args,attrs=None):
        check(any(self.storages[a.root].role == "cipher" for a in args), "alias_source_rejected",self.position)
        refs = [self.read(a) for a in args]; ss = [tuple(a.indices.shape) for a in args]
        if op in ("add","subtract","multiply"):
            try: out = broadcast(*ss)
            except ValueError: raise AliasError("alias_shape_rejected",self.position)
            if self.storages[args[0].root].role == "plain":
                if op in ("add","multiply"): refs.reverse()
                else:
                    neg = self.emit("negate",[refs[1]],ss[1]); refs = [neg,refs[0]]; op = "add"
            # Public expansion is explicit, as required by the existing graph contract.
            cipher = args[0] if self.storages[args[0].root].role == "cipher" else args[1]
            if self.shapes[refs[0]] != out and any(self.storages[a.root].role == "plain" for a in args):
                refs[0] = self.emit("broadcast",[refs[0]],out,dict(shape=list(out)))
        elif op in ("square","relu"): out = ss[0]
        else: raise AliasError("alias_source_rejected",self.position)
        v = self.emit(op,refs,out,attrs)
        return self.allocate(v,out)
    def registered_value(self,args,attributes):
        check(set(attributes) == {"operator","parameters"} and
              type(attributes["operator"]) is str and
              attributes["operator"] in VALUE_OPERATORS and
              type(attributes["parameters"]) is dict and 1 <= len(args) <= 8,
              "alias_call_rejected", self.position)
        check(any(self.storages[value.root].role == "cipher" for value in args) if
              attributes["operator"] in ("matmul","dot") else
              self.storages[args[0].root].role == "cipher",
              "alias_call_rejected", self.position)
        refs = [self.read(value) for value in args]
        op = attributes["operator"]
        params = attributes["parameters"]
        if op in ("linear","batch_norm","polynomial","conv1d","conv2d"):
            check(all(self.storages[value.root].role == "plain" for value in args[1:]),
                  "alias_call_rejected",self.position)
        if op in ("stack","concat"):
            check(all(self.storages[value.root].role == "cipher" for value in args),
                  "alias_call_rejected",self.position)
        # Reuse the existing Agent registry for arity, type, attribute and shape
        # checks. Internal operands are declared as intermediates, not fake
        # request inputs with different admission budgets. Full graph expansion
        # and all aggregate limits are checked once at the end.
        probe_output = self.name("shape_probe")
        probe = dict(format=GRAPH_FORMAT,id="alias_shape_probe",
            inputs=self.graph["inputs"],constants=self.graph["constants"],
            nodes=[dict(id="probe",op=op,inputs=refs,attrs=params,outputs=[probe_output])],
            outputs=[])
        try:
            checker = Builder(probe,resource_profile=self.limits.name)
            checker.origin = "probe"
            for name,value in zip(refs,args):
                if name not in checker.shapes:
                    checker.declare(name,tuple(value.indices.shape),
                                    self.storages[value.root].role)
            names = checker.call(op,refs,params)
            check(len(names) == 1,"alias_call_rejected",self.position)
            result_shape = checker.shapes[names[0]]
        except AliasError:
            raise
        except (ValueError,TypeError,KeyError,IndexError,OverflowError):
            raise AliasError("alias_call_rejected",self.position)
        value = self.emit(op,refs,result_shape,params)
        return self.allocate(value,result_shape)

    def mutation(self,target,result,other=None):
        storage = self.storages[target.root]
        check(storage.role == "cipher", "alias_public_mutation_rejected",self.position)
        flat = target.indices.ravel()
        check(len(np.unique(flat)) == len(flat), "alias_overlap_rejected",self.position)
        if other is not None and other.root == target.root:
            try: other_indices = np.broadcast_to(other.indices,target.indices.shape)
            except ValueError: raise AliasError("alias_shape_rejected",self.position)
            if np.intersect1d(flat,other_indices.ravel()).size:
                check(np.array_equal(target.indices,other_indices), "alias_overlap_rejected",self.position)
        check(result.indices.shape == target.indices.shape, "alias_shape_rejected",self.position)
        update = self.read(result)
        if self.shapes[update] != (len(flat),):
            update = self.emit("reshape",[update],(len(flat),),dict(shape=[len(flat)]))
        if len(flat) == storage.size:
            inverse = np.empty(storage.size,dtype=np.int64); inverse[flat] = np.arange(len(flat))
            restored = update if np.array_equal(inverse,np.arange(storage.size)) else self.emit(
                "public_gather",[update],(storage.size,),dict(axis=0,indices=inverse.tolist()))
        else:
            check(storage.size+len(flat) <= self.limits.tensor_elements, "alias_resource_rejected",self.position)
            joined = self.emit("concat",[storage.value,update],(storage.size+len(flat),),dict(axis=0))
            select = np.arange(storage.size); select[flat] = storage.size+np.arange(len(flat))
            restored = self.emit("public_gather",[joined],(storage.size,),dict(axis=0,indices=select.tolist()))
        old_version = storage.version; storage.value = restored; storage.version += 1
        self.events.append(dict(source_position=self.position,rule="mutation_ssa",storage=target.root,
            previous_version=old_version,new_version=storage.version,written_indices=flat.tolist()))
        return target
    def run(self):
        s = self.source
        check(type(s) is dict and set(s) == {"format","inputs","constants","nodes","outputs"} and
              s["format"] == SOURCE_FORMAT, "alias_source_rejected")
        check(type(s["inputs"]) is list and 1 <= len(s["inputs"]) <= self.limits.inputs and
              type(s["constants"]) is dict and type(s["nodes"]) is list and type(s["outputs"]) is list and
              len(s["nodes"]) <= self.limits.original_nodes, "alias_resource_rejected")
        check(all(type(i) is dict and set(i)=={"name","shape"} and type(i["name"]) is str for i in s["inputs"]), "alias_source_rejected")
        check(len({i["name"] for i in s["inputs"]})==len(s["inputs"]), "alias_source_rejected")
        self.used.update(i["name"] for i in s["inputs"])
        for inp in s["inputs"]:
            check(type(inp) is dict and set(inp) == {"name","shape"} and type(inp["name"]) is str and
                  NAME.fullmatch(inp["name"]) and inp["name"] not in self.values,"alias_source_rejected")
            n = inp["name"];ss = self.tensor_shape(inp["shape"]);self.used.add(n);self.shapes[n] = ss
            self.values[n] = self.allocate(n,ss)
        check(sum(math.prod(x["shape"]) for x in s["inputs"]) <= self.limits.input_elements,"alias_resource_rejected")
        for n,v in s["constants"].items():
            check(type(n) is str and NAME.fullmatch(n) and n not in self.values,"alias_source_rejected")
            ref = self.constant(v);arr = np.asarray(v,dtype=np.float64)
            self.values[n] = self.allocate(ref,arr.shape,"plain",arr)
            self.source_constant_roots[self.values[n].root] = n
            self.public_constant_sources[ref] = dict(source=n,full_identity=True,
                source_shape=list(arr.shape),result_shape=list(arr.shape))
        check(type(self.approximations) is dict and set(self.approximations) <= {"relu"},"alias_approximation_rejected")
        source_ids = set(); self.phase = "source_operation"
        for pos,node in enumerate(s["nodes"]):
            self.position = pos
            check(type(node) is dict and set(node) == {"id","op","inputs","attrs","output"} and
                  type(node["id"]) is str and NAME.fullmatch(node["id"]) and type(node["output"]) is str and
                  NAME.fullmatch(node["output"]) and node["output"] not in self.values and
                  type(node["op"]) is str and type(node["attrs"]) is dict,"alias_source_rejected",pos)
            check(node["id"] not in source_ids,"alias_source_rejected",pos)
            source_ids.add(node["id"])
            op,a = node["op"],node["attrs"];args = self.refs(node["inputs"])
            check(bool(args),"alias_source_rejected",pos);x = args[0];sx = tuple(x.indices.shape)
            before_root = x.root
            if op == "registered":
                result = self.registered_value(args,a);rule = "registered_value:"+a["operator"]
            elif op in ("alias","clone","contiguous"):
                check(len(args) == 1 and not a,"alias_source_rejected",pos)
                if op == "alias":result = x;rule = "alias"
                elif self.storages[x.root].role=="plain":
                    value=self.read(x)
                    result=self.allocate(value,sx,"plain",self.storages[x.root].public.ravel()[x.indices].copy())
                    rule=op+"_public_copy"
                else:
                    contiguous = np.array_equal(x.indices.ravel(),np.arange(x.indices.size)+int(x.indices.flat[0]))
                    if op == "contiguous" and contiguous:result = x;rule = "contiguous_alias"
                    else:
                        value = self.read(x);flat = x.indices.ravel()
                        dense = len(np.unique(flat)) == len(flat) and int(flat.max()-flat.min()+1) == len(flat)
                        if op == "clone" and dense:
                            relative = x.indices-flat.min();inverse = np.empty(len(flat),dtype=np.int64);inverse[relative.ravel()] = np.arange(len(flat))
                            v = value if self.shapes[value] == (len(flat),) else self.emit("reshape",[value],(len(flat),),dict(shape=[len(flat)]))
                            if not np.array_equal(inverse,np.arange(len(flat))):
                                v = self.emit("public_gather",[v],(len(flat),),dict(axis=0,indices=inverse.tolist()))
                            result = self.allocate(v,(len(flat),),indices=relative);rule = "clone_preserve_dense_layout"
                        else:result = self.allocate(value,sx);rule = op+"_copy"
            elif op in ("reshape","view","flatten"):
                check(len(args)==1,"alias_source_rejected",pos)
                if op == "flatten":
                    check(set(a)=={"start_dim","end_dim"},"alias_source_rejected",pos)
                    try:first,last = axis(a["start_dim"],len(sx)),axis(a["end_dim"],len(sx))
                    except ValueError:raise AliasError("alias_shape_rejected",pos)
                    check(first<=last,"alias_shape_rejected",pos)
                    wanted = sx[:first]+(math.prod(sx[first:last+1]),)+sx[last+1:]
                else:
                    check(set(a)=={"shape"},"alias_source_rejected",pos)
                    wanted = reshape_dims(sx,a["shape"],self.limits,pos)
                compatible = compatible_view(x.indices,wanted)
                check(op != "view" or compatible,"alias_view_rejected",pos)
                if compatible:result=self.alias(x.root,x.indices.reshape(wanted));rule=op+"_alias"
                else:
                    value=self.read(x);value=self.emit("reshape",[value],wanted,dict(shape=list(wanted)))
                    result=self.allocate(value,wanted);rule=op+"_copy"
            elif op in ("transpose","permute","slice","expand"):
                check(len(args)==1,"alias_source_rejected",pos)
                try:
                    if op == "transpose":
                        check(set(a)=={"dim0","dim1"},"alias_source_rejected",pos)
                        indices=x.indices.swapaxes(axis(a["dim0"],len(sx)),axis(a["dim1"],len(sx)))
                    elif op == "permute":
                        check(set(a)=={"dims"} and type(a["dims"]) is list,"alias_source_rejected",pos)
                        dims=[axis(i,len(sx)) for i in a["dims"]]
                        check(sorted(dims)==list(range(len(sx))),"alias_shape_rejected",pos)
                        indices=x.indices.transpose(dims)
                    elif op == "slice":
                        check(set(a)=={"index"},"alias_source_rejected",pos)
                        indices=x.indices[decode_index(a["index"],len(sx),pos)]
                    else:
                        check(set(a)=={"shape"},"alias_source_rejected",pos)
                        wanted=tuple(a["shape"]);self.tensor_shape(wanted)
                        check(broadcast(sx,wanted)==wanted,"alias_shape_rejected",pos)
                        indices=np.broadcast_to(x.indices,wanted)
                except (IndexError,ValueError,TypeError,OverflowError):
                    raise AliasError("alias_shape_rejected",pos)
                check(indices.size>0 and indices.ndim>0,"alias_shape_rejected",pos)
                result=self.alias(x.root,indices);rule=op+"_alias"
            elif op in ("axis_slice","view_gather"):
                check(len(args)==1 and set(a)==({"axis","start","stop","step"} if
                      op=="axis_slice" else {"axis","indices"}),
                      "alias_source_rejected",pos)
                try:
                    dim=axis(a["axis"],len(sx))
                    if op=="axis_slice":
                        check(all(v is None or type(v) is int for v in
                              (a["start"],a["stop"],a["step"])) and
                              (a["step"] is None or a["step"]>0),"alias_shape_rejected",pos)
                        spec=[slice(None)]*len(sx)
                        spec[dim]=slice(a["start"],a["stop"],a["step"])
                        indices=x.indices[tuple(spec)]
                    else:
                        check(type(a["indices"]) is list and a["indices"] and
                              all(type(i) is int and 0<=i<sx[dim] for i in a["indices"]),
                              "alias_shape_rejected",pos)
                        indices=np.take(x.indices,a["indices"],axis=dim)
                    check(indices.size>0,"alias_shape_rejected",pos)
                    result=self.alias(x.root,indices);rule=op+"_alias"
                except (ValueError,TypeError,IndexError,OverflowError):
                    raise AliasError("alias_shape_rejected",pos)
            elif op in ("assign","copy"):
                check(not a and len(args) in ((2,3) if op=="assign" else (2,)),
                      "alias_source_rejected",pos)
                target,update=args[:2]
                check(self.storages[target.root].role=="cipher",
                      "alias_public_mutation_rejected",pos)
                other=args[2] if len(args)==3 else update
                if op=="copy":
                    try:indices=np.broadcast_to(update.indices,target.indices.shape)
                    except ValueError:raise AliasError("alias_shape_rejected",pos)
                    expanded=self.alias(update.root,indices)
                    if self.storages[update.root].role=="plain":
                        zero=self.allocate(self.constant(0.),(),"plain",np.asarray(0.))
                        clear=self.new_tensor("multiply",[target,zero])
                        expanded=self.new_tensor("add",[clear,expanded])
                    update=expanded
                result=self.mutation(target,update,other);rule=op+"_mutation"
            elif op in ("add","subtract","multiply"):
                check(len(args)==2 and set(a)=={"inplace","alpha"} and type(a["inplace"]) is bool and
                      type(a["alpha"]) in (int,float) and math.isfinite(a["alpha"]) and abs(a["alpha"])<=1024 and
                      (op!="multiply" or a["alpha"]==1),"alias_source_rejected",pos)
                other=args[1]
                if a["inplace"]:
                    check(self.storages[x.root].role=="cipher","alias_public_mutation_rejected",pos)
                    check(broadcast(sx,tuple(other.indices.shape))==sx,"alias_shape_rejected",pos)
                weighted=other
                if a["alpha"] != 1:
                    alpha_ref=self.constant(a["alpha"]);alpha=self.allocate(alpha_ref,(),"plain",np.asarray(a["alpha"]))
                    if self.storages[other.root].role=="plain":
                        weighted_ref=self.constant(self.storages[other.root].public.ravel()[other.indices.ravel()].reshape(other.indices.shape).tolist())
                        weighted_arr=np.asarray(self.graph["constants"][weighted_ref])*a["alpha"]
                        check(np.isfinite(weighted_arr).all() and np.max(np.abs(weighted_arr))<=1024,"alias_source_rejected",pos)
                        weighted=self.allocate(self.constant(weighted_arr.tolist()),weighted_arr.shape,"plain",weighted_arr)
                    else:weighted=self.new_tensor("multiply",[other,alpha])
                result=self.new_tensor(op,[x,weighted])
                if a["inplace"]:result=self.mutation(x,result,other)
                rule="arithmetic_mutation" if a["inplace"] else "arithmetic_value"
            elif op in ("square","relu"):
                check(len(args)==1 and set(a)==({"inplace"} if op=="relu" else set()),"alias_source_rejected",pos)
                attrs=None
                if op=="relu":
                    check(type(a["inplace"]) is bool and "relu" in self.approximations,"alias_approximation_rejected",pos)
                    try:polynomial_profile(self.approximations["relu"])
                    except (ValueError,TypeError,OverflowError):raise AliasError("alias_approximation_rejected",pos)
                    attrs=dict(approximation=self.approximations["relu"])
                    if a["inplace"]:check(self.storages[x.root].role=="cipher","alias_public_mutation_rejected",pos)
                result=self.new_tensor(op,[x],attrs)
                if op=="relu" and a["inplace"]:result=self.mutation(x,result)
                rule="relu_mutation" if op=="relu" and a["inplace"] else op+"_value"
            else:raise AliasError("alias_source_rejected",pos)
            self.values[node["output"]]=result
            self.events.append(dict(source_position=pos,source_id=node["id"],rule=rule,
                input_storage=before_root,output_storage=result.root,shape=list(result.indices.shape)))
        self.position=-1; self.phase="output_binding"
        check(1<=len(s["outputs"])<=self.limits.outputs,"alias_resource_rejected")
        labels=set()
        for output in s["outputs"]:
            check(type(output) is dict and set(output)=={"name","value"} and type(output["name"]) is str and
                  NAME.fullmatch(output["name"]) and output["name"] not in labels and
                  type(output["value"]) is str and output["value"] in self.values,"alias_source_rejected")
            labels.add(output["name"]);value=self.values[output["value"]]
            check(self.storages[value.root].role=="cipher","alias_source_rejected")
            self.graph["outputs"].append(dict(name=output["name"],value=self.read(value)))
        try:lowered,record=decompose(self.graph,resource_profile=self.limits.name)
        except (ValueError,TypeError,KeyError,IndexError,OverflowError):raise AliasError("alias_graph_rejected",-1)
        return self.graph,lowered,record,self.events

def plan(source,approximations,*,resource_profile="general-tensors-v1"):
    try:limits=resolve(resource_profile)
    except (TypeError,ValueError):raise AliasError("alias_resource_rejected",-1)
    request=plain_data(dict(source=source,approximations=approximations,resource_profile=resource_profile),limits)
    check(type(request["source"]) is dict and set(request["source"]) == {"format","inputs","constants","nodes","outputs"}, "alias_source_rejected")
    worker=Normalizer(request["source"],request["approximations"],limits)
    graph,lowered,decomposition,events=worker.run()
    result=dict(format=FORMAT,source_request=request,source_request_sha256=digest(request),
        mathematical_graph=graph,lowered_graph=lowered,decomposition=decomposition,events=events,
        mathematical_node_origins=worker.node_origins,
        public_constant_sources=worker.public_constant_sources,
        lowered_node_origins={name:worker.node_origins[origin] for name,origin in decomposition["source_map"].items()},
        implementation=implementation(),resource_profile=identity(resource_profile),
        input_storage_assumption="separate contiguous CPU float64 logical inputs",
        production_intake_integrated=False,original_function_accuracy_certified=False,
        registered_value_operators=sorted(VALUE_OPERATORS),
        physical_layout_status="unverified",compiler_status="unverified",encrypted_status="unverified",agent_status="unverified")
    result["binding"]=digest(result);return result
def verify(value,*,expected_source_request_sha256):
    try:
        if type(value) is not dict or value.get("format")!=FORMAT or value["implementation"]!=implementation():raise ValueError()
        if value["binding"]!=digest({k:v for k,v in value.items() if k!="binding"}):raise ValueError()
        if value["source_request_sha256"]!=expected_source_request_sha256 or digest(value["source_request"])!=expected_source_request_sha256:raise ValueError()
        if canonical(plan(**value["source_request"]))!=canonical(value):raise ValueError()
    except (ValueError,TypeError,KeyError,IndexError,OverflowError):raise AliasError("alias_integrity_rejected",-1)
    return value
