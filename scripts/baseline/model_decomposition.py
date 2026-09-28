"""Versioned operator graphs expanded by trusted, bounded decomposition rules.

No user Python execution, I/O, DSL generation, provider calls or layout decisions.
Registry entries are trusted project code; model JSON cannot register callbacks.
"""
import copy
import hashlib
from pathlib import Path
import math
from dataclasses import dataclass
from types import MappingProxyType
from benchmark_graph import FORMAT as CORE_FORMAT, OPS, BOUNDS, NAME, infer, shape, dimensions, validate, canonical, digest, require

FORMAT = "poseidon-operator-graph-v1"
REGISTRY_VERSION = "operator-decomposition-v1"

@dataclass(frozen=True)
class Operator:
    attributes: frozenset
    reference: str
    rule: str
    capabilities: tuple = ("ckks_add_multiply_rotate",)

REGISTRY = MappingProxyType({
    "dot": Operator(frozenset(), "dot", "_rule_dot_matmul"),
    "matmul": Operator(frozenset(), "matmul", "_rule_dot_matmul"),
    "exp": Operator(frozenset({"approximation"}), "exp_taylor", "_rule_approximation"),
    "reciprocal": Operator(frozenset({"approximation"}), "reciprocal_goldschmidt", "_rule_approximation"),
    "rsqrt": Operator(frozenset({"approximation"}), "rsqrt_newton", "_rule_approximation"),
    "rms_norm": Operator(frozenset({"eps", "approximation"}), "rms_norm", "_rule_rms_norm"),
    "softmax": Operator(frozenset({"axis", "exp", "reciprocal"}), "softmax", "_rule_softmax"),
    "rope": Operator(frozenset({"position", "theta"}), "rope_split_half", "_rule_rope"),
})

def profile(op, value):
    require(type(value) is dict, "Explicit approximation required")
    method, parameter = {"exp":("taylor","degree"),"reciprocal":("goldschmidt","iterations"),
                         "rsqrt":("newton","iterations")}[op]
    require(set(value)=={"method","domain",parameter} and value["method"]==method,
            "Unsupported approximation profile")
    domain=value["domain"]
    require(type(domain) is list and len(domain)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in domain)
            and -1024<=domain[0]<domain[1]<=1024, "Approximation domain")
    require(op=="exp" or domain[0]>0, "Positive approximation domain required")
    n=value[parameter]
    require(type(n) is int and 1<=n<=(20 if op=="exp" else 8), "Approximation work budget")
    return value

class Builder:
    def __init__(self, model):
        require(type(model) is dict and set(model)=={"format","id","inputs","constants","nodes","outputs"}
                and model["format"]==FORMAT,"Operator graph fields/version")
        require(len(canonical(model))<=BOUNDS["json_bytes"],"Operator JSON budget")
        require(type(model["nodes"]) is list and 1<=len(model["nodes"])<=64,"Operator node budget")
        self.original=copy.deepcopy(model)
        self.graph=dict(format=CORE_FORMAT,id=model["id"],inputs=copy.deepcopy(model["inputs"]),
                        constants=copy.deepcopy(model["constants"]),nodes=[],outputs=[])
        require(type(model["inputs"]) is list and 1<=len(model["inputs"])<=4,"Input count")
        require(type(model["constants"]) is dict and len(model["constants"])<=32,"Constant count")
        self.shapes={};self.kinds={};self.refs={};self.origins={};self.stack=[];self.counter=0
        self.reserved=set()
        for i in model["inputs"]:
            require(type(i) is dict and set(i)=={"name","shape"},"Input fields")
            self.declare(i["name"],dimensions(i["shape"]),"cipher")
        require(sum(math.prod(s) for s in self.shapes.values())<=256,"Total input budget")
        for n,v in model["constants"].items():self.declare(n,shape(v),"plain")
        self.public={canonical(v):n for n,v in self.graph["constants"].items()}
        for node in model["nodes"]:
            require(type(node) is dict and set(node)=={"id","op","inputs","attrs","outputs"},"Operator node fields")
            require(type(node["id"]) is str and NAME.fullmatch(node["id"]),"Node id")
            for n in node["outputs"]:
                require(type(n) is str and NAME.fullmatch(n) and n not in self.reserved and n not in self.refs,"Output name")
                self.reserved.add(n)

    def declare(self,n,s,k):
        require(type(n) is str and NAME.fullmatch(n) and n not in self.refs,"Duplicate/invalid value")
        self.refs[n]=n;self.shapes[n]=tuple(s);self.kinds[n]=k
    def fresh(self):
        while True:
            n="generated_"+str(self.counter);self.counter+=1
            if n not in self.refs and n not in self.shapes and n not in self.reserved:return n
    def constant(self,v):
        shape(v);key=canonical(v)
        if key in self.public:return self.public[key]
        require(len(self.graph["constants"])<32,"Decomposition constant budget")
        n=self.fresh();self.graph["constants"][n]=v;self.public[key]=n
        self.shapes[n]=shape(v);self.kinds[n]="plain";return n
    def call(self,op,refs,attrs=None):
        attrs={} if attrs is None else attrs
        require(type(refs) is list and all(r in self.shapes for r in refs) and refs,"Undefined operand")
        require(self.kinds[refs[0]]=="cipher","First operand must be encrypted")
        if op in OPS:
            require(type(attrs) is dict and set(attrs)==OPS[op],"Core operator attributes")
            ss=infer(op,[self.shapes[r] for r in refs],attrs)
            require(len(self.graph["nodes"])<64,"Expanded graph node budget")
            names=[self.fresh() for _ in ss];nid=self.fresh()
            for n,s in zip(names,ss):self.shapes[n]=dimensions(list(s));self.kinds[n]="cipher"
            self.graph["nodes"].append(dict(id=nid,op=op,inputs=refs,attrs=copy.deepcopy(attrs),outputs=names))
            self.origins[nid]=self.origin
            return names
        require(op in REGISTRY,"Unregistered operator: "+str(op))
        require(type(attrs) is dict and set(attrs)==REGISTRY[op].attributes,"High-level operator attributes")
        require(op not in self.stack and len(self.stack)<8,"Recursive decomposition")
        self.stack.append(op)
        try:return self.expand(op,refs,attrs)
        finally:self.stack.pop()
    def one(self,op,refs,attrs=None):return self.call(op,refs,attrs)[0]
    def scale(self,x,c):return self.one("multiply",[x,self.constant(c)])
    def add(self,x,c):return self.one("add",[x,self.constant(c)])
    def expand(self,op,refs,a):
        # Dispatch is registered per operator, never by graph/model identity.
        return getattr(self,REGISTRY[op].rule)(op,refs,a)

    def _rule_dot_matmul(self,op,refs,a):
        x=refs[0];s=self.shapes[x]
        require(len(refs)==2 and self.kinds[refs[1]]=="cipher","Encrypted dot/matmul operands")
        t=self.shapes[refs[1]]
        if op=="dot":
            require(len(s)==len(t)==1 and s==t,"Dot shape")
            z=self.one("multiply",refs)
            return [self.one("sum",[z],dict(axes=[0],keepdims=False))]
        require(len(s)==len(t)==2 and s[1]==t[0],"Rank-2 matmul shape")
        l=self.one("reshape",[x],dict(shape=[s[0],s[1],1]))
        r=self.one("reshape",[refs[1]],dict(shape=[1,t[0],t[1]]))
        z=self.one("multiply",[l,r])
        return [self.one("sum",[z],dict(axes=[1],keepdims=False))]

    def _rule_rope(self,op,refs,a):
        x=refs[0];s=self.shapes[x]
        require(len(refs)==1 and len(s)==1 and s[0]%2==0,"RoPE even vector")
        require(type(a["position"]) is int and 0<=a["position"]<=65536 and type(a["theta"]) in (int,float)
                and math.isfinite(a["theta"]) and 1<a["theta"],"RoPE public parameters")
        half=s[0]//2
        l,r=self.call("split",[x],dict(axis=0,sections=[half,half]))
        rot=self.one("concat",[self.one("negate",[r]),l],dict(axis=0))
        angles=[a["position"]/a["theta"]**(2*i/s[0]) for i in range(half)]*2
        lc=self.one("multiply",[x,self.constant([math.cos(t) for t in angles])])
        rs=self.one("multiply",[rot,self.constant([math.sin(t) for t in angles])])
        return [self.one("add",[lc,rs])]

    def _rule_rms_norm(self,op,refs,a):
        x=refs[0];s=self.shapes[x]
        require(len(refs)==2 and self.kinds[refs[1]]=="plain" and self.shapes[refs[1]]==(s[-1],),"RMSNorm gain")
        require(type(a["eps"]) in (int,float) and math.isfinite(a["eps"]) and 0<a["eps"]<=1,"RMSNorm epsilon")
        square=self.one("square",[x])
        mean=self.one("mean",[square],dict(axes=[-1],keepdims=True))
        variance=self.add(mean,a["eps"])
        inv=self.one("rsqrt",[variance],dict(approximation=a["approximation"]))
        return [self.one("multiply",[self.one("multiply",[x,inv]),refs[1]])]

    def _rule_softmax(self,op,refs,a):
        x=refs[0];s=self.shapes[x]
        require(len(refs)==1 and type(a["axis"]) is int and -len(s)<=a["axis"]<len(s),"Softmax axis")
        exp=self.one("exp",[x],dict(approximation=a["exp"]))
        total=self.one("sum",[exp],dict(axes=[a["axis"]],keepdims=True))
        inv=self.one("reciprocal",[total],dict(approximation=a["reciprocal"]))
        return [self.one("multiply",[exp,inv])]

    def _rule_approximation(self,op,refs,a):
        x=refs[0];s=self.shapes[x]
        require(len(refs)==1,"Approximation arity")
        p=profile(op,a["approximation"])
        if op=="exp":
            coeff=[1/math.factorial(i) for i in range(p["degree"]+1)]
            return [self.one("polynomial",[x,self.constant(coeff)],dict(basis="power"))]
        upper=p["domain"][1]
        if op=="reciprocal":
            residual=self.add(self.scale(x,-1/upper),1.)
            result=self.scale(self.add(residual,1.),1/upper)
            for _ in range(p["iterations"]-1):
                residual=self.one("square",[residual])
                result=self.one("multiply",[result,self.add(residual,1.)])
            return [result]
        result=self.add(self.scale(x,0.),1/math.sqrt(upper))
        for _ in range(p["iterations"]):
            yy=self.one("square",[result])
            term=self.one("multiply",[x,yy])
            result=self.one("multiply",[result,self.add(self.scale(term,-.5),1.5)])
        return [result]

def decompose(model):
    b=Builder(model);ids=set()
    for node in model["nodes"]:
        require(node["id"] not in ids,"Duplicate node id");ids.add(node["id"]);b.origin=node["id"]
        require(type(node["inputs"]) is list and all(type(r) is str and r in b.refs for r in node["inputs"]),"Forward/undefined operator input")
        values=b.call(node["op"],[b.refs[r] for r in node["inputs"]],node["attrs"])
        require(type(node["outputs"]) is list and len(node["outputs"])==len(values),"Output arity")
        for name,value in zip(node["outputs"],values):b.refs[name]=value
    require(type(model["outputs"]) is list,"Output declarations")
    for o in model["outputs"]:
        require(type(o) is dict and set(o)=={"name","value"} and o["value"] in b.refs,"Named output")
        b.graph["outputs"].append(dict(name=o["name"],value=b.refs[o["value"]]))
    validate(b.graph)
    binding=dict(format="poseidon-decomposition-binding-v1",registry_version=REGISTRY_VERSION,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_sha256=digest(model),lowered_sha256=digest(b.graph),source_map=b.origins,
        required_capabilities=sorted({c for n in model["nodes"] if n["op"] in REGISTRY for c in REGISTRY[n["op"]].capabilities}),
        approximation_profiles={n["id"]:n["attrs"] for n in model["nodes"] if n["op"] in ("exp","reciprocal","rsqrt","rms_norm","softmax")},
        no_dsl_answer=True)
    binding["binding_sha256"]=digest(binding)
    return b.graph,binding
