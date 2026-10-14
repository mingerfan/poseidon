"""Conservative scalar extraction/repacking baseline. Never used to compute reference."""
import math
import numpy as np
from benchmark_graph import validate, require
from unified_graph_contract import validate_request

class Emitter:
    def __init__(self,request,*,singleton=False):
        self.request=request;self.period=request["layout"]["input_slot_period"];self.lines=[]
        self.constants={float(v):k for k,v in request["public_constants"].items() if type(v) in (int,float)}
        self.zero=Scalar(self,"zero_ct",True)
        self.singleton=singleton;self.masked_constants={}
    def literal(self,x):
        x=float(x);require(x in self.constants,"Unprepared derived scalar constant")
        return self.constants[x]
    def emit(self,text,zero=False):
        name="v"+str(len(self.lines));self.lines.append("    "+name+" = "+text)
        require(len(self.lines)<=1024,"Deterministic baseline operation budget")
        return Scalar(self,name,zero)
    def constant_cipher(self,value):
        if value not in self.masked_constants:
            base=self.emit("zero_ct + "+self.literal(value))
            self.masked_constants[value]=self.emit(base.name+" * mask0")
        return self.masked_constants[value]
    def rotate(self,x,k):
        k%=self.period
        if getattr(self,"capacity_inline_rotations",False):
            text=x.name
            for i in range(self.period.bit_length()-1):
                if k&(1<<i):text+=".rotate("+str(1<<i)+")"
            return x if text==x.name else self.emit(text)
        for i in range(self.period.bit_length()-1):
            if k&(1<<i):x=self.emit(x.name+".rotate("+str(1<<i)+")")
        return x
    def input(self,name,n):
        out=[]
        for i in range(n):
            x=self.rotate(Scalar(self,name),i)
            x=self.emit(x.name+" * mask0")
            if not self.singleton:
                for j in range(self.period.bit_length()-1):
                    x=x+self.rotate(x,1<<j)
            out.append(x)
        return np.array(out,dtype=object)
    def pack(self,values):
        result=self.zero
        for i,v in enumerate(values.flat):
            if v.zero:term=self.zero
            elif self.singleton:term=self.rotate(v,-i)
            elif "mask"+str(i) in self.request["public_constants"]:
                term=self.emit(v.name+" * mask"+str(i))
            else:
                # Values are replicated scalars. Keep the same immutable mask0
                # and rotate the masked ciphertext; never rotate a Plain Expr.
                term=self.rotate(self.emit(v.name+" * mask0"),-i)
            result=result+term
        return result

class Scalar:
    __array_priority__=10000
    def __init__(self,emitter,name,zero=False):self.e=emitter;self.name=name;self.zero=zero
    def __add__(self,other):
        if isinstance(other,Scalar):
            if other.zero:return self
            if self.zero:return other
            return self.e.emit(self.name+" + "+other.name)
        if float(other)==0:return self
        if self.e.singleton:return self+self.e.constant_cipher(float(other))
        return self.e.emit(self.name+" + "+self.e.literal(other))
    __radd__=__add__
    def __neg__(self):return self if self.zero else self.e.emit("-"+self.name)
    def __sub__(self,other):
        if isinstance(other,Scalar):
            if self.name==other.name:return self.e.zero
            if other.zero:return self
            return self.e.emit(self.name+" - "+other.name)
        if self.e.singleton:return self-self.e.constant_cipher(float(other)) if float(other)!=0 else self
        return self.e.emit(self.name+" - "+self.e.literal(other)) if float(other)!=0 else self
    def __rsub__(self,other):return -self+other
    def __mul__(self,other):
        if isinstance(other,Scalar):
            if self.zero or other.zero:return self.e.zero
            return self.e.emit(self.name+" * "+other.name)
        if self.zero or float(other)==0:return self.e.zero
        if float(other)==1:return self
        return self.e.emit(self.name+" * "+self.e.literal(other))
    __rmul__=__mul__
    def __truediv__(self,other):return self*(1.0/float(other))
    def __pow__(self,k):
        require(type(k) is int and 0<=k<=128,"Rule exponent")
        if k==0:return self.e.zero+1.0
        if k==1:return self
        a=self**(k//2);v=a*a
        return v*self if k%2 else v


def emit_spatial(op,xs,a,em):
    x=xs[0];rank=1 if op.endswith("1d") else 2
    batched=x.ndim==rank+2
    if not batched:x=x[None]
    conv=op.startswith("conv");w=xs[1] if conv else None
    k=w.shape[2:] if conv else a["kernel"];d=a["dilation"] if conv else [1]*rank
    co=w.shape[0] if conv else x.shape[1];groups=a["groups"] if conv else 1
    outsp=tuple((n+2*p-dil*(kk-1)-1)//st+1 for n,p,dil,kk,st in zip(x.shape[2:],a["padding"],d,k,a["stride"]))
    out=np.empty((x.shape[0],co,*outsp),dtype=object)
    for dst in np.ndindex(out.shape):
        batch,ch,*coord=dst;acc=em.zero;valids=0
        channels=x.shape[1]//groups
        for ci in (range((ch//(co//groups))*channels,(ch//(co//groups)+1)*channels) if conv else [ch]):
            for kk in np.ndindex(tuple(k)):
                pos=tuple(i*st-pad+j*dil for i,st,pad,j,dil in zip(coord,a["stride"],a["padding"],kk,d))
                if all(0<=j<n for j,n in zip(pos,x.shape[2:])):
                    value=x[(batch,ci,*pos)]
                    acc=acc+(value*float(w[(ch,ci%channels,*kk)]) if conv else value);valids+=1
        if conv and len(xs)==3:acc=acc+float(xs[2][ch])
        if not conv:acc=acc/(math.prod(k) if a["count_include_pad"] else valids)
        out[dst]=acc
    return out if batched else out[0]



class PackedPrefix:
    """Trusted lazy C-order prefix. A cell remains a whole ciphertext Expr."""
    def __init__(self,emitter,expr,shape):
        self.emitter=emitter;self.expr=expr;self.shape=tuple(shape);self._array=None
        require(1<=math.prod(self.shape)<=emitter.period,"Packed prefix capacity")
    def unpack(self):
        if self._array is None:
            self._array=self.emitter.input(self.expr.name,math.prod(self.shape)).reshape(self.shape)
        return self._array

def power_polynomial(x, coefficients, zero):
    """Grouped power-basis evaluation using the immutable public coefficients."""
    require(0 < len(coefficients) <= 129, "Rule polynomial coefficient budget")
    parts = []
    for i in range(0, len(coefficients), 2):
        even = float(coefficients[i])
        odd = float(coefficients[i+1]) if i+1 < len(coefficients) else 0.
        parts.append(x*odd+even if odd else zero+even)
    if len(parts) == 1:
        return parts[0]
    base = x*x
    while len(parts) > 1:
        pairs = []
        for i in range(0, len(parts), 2):
            pairs.append(parts[i]+parts[i+1]*base if i+1 < len(parts) else parts[i])
        parts = pairs
        if len(parts) > 1:
            base = base*base
    return parts[0]


def packed_elementwise(op,xs,attrs,outshape,*,balanced_chebyshev=False,balanced_power=False,public_lane_factors=False):
    """Only shape-proven lane-local operations; all other operators unpack."""
    x=xs[0]
    if not isinstance(x,PackedPrefix):return None
    a=x.expr;em=x.emitter
    if op in ("flatten","reshape"):
        return PackedPrefix(em,a,outshape)
    if tuple(outshape)!=x.shape:return None
    if op in ("add","subtract","multiply"):
        other=xs[1]
        if isinstance(other,PackedPrefix):
            if other.shape!=x.shape:return None
            b=other.expr
        elif (isinstance(other,np.ndarray) and other.size
              and isinstance(other.reshape(-1)[0],Scalar)
              and all(isinstance(v,Scalar) and v.e is em
                      and v.name == other.reshape(-1)[0].name for v in other.flat)):
            b=other.reshape(-1)[0]
        elif (isinstance(other,np.ndarray) and other.size
              and np.issubdtype(other.dtype,np.number)
              and np.all(other==other.reshape(-1)[0])):
            b=float(other.reshape(-1)[0])
        elif (public_lane_factors and op=="multiply" and isinstance(other,np.ndarray)
              and np.issubdtype(other.dtype,np.number)):
            try:weights=np.broadcast_to(other,x.shape)
            except ValueError:return None
            result=em.zero
            for i,weight in enumerate(weights.reshape(-1)):
                if float(weight)==0:continue
                name="mask"+str(i)
                if name not in em.request["public_constants"]:return None
                term=a*float(weight)
                result=result+em.emit(term.name+" * "+name)
            return PackedPrefix(em,result,outshape)
        else:return None
        y=a+b if op=="add" else a-b if op=="subtract" else a*b
    elif op=="negate":y=-a
    elif op=="square":y=a*a
    elif op=="power":y=a**attrs["exponent"]
    elif op=="polynomial":
        coeff=xs[1];y=em.zero
        if attrs["basis"]=="power":
            if balanced_power:y=power_polynomial(a,coeff,em.zero)
            else:
                for c in reversed(coeff):y=y*a+float(c)
        elif balanced_chebyshev:
            from chebyshev_lowering import balanced
            y=balanced(a,coeff,em.zero)
        else:
            t0=em.zero+1.;y=t0*float(coeff[0])
            if len(coeff)>1:
                t1=a;y=y+t1*float(coeff[1])
                for c in coeff[2:]:
                    t2=a*t1*2.-t0;y=y+t2*float(c);t0,t1=t1,t2
    else:return None
    return PackedPrefix(em,y,outshape)

def packed_reduction(op, xs, attrs, outshape):
    """Whole-period reductions only; no tail slots or partial axes are consumed."""
    x = xs[0]
    if not isinstance(x, PackedPrefix):
        return None
    axes = tuple(axis % len(x.shape) for axis in attrs["axes"])
    if set(axes) != set(range(len(x.shape))) or math.prod(x.shape) != x.emitter.period:
        return None
    result = x.expr
    for i in range(x.emitter.period.bit_length()-1):
        result = result+x.emitter.rotate(result, 1 << i)
    if op == "mean":
        result = result/x.emitter.period
    return np.full(tuple(outshape), result, dtype=object)


def has_packed_reduction(request):
    from unified_chunk_layout import ABI as CHUNK_ABI
    if request["layout"]["execution_abi"] == CHUNK_ABI:
        return False
    shapes = validate(request["model"])["shapes"]
    period = request["layout"]["input_slot_period"]
    for node in request["model"]["nodes"]:
        if node["op"] in ("sum", "mean"):
            shape = shapes[node["inputs"][0]]
            axes = {i % len(shape) for i in node["attrs"]["axes"]}
            if math.prod(shape) == period and axes == set(range(len(shape))):
                return True
    return False


def lower(request,*,packed_prefix=False,balanced_chebyshev=False,singleton=False,balanced_power=False,packed_reductions=False):
    validate_request(request);g=request["model"];check=validate(g);em=Emitter(request,singleton=singleton)
    require(not (singleton and packed_prefix),"Incompatible lowering representations")
    values={k:np.asarray(v,dtype=np.float64) for k,v in g["constants"].items()}
    from unified_chunk_layout import ABI as CHUNK_ABI
    chunk=request["layout"]["execution_abi"]==CHUNK_ABI
    require(not packed_prefix or not chunk,"Packed-prefix fallback does not change chunk ABI")
    for spec in g["inputs"]:
        bindings=[b for b in request["layout"]["inputs"] if b["name"]==spec["name"]]
        if packed_prefix:
            values[spec["name"]]=PackedPrefix(em,Scalar(em,bindings[0]["dsl_name"]),spec["shape"])
            continue
        parts=[em.input(b["dsl_name"],b.get("elements",math.prod(spec["shape"]))) for b in bindings]
        values[spec["name"]]=np.concatenate(parts).reshape(spec["shape"])
    for node in g["nodes"]:
        op=node["op"];a=node["attrs"];xs=[values[r] for r in node["inputs"]]
        if packed_reductions and op in ("sum", "mean"):
            y=packed_reduction(op,xs,a,check["shapes"][node["outputs"][0]])
            if y is not None:
                values[node["outputs"][0]]=y
                continue
        if packed_prefix and len(node["outputs"])==1:
            y=packed_elementwise(op,xs,a,check["shapes"][node["outputs"][0]],balanced_chebyshev=balanced_chebyshev,balanced_power=balanced_power,public_lane_factors=balanced_power)
            if y is not None:
                values[node["outputs"][0]]=y;continue
        xs=[v.unpack() if isinstance(v,PackedPrefix) else v for v in xs];x=xs[0]
        if op=="add":y=x+xs[1]
        elif op=="subtract":y=x-xs[1]
        elif op=="multiply":y=x*xs[1]
        elif op=="negate":y=-x
        elif op=="square":y=x*x
        elif op=="power":y=x**a["exponent"]
        elif op=="linear":
            w=xs[1];y=np.empty((*x.shape[:-1],w.shape[0]),dtype=object)
            for i in np.ndindex(y.shape):
                v=em.zero
                for j in range(w.shape[1]):v=v+x[(*i[:-1],j)]*float(w[i[-1],j])
                y[i]=v+(float(xs[2][i[-1]]) if len(xs)==3 else 0)
        elif op=="flatten":y=x.reshape(-1)
        elif op=="reshape":y=x.reshape(a["shape"])
        elif op=="transpose":y=np.swapaxes(x,a["dim0"],a["dim1"])
        elif op=="permute":y=x.transpose(a["dims"])
        elif op=="rotate":y=np.roll(x.reshape(-1),-a["step"]).reshape(x.shape)
        elif op=="concat":y=np.concatenate(xs,axis=a["axis"])
        elif op=="stack":y=np.stack(xs,axis=a["axis"])
        elif op=="slice":
            idx=[slice(None)]*x.ndim;idx[a["axis"]]=slice(a["start"],a["stop"],a["step"]);y=x[tuple(idx)]
        elif op=="split":y=np.split(x,np.cumsum(a["sections"])[:-1],axis=a["axis"])
        elif op in ("sum","mean"):
            axes=tuple(v%x.ndim for v in a["axes"])
            y=np.asarray(np.sum(x,axis=axes,keepdims=a["keepdims"]),dtype=object)
            if op=="mean":y=np.asarray(y/math.prod(x.shape[i] for i in axes),dtype=object)
            if y.ndim==0:y=y.reshape(1)
        elif op=="batch_norm":
            mean,var,gamma,beta=xs[1:];shape=[1,x.shape[1]]+[1]*(x.ndim-2)
            gain=gamma/np.sqrt(var+a["eps"]);bias=beta-mean*gain
            y=x*gain.reshape(shape)+bias.reshape(shape)
        elif op=="polynomial":
            coeff=xs[1];y=np.full(x.shape,em.zero,dtype=object)
            if a["basis"]=="power":
                if balanced_power:
                    for i in np.ndindex(x.shape):y[i]=power_polynomial(x[i],coeff,em.zero)
                else:
                    for c in reversed(coeff):y=y*x+float(c)
            elif balanced_chebyshev:
                from chebyshev_lowering import balanced
                for i in np.ndindex(x.shape):y[i]=balanced(x[i],coeff,em.zero)
            else:
                t0=np.full(x.shape,em.zero+1.0,dtype=object);y=t0*float(coeff[0])
                if len(coeff)>1:
                    t1=x;y=y+t1*float(coeff[1])
                    for c in coeff[2:]:
                        t2=x*t1*2.0-t0;y=y+t2*float(c);t0,t1=t1,t2
        elif op.startswith(("conv","avg_pool")):y=emit_spatial(op,xs,a,em)
        else:raise ValueError("No deterministic emitter for "+op)
        for name,v in zip(node["outputs"],y if op=="split" else [y]):
            values[name]=np.asarray(v,dtype=object)
            require(list(values[name].shape)==check["shapes"][name],"Rule shape mismatch")
    results=[]
    for output,binding in zip(g["outputs"],request["layout"]["outputs"]):
        value=values[output["value"]]
        if isinstance(value,PackedPrefix):
            results.append(value.expr.name);continue
        if chunk:
            for part in binding["chunks"]:
                start=part["offset"]
                results.append(em.pack(value.reshape(-1)[start:start+part["elements"]]).name)
        else:results.append(em.pack(value).name)
    names=[x["dsl_name"] for x in request["layout"]["inputs"]]+["zero_ct"]
    source='@hc.func("'+",".join("c" for _ in names)+'")\ndef golden('+", ".join(names)+'):\n'
    source+="\n".join(em.lines)+"\n    return ["+", ".join(results)+"]\n"
    require(len(source.encode())<=65536,"Rule source budget")
    return source


def candidate_source(request):
    """Return a checked manual baseline; request/reference preparation is independent.

    Preserve historical and directed sources. On recognized deterministic
    resource-budget failures, retry a generic lane-local prefix lowering. No budget is increased, no graph ID is inspected, and no
    provider output passes through this function. Poseidon CPU free baselines
    may use a bounded, type-checked DAG schedule that preserves the graph and
    uses shape-proven whole-period reductions, balanced powers and bounded
    factor scheduling. Depth is a diagnostic; compilation and numerical
    qualification remain required. Historical and directed sources retain their
    original path.
    """
    from unified_graph_contract import validate_candidate
    from unified_capacity_layout import ABIS as CAPACITY_ABIS
    if request["layout"]["execution_abi"] in CAPACITY_ABIS:
        require("construction_exercise" not in request,"Capacity directed baseline not yet verified")
        from unified_capacity_lowering import lower as lower_capacity
        from unified_capacity_constants import request_version,CHANNEL_VERSION,TILE_VERSION,INDEX_VERSION
        enabled=(request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu"
                 and request_version(request) in (TILE_VERSION,CHANNEL_VERSION,INDEX_VERSION))
        return lower_capacity(request,semantic_pairs=enabled)
    from unified_public_contract import CONTRACT as PUBLIC
    attempts=[]
    selected_schedule=None
    resource_errors={"Expanded ciphertext operation budget exceeded",
                     "Deterministic baseline operation budget", "AST node limit"}
    def build(packed,singleton=False):
        nonlocal selected_schedule
        selected_schedule=None
        exercise=request.get("construction_exercise")
        balanced_power=request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu" and exercise is None
        reduce_packed=balanced_power and not singleton and has_packed_reduction(request)
        source=lower(request,packed_prefix=packed or reduce_packed,singleton=singleton,
                     balanced_power=balanced_power,packed_reductions=reduce_packed)
        if (request.get("compiler_configuration",{}).get("backend")=="poseidon_cpu"
                and exercise is None):
            # This is the trusted rule baseline only. Provider/replay candidates
            # never enter candidate_source and are validated without rewriting.
            from unified_graph_scheduling import schedule_source
            from unified_graph_quadratic import schedule_quadratic_source
            original=source
            try:
                scheduled,record=schedule_quadratic_source(original,request)
                if not record["applied"]:
                    quadratic_record=record
                    scheduled,record=schedule_source(original,request)
                    record=dict(record,quadratic=quadratic_record)
                if record["applied"]:
                    try:
                        validate_candidate(dict(schema=1,request_id=request["request_id"],
                                                hecate_source=scheduled),request)
                    except ValueError as error:
                        if str(error) not in resource_errors:raise
                        record=dict(record,applied=False,rejected=str(error))
                    else:source=scheduled
                selected_schedule=dict(record,power_basis_evaluation="grouped-power-v1",whole_period_reductions=reduce_packed)
            except ValueError as error:
                if str(error) not in {"Trusted schedule node budget", "Trusted schedule recursion budget",
                                      "Trusted schedule work budget", "Trusted schedule serialization depth",
                                      "Deterministic baseline operation budget", "Rule source budget"}:raise
                selected_schedule=dict(version="poseidon-rule-scheduling-v4",applied=False,
                                       rejected=str(error),request_changed=False,reference_changed=False)
        exercise=request.get("construction_exercise")
        if exercise is not None:
            if request.get("construction_profile")==PUBLIC:
                from unified_public_exercises import golden_variant
                source=golden_variant(source,exercise["id"],request=request)
            else:
                from unified_graph_exercises import golden_variant
                source=golden_variant(source,exercise["id"])
        validate_candidate(dict(schema=1,request_id=request["request_id"],hecate_source=source),request)
        return source
    try:
        source=build(False)
        record=dict(strategy="packed-reduction-v1" if selected_schedule and selected_schedule.get("whole_period_reductions") else "scalar-v1",fallback=False)
        if selected_schedule is not None:record["scheduling"]=selected_schedule
        return source,record
    except ValueError as error:
        if (str(error) not in {"Expanded ciphertext operation budget exceeded",
                              "Deterministic baseline operation budget", "AST node limit"} or
            request["layout"]["execution_abi"]!="unified-periodic-inputs-v1"):raise
        attempts.append(dict(strategy="scalar-v1",rejected=str(error)))
    try:
        source=build(True)
        record=dict(strategy="packed-prefix-v1",fallback=True,prior_attempts=attempts,
                    emitted_cipher_limit=256,request_changed=False,reference_changed=False)
        if selected_schedule is not None:record["scheduling"]=selected_schedule
        return source,record
    except ValueError as error:
        if str(error) not in {"Expanded ciphertext operation budget exceeded",
                              "Deterministic baseline operation budget", "AST node limit"}:raise
        attempts.append(dict(strategy="packed-prefix-v1",rejected=str(error)))
    # Represent each extracted scalar only at lane zero. This avoids replicating
    # every input across the period before reductions. Public additions are
    # masked; output packing rotates disjoint singletons into their bound lanes.
    source=build(False,singleton=True)
    record=dict(strategy="singleton-v1",fallback=True,prior_attempts=attempts,
                request_changed=False,reference_changed=False)
    if selected_schedule is not None:record["scheduling"]=selected_schedule
    return source,record
