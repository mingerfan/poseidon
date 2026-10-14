"""Trusted prototype: replicated quadratic expressions to packed correlations.

Only public coefficient algebra and declared zero-padded periodic input bindings
are used. Provider candidates never pass through this prototype.
"""
from collections import defaultdict, Counter
import math
import numpy as np
from benchmark_graph import require
from unified_graph_scheduling import Schedule

VERSION="poseidon-packed-quadratic-v1"

class Budget(Exception):
    pass

class Quadratic:
    def __init__(self, schedule, request):
        self.s=schedule
        self.request=request
        self.period=request["layout"]["input_slot_period"]
        self.bindings={b["dsl_name"]:int(np.prod(b["shape"])) for b in request["layout"]["inputs"]}
        self.states={}
        self.work=0
        self.max_work=262144
        self.rewrites=[]
        self.literals=[]
    def spend(self,n):
        self.work+=n
        if self.work>self.max_work:raise Budget("Quadratic coefficient work budget")
    def clean(self,state):
        if len(state)>64:return None
        if any(not np.all(np.isfinite(v)) for v in state.values()):return None
        return {k:v for k,v in state.items() if np.any(v!=0)}
    def state(self,n):
        if n.index in self.states:return self.states[n.index]
        self.spend(self.period)
        if n.op=="name":
            if n.plain:
                if n.value in self.request["public_constants"]:
                    value=self.request["public_constants"][n.value]
                else:
                    try:value=float(n.value)
                    except ValueError:value=None
                if value is None:result=None
                else:
                    value=np.asarray(value,dtype=np.float64).reshape(-1)
                    result={():np.full(self.period,value[0])} if len(value)==1 else {():value.copy()} if len(value)==self.period else None
            elif n.value in self.bindings:
                result={}
                for i in range(self.bindings[n.value]):
                    coefficient=np.zeros(self.period);coefficient[i]=1.
                    result[((n.value,i),)]=coefficient
            elif n.value=="zero_ct" and self.s.trusted_zero:result={}
            else:result=None
        else:
            children=[self.state(c) for c in n.inputs]
            if any(v is None for v in children):result=None
            elif n.op in ("add","subtract"):
                a,b=children;result={k:v.copy() for k,v in a.items()}
                sign=1 if n.op=="add" else -1
                for key,value in b.items():
                    result[key]=result.get(key,np.zeros(self.period))+sign*value
            elif n.op=="negate":result={k:-v for k,v in children[0].items()}
            elif n.op=="rotate":result={k:np.roll(v,-n.value) for k,v in children[0].items()}
            elif n.op=="multiply":
                a,b=children;result={}
                self.spend(len(a)*len(b)*self.period)
                if any(len(x)+len(y)>2 for x in a for y in b):result=None
                else:
                    for x,c in a.items():
                        for y,d in b.items():
                            key=tuple(sorted(x+y))
                            result[key]=result.get(key,np.zeros(self.period))+c*d
            else:result=None
        result=self.clean(result) if result is not None else None
        self.states[n.index]=result
        return result
    def number(self,value):
        require(math.isfinite(value) and abs(value)<=1024,"Derived public literal bound")
        self.literals.append(value)
        return self.s.node("name",value=repr(float(value)),plain=True)
    def rotate(self,n,step):
        step%=self.period
        for i in range(self.period.bit_length()-1):
            if step&(1<<i):n=self.s.node("rotate",(n,),value=1<<i)
        return n
    def scaled(self,n,c):
        if c==1:return n
        if c==-1:return self.s.node("negate",(n,))
        return self.s.node("multiply",(n,self.number(c)))
    def contraction(self,n):
        state=self.state(n)
        if n.plain or state is None or not any(len(k)==2 for k in state):return None
        if any(len(k)==1 or not np.all(v==v[0]) for k,v in state.items()):return None
        groups=defaultdict(dict)
        for key,value in state.items():
            if not key:continue
            a,b=key
            if a[0]==b[0]:
                offset=min((a[1]-b[1])%self.period,(b[1]-a[1])%self.period)
            else:
                offset=(a[1]-b[1])%self.period
            groups[(a[0],b[0],offset)][key]=float(value[0])
        projections=defaultdict(list)
        for (a,b,offset),coefficients in sorted(groups.items()):
            expected=Counter()
            for j in range(self.bindings[b]):
                i=(j+offset)%self.period
                if i<self.bindings[a]:
                    expected[tuple(sorted(((a,i),(b,j))))]+=1
            if set(expected)!=set(coefficients):return None
            weights=[coefficients[k]/count for k,count in expected.items()]
            if not weights or any(w!=weights[0] for w in weights):return None
            value=weights[0]
            if not math.isfinite(value) or abs(value)>1024:return None
            if value==0:return None
            projections[(a,b)].append((offset,value))
        bias=float(state.get((),np.zeros(self.period))[0])
        if not math.isfinite(bias) or abs(bias)>1024:return None
        result=None
        for (a,b),terms in sorted(projections.items()):
            projection=None
            for step,coefficient in terms:
                term=self.scaled(self.rotate(self.s.leaf(a),step),coefficient)
                projection=term if projection is None else self.s.node("add",(projection,term))
            dot=self.s.node("multiply",(projection,self.s.leaf(b)))
            for i in range(self.period.bit_length()-1):
                dot=self.s.node("add",(dot,self.rotate(dot,1<<i)))
            result=dot if result is None else self.s.node("add",(result,dot))
        if bias:result=self.s.node("add",(result,self.number(bias)))
        if result.depth>=n.depth:return None
        self.rewrites.append(dict(original_node=n.index,original_depth=n.depth,
            new_depth=result.depth,quadratic_terms=len(state),projection_groups=len(projections)))
        return result
    def transform(self,outputs):
        reachable={}
        def visit(n,depth=0):
            if depth>256:raise Budget("Quadratic DAG depth budget")
            if n.index in reachable:return
            for child in n.inputs:visit(child,depth+1)
            reachable[n.index]=n
        for n in outputs:visit(n)
        if len(reachable)>1024:raise Budget("Quadratic reachable-node budget")
        replacements={}
        for n in sorted(reachable.values(),key=lambda n:n.index):
            inputs=tuple(replacements[c.index] for c in n.inputs)
            clone=self.s.node(n.op,inputs,value=n.value,plain=n.plain)
            replacement=self.contraction(clone)
            replacements[n.index]=replacement or clone
        return [replacements[n.index] for n in outputs]

def simplify(source,request):
    record=dict(version=VERSION,applied=False,request_changed=False,reference_changed=False,provider_candidate_rewritten=False)
    layout=request["layout"]
    if layout["execution_abi"]!="unified-periodic-inputs-v1" or layout["input_slot_period"]>16 or sum(int(np.prod(b["shape"])) for b in layout["inputs"])>16:
        return source,dict(record,reason="Outside bounded quadratic scheduling scope")
    s=Schedule(request);outputs=s.parse(source,rewrite=False)
    q=Quadratic(s,request)
    try:
        outputs=q.transform(outputs)
        if not q.rewrites:return source,dict(record,reason="No complete lower-depth packed quadratic contraction",coefficient_work=q.work)
        result,operations=s.source(source,outputs)
    except Budget as error:
        return source,dict(record,reason=str(error),coefficient_work=q.work)
    record.update(applied=True,rewrites=q.rewrites,coefficient_work=q.work,
        derived_scalar_literals=sorted(set(q.literals)),emitted_operations=operations,
        exact_zero_coefficients_only=True,no_plain_plain_operations=True)
    return result,record
