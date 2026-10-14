"""Bounded developer prototype for public polynomial and shared-factor schedules."""
import math
from collections import Counter
from benchmark_graph import require
from unified_graph_scheduling import Schedule
from packed_quadratic_prototype import Quadratic

def estrin(x, coefficients, zero):
    def group(coeffs, base):
        if len(coeffs)==1:
            return zero+float(coeffs[0])
        pairs=[]
        for i in range(0,len(coeffs),2):
            even=float(coeffs[i])
            odd=float(coeffs[i+1]) if i+1<len(coeffs) else 0.
            pairs.append(base*odd+even if odd else zero+even)
        return combine(pairs,base*base)
    def combine(parts,base):
        if len(parts)==1:return parts[0]
        pairs=[]
        for i in range(0,len(parts),2):
            pairs.append(parts[i]+parts[i+1]*base if i+1<len(parts) else parts[i])
        return combine(pairs,base*base) if len(pairs)>1 else pairs[0]
    return group(coefficients,x)

def shared_factors(s,outputs):
    memo={};changes=0
    def plain(n):
        return n.plain or bool(n.inputs) and all(plain(c) for c in n.inputs)
    def candidates(n):
        values={n} if not n.plain else set()
        if n.op=="multiply":
            for c in n.inputs:values.update(candidates(c))
        return values
    def remove(n,common):
        if n is common:return None
        if n.op!="multiply":return n
        a,b=n.inputs
        if common in candidates(a):
            v=remove(a,common)
            return b if v is None else s.node("multiply",(v,b))
        if common in candidates(b):
            v=remove(b,common)
            return a if v is None else s.node("multiply",(a,v))
        return n
    def rewrite(n):
        nonlocal changes
        if n.index in memo:return memo[n.index]
        cs=tuple(rewrite(c) for c in n.inputs)
        result=s.node(n.op,cs,value=n.value,plain=n.plain)
        if n.op in ("add","subtract") and all(not c.plain for c in cs):
            left,right=cs
            shared=candidates(left)&candidates(right)
            shared.discard(left);shared.discard(right)
            if shared:
                common=max(shared,key=lambda c:(c.depth,len(candidates(c)), -c.index))
                a,b=remove(left,common),remove(right,common)
                if a is not None and b is not None and not (plain(a) and plain(b)):
                    result=s.node("multiply",(s.node(n.op,(a,b)),common))
                    changes+=1
        memo[n.index]=result
        return result
    return [rewrite(n) for n in outputs],changes

def simplify(source,request):
    s=Schedule(request);outputs=s.parse(source,rewrite=False)
    q=Quadratic(s,request);outputs=q.transform(outputs)
    outputs,shared=shared_factors(s,outputs)
    result,ops=s.source(source,outputs)
    return result,dict(version="public-polynomial-correlation-prototype-v2",
        quadratic_rewrites=q.rewrites,coefficient_work=q.work,shared_factors=shared,
        emitted_operations=ops,provider_candidate_rewritten=False)
