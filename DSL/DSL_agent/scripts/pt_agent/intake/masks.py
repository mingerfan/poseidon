"""Prove a binary key mask's two values at every static attention coordinate."""

from dataclasses import dataclass
import numpy as np


class MaskError(ValueError):
    pass


@dataclass
class Expression:
    zero: np.ndarray
    one: np.ndarray
    key: np.ndarray
    source: str | None


def expression(graph, name, mask_inputs):
    cache = {}

    def evaluate(name):
        if name in cache:
            return cache[name]
        if name in graph.weights:
            x = np.asarray(graph.weights[name])
            result = Expression(x,x,np.full(x.shape,-1,dtype=int),None)
        elif name in mask_inputs:
            shape=graph.tensors[name].shape
            if len(shape)!=2 or shape[0]!=1:
                raise MaskError('暂不支持的掩码形式: key mask shape must be [1,Lk]')
            result=Expression(np.zeros(shape),np.ones(shape),np.arange(shape[-1]).reshape(shape),name)
        elif name in graph.inputs:
            raise MaskError('attention mask depends on encrypted scores or features')
        else:
            node=graph.producer(name);a=node.attrs
            args=[evaluate(v) for v in node.inputs]
            source={v.source for v in args if v.source is not None}
            if len(source)>1:
                raise MaskError('暂不支持的掩码形式: multiple key masks')
            x=args[0]
            def function(vs):
                v=vs[0]
                if node.op in ('cast','identity'):return v
                if node.op=='reshape':return v.reshape(a['shape'])
                if node.op=='broadcast':return np.broadcast_to(v,a['shape'])
                if node.op=='permute':return v.transpose(a['dims'])
                if node.op=='slice':
                    s=[slice(None)]*v.ndim;s[a['axis']]=slice(a['start'],a['end'],a['step'])
                    return v[tuple(s)]
                if node.op=='slice_scatter':
                    out=v.copy();s=[slice(None)]*v.ndim
                    s[a['axis']]=slice(a['start'],a['end'],a['step']);out[tuple(s)]=vs[1]
                    return out
                y=(vs[1] if len(vs)>1 else graph.weights[a['const']] if 'const' in a else a.get('scalar'))
                if node.op=='add':return v+y
                if node.op=='sub':return y-v if a.get('reverse') else v-y
                if node.op=='mul':return v*y
                if node.op=='compare':
                    return getattr(np,dict(eq='equal',ne='not_equal',lt='less',le='less_equal',gt='greater',ge='greater_equal')[a['predicate']])(v,y)
                if node.op=='where':return np.where(v,vs[1],vs[2])
                raise MaskError('暂不支持的掩码形式: '+node.op)
            with np.errstate(over='ignore',invalid='ignore'):
                zero=function([v.zero for v in args]);one=function([v.one for v in args])
            if node.op in ('reshape','broadcast','permute','slice','cast','identity'):
                # Apply the same shape operation to the symbolic key index.
                old=x.zero;x.zero=x.key
                key=function([x.zero])
                x.zero=old
            elif node.op=='slice_scatter':
                old=[v.zero for v in args]
                key=function([v.key for v in args])
            else:
                shape=np.broadcast_shapes(*(v.zero.shape for v in args))
                key=np.full(shape,-1,dtype=int)
                for v in args:
                    k=np.broadcast_to(v.key,shape)
                    if np.any((key>=0)&(k>=0)&(key!=k)):
                        raise MaskError('暂不支持的掩码形式: a score depends on multiple key positions')
                    key=np.maximum(key,k)
            result=Expression(np.asarray(zero),np.asarray(one),np.asarray(key),next(iter(source),None))
        cache[name]=result
        return result
    return evaluate(name)


def recognize(graph, name, mask_inputs, *, length, keys):
    expr=expression(graph,name,mask_inputs)
    shape=(1,1,length,keys)
    try:
        zero=np.broadcast_to(expr.zero,shape)
        one=np.broadcast_to(expr.one,shape)
        key=np.broadcast_to(expr.key,shape)
    except ValueError as exc:
        raise MaskError('暂不支持的掩码形式: not a per-key/static mask') from exc
    allowed=lambda x:(x==0)|(x<=-1e4)
    if not allowed(zero).all() or not allowed(one).all():
        raise MaskError('暂不支持的掩码形式: additive values must be zero or <= -1e4')
    keep=one==0
    causal=np.arange(keys)[None,:]<=np.arange(length)[:,None]
    if keep.all():kind='all'
    elif np.array_equal(keep[0,0],causal):kind='causal'
    else:raise MaskError('暂不支持的掩码形式: static keep must be all or causal')
    if expr.source is not None:
        expected=np.broadcast_to(np.arange(keys),shape)
        if np.any(keep & ((zero>-1e4)|(key!=expected))):
            raise MaskError('暂不支持的掩码形式: expected keep_static[i,j] * m[j]')
    elif not np.array_equal(zero,one):
        raise MaskError('暂不支持的掩码形式: nonstatic mask')
    return kind,expr.source
