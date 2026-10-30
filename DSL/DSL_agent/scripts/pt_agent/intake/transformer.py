"""Normalize static token views and attention semantics before DSL planning.

Index maps prove the head/GQA views; masks have a separate binary algebra.
Neither path uses model names, observed model outputs or generated answers.
"""

from dataclasses import dataclass
import math
import numpy as np

from .graph import Graph, Node, TensorInfo
from .masks import MaskError, recognize


class Unsupported(ValueError):
    pass


@dataclass
class View:
    root: str
    L: int
    D: int
    index: np.ndarray
    sign: np.ndarray
    score: bool = False


class Normalizer:
    def __init__(self, graph):
        self.original=graph
        self.graph=Graph(graph.name,[],{},dict(graph.weights),list(graph.inputs),[])
        self.cache={};self.masks=set()
        for name in graph.inputs:
            t=graph.tensors[name]
            self.graph.tensors[name]=t
            if len(t.shape)==3 and t.shape[0]==1:
                self.cache[name]=self.view(name,t.shape[1],t.shape[2],t.shape)
            elif len(t.shape)==2 and t.shape[0]==1:
                self.masks.add(name)
                self.graph.tensors[name]=TensorInfo(name,t.shape,'mask')
        self.active=None

    def view(self, name, L, D, shape, *, index=None, sign=None, score=False):
        index=np.arange(math.prod(shape)).reshape(shape) if index is None else index
        return View(name,L,D,index,np.ones(index.shape,dtype=np.int8) if sign is None else sign,score)

    def emit(self, node, op, inputs, attrs, shape, *, name=None):
        name=name or node.outputs[0]
        self.graph.nodes.append(Node(f'n{len(self.graph.nodes)}',op,list(inputs),[name],attrs,node.src))
        self.graph.tensors[name]=TensorInfo(name,tuple(shape),'act')
        return name

    def shape(self, name):
        return self.original.tensors[name].shape

    def strip_cast(self, name):
        node=self.original.producer(name)
        while node is not None and node.op in ('cast','identity'):
            if node.op=='cast' and node.attrs['dtype'] not in ('torch.float32','torch.float64'):
                raise Unsupported('暂不支持 non-floating cast')
            name=node.inputs[0];node=self.original.producer(name)
        return name

    def depends_only_on_mask(self, name):
        if name in self.original.weights:return True
        if name in self.original.inputs:return name in self.masks
        node=self.original.producer(name)
        return node is not None and all(self.depends_only_on_mask(v) for v in node.inputs)

    def canonical_map(self, view):
        if view.score:
            raise Unsupported('暂不支持 score layout in a token operation')
        labels=view.index//view.D
        candidates=[]
        for axis,size in enumerate(view.index.shape):
            if size>view.L:continue
            index=np.moveaxis(view.index,axis,0).reshape(size,-1)
            if index.shape[1]!=view.D:continue
            if np.all(index//view.D == (index//view.D)[:,:1]):
                candidates.append((index,np.moveaxis(view.sign,axis,0).reshape(size,-1)))
        if not candidates:
            raise Unsupported('暂不支持 view: only token/head split, merge and static last-axis shuffle are lowerable')
        return candidates[0]

    def materialize(self, view, node):
        index,sign=self.canonical_map(view)
        rows=index[:,0]//view.D
        root=view.root;L=len(index)
        if not np.array_equal(rows,np.arange(view.L)):
            if not len(rows) or not np.array_equal(rows,np.arange(rows[0],rows[0]+len(rows))):
                raise Unsupported('暂不支持 noncontiguous token view')
            root=self.emit(node,'slice',[root],dict(axis=1,start=int(rows[0]),end=int(rows[-1])+1,step=1),
                           (1,L,view.D),name=node.outputs[0]+'.tokens')
        local=index%view.D
        if not np.all(local==local[:1]) or not np.all(sign==sign[:1]):
            raise Unsupported('暂不支持 token-dependent static reorder')
        if not np.array_equal(local[0],np.arange(view.D)) or not np.all(sign==1):
            root=self.emit(node,'shuffle',[root],dict(index=tuple(map(int,local[0])),sign=tuple(map(int,sign[0]))),
                           (1,L,view.D),name=node.outputs[0]+'.shuffle')
        return root,L,view.D

    def pointwise_constant(self, node, view, value):
        value=np.broadcast_to(value,self.shape(node.outputs[0]))
        if view.score:
            return value.reshape(self.graph.tensors[view.root].shape)
        index,sign=self.canonical_map(view)
        # Interpret public data in the same logical token/head order as x.
        for axis,size in enumerate(view.index.shape):
            transformed=np.moveaxis(view.index,axis,0).reshape(size,-1)
            if transformed.shape==index.shape and np.array_equal(transformed,index):
                result=np.moveaxis(value,axis,0).reshape(size,-1)
                return result.reshape(1,size,view.D)
        raise Unsupported('暂不支持 public position-constant view')

    def rms_pattern(self, node):
        if node.op!='mul' or 'const' not in node.attrs:return None
        end=self.original.producer(self.strip_cast(node.inputs[0]))
        if end is None or end.op!='mul' or len(end.inputs)!=2:return None
        for source,inverse in (end.inputs,end.inputs[::-1]):
            inv=self.original.producer(self.strip_cast(inverse))
            if inv is None or inv.op!='act' or inv.attrs['fn']!='rsqrt':continue
            add=self.original.producer(inv.inputs[0])
            if add is None or add.op!='add' or 'scalar' not in add.attrs:continue
            mean=self.original.producer(add.inputs[0])
            if mean is None or mean.op!='mean' or not mean.attrs['keepdim']:continue
            square=self.original.producer(mean.inputs[0])
            if square is None or square.op!='act' or square.attrs.get('fn')!='pow' or square.attrs.get('exponent')!=2:continue
            if self.strip_cast(square.inputs[0])!=self.strip_cast(source):continue
            if tuple(mean.attrs['axes'])!=(len(self.shape(square.inputs[0]))-1,):continue
            return self.strip_cast(source),node.attrs['const'],add.attrs['scalar']
        return None

    def get(self, name):
        if name in self.cache:return self.cache[name]
        node=self.original.producer(name)
        if node is None:raise Unsupported('暂不支持 non-token input '+name)
        self.active=node
        a=node.attrs;shape=self.shape(name)
        rms=self.rms_pattern(node)
        if rms is not None:
            source,w,eps=rms
            root,L,D=self.materialize(self.get(source),node)
            out=self.emit(node,'rmsnorm',[root],dict(weight=w,eps=eps),(1,L,D))
            result=self.view(out,L,D,shape)
        elif node.op in ('reshape','permute','broadcast','slice','cast','identity'):
            x=self.get(node.inputs[0]);index=x.index;sign=x.sign
            if node.op=='reshape':index=index.reshape(shape);sign=sign.reshape(shape)
            elif node.op=='permute':index=index.transpose(a['dims']);sign=sign.transpose(a['dims'])
            elif node.op=='broadcast':index=np.broadcast_to(index,shape);sign=np.broadcast_to(sign,shape)
            elif node.op=='slice':
                parts=[slice(None)]*index.ndim;parts[a['axis']]=slice(a['start'],a['end'],a['step'])
                index=index[tuple(parts)];sign=sign[tuple(parts)]
            elif node.op=='cast' and a['dtype'] not in ('torch.float32','torch.float64'):
                raise Unsupported('暂不支持 non-floating cast')
            result=View(x.root,x.L,x.D,index,sign,x.score)
        elif node.op=='concat':
            parts=[self.get(v) for v in node.inputs]
            if len({v.root for v in parts})!=1:
                raise Unsupported('暂不支持 nonstatic/cross-token concatenation')
            x=parts[0]
            result=View(x.root,x.L,x.D,np.concatenate([v.index for v in parts],axis=a['axis']),
                        np.concatenate([v.sign for v in parts],axis=a['axis']))
        elif node.op=='mul' and a.get('scalar')==-1:
            x=self.get(node.inputs[0]);result=View(x.root,x.L,x.D,x.index,-x.sign,x.score)
        elif node.op=='linear':
            x=self.get(node.inputs[0]);root,L,D=self.materialize(x,node)
            O=self.original.weights[a['weight']].shape[0]
            out=self.emit(node,'linear',[root],dict(a),(1,L,O))
            result=self.view(out,L,O,shape)
        elif node.op in ('layernorm','act'):
            x=self.get(node.inputs[0]);root,L,D=self.materialize(x,node)
            if node.op=='layernorm' and tuple(a['normalized_shape'])!=(D,):
                raise Unsupported('暂不支持 normalization outside the last token dimension')
            out=self.emit(node,node.op,[root],dict(a),(1,L,D))
            result=self.view(out,L,D,shape,index=x.index.copy())
        elif node.op in ('add','sub','mul'):
            x=self.get(node.inputs[0])
            if x.score:root,L,D=x.root,x.L,x.D
            else:root,L,D=self.materialize(x,node)
            attrs=dict(a)
            inputs=[root]
            if len(node.inputs)==2:
                other=self.get(node.inputs[1])
                otherroot,otherL,otherD=self.materialize(other,node)
                if (L,D)!=(otherL,otherD):raise Unsupported('暂不支持 token broadcast between encrypted features')
                inputs.append(otherroot)
            else:
                value=self.original.weights[a['const']] if 'const' in a else np.asarray(a['scalar'])
                if a.get('reverse'):
                    raise Unsupported('暂不支持 reversed feature subtraction')
                value=self.pointwise_constant(node,x,value)
                key='u4.public.'+node.id
                self.graph.weights[key]=np.array(value,dtype=np.float64)
                attrs=dict(const=key)
            if x.score:
                outshape=self.graph.tensors[root].shape
            else:outshape=(1,L,D)
            out=self.emit(node,node.op,inputs,attrs,outshape)
            result=self.view(out,L,D,shape,index=x.index.copy(),score=x.score)
        elif node.op=='matmul':
            left,right=[self.get(v) for v in node.inputs]
            if left.score:
                s=self.graph.tensors[left.root].shape;H,L,K=s[1:]
                vshape=self.shape(node.inputs[1]);dh=vshape[-1]
                Hkv=right.D//dh
                expected=np.arange(right.L*right.D).reshape(right.L,Hkv,dh).transpose(1,0,2)
                expected=np.repeat(expected,H//Hkv,axis=0)
                if not np.array_equal(right.index.reshape(H,K,dh),expected) or not np.all(right.sign==1):
                    raise Unsupported('暂不支持 V head/GQA view')
                out=self.emit(node,'matmul',[left.root,right.root],dict(transpose_b=False,heads=H,kv_heads=Hkv,scale=None),
                              (1,L,H*dh))
                index=np.arange(L*H*dh).reshape(L,H,dh).transpose(1,0,2).reshape(shape)
                result=self.view(out,L,H*dh,shape,index=index)
            else:
                qshape=self.shape(node.inputs[0]);kshape=self.shape(node.inputs[1])
                H=math.prod(qshape[:-2]);L,dh=qshape[-2:];K=kshape[-1]
                Hkv=right.D//dh
                if not Hkv or H%Hkv or left.D!=H*dh or left.L!=L or right.L!=K:
                    raise Unsupported('暂不支持 Q/K attention dimensions')
                q=np.arange(L*left.D).reshape(L,H,dh).transpose(1,0,2)
                k=np.arange(K*right.D).reshape(K,Hkv,dh).transpose(1,2,0)
                k=np.repeat(k,H//Hkv,axis=0)
                if (not np.array_equal(left.index.reshape(H,L,dh),q) or not np.array_equal(right.index.reshape(H,dh,K),k)
                        or not np.all(left.sign==1) or not np.all(right.sign==1)):
                    raise Unsupported('暂不支持 Q/K head split/transpose/GQA view')
                out=self.emit(node,'matmul',[left.root,right.root],dict(transpose_b=True,heads=H,kv_heads=Hkv,scale=None),
                              (1,H,L,K))
                result=self.view(out,L,H*K,shape,score=True)
        elif node.op=='softmax':
            source=self.strip_cast(node.inputs[0]);add=self.original.producer(source)
            mask=None;keep='all'
            if add is not None and add.op=='add' and len(add.inputs)==2:
                candidates=[i for i,v in enumerate(add.inputs) if self.depends_only_on_mask(v)]
                if len(candidates)==1:
                    i=candidates[0];source=add.inputs[1-i]
                    x=self.get(source);s=self.graph.tensors[x.root].shape
                    keep,mask=recognize(self.original,add.inputs[i],self.masks,length=s[-2],keys=s[-1])
            x=self.get(source)
            if not x.score:raise Unsupported('暂不支持 non-attention frontend softmax')
            s=self.graph.tensors[x.root].shape
            out=self.emit(node,'softmax',[x.root]+([mask] if mask else []),dict(axis=-1,keep=keep,mask=mask),s)
            result=self.view(out,x.L,x.D,shape,index=x.index.copy(),score=True)
        else:
            raise Unsupported('暂不支持 '+node.op)
        self.cache[name]=result
        return result

    def finish(self, *, last_position=False):
        for name in self.original.outputs:
            view=self.get(name)
            node=self.original.producer(name)
            root,L,D=self.materialize(view,node) if not view.score else (view.root,view.L,view.D)
            shape=self.shape(name)
            if len(shape)==2 and L==1:
                root=self.emit(node,'flatten',[root],dict(start_dim=1,end_dim=2),(1,D),name=name+'.flat')
            self.graph.outputs.append(root)
        # Scalar score multiplications fold into a Q projection whenever its
        # normalized input is that projection. Keep shared/public shifts.
        removed=set()
        for node in self.graph.nodes:
            if node.op!='mul' or 'const' not in node.attrs:continue
            producer=self.graph.producer(node.inputs[0])
            c=self.graph.weights[node.attrs['const']]
            if producer is None or producer.op!='matmul' or not producer.attrs['transpose_b'] or not np.all(c==c.flat[0]):continue
            q=self.graph.producer(producer.inputs[0])
            if q is not None and q.op=='linear' and len(self.graph.consumers(q.outputs[0]))==1:
                for field in ('weight','bias'):
                    if q.attrs.get(field) is not None:
                        key='u4.scaled.'+q.id+'.'+field
                        self.graph.weights[key]=self.graph.weights[q.attrs[field]]*float(c.flat[0])
                        q.attrs[field]=key
                for following in self.graph.nodes:
                    following.inputs=[producer.outputs[0] if v==node.outputs[0] else v for v in following.inputs]
                removed.add(node.id)
            else:
                producer.attrs['scale']=float(c.flat[0])
                for following in self.graph.nodes:
                    following.inputs=[producer.outputs[0] if v==node.outputs[0] else v for v in following.inputs]
                removed.add(node.id)
        self.graph.nodes=[n for n in self.graph.nodes if n.id not in removed]
        if last_position:
            out=self.graph.outputs[0];head=self.graph.producer(out)
            if head is None or head.op!='linear' or self.graph.tensors[head.inputs[0]].shape[:2]!=(1,L):
                raise Unsupported('暂不支持 causal-LM output without a final token linear head')
            sel='sel'
            if sel in self.graph.tensors:raise Unsupported('reserved client selection input already exists')
            self.graph.inputs.append(sel)
            self.graph.tensors[sel]=TensorInfo(sel,(1,L),'mask')
            source=head.inputs[0];dim=self.graph.tensors[source].shape[-1];selected=source+'.last'
            index=self.graph.nodes.index(head)
            self.graph.nodes.insert(index,Node('select_last','select',[source,sel],[selected],{},head.src))
            self.graph.tensors[selected]=TensorInfo(selected,(1,1,dim),'act')
            head.inputs[0]=selected
            self.graph.tensors[out]=TensorInfo(out,(1,1,D),'act')
        used=set(self.graph.inputs)|set(self.graph.outputs)|{v for n in self.graph.nodes for v in n.inputs+n.outputs}
        self.graph.tensors={k:v for k,v in self.graph.tensors.items() if k in used}
        self.graph.intake_stats=dict(getattr(self.original,'intake_stats',{}),
                                     transformer_normalized=True,source_nodes=len(self.original.nodes),
                                     floating_casts='float64 arithmetic; framework float32 rounding is not reproduced')
        return self.graph


def normalize(graph, *, last_position=False, strict=True):
    if not any(len(graph.tensors[name].shape)==3 for name in graph.inputs):return graph
    worker=Normalizer(graph)
    try:
        return worker.finish(last_position=last_position)
    except (Unsupported,MaskError) as exc:
        node=worker.active
        diag=dict(op_id=node.id if node else None,lineno=graph.nodes.index(node)+1 if node else 0,
                  source=node.src if node else graph.name,reason=str(exc),severity='error')
        if strict:raise Unsupported(str(diag)) from exc
        graph.lowering_errors=[diag]
        return graph
