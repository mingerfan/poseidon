
"""Sparse virtual 65536-ring on bounded, P-periodic physical ciphertext Exprs.

Every virtual slot is retained until an explicit checked logical-output projection.
Public support masks prove zero regions; encrypted values are never inspected.
No helper function is replaced, and no nonperiodic constant is truncated.
"""
import math,hashlib,struct
from dataclasses import dataclass
from benchmark_graph import require

NT=65536

@dataclass(frozen=True)
class Block:
    value:object
    support:int
    public:bool=False

class Context:
    def __init__(self,period,limit=1024,max_blocks=64):
        require(type(period) is int and period in (4,8,16,32,64,128,256),"Virtual block period")
        require(type(limit) is int and 1<=limit<=1024 and type(max_blocks) is int and 1<=max_blocks<=64,"Virtual resource bounds")
        self.period=period;self.limit=limit;self.max_blocks=max_blocks;self.blocks=NT//period
        self.full=(1<<period)-1;self.physical=[];self.events=[];self.constants=[];self.peak_blocks=0;self.zero_elisions=0
    def spend(self,operation):
        require(len(self.physical)<self.limit,"Virtual physical operation budget")
        self.physical.append(operation)
    def check(self,blocks):
        require(type(blocks) is dict and len(blocks)<=self.max_blocks,"Virtual live block budget")
        for index,b in blocks.items():
            require(type(index) is int and 0<=index<self.blocks and type(b) is Block and
                    type(b.support) is int and 0<b.support<=self.full,"Virtual block identity/support")
            if b.public:
                require(type(b.value) is tuple and len(b.value)==self.period and
                        all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1024 for v in b.value),"Virtual public block")
                require(b.support==self.support(b.value),"Virtual public support")
        self.peak_blocks=max(self.peak_blocks,len(blocks))
    @staticmethod
    def support(values):
        return sum(1<<i for i,v in enumerate(values) if v!=0)
    def public_block(self,values):
        values=tuple(float(v) for v in values);mask=self.support(values)
        if not mask:return None
        require(all(math.isfinite(v) and abs(v)<=1024 for v in values),"Virtual derived public constant")
        return Block(values,mask,True)
    def constant(self,value):
        dtype=None
        if hasattr(value,"detach"):
            dtype=str(value.dtype);value=value.detach().cpu().reshape(-1).tolist()
        elif hasattr(value,"tolist"):value=value.reshape(-1).tolist()
        if type(value) in (int,float):
            require(value==0,"Virtual helper scalar must be exact public zero")
            self.constants.append(dict(kind="scalar_zero"));return {}
        require(type(value) in (list,tuple) and len(value)==NT,"Virtual constant must retain all 65536 entries")
        require(all(type(v) in (int,float) and math.isfinite(v) and abs(v)<=1024 for v in value),"Virtual full constant values")
        digest=hashlib.sha256(struct.pack("<"+str(NT)+"d",*value)).hexdigest()
        blocks={}
        for b in range(self.blocks):
            part=self.public_block(value[b*self.period:(b+1)*self.period])
            if part is not None:blocks[b]=part
        self.check(blocks)
        self.constants.append(dict(kind="full_virtual_vector",entries=NT,dtype=dtype,sha256=digest,
                                   nonzero_blocks=sorted(blocks),nonzero_values=sum(b.support.bit_count() for b in blocks.values())))
        return blocks
    def footprint(self,blocks):
        return [[i,b.support,"public" if b.public else "cipher"] for i,b in sorted(blocks.items())]
    def event(self,operation,inputs,result,**fields):
        require(len(self.events)<2048,"Virtual logical operation budget")
        self.check(result)
        self.events.append(dict(operation=operation,inputs=[self.footprint(x) for x in inputs],
                                output=self.footprint(result),**fields))
    def mask(self,block,keep):
        support=block.support&keep
        if not support:self.zero_elisions+=1;return None
        if support==block.support:return block
        values=[float(bool(support&(1<<i))) for i in range(self.period)]
        if block.public:return self.public_block([v*m for v,m in zip(block.value,values)])
        self.spend("mask");return Block(block.value*values,support)
    def merge(self,left,right,op):
        if op=="multiply":
            if left is None or right is None or not left.support&right.support:
                self.zero_elisions+=1;return None
        elif right is None:return left
        elif left is None:
            if op=="add":return right
            if right.public:return self.public_block([-v for v in right.value])
            self.spend("negate");return Block(-right.value,right.support)
        if left.public and right.public:
            return self.public_block([(a+b if op=="add" else a-b if op=="subtract" else a*b)
                                      for a,b in zip(left.value,right.value)])
        a=list(left.value) if left.public else left.value
        b=list(right.value) if right.public else right.value
        self.spend(op)
        value=a+b if op=="add" else a-b if op=="subtract" else a*b
        support=left.support&right.support if op=="multiply" else left.support|right.support
        return Block(value,support)
    def rotate_block(self,block,step):
        if not step:return block
        p=self.period;support=((block.support>>step)|(block.support<<(p-step)))&self.full
        if block.public:
            return self.public_block(block.value[step:]+block.value[:step])
        result=block.value
        for j in range(p.bit_length()-1):
            if step&(1<<j):self.spend("rotate");result=result.rotate(1<<j)
        return Block(result,support)
    def report(self):
        return dict(virtual_slots=NT,physical_slots=16384,block_period=self.period,max_live_blocks=self.max_blocks,
                    peak_live_blocks=self.peak_blocks,physical_operations=list(self.physical),events=self.events,
                    full_constants=self.constants,public_support_zero_elisions=self.zero_elisions,
                    secret_values_inspected=False,bootstrap_removed=False,nonperiodic_constant_truncation=False)

class VirtualExpr:
    __array_priority__=10000
    def __init__(self,context,blocks):
        context.check(blocks);self.context=context;self.values=blocks
    @classmethod
    def input(cls,context,cipher,count):
        require(type(count) is int and 1<=count<=context.period,"Virtual input prefix")
        block=Block(cipher,context.full);block=context.mask(block,(1<<count)-1)
        result=cls(context,{0:block})
        context.event("input",[],result.values,count=count)
        return result
    @classmethod
    def input_chunks(cls,context,ciphers,count):
        """Bind C-order chunks without collapsing distinct encrypted blocks.

        This is a trusted representation primitive; candidates cannot instantiate
        it. The existing public request gate stays closed until joint bindings
        and their tracing audits are connected.
        """
        p=context.period
        require(type(count) is int and 1<=count<=256,"Virtual chunk logical size")
        require(type(ciphers) in (list,tuple) and len(ciphers)==(count+p-1)//p
                and 1<=len(ciphers)<=4,"Virtual chunk input count")
        blocks={}
        for index,cipher in enumerate(ciphers):
            length=min(p,count-index*p)
            blocks[index]=context.mask(Block(cipher,context.full),(1<<length)-1)
        result=cls(context,blocks)
        context.event("chunked_input",[],result.values,count=count,
                      chunks=[dict(block=i,offset=i*p,elements=min(p,count-i*p))
                              for i in range(len(ciphers))])
        return result
    def export_chunk(self,offset,count,total,zero_cipher):
        """Project one declared output chunk; retain the complete virtual value."""
        c=self.context;p=c.period
        require(type(total) is int and 1<=total<=256 and (total+p-1)//p<=4,
                "Virtual chunk output total")
        require(type(offset) is int and 0<=offset<total and offset%p==0
                and type(count) is int and count==min(p,total-offset),
                "Virtual chunk output selection")
        index=offset//p
        selected=c.mask(self.values[index],(1<<count)-1) if index in self.values else None
        public=selected is None or selected.public
        if public:
            zero=c.mask(Block(zero_cipher,c.full),(1<<count)-1)
            if selected is None:value=zero.value
            else:c.spend("public_materialize");value=zero.value+list(selected.value)
        else:value=selected.value
        c.event("chunk_output_projection",[self.values],({index:selected} if selected else {}),
                selected_start=offset,selected_count=count,logical_total=total,public_only=public,
                discarded_nonzero_blocks=sorted(i for i in self.values if i!=index),
                encrypted_zero_argument_used=public)
        return value
    def binary(self,other,op,reverse=False):
        c=self.context
        if isinstance(other,VirtualExpr):
            require(other.context is c,"Mixed virtual contexts");values=other.values
        else:values=c.constant(other)
        left,right=(values,self.values) if reverse else (self.values,values);result={}
        for index in sorted(set(left)|set(right)):
            block=c.merge(left.get(index),right.get(index),op)
            if block is not None:result[index]=block
        c.event(op,[left,right],result)
        return type(self)(c,result)
    def __add__(self,x):return self.binary(x,"add")
    def __radd__(self,x):return self.binary(x,"add",True)
    def __sub__(self,x):return self.binary(x,"subtract")
    def __rsub__(self,x):return self.binary(x,"subtract",True)
    def __mul__(self,x):return self.binary(x,"multiply")
    def __rmul__(self,x):return self.binary(x,"multiply",True)
    def __neg__(self):return self.binary(0,"subtract",True)
    def rotate(self,step):
        require(type(step) is int,"Virtual integer rotation")
        c=self.context;p=c.period;normalized=step%NT;q,r=divmod(normalized,p);result={}
        for index,block in sorted(self.values.items()):
            rotated=c.rotate_block(block,r);target=(index-q)%c.blocks
            parts=[(target,c.full)] if r==0 else [(target,(1<<(p-r))-1),((target-1)%c.blocks,c.full^((1<<(p-r))-1))]
            for dest,mask in parts:
                piece=c.mask(rotated,mask)
                if piece is not None:
                    joined=c.merge(result.get(dest),piece,"add")
                    if joined is not None:result[dest]=joined
        c.event("rotate",[self.values],result,requested=step,normalized=normalized,block_shift=q,lane_shift=r)
        return type(self)(c,result)
    def export_prefix(self,count,zero_cipher):
        c=self.context;require(type(count) is int and 1<=count<=c.period,"Virtual output prefix")
        selected=c.mask(self.values.get(0), (1<<count)-1) if 0 in self.values else None
        public=selected is None or selected.public
        if public:
            # The runner's existing encrypted-zero argument is explicit at the
            # candidate callsite. Never derive a fake zero via x-x or decrypt.
            zero=c.mask(Block(zero_cipher,c.full),(1<<count)-1)
            if selected is None:value=zero.value
            else:c.spend("public_materialize");value=zero.value+list(selected.value)
        else:value=selected.value
        c.event("output_projection",[self.values],({0:selected} if selected else {}),
                selected_start=0,selected_count=count,public_only=public,
                discarded_nonzero_blocks=sorted(i for i in self.values if i!=0),
                encrypted_zero_argument_used=public)
        return value

def proxy_type(expr_class):
    cls=type("TrustedVirtualExpr",(VirtualExpr,expr_class),{})
    for name,value in VirtualExpr.__dict__.items():
        if name not in ("__dict__","__weakref__","__module__","__doc__"):setattr(cls,name,value)
    return cls
