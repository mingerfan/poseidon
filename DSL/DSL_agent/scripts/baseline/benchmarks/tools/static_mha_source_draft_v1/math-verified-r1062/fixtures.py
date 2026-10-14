"""Small deterministic source fixtures for the independent MHA draft audit."""
import torch
from mha_source_plan import public_mask,from_fx_call,verify

APPROX=dict(exp=dict(method="taylor",domain=[-.5,.5],degree=6),
            reciprocal=dict(method="goldschmidt",domain=[.5,8.],iterations=8))
CASES=[
 dict(id="self_unbatched",e=2,h=1,n=1,l=2,s=2,batch_first=False,unbatched=True,self_attention=True),
 dict(id="self_multihead",e=4,h=2,n=1,l=2,s=2,batch_first=True,self_attention=True,average=False),
 dict(id="cross_sequence_first",e=2,h=1,n=2,l=2,s=3,batch_first=False,need=False,dropout=.25),
 dict(id="separate_dimensions",e=4,h=2,n=1,l=2,s=3,batch_first=True,kdim=2,vdim=3),
 dict(id="boolean_and_padding",e=2,h=1,n=1,l=2,s=3,batch_first=True,mask="boolean_padding"),
 dict(id="per_head_finite_bias",e=4,h=2,n=1,l=2,s=2,batch_first=True,mask="per_head_float",average=False),
 dict(id="negative_infinity_padding",e=2,h=1,n=1,l=2,s=3,batch_first=True,mask="padding_neginf"),
 dict(id="bias_kv",e=2,h=1,n=1,l=2,s=2,batch_first=True,bias_kv=True,average=False),
 dict(id="zero_attention",e=2,h=1,n=1,l=2,s=2,batch_first=True,zero=True),
 dict(id="bias_and_zero_padding",e=4,h=2,n=1,l=2,s=3,batch_first=True,bias_kv=True,zero=True,need=False,dropout=.1,mask="tail_padding"),
 dict(id="causal_hint",e=2,h=1,n=1,l=2,s=3,batch_first=True,need=False,causal=True,mask="causal"),
 dict(id="causal_extra_tokens",e=2,h=1,n=1,l=3,s=2,batch_first=True,bias_kv=True,zero=True,need=False,causal=True,mask="causal"),
]
def mask_tensors(case):
    n,h,l,s=(case[k] for k in ("n","h","l","s"));kind=case.get("mask")
    attn=padding=None
    if kind=="boolean_padding":
        attn=torch.tensor([[False,True,False],[True,False,False]],dtype=torch.bool)
        padding=torch.tensor([[False,True,False]],dtype=torch.bool)
    elif kind=="per_head_float":
        attn=torch.linspace(-.05,.05,n*h*l*s,dtype=torch.float64).reshape(n*h,l,s)
    elif kind=="padding_neginf":
        padding=torch.tensor([[0.,float("-inf"),0.]],dtype=torch.float64)
    elif kind=="tail_padding":
        padding=torch.tensor([[False,False,True]],dtype=torch.bool)
    elif kind=="causal":
        attn=torch.arange(s)[None,:]>torch.arange(l)[:,None]
    return attn,padding
def module(case,index):
    torch.manual_seed(11501+index)
    m=torch.nn.MultiheadAttention(case["e"],case["h"],dropout=case.get("dropout",0.),
        bias=case.get("bias",True),add_bias_kv=case.get("bias_kv",False),add_zero_attn=case.get("zero",False),
        kdim=case.get("kdim"),vdim=case.get("vdim"),batch_first=case["batch_first"],dtype=torch.float64).eval()
    with torch.no_grad():
        for p in m.parameters():p.uniform_(-.2,.2)
    return m
class CrossFixture(torch.nn.Module):
    def __init__(self,m,case):
        super().__init__();self.attention=m;self.options=dict(need_weights=case.get("need",True),average_attn_weights=case.get("average",True),is_causal=case.get("causal",False))
        a,p=mask_tensors(case);self.register_buffer("attn_mask",a);self.register_buffer("padding_mask",p)
    def forward(self,q,k,v):
        return self.attention(q,k,v,key_padding_mask=self.padding_mask,attn_mask=self.attn_mask,**self.options)
class SelfFixture(CrossFixture):
    def forward(self,q):
        return self.attention(q,q,q,key_padding_mask=self.padding_mask,attn_mask=self.attn_mask,**self.options)
def input_shape(case,role):
    n,l,s,e=(case[k] for k in ("n","l","s","e"))
    length=l if role=="query" else s
    width=e if role=="query" else case.get("kdim",e) if role=="key" else case.get("vdim",e)
    if case.get("unbatched"):return [length,width]
    return [n,length,width] if case["batch_first"] else [length,n,width]
def capture(case,index=0):
    m=module(case,index)
    fixture=(SelfFixture if case.get("self_attention") else CrossFixture)(m,case).eval()
    graph=torch.fx.symbolic_trace(fixture)
    bindings={};shape_map={"q":input_shape(case,"query"),"k":input_shape(case,"key"),"v":input_shape(case,"value")}
    calls=[]
    for node in graph.graph.nodes:
        if node.op=="placeholder":bindings[node]=dict(kind="cipher",name=node.target,shape=shape_map[node.target],dtype="float64")
        elif node.op=="get_attr":bindings[node]=public_mask(graph.get_buffer(node.target))
        elif node.op=="call_module":calls.append(node)
        elif node.op!="output":raise ValueError("Unexpected fixture node")
    if len(calls)!=1:raise ValueError("Known one-module fixture")
    node=calls[0];state=graph.get_submodule(node.target)
    plan=from_fx_call(node,state,bindings,approximation=APPROX)
    verify(plan,expected_source_request_sha256=plan["source_request_sha256"])
    return fixture,graph,plan
