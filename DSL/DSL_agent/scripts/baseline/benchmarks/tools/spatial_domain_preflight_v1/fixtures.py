"""New finite v4 domain-composition fixtures; existing models are unchanged."""
import copy
def node(name,op,refs,value,attrs):
    return dict(id=name,op=op,inputs=refs,outputs=[value],attrs=attrs)
def polynomial(domain):
    return dict(method="polynomial",basis="power",domain=domain,coefficients=[0.,0.,1.])
def resize(ref,name="resize",size=(4,),mode="nearest",align=None):
    return node(name,"resize",[ref],name+"_value",dict(size=list(size),scale_factor=None,mode=mode,
     align_corners=align,recompute_scale_factor=None,antialias=False))
def convolution(ref,rank,name="conv_t"):
    return node(name,"conv_transpose"+str(rank)+"d",[ref,"w"],name+"_value",
     dict(stride=[1]*rank,padding=[0]*rank,dilation=[1]*rank,groups=1,output_padding=[0]*rank,output_size=None))
def model(name,shape,nodes,constants=None):
    return dict(format="poseidon-operator-graph-v4",id=name,inputs=[dict(name="x",shape=list(shape))],
     constants=constants or {},nodes=nodes,outputs=[dict(name="out",value=nodes[-1]["outputs"][0])])
def models():
    result={}
    for rank in(1,2):
        shape=(1,1,2)if rank==1 else(1,1,2,2)
        r=resize("x",size=(4,)if rank==1 else(3,3),mode="nearest"if rank==1 else"bilinear",align=None if rank==1 else False)
        result["resize"+str(rank)+"_gelu"]=model("resize"+str(rank)+"_gelu",shape,[r,
         node("activation","gelu",[r["outputs"][0]],"y",dict(approximation=polynomial([-1.,1.])))])
    for rank in(1,2):
        shape=(1,1,2)if rank==1 else(1,1,1,2)
        c=convolution("x",rank);w=[[[2.]]]if rank==1 else[[[[2.]]]]
        attrs=dict(approximation=polynomial([-1.,1.]))if rank==1 else dict(approximation=dict(method="newton",domain=[1.,5.],iterations=1))
        result["transpose"+str(rank)+"_nonlinear"]=model("transpose"+str(rank)+"_nonlinear",shape,
         [c,node("activation","silu"if rank==1 else"sqrt",[c["outputs"][0]],"y",attrs)],{"w":w})
    first=node("first","gelu",["x"],"first_value",dict(approximation=polynomial([-2.,2.])))
    r=resize("first_value")
    result["two_domains"]=model("two_domains",(1,1,2),[first,r,
     node("second","exp",[r["outputs"][0]],"y",dict(approximation=dict(method="taylor",domain=[-1.,1.],degree=1)))])
    shift=node("public_shift","subtract",["offset","x"],"shifted",{})
    r=resize("shifted")
    result["public_left_spatial"]=model("public_left_spatial",(1,1,2),[shift,r,
     node("activation","exp",[r["outputs"][0]],"y",dict(approximation=dict(method="taylor",domain=[-1.,1.],degree=1)))],
     {"offset":[[[.5,.5]]]})
    return copy.deepcopy(result)
