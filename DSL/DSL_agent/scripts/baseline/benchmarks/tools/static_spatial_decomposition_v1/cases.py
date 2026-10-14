"""Additional spatial models; never mutates the frozen 1,200-model benchmark."""
import copy
import numpy as np

def resize(identifier,shape,size=None,factor=None,mode="nearest",align=None,recompute=None):
    attrs=dict(size=size,scale_factor=factor,mode=mode,align_corners=align,
               recompute_scale_factor=recompute,antialias=False)
    return dict(format="poseidon-operator-graph-v1",id=identifier,inputs=[dict(name="x",shape=shape)],
                constants={},nodes=[dict(id="resize",op="resize",inputs=["x"],attrs=attrs,outputs=["y"])],
                outputs=[dict(name="output",value="y")])

def transpose(identifier,shape,cout,kernel,stride,padding,dilation,groups=1,output_padding=None,output_size=None,bias=True):
    rank=len(kernel);cin=shape[-rank-1]
    rng=np.random.default_rng(20261011+sum(map(ord,identifier)))
    constants=dict(weight=rng.uniform(-.3,.3,(cin,cout//groups,*kernel)).tolist())
    refs=["x","weight"]
    if bias:constants["bias"]=rng.uniform(-.1,.1,cout).tolist();refs.append("bias")
    attrs=dict(stride=stride,padding=padding,dilation=dilation,groups=groups,
               output_padding=output_padding or [0]*rank,output_size=output_size)
    return dict(format="poseidon-operator-graph-v1",id=identifier,inputs=[dict(name="x",shape=shape)],
                constants=constants,nodes=[dict(id="transposed",op=f"conv_transpose{rank}d",
                inputs=refs,attrs=attrs,outputs=["y"])],outputs=[dict(name="output",value="y")])

def models():
    result=[]
    for mode in ("nearest","nearest-exact","linear"):
        for context in range(3):
            ident=f"resize_1d_{mode.replace('-','_')}_{context}"
            shape=[1,1,3] if context==0 else [2,2,3] if context==1 else [1,2,4]
            kw=dict(size=[5]) if context==0 else dict(factor=[1.7]) if context==1 else dict(factor=[1.15],recompute=True)
            result.append(resize(ident,shape,mode=mode,align=(context==0) if mode=="linear" else None,**kw))
    for mode in ("nearest","nearest-exact","bilinear"):
        for context in range(3):
            ident=f"resize_2d_{mode.replace('-','_')}_{context}"
            shape=[1,1,2,3] if context==0 else [1,2,3,4] if context==1 else [2,1,4,2]
            kw=dict(size=[3,4]) if context==0 else dict(factor=[1.7,.7]) if context==1 else dict(factor=[.7,1.6],recompute=True)
            result.append(resize(ident,shape,mode=mode,align=(context==0) if mode=="bilinear" else None,**kw))
    specs1=[
      ([1,4],2,[3],[1],[0],[1],1,[0],None,False),
      ([1,4,3],2,[2],[2],[0],[1],2,[0],None,True),
      ([1,2,5],2,[3],[1],[2],[2],2,[0],None,True),
      ([1,2,3],2,[3],[2],[1],[1],1,[1],None,True),
      ([1,1,3],2,[2],[1],[0],[2],1,[1],None,False),
      ([1,1,3],2,[3],[2],[1],[1],1,[0],[999,999,6],True),
      ([1,1,6],2,[2],[2],[4],[2],1,[1],None,True),
      ([1,1,1],1,[1],[3],[1],[1],1,[2],None,True),
      ([2,3],4,[2],[2],[1],[1],2,[1],[999,5],False)]
    specs2=[
      ([1,2,3],2,[2,2],[1,1],[0,0],[1,1],1,[0,0],None,False),
      ([1,4,2,3],2,[2,3],[2,1],[0,1],[1,1],2,[1,0],None,True),
      ([1,2,3,4],2,[3,2],[1,1],[2,1],[2,2],2,[0,0],None,True),
      ([1,2,2,3],2,[3,3],[2,2],[1,1],[1,1],1,[1,1],None,True),
      ([1,1,3,3],2,[2,2],[1,1],[0,0],[2,2],1,[1,1],None,False),
      ([1,1,2,3],2,[3,3],[2,2],[1,1],[1,1],1,[0,0],[999,999,4,6],True),
      ([1,1,4,4],2,[2,2],[2,2],[3,3],[2,2],1,[1,1],None,True),
      ([1,1,1,2],1,[1,1],[3,2],[1,0],[1,1],1,[2,1],None,True),
      ([2,2,3],4,[2,3],[2,2],[1,1],[1,1],2,[1,1],None,False)]
    for rank,specs in ((1,specs1),(2,specs2)):
        for i,spec in enumerate(specs):result.append(transpose(f"transpose_{rank}d_{i}",*spec))
    for i,source in enumerate([result[j] for j in (0,8,9,17,21,30)]):
        m=copy.deepcopy(source);m["id"]="spatial_composition_"+str(i)
        tail="y"
        if i in (0,2,4):
            m["nodes"].append(dict(id="square",op="square",inputs=[tail],attrs={},outputs=["sq"]));tail="sq"
        else:
            rank=len(m["inputs"][0]["shape"])
            m["nodes"].append(dict(id="reduce",op="mean",inputs=[tail],
                attrs=dict(axes=list(range(2,rank)) or [-1],keepdims=True),outputs=["reduced"]));tail="reduced"
        m["outputs"]=[dict(name="original",value="y"),dict(name="composed",value=tail)]
        result.append(m)
    assert len(result)==42
    return result
