
"""Immutable two-axis stride-two slice binding to actual poly.Func.HE_DS."""
import math,copy
from benchmark_graph import validate,require
from upstream_adapters import spatial_mapped

ADAPTER="mapped-downsample-v1"

def inner_binding(channels,height,width,period):
    require(height>=2 and width>=2 and height&(height-1)==0 and width&(width-1)==0,"DS tile shape")
    require(channels*height*width<=period,"DS tile exceeds P")
    ho,wo=height//2,width//2;to=(channels+3)//4
    pi=period//(1<<(channels*height*width-1).bit_length())
    packed=4*ho*wo*to;po=period//(1<<(packed-1).bit_length())
    require(pi>=1 and po>=1,"DS single-expression closure")
    geometry=dict(nt=period,bb=1.,fh=1,fw=1,s=2,hi=height,wi=width,ki=1,ci=channels,co=channels,
                  ho=ho,wo=wo,ko=2,ti=channels,to=to,ni=1,no=1,pi=pi,po=po,q=(channels+pi-1)//pi)
    center=4*ho*wo*channels//8;output_block=period//po;closure=period//pi
    positions=[(height*width*(c//4)+width*(2*h+(c%4)//2)+2*w+c%2+center)%output_block
               for c in range(channels) for h in range(ho) for w in range(wo)]
    require(len(positions)==len(set(positions)) and max(positions)<period,"DS output positions collide")
    rot=lambda n:(n%period).bit_count()
    work=4+sum(1+rot(-closure*(1<<j)) for j in range(pi.bit_length()-1))
    work+=sum(2+rot(height*width*(c-c//4)-width*((c%4)//2)-c%2) for c in range(channels))
    work+=rot(-center)+sum(1+rot(-output_block*(1<<j)) for j in range(po.bit_length()-1))
    if positions!=list(range(len(positions))):
        work+=sum(1+rot(src-dst) for dst,src in enumerate(positions))+len(positions)-1
    require(work<=1024,"DS inner work budget")
    return dict(schema=1,kind="downsample",helper="HE_DS",node_id="ds_inner",input_value="input0",output_value="out",
                input_shape=[1,channels,height,width],output_shape=[1,channels,ho,wo],period=period,
                geometry=geometry,parameters={},input_closure_period=closure,output_positions=positions,work=work)

def bind_node(model,node_id,period):
    checked=validate(model);nodes=[n for n in model["nodes"] if n["id"]==node_id]
    require(len(nodes)==1 and nodes[0]["op"]=="slice","DS final slice identity")
    last=nodes[0];producers={v:n for n in model["nodes"] for v in n["outputs"]}
    first=producers.get(last["inputs"][0])
    require(first is not None and first["op"]=="slice","DS requires two spatial slice producer edges")
    shape=checked["shapes"][first["inputs"][0]];outshape=checked["shapes"][last["outputs"][0]]
    require(len(shape)==4 and shape[0]==1,"DS batch-one rank4")
    require(type(period) is int and period in (4,8,16,32,64,128,256),"DS period")
    require(max(math.prod(shape),math.prod(outshape))<=period,"DS intermediate exceeds P")
    axes={n["attrs"]["axis"]%4 for n in (first,last)}
    require(axes=={2,3} and all(n["attrs"]["step"]==2 for n in (first,last)),"DS requires distinct spatial axes, step two")
    indices={}
    for node in (first,last):
        a=node["attrs"];axis=a["axis"]%4
        indices[axis]=list(range(*slice(a["start"],a["stop"],a["step"]).indices(shape[axis])))
    require(indices[2] and indices[3],"DS empty slice")
    _,ci,hi,wi=shape;ho,wo=len(indices[2]),len(indices[3])
    require(outshape==[1,ci,ho,wo],"DS mathematical output binding")
    calls=[];choices=[]
    powers=(2,4,8,16,32,64,128,256)
    for th in powers:
        for tw in powers:
            if th>1<<(2*ho-1).bit_length() or tw>1<<(2*wo-1).bit_length() or th*tw>period:continue
            channels=min(ci,period//(th*tw));cy,cx=th//2,tw//2
            count=((ci+channels-1)//channels)*((ho+cy-1)//cy)*((wo+cx-1)//cx)
            choices.append((count,th*tw,th,tw,channels))
    require(choices,"DS no bounded tile")
    _,_,th,tw,channels=min(choices);cy,cx=th//2,tw//2
    for c0 in range(0,ci,channels):
        count=min(channels,ci-c0);inner=inner_binding(count,th,tw,period)
        for h0 in range(0,ho,cy):
            for w0 in range(0,wo,cx):
                cells=[];targets=[];origin_y=indices[2][h0];origin_x=indices[3][w0]
                for c in range(count):
                    for y in range(th):
                        for x in range(tw):
                            yy,xx=origin_y+y,origin_x+x
                            if 0<=yy<hi and 0<=xx<wi:
                                cells.append(((c*th+y)*tw+x,((c+c0)*hi+yy)*wi+xx,1.))
                    for h in range(h0,min(ho,h0+cy)):
                        for w in range(w0,min(wo,w0+cx)):
                            targets.append((((c+c0)*ho+h)*wo+w,(c*cy+h-h0)*cx+w-w0,1.))
                calls.append(dict(inner=copy.deepcopy(inner),before=spatial_mapped.mapping(cells,period),
                                  after=spatial_mapped.mapping(targets,period)))
    require(1<=len(calls)<=64,"DS invocation budget")
    work=sum(c["inner"]["work"]+spatial_mapped.map_work(c["before"])+spatial_mapped.map_work(c["after"])
             for c in calls)+len(calls)-1
    require(work<=1024,"DS total work exceeds 1024")
    return dict(schema=1,adapter=ADAPTER,helper="HE_DS",node_id=node_id,first_node_id=first["id"],
                input_value=first["inputs"][0],output_value=last["outputs"][0],input_shape=shape,output_shape=outshape,
                period=period,work=work,calls=calls,row_indices=indices[2],column_indices=indices[3],
                original_slices=[copy.deepcopy(first),copy.deepcopy(last)],
                public_zero_policy="exact_public_zero_intermediates_only_fail_public_only_result",
                transformation="fixed full input tiles; actual upstream downselection, centering and duplication; canonical output maps")

def apply(binding,cipher,helpers,mpcb):
    require(binding["adapter"]==ADAPTER,"DS adapter identity")
    return spatial_mapped.execute_plan(binding,cipher,helpers,mpcb)

def verify_record(binding,record):
    require(binding["adapter"]==ADAPTER,"DS adapter identity")
    spatial_mapped.verify_plan(binding,record)

def probe(binding,x):
    _,ci,hi,wi=binding["input_shape"]
    out=[x[(c*hi+y)*wi+z] for c in range(ci) for y in binding["row_indices"] for z in binding["column_indices"]]
    return tuple(out+[0.]*(binding["period"]-len(out)))
