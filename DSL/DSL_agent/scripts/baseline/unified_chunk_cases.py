"""Small arbitrary-DAG contexts for the explicit chunk ABI, no expected answers."""
from benchmark_suite import Builder

def cases():
    rows=[]
    def add(name,b,value,period=4,source=None,profile=None,exercise=None):
        rows.append(dict(name=name,model=b.finish(*(value if type(value) is list else [value])),
                         period=period,source=source,profile=profile,exercise=exercise))
    b=Builder([(9,)]);add("tail_negate",b,b.node("negate",["input0"]))
    b=Builder([(9,)]);add("whole_rotate",b,b.node("rotate",["input0"],step=2))
    b=Builder([(9,)]);parts=b.node("split",["input0"],axis=0,sections=[4,5]);add("split_reorder",b,list(reversed(parts)))
    b=Builder([(9,)]);w=b.const([[((i+2*j)%5-2)/8 for j in range(9)] for i in range(3)])
    add("linear_cross_chunks",b,b.node("linear",["input0",w,b.const([.1,-.2,.3])]))
    b=Builder([(5,)]);x=b.node("linear",["input0",b.const([[.125*(i+j-2) for j in range(5)] for i in range(3)])])
    x=b.node("square",[x]);add("mlp_cross_chunks",b,b.node("linear",[x,b.const([[.25,-.125,.5],[-.25,.5,.125]])]))
    b=Builder([(5,),(1,5)]);x=b.node("reshape",["input1"],shape=[5]);x=b.node("add",["input0",x])
    add("different_shape_residual",b,[x,b.node("sum",[x],axes=[0],keepdims=False)])
    b=Builder([(1,1,9)]);w=b.const([[[.25,-.5,.25]]]);x=b.node("conv1d",["input0",w],stride=[1],padding=[0],dilation=[1],groups=1)
    add("conv1d_cross_chunks",b,x)
    b=Builder([(1,1,3,3)]);w=b.const([[[[.25,-.25],[.5,.125]]]])
    x=b.node("conv2d",["input0",w],stride=[1,1],padding=[0,0],dilation=[1,1],groups=1)
    add("conv2d_cross_chunks",b,x)
    b=Builder([(1,2,3)]);x=b.node("batch_norm",["input0",b.const([.1,-.2]),b.const([.5,.25]),b.const([.75,-.5]),b.const([0.,.2])],eps=1e-5)
    add("batchnorm_pool",b,b.node("avg_pool1d",[x],kernel=[2],stride=[1],padding=[0],count_include_pad=True))
    b=Builder([(9,),(9,)]);x=b.node("concat",["input0","input1"],axis=0)
    add("concat_large_intermediate",b,b.node("sum",[x],axes=[0],keepdims=False),8)
    b=Builder([(1,2,2,2)]);x=b.node("permute",["input0"],dims=[0,3,1,2]);add("rank4_permute",b,x)
    b=Builder([(3,)]);w=b.const([[((i+j)%3-1)/4 for j in range(3)] for i in range(9)])
    add("expanded_output",b,b.node("linear",["input0",w]))
    b=Builder([(8,)]);add("four_named_outputs",b,b.node("split",["input0"],axis=0,sections=[2,2,2,2]))
    b=Builder([(256,)]);add("maximum_total_elements",b,b.node("negate",["input0"]),64,
        '@hc.func("c,c,c,c,c")\ndef golden(x,y,z,t,zero_ct):\n return [-x,-y,-z,-t]\n')
    for name,index,exercise,profile in (
        ("native_star_chunks",0,"unified-star",None),
        ("native_loop_chunks",3,"unified-loop",None),
        ("native_view_chunks",10,"unified-view",None)):
        row=dict(rows[index],name=name,exercise=exercise,profile=profile);rows.append(row)
    from unified_public_exercises import SPECS
    key=next(k for k,v in SPECS.items() if v[0]=="public.loop")
    rows.append(dict(rows[0],name="public_loop_chunks",exercise=key,profile="hecate-unified-public-v1"))
    return rows

def source_for(row,request):
    from unified_graph_lowering import lower
    source=row["source"] or lower(request)
    if row.get("exercise"):
        if row.get("profile"):
            from unified_public_exercises import golden_variant
            return golden_variant(source,row["exercise"],request=request)
        from unified_graph_exercises import golden_variant
        return golden_variant(source,row["exercise"])
    return source
