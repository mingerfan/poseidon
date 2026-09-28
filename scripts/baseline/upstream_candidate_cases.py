"""Manual regression fixtures, never model dispatch or provider-visible answers."""
from benchmark_suite import Builder,helper_candidates,coefficient_tables
from benchmark_graph import digest

def polynomial(b,x):
    _,coeff=coefficient_tables()
    y=b.node("polynomial",[x,b.const([v if i%2 else 0.0 for i,v in enumerate(coeff)])],basis="chebyshev")
    y=b.node("add",[y,b.const(0.5)])
    return b.node("multiply",[x,y])

def cases():
    rows=[]
    def add(name,model,source,**meta):
        model['id']=name
        rows.append(dict(name=name,model=model,source=source,model_sha256=digest(model),
                         fixture='manual_not_agent',**meta))
    header='@hc.func("c,c")\ndef golden(x,zero_ct):\n'
    for i,(g,meta) in enumerate((g,m) for g,m in helper_candidates() if m.get('helper')=='HE_SiLU'):
        add('silu_frozen_'+str(i),g,header+'    return HE_SiLU(x)\n',expected_calls=1,ideal='silu')
    b=Builder([(2,),(1,2)])
    y=b.node('reshape',['input1'],shape=[2]);q=b.node('add',['input0',y]);q=b.node('multiply',[q,b.const(.25)])
    h=polynomial(b,q);a=b.node('add',[h,'input0']);z=b.node('subtract',['input0',y])
    add('silu_multi_residual',b.finish(a,z),'@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n    h = HE_SiLU((x+y)*0.25)\n    return [h+x,x-y]\n',expected_calls=1)
    b=Builder([(1,2,2)]);h=polynomial(b,'input0');h=b.node('multiply',[h,b.const(.5)]);h=b.node('add',[h,b.const(.125)])
    add('silu_nested_native',b.finish(h),'@hc.func("c")\ndef f(v):\n    return HE_SiLU(v)\n'+header+'    return f(x)*0.5+0.125\n',expected_calls=1)
    b=Builder([(4,)]);h=polynomial(b,'input0');r=b.node('rotate',[h],step=1)
    add('silu_shared_rotation',b.finish(h,r),header+'    h = HE_SiLU(x)\n    return [h,h.rotate(1)]\n',expected_calls=1)
    b=Builder([(4,)]);h=polynomial(b,'input0')
    add('silu_array_starred',b.finish(h),header+'    a = np.array([x],dtype=object)\n    return HE_SiLU(*a)\n',expected_calls=1)
    b=Builder([(2,)]);z=b.node('square',['input0'])
    add('silu_unused_definition',b.finish(z),'@hc.func("c")\ndef unused(v):\n    return HE_SiLU(v)\n'+header+'    return x*x\n',expected_calls=1,contribution_negative=True)
    return rows
