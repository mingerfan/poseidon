"""Manual models/candidates for actual bound BN; no family-based runtime lowering."""
from benchmark_suite import Builder
from benchmark_graph import digest
from upstream_adapters.scenarios import supplemental_rows

def bn(b,x):
    c=b.shapes[x][1]
    return b.node('batch_norm',[x,b.const([.03125*(i+1) for i in range(c)]),b.const([1.+.125*i for i in range(c)]),
                               b.const([.5-.0625*i for i in range(c)]),b.const([.015625*(i-1) for i in range(c)])],eps=1e-5)

def cases():
    rows=[];header='@hc.func("c,c")\ndef golden(x,zero_ct):\n'
    def add(name,g,source,calls=1,negative=False):
        g['id']=name;rows.append(dict(name=name,model=g,source=source,model_sha256=digest(g),expected_calls=calls,
                                     contribution_negative=negative,fixture='manual_not_agent'))
    for i,row in enumerate(supplemental_rows()):
        add('bn_asymmetric_'+str(i),row['model'],header+'    return HE_BN0(x)\n')
    b=Builder([(1,2,2),(1,2)])
    q=b.node('square',['input0']);x=bn(b,q);x=b.node('add',[x,'input0']);y=bn(b,'input1')
    add('bn_multi_residual',b.finish(x,y),'@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n    return [HE_BN0(x*x)+x,HE_BN1(y)]\n',2)
    b=Builder([(1,2)])
    h=bn(b,'input0');out=b.node('linear',[h,b.const([[.5,.25],[-.25,.5]]),b.const([.125,-.125])])
    reduction='    a = (h*0.5+h.rotate(1)*0.25+0.125)*mask0\n    b = (h.rotate(1).rotate(2)*(-0.25)+h*0.5-0.125)*mask1\n'
    add('bn_then_linear',b.finish(out),header+'    h = HE_BN0(x)\n'+reduction+'    return a+b\n')
    b=Builder([(1,2)]);h=b.node('linear',['input0',b.const([[.5,.25],[-.25,.5]]),b.const([.125,-.125])]);out=bn(b,h)
    add('linear_then_bn',b.finish(out),header+'    h = x\n'+reduction+'    return HE_BN0(a+b)\n')
    b=Builder([(1,2,2)]);h=bn(b,'input0')
    add('bn_nested_native',b.finish(h),'@hc.func("c")\ndef inner(v):\n    return HE_BN0(v)\n'+header+'    return inner(x)\n')
    b=Builder([(1,2,2)]);h=bn(b,'input0')
    add('bn_array_starred',b.finish(h),header+'    a = np.array([x],dtype=object)\n    return HE_BN0(*a)\n')
    b=Builder([(1,2)]);h=bn(b,'input0');h=b.node('square',[h])
    add('bn_public_loop',b.finish(h),header+'    h = x\n    for i in range(1):\n        h = HE_BN0(h)\n    return h*h\n')
    b=Builder([(1,2)]);bn(b,'input0');h=b.node('square',['input0'])
    # The graph retains an unused BN node and the candidate an unused definition.
    add('bn_unused_definition',b.finish(h),'@hc.func("c")\ndef unused(v):\n    return HE_BN0(v)\n'+header+'    return x*x\n',negative=True)
    return rows
