"""Mathematical concat fixtures and actual helper programs; no Agent claims."""
from benchmark_suite import Builder,helper_candidates
from benchmark_graph import digest

def cases():
    rows=[]
    def add(name,model,source,helpers=('HE_Concat0',)):
        model['id']=name;rows.append(dict(name=name,model=model,source=source,required_helpers=list(helpers),
                                         model_sha256=digest(model),configuration='seal-cpu-eva-w45-v1'))
    header='@hc.func("c,c,c")\ndef golden(x,y,zero_ct):\n'
    for i,(g,meta) in enumerate((g,m) for g,m in helper_candidates() if m['helper']=='HE_Concat'):
        add('concat_frozen_'+str(i),g,header+'    return HE_Concat0(x,y)\n')
    for i,shape in enumerate([(1,3),(1,2,3),(1,2,2,3),(1,8,4,4)]):
        b=Builder([shape,shape]);out=b.node('concat',['input0','input1'],axis=1)
        add('concat_rank_'+str(i),b.finish(out),header+'    return HE_Concat0(x,y)\n')
    b=Builder([(3,),(3,)]);x=b.node('square',['input0']);h=b.node('concat',[x,'input1'],axis=0);y=b.node('negate',[h])
    add('concat_square_negate',b.finish(y),header+'    return -HE_Concat0(x*x,y)\n')
    b=Builder([(2,),(2,)]);h=b.node('concat',['input0','input1'],axis=0);r=b.node('rotate',[h],step=1)
    add('concat_shared_rotation',b.finish(h,r),header+'    h = HE_Concat0(x,y)\n    return [h,h.rotate(1)]\n')
    b=Builder([(1,2),(1,2)]);h=b.node('concat',['input0','input1'],axis=1)
    weight=[[.5,.25,-.125,.375],[-.25,.5,.25,.125]]
    out=b.node('linear',[h,b.const(weight),b.const([.125,-.125])])
    source=header+'    h = HE_Concat0(x,y)\n    a = (h*0.5+h.rotate(1)*0.25-h.rotate(2)*0.125+h.rotate(1).rotate(2)*0.375+0.125)*mask0\n    b = (h.rotate(1).rotate(2)*(-0.25)+h*0.5+h.rotate(1)*0.25+h.rotate(2)*0.125-0.125)*mask1\n    return a+b\n'
    add('concat_then_linear',b.finish(out),source)
    b=Builder([(3,),(3,)]);h=b.node('concat',['input0','input1'],axis=0)
    add('concat_nested_starred',b.finish(h),'@hc.func("c,c")\ndef join(a,b):\n    return HE_Concat0(a,b)\n'+header+'    a = np.array([x,y],dtype=object)\n    return join(*a)\n')
    return rows
