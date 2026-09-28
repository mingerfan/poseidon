"""Small spatial fixtures; helper failures remain declared, not removed."""
from benchmark_suite import Builder
from benchmark_graph import digest

def models():
    rows=[]
    for c,h,w in ((1,2,2),(1,4,4),(2,4,4),(3,2,4)):
        for kernel,stride in ((1,1),(3,1),(3,2)):
            b=Builder([(1,c,h,w)]);co=2
            weights=[[[[((i+j+u+v)%5-2)/32 for v in range(kernel)] for u in range(kernel)] for j in range(c)] for i in range(co)]
            out=b.node("conv2d",["input0",b.const(weights),b.const([.1,-.2])],
                stride=[stride]*2,padding=[kernel//2]*2,dilation=[1,1],groups=1)
            rows.append(("conv_"+"_".join(map(str,(c,h,w,kernel,stride))),b.finish(out)))
        for kernel,stride,pad in ((2,2,0),(3,1,1),(3,2,1)):
            b=Builder([(1,c,h,w)])
            out=b.node("avg_pool2d",["input0"],kernel=[kernel]*2,stride=[stride]*2,padding=[pad]*2,count_include_pad=True)
            rows.append(("avg_"+"_".join(map(str,(c,h,w,kernel,stride))),b.finish(out)))
        b=Builder([(1,c,h,w)])
        out=b.node("avg_pool2d",["input0"],kernel=[h,w],stride=[h,w],padding=[0,0],count_include_pad=True)
        rows.append(("pool_"+"_".join(map(str,(c,h,w))),b.finish(out)))
    return rows

def cases():
    rows=[]
    for name,model in models():
        model['id']=name
        helper=('HE_Conv0' if name.startswith('conv') else 'HE_Pool0' if name.startswith('pool') or name=='avg_1_2_2_2_2' else 'HE_Avg0')
        # A local window covering the entire input binds the genuine global helper.
        if name.startswith('avg'):
            from upstream_adapters.spatial_node import bind_node
            from unified_graph_contract import layout
            try:helper=bind_node(model,model['nodes'][0]['id'],layout(model)['input_slot_period'])['helper']+'0'
            except ValueError:pass
        rows.append(dict(name=name,model=model,source='@hc.func("c,c")\ndef golden(x,zero_ct):\n    return '+helper+'(x)\n',
            model_sha256=digest(model),required_helpers=[helper],configuration='seal-cpu-eva-w45-v1',profile='upstream-poly-spatial-v4'))
    return rows

def directed_cases():
    import copy
    rows=[]
    seeds=('conv_1_2_2_1_1','avg_1_4_4_2_2','pool_2_4_4')
    for seed in seeds:
        base=next(r for r in cases() if r['name']==seed)
        rows.append(base)
        for context in ('pre_negate','post_square'):
            row=copy.deepcopy(base);g=row['model'];node=g['nodes'][0]
            if context=='pre_negate':
                g['nodes'].insert(0,dict(id='spatial_pre',op='negate',inputs=['input0'],outputs=['spatial_pre_value'],attrs={}))
                node['inputs'][0]='spatial_pre_value'
                row['source']=row['source'].replace('(x)\n','(-x)\n')
            else:
                value=node['outputs'][0]
                g['nodes'].append(dict(id='spatial_post',op='square',inputs=[value],outputs=['spatial_post_value'],attrs={}))
                g['outputs'][0]['value']='spatial_post_value'
                call=row['required_helpers'][0]+'(x)'
                row['source']=row['source'].replace('    return '+call,'    h = '+call+'\n    return h*h')
            row['name']=seed+'_'+context;g['id']=row['name'];row['model_sha256']=digest(g);rows.append(row)
    return rows
