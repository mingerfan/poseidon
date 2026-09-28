"""Frozen spatial corpus and asymmetric manual fixtures; never provider answers."""
import copy
from benchmark_suite import generate,Builder
from benchmark_graph import digest
def corpus():
    return [r['model'] for r in generate() if r['metadata'].get('helper') in ('HE_Conv','HE_Avg','HE_Pool')]

def asymmetric():
    rows=[]
    for ci,h,w,co,k,s,pad in ((1,3,3,1,2,1,0),(2,3,3,2,2,1,0),(2,4,4,2,2,2,1),(1,3,5,2,2,1,0),(3,3,3,3,2,1,0)):
        b=Builder([(1,ci,h,w)])
        weight=[[[[(a*7+j*3+u*2+v-3)/32 for v in range(k)] for u in range(k)] for j in range(ci)] for a in range(co)]
        out=b.node('conv2d',['input0',b.const(weight),b.const([(.125 if a%2 else -.0625) for a in range(co)])],
            stride=[s,s],padding=[pad,pad],dilation=[1,1],groups=1)
        rows.append(b.finish(out))

    for groups,dilation in ((2,1),(1,2)):
        b=Builder([(1,2,3,3)])
        w=[[[[(c*9+j*4+u*2+v-3)/32 for v in range(2)] for u in range(2)] for j in range(2//groups)] for c in range(2)]
        y=b.node('conv2d',['input0',b.const(w),b.const([.1,-.15])],stride=[1,1],padding=[1,1],
                 dilation=[dilation,dilation],groups=groups)
        rows.append(b.finish(y))
    for include in (True,False):
        b=Builder([(1,2,4,4)])
        out=b.node('avg_pool2d',['input0'],kernel=[2,2],stride=[2,2],padding=[1,1],count_include_pad=include)
        rows.append(b.finish(out))
    b=Builder([(1,3,3,3)])
    out=b.node('avg_pool2d',['input0'],kernel=[2,2],stride=[1,1],padding=[0,0],count_include_pad=False)
    rows.append(b.finish(out))
    return rows

def source(model,cap):
    binding_by_node={s['binding']['node_id']:name for name,s in cap['helpers'].items() if 'binding' in s}
    env={entry['name']:('x','y','z','t')[i] for i,entry in enumerate(model['inputs'])}
    lines=['@hc.func("'+','.join(['c']*(len(env)+1))+'")','def golden('+','.join(list(env.values())+['zero_ct'])+'):']
    required=[]
    for i,node in enumerate(model['nodes']):
        lhs='v'+str(i);arg=env[node['inputs'][0]]
        if node['op']=='negate':expr='-'+arg
        elif node['op']=='square':expr=arg+'*'+arg
        else:
            name=binding_by_node[node['id']];expr=name+'('+arg+')';required.append(name)
        lines.append('    '+lhs+' = '+expr);env[node['outputs'][0]]=lhs
    lines.append('    return '+env[model['outputs'][0]['value']])
    return chr(10).join(lines)+chr(10),required

def row(model,name=None):
    from upstream_candidate_helpers import manifest,MAPPED_PROFILE
    g=copy.deepcopy(model)
    if name is not None:g['id']=name
    cap=manifest(g,MAPPED_PROFILE);program,required=source(g,cap)
    return dict(name=g['id'],model=g,model_sha256=digest(g),source=program,required_helpers=required,
        expected_calls=len(required),configuration='seal-cpu-eva-w45-v1',profile=MAPPED_PROFILE)

def cases():
    return [row(g) for g in corpus()]+[row(g,'mapped_asymmetric_'+str(i)) for i,g in enumerate(asymmetric())]

def directed_cases():
    items={g['id']:g for g in corpus()}
    selected=[row(items['bench_helper_'+str(i).zfill(4)]) for i in (16,18,21,48,50,53,64,66)]
    g=copy.deepcopy(items['bench_helper_0064']);previous=g['outputs'][0]['value']
    g['nodes'].append(dict(id='mapped_square',op='square',attrs={},inputs=[previous],outputs=['mapped_square_value']))
    g['outputs'][0]['value']='mapped_square_value';selected.append(row(g,'mapped_pool_then_square'))
    return selected
