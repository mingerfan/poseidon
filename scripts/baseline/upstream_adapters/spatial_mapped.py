
"""Exact public gather/scatter plans around real upstream spatial calls.

Mappings transform only layout, fixed zero padding and fixed public coefficients.
They never replace Conv/Avg/Pool with a rule-generated mathematical implementation.
The existing v4 binding remains preferred and byte-identical.
"""
import copy,math
from benchmark_graph import validate,require,canonical
from upstream_adapters import spatial_node
from upstream_adapters.periodic_ring import proxy_type

ADAPTER="windowed-spatial-v1"

def mapping(groups,period):
    """Partition a public sparse map by actual ciphertext rotation displacement."""
    result={}
    for dst,src,factor in groups:
        require(type(dst) is type(src) is int and 0<=dst<period and 0<=src<period,"Spatial map index")
        require(type(factor) in (int,float) and math.isfinite(factor) and abs(factor)<=1024,"Spatial map coefficient")
        step=(src-dst)%period
        values=result.setdefault(step,[0.]*period)
        values[dst]+=factor
    rows=[dict(step=k,mask=v) for k,v in sorted(result.items())]
    require(rows and all(math.isfinite(x) and abs(x)<=1024 for r in rows for x in r['mask']),"Empty or oversized spatial map")
    return rows

def map_work(rows):
    return sum(r['step'].bit_count()+1 for r in rows)+len(rows)-1

def transform(value,rows):
    out=None
    for row in rows:
        term=value.rotate(row['step'])*row['mask']
        out=term if out is None else out+term
    return out

def inner_model(shape,kind,params=None):
    # Minimal declarative inner view; no benchmark generator or DSL answer loaded
    # into the candidate sandbox.
    from benchmark_graph import FORMAT
    params=params or {};constants={};inputs=['input0']
    if kind=='conv':
        weights=params['weight'];fh=len(weights[0][0]);fw=len(weights[0][0][0])
        constants=dict(c0=weights,c1=params['bias']);inputs+=['c0','c1'];op='conv2d'
        attrs=dict(stride=[1,1],padding=[fh//2,fw//2],dilation=[1,1],groups=1)
    elif kind=='pool':op='mean';attrs=dict(axes=[2,3],keepdims=True)
    else:op='avg_pool2d';attrs=dict(kernel=[2,2],stride=[2,2],padding=[0,0],count_include_pad=True)
    return dict(format=FORMAT,id='spatial_inner',inputs=[dict(name='input0',shape=list(shape))],
        constants=constants,nodes=[dict(id='inner',op=op,attrs=attrs,inputs=inputs,outputs=['inner_value'])],
        outputs=[dict(name='output0',value='inner_value')])


def avg_inner(channels,period):
    # A 2x2 closure must call HE_Avg, rather than the full-spatial HE_Pool alias.
    # Geometry follows the same checked InferShapes equations as v4.
    g=dict(nt=period,bb=1.,fh=2,fw=2,s=2,hi=2,wi=2,ki=1,ci=channels,co=channels,
           ho=1,wo=1,ko=2,ti=channels,to=(channels+3)//4,ni=1,no=1,
           pi=period//(1<<(channels*4-1).bit_length()),
           po=period//(4*((channels+3)//4)),q=0)
    # po must use the next power-of-two packed footprint.
    packed=4*((channels+3)//4);g['po']=period//(1<<(packed-1).bit_length())
    require(channels*4<=period and g['pi']>=1 and g['po']>=1,"Window Avg packing exceeds P")
    g['q']=(channels+g['pi']-1)//g['pi'];bits=period.bit_length()-1
    return dict(schema=1,kind='avg',helper='HE_Avg',node_id='inner_avg',input_value='input0',output_value='out',
        input_shape=[1,channels,2,2],output_shape=[1,channels,1,1],period=period,geometry=g,parameters={},
        input_closure_period=period//g['pi'],output_positions=list(range(channels)),
        work=4+4*bits+4*(bits+2)+channels*(bits+2))

def bind_node(model,node_id,period):
    try:return spatial_node.bind_node(model,node_id,period)
    except ValueError as error:old_reason=str(error)
    checked=validate(model);node=next(n for n in model['nodes'] if n['id']==node_id)
    shape=checked['shapes'][node['inputs'][0]];outshape=checked['shapes'][node['outputs'][0]]
    require(len(shape)==4 and shape[0]==1,"Mapped spatial requires batch-one rank4")
    require(period in (4,8,16,32,64,128,256) and type(period) is int,"Mapped spatial period")
    require(max(math.prod(shape),math.prod(outshape))<=period,"Mapped spatial intermediate exceeds P")
    _,ci,hi,wi=shape;op=node['op'];a=node['attrs'];calls=[]
    if op=='mean':
        require({x%4 for x in a['axes']}=={2,3},"Mapped HE_Pool requires spatial axes")
        family='HE_Pool';size=1<<(hi*wi-1).bit_length()
        g=inner_model([1,1,1,size],'pool');inner=spatial_node.bind_node(g,g['nodes'][0]['id'],period)
        for c in range(ci):
            before=mapping([(i,c*hi*wi+i,1.) for i in range(hi*wi)],period)
            after=mapping([(c,0,size/(hi*wi))],period)
            calls.append(dict(inner=inner,before=before,after=after))
    elif op=='avg_pool2d':
        require(a['kernel']==[2,2],"Mapped HE_Avg currently requires 2x2 kernel")
        family='HE_Avg';co,ho,wo=outshape[1:];inner=avg_inner(ci,period)
        sy,sx=a['stride'];py,px=a['padding']
        for h in range(ho):
            for w in range(wo):
                cells=[];valid=0
                for u in range(2):
                    for v in range(2):
                        y,x=h*sy-py+u,w*sx-px+v
                        if 0<=y<hi and 0<=x<wi:
                            valid+=1
                            for c in range(ci):cells.append((c*4+u*2+v,(c*hi+y)*wi+x,1.))
                divisor=4 if a['count_include_pad'] else valid
                require(divisor>0,"Mapped average empty divisor")
                before=mapping(cells,period)
                after=mapping([(c*ho*wo+h*wo+w,c,4/divisor) for c in range(ci)],period)
                calls.append(dict(inner=inner,before=before,after=after))
    elif op=='conv2d':
        original_weight=copy.deepcopy(model['constants'][node['inputs'][1]])
        co=len(original_weight);raw_h=len(original_weight[0][0]);raw_w=len(original_weight[0][0][0])
        dy,dx=a['dilation'];groups=a['groups'];in_group=ci//groups;out_group=co//groups
        kh=(raw_h-1)*dy+1;kw=(raw_w-1)*dx+1
        require(1<=kh<=3 and 1<=kw<=3,"Mapped HE_Conv effective kernels up to 3")
        # Group separation and dilation add exact zero coefficients only.
        weight=[[[[original_weight[c][j%in_group][u//dy][v//dx]
                   if j//in_group==c//out_group and u%dy==0 and v%dx==0 else 0.
                   for v in range(kw)] for u in range(kh)] for j in range(ci)] for c in range(co)]
        family='HE_Conv';fh=kh if kh%2 else kh+1;fw=kw if kw%2 else kw+1
        # Add only zero coefficients at the end of even kernels. Select the
        # corresponding centered output position, not a changed convolution.
        padded=[[[[weight[c][j][u][v] if u<kh and v<kw else 0. for v in range(fw)]
                   for u in range(fh)] for j in range(ci)] for c in range(co)]
        bias=copy.deepcopy(model['constants'][node['inputs'][2]]) if len(node['inputs'])==3 else [0.]*co
        sy,sx=a['stride'];py,px=a['padding'];ho,wo=outshape[2:]
        choices=[]
        for th in (1,2,4,8,16,32,64,128,256):
            for tw in (1,2,4,8,16,32,64,128,256):
                if th<max(kh,fh//2+1) or tw<max(kw,fw//2+1) or max(ci,co)*th*tw>period:continue
                cy=(th-kh)//sy+1;cx=(tw-kw)//sx+1
                choices.append(((ho+cy-1)//cy*((wo+cx-1)//cx),th*tw,th,tw,cy,cx))
        require(choices,"Mapped Conv tile does not fit P")
        _,_,th,tw,cy,cx=min(choices)
        g=inner_model([1,ci,th,tw],'conv',dict(weight=padded,bias=bias))
        inner=spatial_node.bind_node(g,g['nodes'][0]['id'],period)
        for h0 in range(0,ho,cy):
            for w0 in range(0,wo,cx):
                cells=[];origin_y=h0*sy-py;origin_x=w0*sx-px
                for c in range(ci):
                    for u in range(th):
                        for v in range(tw):
                            y,x=origin_y+u,origin_x+v
                            if 0<=y<hi and 0<=x<wi:cells.append(((c*th+u)*tw+v,(c*hi+y)*wi+x,1.))
                targets=[]
                for c in range(co):
                    for h in range(h0,min(ho,h0+cy)):
                        for w in range(w0,min(wo,w0+cx)):
                            src=(c*th+(h-h0)*sy+fh//2)*tw+(w-w0)*sx+fw//2
                            targets.append(((c*ho+h)*wo+w,src,1.))
                calls.append(dict(inner=inner,before=mapping(cells,period),after=mapping(targets,period)))
    else:raise ValueError("No mapped spatial helper")
    require(1<=len(calls)<=64,"Mapped spatial invocation budget")
    work=sum(c['inner']['work']+map_work(c['before'])+map_work(c['after']) for c in calls)+len(calls)-1
    require(work<=1024,"Mapped spatial reserved work exceeds 1024")
    return dict(schema=1,adapter=ADAPTER,helper=family,node_id=node_id,input_value=node['inputs'][0],
        output_value=node['outputs'][0],input_shape=shape,output_shape=outshape,period=period,work=work,
        original=dict(op=op,attrs=copy.deepcopy(a),weight=copy.deepcopy(model['constants'][node['inputs'][1]]) if op=='conv2d' else None,
                      bias=copy.deepcopy(model['constants'][node['inputs'][2]]) if op=='conv2d' and len(node['inputs'])==3 else None),
        calls=calls,v4_blocker=old_reason,public_zero_policy='exact_public_zero_intermediates_only_fail_public_only_result',transformation='fixed public sparse layout maps around unchanged upstream calls')


def zero_aware_proxy(expr_class):
    """Elide exact public zero intermediates; never create a transparent cipher.

    No near-zero threshold, encrypted-value inspection or decrypt/re-encrypt.
    A public-only final result is rejected by the ordinary result boundary.
    Existing upstream methods and v4 Expr behavior are not modified.
    """
    cls=proxy_type(expr_class);base_binary=cls.binary;base_rotate=cls.rotate
    def is_zero(value):
        return value is None or type(value) in (int,float) and value==0 or type(value) is list and all(v==0 for v in value)
    def folded(self,op):
        self.spend(op);self.record.setdefault('public_zero_eliminations',[]).append(dict(index=len(self.record['operations'])-1,operation=op))
    def binary(self,other,operation,reverse=False):
        value=self.other(other);left,right=(value,self.value) if reverse else (self.value,value)
        zl,zr=is_zero(left),is_zero(right)
        if operation=='multiply' and (zl or zr):
            folded(self,operation);return type(self)(None,self.period,self.record,self.limit)
        if operation in ('add','subtract') and (zl or zr):
            folded(self,operation)
            if zl and zr:result=None
            elif operation=='subtract' and zl:
                require(not isinstance(right,list) and type(right) not in (int,float),"Public-only helper expression")
                result=-right
            else:result=right if zl else left
            require(result is None or not isinstance(result,list) and type(result) not in (int,float),"Public-only helper expression")
            return type(self)(result,self.period,self.record,self.limit)
        return base_binary(self,other,operation,reverse)
    def rotate(self,step):
        if self.value is not None:return base_rotate(self,step)
        require(type(step) is int,"Upstream rotation must be integral")
        n=step%self.period;steps=[1<<j for j in range(self.period.bit_length()-1) if n&(1<<j)]
        self.record['rotations'].append(dict(requested=step,normalized=n,steps=steps))
        for part in steps:folded(self,'rotate')
        return type(self)(None,self.period,self.record,self.limit)
    cls.binary=binary;cls.rotate=rotate
    return cls

def apply(binding,cipher,helpers,mpcb):
    if binding.get('adapter')!=ADAPTER:return spatial_node.apply(binding,cipher,helpers,mpcb)
    return execute_plan(binding,cipher,helpers,mpcb)

def execute_plan(binding,cipher,helpers,mpcb):
    p=binding['period'];record=dict(operations=[],rotations=[])
    proxy=proxy_type(helpers.hc.Expr);x=proxy(cipher,p,record,binding['work']);out=None;actual=[]
    for call in binding['calls']:
        mapped=transform(x,call['before'])
        value,observation=spatial_node.apply(call['inner'],mapped.value,helpers,mpcb,proxy_factory=zero_aware_proxy)
        require(observation['helper']==binding['helper'],"Mapped helper family changed")
        actual.append(observation);term=transform(proxy(value,p,record,binding['work']),call['after'])
        out=term if out is None else out+term
    count=len(record['operations'])+sum(c['operation_count'] for c in actual)
    require(count<=binding['work'],"Mapped helper actual work")
    return out.value,dict(adapter=binding['adapter'],helper=binding['helper'],inner_calls=actual,
        mapping_operations=len(record['operations']),mapping_rotations=record['rotations'],operation_count=count,
        unchanged_upstream_functions=True,bootstrap_removed=False,output_contribution_proven=False)

def verify_record(binding,record):
    if binding.get('adapter')!=ADAPTER:return spatial_node.verify_record(binding,record)
    return verify_plan(binding,record)

def verify_plan(binding,record):
    require(type(record) is dict and set(record)=={'adapter','helper','inner_calls','mapping_operations','mapping_rotations','operation_count',
        'unchanged_upstream_functions','bootstrap_removed','output_contribution_proven'},"Mapped record fields")
    require(record['adapter']==binding['adapter'] and record['helper']==binding['helper'] and
        record['unchanged_upstream_functions'] is True and record['bootstrap_removed'] is False and
        record['output_contribution_proven'] is False,"Mapped actual identity")
    require(type(record['inner_calls']) is list and len(record['inner_calls'])==len(binding['calls']),"Mapped actual call count")
    count=record['mapping_operations']
    expected=sum(map_work(c['before'])+map_work(c['after']) for c in binding['calls'])+len(binding['calls'])-1
    require(type(count) is int and count==expected,"Mapped actual mapping work")
    rotations=[dict(requested=r['step'],normalized=r['step'],
        steps=[1<<i for i in range(binding['period'].bit_length()-1) if r['step']&(1<<i)])
        for c in binding['calls'] for side in ('before','after') for r in c[side]]
    require(canonical(record['mapping_rotations'])==canonical(rotations),"Mapped actual rotations")
    for call,actual in zip(binding['calls'],record['inner_calls']):
        core=dict(actual);eliminated=core.pop('public_zero_eliminations')
        require(type(eliminated) is list and len(eliminated)<=core['operation_count'],"Public zero record budget")
        indices=[]
        for item in eliminated:
            require(type(item) is dict and set(item)=={'index','operation'} and type(item['index']) is int and
                    0<=item['index']<core['operation_count'] and item['operation'] in ('add','subtract','multiply','rotate'),
                    "Public zero record fields")
            indices.append(item['index'])
        require(indices==sorted(set(indices)),"Public zero record order")
        spatial_node.verify_record(call['inner'],core);count+=actual['operation_count']
    require(type(record['operation_count']) is int and record['operation_count']==count<=binding['work'],"Mapped actual total work")

def probe(binding,x):
    """Independent finite public influence evaluator, not a reference oracle."""
    p=binding['period'];_,ci,hi,wi=binding['input_shape'];orig=binding['original'];a=orig['attrs'];op=orig['op']
    if op=='mean':out=[sum(x[c*hi*wi:(c+1)*hi*wi])/(hi*wi) for c in range(ci)]
    else:
        co,ho,wo=binding['output_shape'][1:]
        sy,sx=a['stride'];py,px=a['padding']
        kh,kw=(len(orig['weight'][0][0]),len(orig['weight'][0][0][0])) if op=='conv2d' else a['kernel']
        out=[]
        for c in range(co):
            for h in range(ho):
                for w in range(wo):
                    total=(orig['bias'][c] if orig['bias'] is not None else 0.) if op=='conv2d' else 0.
                    valid=0
                    for j in (range((c//(co//a['groups']))*(ci//a['groups']), (c//(co//a['groups'])+1)*(ci//a['groups'])) if op=='conv2d' else [c]):
                        for u in range(kh):
                            for v in range(kw):
                                dy,dx=a['dilation'] if op=='conv2d' else (1,1)
                                y,z=h*sy-py+u*dy,w*sx-px+v*dx
                                if 0<=y<hi and 0<=z<wi:
                                    valid+=1;v0=x[(j*hi+y)*wi+z]
                                    total+=v0*orig['weight'][c][j%(ci//a['groups'])][u][v] if op=='conv2d' else v0
                    if op!='conv2d':total/=kh*kw if a['count_include_pad'] else valid
                    out.append(total)
    return tuple(out+[0.]*(p-len(out)))
