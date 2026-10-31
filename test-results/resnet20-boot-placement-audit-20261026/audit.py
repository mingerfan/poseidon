import ast, re, gzip, json, math, hashlib, sys
from pathlib import Path
from functools import lru_cache
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
STRATEGY=sys.argv[1] if len(sys.argv)>1 else 'optimized'
FILE=(ROOT/'test-results/relu-compiler-20261024/double-align/optimized/logical.mlir.gz' if STRATEGY=='latest' else ROOT/f'test-results/model-regression-20261021/server/resnet20-after/{STRATEGY}/logical.mlir.gz')
lines=gzip.decompress(FILE.read_bytes()).decode().splitlines()
ops={}
for i,line in enumerate(lines,1):
    m=re.match(r'\s*(%\d+) = "ckks\.(\w+)"\(([^)]*)\)',line)
    if not m: continue
    result,op,arguments=m.groups()
    value=re.search(r'payload = dense<([^>]+)>',line)
    ops[result]=dict(id=result,op=op,args=re.findall(r'%[\w]+',arguments),scalar=float(value[1]) if value else None,line=i,text=line)
identity={'bootstrapc','rescalec','modswitchc','relinearize'}
@lru_cache(None)
def strip(v):
    o=ops.get(v)
    if o is None:return v
    if o['op'] in identity:return strip(o['args'][0])
    if o['op']=='mulcp' and ops[o['args'][1]]['scalar']==1:return strip(o['args'][0])
    return v
halves={v for v,o in ops.items() if o['op']=='encode' and o['scalar']==0.5}
adds=[v for v,o in ops.items() if o['op']=='addcp' and o['args'][1] in halves]
# Canonical expressions, ignoring scale management and treating x as a leaf.
registry={}
def intern(x):
    if x not in registry:registry[x]=len(registry)
    return registry[x]
class Expr:
    def __init__(self, sig):self.sig=sig
    @staticmethod
    def c(x):return Expr(intern(('c',format(float(x),'.12g'))))
    @staticmethod
    def op(kind,a,b):
        if not isinstance(a,Expr):a=Expr.c(a)
        if not isinstance(b,Expr):b=Expr.c(b)
        if kind=='mul' and a.sig==Expr.c(2).sig:return Expr(intern(('add',b.sig,b.sig)))
        if kind=='mul' and b.sig==Expr.c(2).sig:return Expr(intern(('add',a.sig,a.sig)))
        return Expr(intern((kind,*sorted([a.sig,b.sig]))))
    def __add__(self,b):return self.op('add',self,b)
    def __radd__(self,b):return self.op('add',b,self)
    def __mul__(self,b):return self.op('mul',self,b)
    def __rmul__(self,b):return self.op('mul',b,self)
    def __neg__(self):return Expr(intern(('neg',self.sig)))
    def __sub__(self,b):return self+-b
    def __rsub__(self,b):return b+-self
source=ROOT/'test-results/model-regression-20261021/source/ResNet20.py'
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='component')
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
polybase=ROOT/'third_party/ckks-runtime/third_party/dacapo/python/poly/poly/data'
co=[float(x) for x in (ROOT/'bench/resnet20_s2c_first/support/data/resnet20/relu_param/d13.txt').read_text().split()]
p1=ns['component']((polybase/'tree15.txt').read_text().splitlines(),co[:16],15,4,2.,Expr.c)
p2=ns['component']((polybase/'tree15.txt').read_text().splitlines(),co[16:32],15,4,1.7,Expr.c)
p3=ns['component']((polybase/'tree27.txt').read_text().splitlines(),co[32:],27,8,2.,Expr.c)
relus=[]
for idx,a in enumerate(adds):
    matches=[(v,o) for v,o in ops.items() if o['op']=='mulcc' and a in [strip(x) for x in o['args']]]
    assert len(matches)==1,(a,matches)
    final,o=matches[0]
    x=next(strip(v) for v in o['args'] if strip(v)!=a)
    @lru_cache(None)
    def signature(v):
        v=strip(v)
        if v==x:return Expr(intern(('x',idx)))
        oo=ops[v];op=oo['op'];ar=oo['args']
        if op=='encode':
            if oo['scalar'] is None:raise ValueError('nonuniform plaintext')
            return Expr.c(oo['scalar'])
        if op in ['mulcp','mulcc']:return signature(ar[0])*signature(ar[1])
        if op in ['addcp','addcc']:return signature(ar[0])+signature(ar[1])
        if op=='negatec':return -signature(ar[0])
        raise ValueError(op)
    e1=p1(signature(x));e2=p2(e1);e3=p3(e2)
    expected=signature(x)*(e3+Expr.c(0.5))
    assert signature(final).sig==expected.sig,('signature mismatch',idx,x,final)
    end1=[];end2=[];end3=[]
    for v,oo in ops.items():
        if oo['line']<=ops[x]['line'] or oo['line']>=ops[final]['line']:continue
        try:s=signature(v).sig
        except (ValueError,KeyError):continue
        if s==e1.sig:end1.append(v)
        if s==e2.sig:end2.append(v)
        if s==e3.sig:end3.append(v)
    name='stem.relu' if idx==0 else f'layer{(idx-1)//6+1}.{((idx-1)%6)//2}.relu{1+(idx-1)%2}'
    boots=[v for v,oo in ops.items() if oo['op']=='bootstrapc' and ops[x]['line']<oo['line']<ops[final]['line']]
    classified=[]
    for b in boots:
        try:sig=signature(b).sig
        except (ValueError,KeyError):sig=None
        label={e1.sig:'after_p1',e2.sig:'after_p2',e3.sig:'after_p3',signature(x).sig:'saved_relu_input'}.get(sig,'other_live_value')
        classified.append(dict(id=b,line=ops[b]['line'],where=label,input=ops[b]['args'][0]))
    relus.append(dict(name=name,input=x,final_multiply=final,p1=end1[0],p2=end2[0],p3=end3[0],boots=classified))
claimed={b['id'] for r in relus for b in r['boots']}
rest=[]
for v,o in ops.items():
    if o['op']=='bootstrapc' and v not in claimed:
        prior=[r for r in relus if ops[r['final_multiply']]['line']<o['line']]
        after=[r for r in relus if ops[r['input']]['line']>o['line']]
        rest.append(dict(id=v,line=o['line'],input=o['args'][0],previous_relu=prior[-1]['name'] if prior else None,next_relu=after[0]['name'] if after else None,preceding=lines[o['line']-4:o['line']-1],following=lines[o['line']:o['line']+5]))
# Every extra live value during relu1 must be the enclosing block's shortcut.
for i,r in enumerate(relus):
    for b in r['boots']:
        if b['where']=='other_live_value':
            assert r['name'].endswith('relu1'),r['name']
            residual_input=ops[relus[i+1]['input']]
            assert residual_input['op']=='addcc'
            assert strip(b['input']) in [strip(v) for v in residual_input['args']],(r['name'],b)
            b['where']='saved_residual_shortcut'
        elif b['where']=='saved_relu_input':
            b['where']=('before_relu2' if ops[b['id']]['line']<ops[r['p1']]['line']
                        else 'saved_relu_input_at_polynomial_boundary')
all_boots=[b for r in relus for b in r['boots']]+rest
assert len({b['id'] for b in all_boots})==sum(o['op']=='bootstrapc' for o in ops.values())
for b in all_boots:
    b['instruction']=ops[b['id']]['text'].strip()
    states=re.findall(r'!ckks.poly<(\d+) \* (\d+) \* (\d+)>',b['instruction'])
    b['input_physical_level']=int(states[0][2])+3
    b['output_physical_level']=int(states[-1][2])+3
    b['input_scale_log2']=int(states[0][1])
    b['output_scale_log2']=int(states[-1][1])
print(json.dumps(dict(strategy=STRATEGY, source=str(FILE.relative_to(ROOT)), source_sha256=hashlib.sha256(FILE.read_bytes()).hexdigest(),
    method='Match all 19 ReLU arithmetic expression DAGs to source component(p1,p2,p3), ignoring management operations and scale-alignment multiplication by encoded 1; commutative operand order normalized; constants matched to 12 significant digits. Nonuniform convolution constants are not compared. No CKKS numerical test was run.',
    relus=relus,other_boots=rest,totals=dict(Counter(b['where'] for r in relus for b in r['boots'])),
    total_boots=sum(o['op']=='bootstrapc' for o in ops.values())),indent=2))
