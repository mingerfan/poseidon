#!/usr/bin/env python3
"""Manual experiment: one extra Q per branch for precise actual-scale alignment.
This is a proposal test, not compiler-generated production behavior.
"""
import hashlib,json,math
from pathlib import Path
base=Path(__file__).resolve().parent;out=base/'precision-prototype';out.mkdir(exist_ok=True)
spec=json.loads((base/'depth9-gpu1/operator-spec.json').read_text());spec_text=json.dumps(spec,indent=2)+'\n';(out/'operator-spec.json').write_text(spec_text)
q=spec['context']['rns_moduli'];ctx=spec['context']['context_id'];host={'kind':'host','rank':0};device={'kind':'device','rank':0,'index':0}
values=[];steps=[];comm=0

def value(kind,place,level,scale,components=2):
 i=str(len(values));values.append(dict(id=i,kind=kind,place=place,context=ctx,level=level,scale_log2=scale,ntt=True,components=components));return i

def instruction(**body):body['ordinal']=len(steps);steps.append(body)
def transfer(i):
 global comm
 src=values[int(i)];o=value(src['kind'],device,src['level'],src['scale_log2'],src['components']);instruction(kind='transfer',transfer_id=str(comm),hint='point_to_point',inputs=[i],outputs=[o],sources=[src['place']],destinations=[device],output_kinds=[src['kind']]);comm+=1;return o

def plain(level,scale,number):
 h=value('plaintext',host,level,scale,1);instruction(kind='encode',payload={'kind':'inline','values':[number]*8192},output=h);return transfer(h)

def mul(a,b):
 av,bv=values[int(a)],values[int(b)];assert av['level']==bv['level'];o=value('ciphertext',device,av['level'],av['scale_log2']+bv['scale_log2']);instruction(kind='compute',op='mul_cp',place=device,inputs=[a,b],output=o);return o

def rescale(a):
 av=values[int(a)];level=av['level']-1;scale=av['scale_log2']-math.log2(q[av['level']]);o=value('ciphertext',device,level,scale);instruction(kind='compute',op='rescale',place=device,inputs=[a],output=o,attrs={'target_level':level,'target_scale_log2':scale});return o

def modswitch(a,level):
 av=values[int(a)];o=value('ciphertext',device,level,av['scale_log2']);instruction(kind='compute',op='mod_switch',place=device,inputs=[a],output=o,attrs={'target_level':level});return o

input_id=value('ciphertext',host,27,40);x=transfer(input_id);a=x;s=40
for n in range(3):
 av=values[int(a)];p=plain(av['level'],40,1.01);a=mul(a,p);s+=40
 while s>=70:a=rescale(a);s-=30
b=modswitch(x,values[int(a)]['level'])
av,bv=values[int(a)],values[int(b)];gap=av['scale_log2']-bv['scale_log2']
a=rescale(mul(a,plain(av['level'],30,1)))
b=rescale(mul(b,plain(bv['level'],30+gap,1)))
av,bv=values[int(a)],values[int(b)];assert abs(av['scale_log2']-bv['scale_log2'])<1e-10
result=value('ciphertext',device,av['level'],av['scale_log2']);instruction(kind='compute',op='add_cc',place=device,inputs=[a,b],output=result)
plan=dict(format_version=5,plan_id='3200',target=dict(target_id=spec['target_id'],world_size=1,device_counts=[1],capability_version=1,operator_spec=dict(id=spec['spec_id'],version=spec['version'],source_sha256='sha256:'+hashlib.sha256(spec_text.encode()).hexdigest())),values=values,external_inputs=[input_id],initialization=[],execution=steps,finalization=[],final_outputs=[result])
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
fixture=dict(format_version=1,model={'name':'manual one-Q high-precision alignment proposal; not production compiler'},context=dict(poly_degree=16384,q_moduli_bits=[30]*50,p_moduli_bits=[30]*8,default_scale_log2=40),tolerances=dict(atol=.001,rtol=.0001),inputs=[dict(name='x',values=[.25])],expected_outputs=[dict(name='out',values=[.25*1.01**3+.25])])
(out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n');print(out)
