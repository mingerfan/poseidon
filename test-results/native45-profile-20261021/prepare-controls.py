#!/usr/bin/env python3
import copy
import json
from pathlib import Path

base=Path(__file__).resolve().parent
original=base/'greedy-depth20'
plan=json.loads((original/'result.chains.runtime-plan.json').read_text())
boot=next(s for s in plan['execution'] if s.get('op')=='boot')
prototype=plan['values'][0]
checks=[('full-exact-sparse',49,40.,[.25]),('low-exact-sparse',4,40.,[.25]),
 ('mid-drift-sparse',14,40.14416336611769,[.25]),('low-high-scale-sparse',4,60.37289450508031,[.25]),
 ('low-scale50-sparse',5,50.34771237783504,[.25]),
 ('full-exact-dense',49,40.,[(i%17+1)/32. for i in range(8192)])]
plan.update(values=[],external_inputs=[],initialization=[],execution=[],finalization=[],final_outputs=[])
fixture=json.loads((original/'fixture.json').read_text());fixture['model']['name']='native45 direct encrypted inputs; input state controls'
fixture['inputs']=[];fixture['expected_outputs']=[]
for i,(name,level,scale,values) in enumerate(checks):
 a,b,c=[str(3*i+j) for j in range(3)]
 host=dict(kind='host',rank=0);device=dict(kind='device',rank=0,index=0)
 for id,place,l,s in [(a,host,level,scale),(b,device,level,scale),(c,device,22,45.)]:
  desc=copy.deepcopy(prototype);desc.update(id=id,place=place,level=l,scale_log2=s);plan['values'].append(desc)
 plan['external_inputs'].append(a);plan['final_outputs'].append(c)
 plan['initialization'].append(dict(destinations=[device],hint='point_to_point',inputs=[a],kind='transfer',ordinal=i,output_kinds=['ciphertext'],outputs=[b],sources=[host],transfer_id=str(i)))
 op=copy.deepcopy(boot);op.update(inputs=[b],output=c,ordinal=len(checks)+i);plan['execution'].append(op)
 fixture['inputs'].append(dict(name='input-'+name,values=values));fixture['expected_outputs'].append(dict(name='output-'+name,values=values))
out=base/'direct-input-controls';out.mkdir(exist_ok=True)
(out/'result.chains.runtime-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(out/'fixture.json').write_text(json.dumps(fixture,indent=2)+'\n');(out/'operator-spec.json').write_bytes((original/'operator-spec.json').read_bytes())
(out/'cases.json').write_text(json.dumps([dict(name=n,input_level=l,input_scale_log2=s) for n,l,s,_ in checks],indent=2)+'\n')
