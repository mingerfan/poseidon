import json, math, sys
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parent
spec=json.load(open(sys.argv[1]));primes=spec['context']['rns_moduli'];report={}
for beam in (4,32):
 p=json.load(open(root/f'paired-8-runtime-beam{beam}.graph.runtime-plan.json'))
 desc={v['id']:v for v in p['values']};values=dict(zip(p['external_inputs'],(.8,.3)))
 counts=Counter();max_scale_error=0.
 for phase in ('initialization','execution','finalization'):
  for s in p[phase]:
   kind=s['kind']
   if kind=='release':continue
   op=s.get('op',kind);counts[op]+=1
   if kind=='encode':
    payload=s['payload']['values'];assert all(v==payload[0] for v in payload)
    out=s['output'];scale=desc[out]['scale_log2'];delta=2.**scale
    values[out]=round(payload[0]*delta)/delta
    continue
   out=s['outputs'][0] if kind=='transfer' else s['output'];ins=s['inputs'];a=desc[ins[0]];b=desc[out]
   scale=a['scale_log2'];level=a['level'];value=values[ins[0]]
   if op.startswith(('add_','mul_')):
    second=desc[ins[1]];assert level==second['level']
    if op.startswith('mul_'):scale+=second['scale_log2'];value*=values[ins[1]]
    else:assert abs(scale-second['scale_log2'])<1e-8;value+=values[ins[1]]
   elif op=='rescale':scale-=math.log2(primes[level]);level-=1
   elif op=='mod_switch':assert b['level']<level;level=b['level']
   elif op=='boot':
    bp=spec['boot_profiles'][0];assert bp['input_level_min']<=level<=bp['input_level_max']
    if 'input_scale_log2' in bp: assert abs(scale-bp['input_scale_log2'])<1e-8
    level=bp['output_level'];scale=bp['output_scale_log2']
   else:assert op in ('transfer','copy','relinearize','rotate'),op
   max_scale_error=max(max_scale_error,abs(scale-b['scale_log2']));assert max_scale_error<1e-8
   assert level==b['level']
   if b['kind']=='ciphertext':assert level>=spec['levels']['lower_bound'] and 40-1e-8<=scale<sum(math.log2(q) for q in primes[:level+1])
   values[out]=value
 r,y=.8,.3
 for _ in range(8):r=r*r*(.25*r+.75);y=y*(.5*r+1)
 out=p['final_outputs'][0];err=abs(values[out]-y);assert err<1e-9
 assert abs(desc[out]['scale_log2']-40)<1e-8
 report[f'beam{beam}']=dict(ops=dict(counts),expected=y,actual=values[out],absolute_error=err,output=desc[out],max_scale_error=max_scale_error)
# Construct a 0-Boot schedule for the two-output linear probe, independent of the planner.
removed=sum(math.log2(q) for q in primes[21:23]);plain_scale=removed
raw_scale=40+plain_scale;out_scale=raw_scale-removed
assert plain_scale>=40 and out_scale==40 and raw_scale+12<sum(math.log2(q) for q in primes[:23])
report['linear_oracle']=dict(input_q=23,input_scale=40,plaintext_scale=plain_scale,product_scale=raw_scale,output_q=21,output_scale=out_scale,boots=0,total_rescales_two_outputs=4,scope='Scale/level feasibility plus uniform scalar plaintext rounding; no encrypted execution')
for terms in (64,256,1024):
 errs=[]
 for branch in range(2):
  weights=[(1+branch*.2+j/(terms*10))/terms for j in range(terms)]
  delta=2**plain_scale;quant=[round(w*delta)/delta for w in weights]
  errs.append(abs(sum(quant)-sum(weights)))
 assert max(errs)<1e-12
 report['linear_oracle'][str(terms)]=dict(max_uniform_input_error=max(errs),sum_abs_weights=[sum((1+branch*.2+j/(terms*10))/terms for j in range(terms)) for branch in range(2)])
report['limitations']='Boot is identity; full-slot constants are rounded, but ciphertext encryption, RS rounding and Boot approximation errors are not simulated.'
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
