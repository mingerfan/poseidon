from pathlib import Path
import json
import math
import sys

root = Path('/home/xuming/poseidon-manual-boot-20261026/resnet20-optimized-20261027T055506Z/placement-audit/variable-chain32/gpu')
name = sys.argv[1]
plan = json.loads(next((root/(name+'-artifacts')).glob('*.runtime-plan.json')).read_text())
desc = {v['id']:v for v in plan['values']}
samples = (-.75,-.5,-.25,-.125,0.,.125,.25,.5,.75,.875)
plain = {}
for op in plan['initialization']:
    if op['kind'] == 'encode':
        values = op['payload']['values']
        assert all(v == values[0] for v in values)
        c = values[0]
        scale = 2**desc[op['output']]['scale_log2']
        plain[op['output']] = dict(intended=c,quantized=round(c*scale)/scale,scale_log2=desc[op['output']]['scale_log2'])


def replay(x,quantized):
    v = {oid:x for oid in plan['external_inputs']}
    for op in plan['initialization']+plan['execution']+plan['finalization']:
        kind = op.get('op',op['kind'])
        output = op.get('outputs',[op['output']] if 'output' in op else [])
        if not output:
            continue
        oid = output[0]
        if kind == 'encode':
            v[oid] = plain[oid]['quantized' if quantized else 'intended']
            continue
        args = [v[a] for a in op['inputs']]
        if kind in ('mul_cc','mul_cp'):
            v[oid] = args[0]*args[1]
        elif kind in ('add_cc','add_cp'):
            v[oid] = args[0]+args[1]
        elif kind == 'negate':
            v[oid] = -args[0]
        else:
            assert kind in ('transfer','mod_switch','rescale','relinearize')
            v[oid] = args[0]
    return v[plan['final_outputs'][0]]


result = dict(name=name,model='Constant plaintext coefficient rounding only; no encryption/rescale rounding or noise modeled',samples=[dict(input=x,intended=replay(x,False),quantized=replay(x,True)) for x in samples],low_scale_units=[dict(id=oid,**p) for oid,p in plain.items() if p['intended']==1 and p['scale_log2']<10])
gpu = json.loads((root/(name+'-summary.json')).read_text())
if 'samples' in gpu:
    result['max_abs_gpu_vs_quantized_model'] = max(abs(g-a['quantized']) for g,a in zip(gpu['samples']['actual'],result['samples']))
    full = json.loads((root/(name+'-report.json')).read_text())
    result['max_abs_gpu_vs_quantized_model_all_slots'] = max(abs(value-result['samples'][i%len(samples)]['quantized']) for i,value in enumerate(full['actual']))
result['max_abs_quantization_error'] = max(abs(a['intended']-a['quantized']) for a in result['samples'])
(root/(name+'-quantization.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='low_scale_units'},ensure_ascii=False),flush=True)
