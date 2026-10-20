"""Compare model arithmetic while ignoring verified CKKS identity bookkeeping."""
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'resnet18-model-v2'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def signature(path):
    plan = json.loads(path.read_text())
    values = {id:digest(['input',index]) for index,id in enumerate(plan['external_inputs'])}
    one_values = set()
    ones = 'sha256:'+hashlib.sha256(struct.pack('<d',1.0)*32768).hexdigest()
    operations = Counter()
    for phase in ('initialization','execution','finalization'):
        for instruction in plan[phase]:
            kind = instruction['kind']
            if kind in ('transfer','replicate'):
                source, = instruction['inputs']
                for output in instruction['outputs']:
                    values[output] = values[source]
                    if source in one_values:one_values.add(output)
            elif kind=='encode':
                output = instruction['output']
                values[output] = digest(['constant',instruction['payload']])
                if instruction['payload'].get('content')==ones:one_values.add(output)
            elif kind=='compute':
                op,inputs,output = instruction['op'],instruction['inputs'],instruction['output']
                operations[op] += 1
                if op in ('boot','rescale','mod_switch','relinearize'):
                    if op=='boot':assert instruction['attrs']['implementation']=='native'
                    values[output] = values[inputs[0]]
                elif op=='mul_cp' and inputs[1] in one_values:
                    values[output] = values[inputs[0]]
                else:
                    operands = [values[id] for id in inputs]
                    if op in ('add_cc','mul_cc'):operands.sort()
                    values[output] = digest([op,instruction.get('attrs',{}),operands])
            else:
                assert kind in ('release','fence'),kind
    return [values[id] for id in plan['final_outputs']],operations


if __name__=='__main__':
    oldpath, = (BASE/'compiled-jit').glob('*.runtime-plan.json')
    newpath, = (ROOT/'compiled-full').glob('*.runtime-plan.json')
    before,old_counts = signature(oldpath)
    after,new_counts = signature(newpath)
    assert before==after,'compiled logical model changed beyond CKKS identity bookkeeping'
    report = dict(status='PASS',logical_model_unchanged=True,logical_output_sha256=before,
                  old_operations=old_counts,new_operations=new_counts,
                  ignored='Native identity refresh, modulus/rescale/relinearize and multiply by literal all-ones; original GPU error gates remain mandatory')
    with (ROOT/'arithmetic-audit.json').open('x') as output:
        json.dump(report,output,indent=2)
    print(json.dumps(report,indent=2))
