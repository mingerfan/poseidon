"""Check that deferred encoding changes no arithmetic, payload or CKKS type."""
import argparse
import hashlib
import json
from pathlib import Path


def signature(path):
    p = json.loads(path.read_text())
    phases = [*p['initialization'], *p['execution'], *p['finalization']]
    aliases = {}
    for instruction in phases:
        if instruction['kind'] == 'transfer':
            for output,input_id in zip(instruction['outputs'],instruction['inputs']):
                aliases[output] = input_id
    def logical(value):
        while value in aliases:
            value = aliases[value]
        return value
    computes = []
    for instruction in phases:
        if instruction['kind'] == 'compute':
            body = {k:v for k,v in instruction.items() if k not in ('ordinal','reuse_input')}
            body['inputs'] = [logical(v) for v in body['inputs']]
            computes.append(body)
    constants = {i['output']:i['payload'] for i in phases if i['kind']=='encode'}
    values = {v['id']:v for v in p['values'] if v['id'] not in aliases}
    transferred_types = sorted(json.dumps(dict(source=logical(v['id']),
        type={k:x for k,x in v.items() if k != 'id'}),sort_keys=True)
        for v in p['values'] if v['id'] in aliases)
    return dict(computes=computes,constants=constants,values=values,transferred_types=transferred_types,
                inputs=p['external_inputs'],outputs=p['final_outputs'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('eager',type=Path)
    parser.add_argument('deferred',type=Path)
    args = parser.parse_args()
    eager,deferred = signature(args.eager),signature(args.deferred)
    for field in eager:
        assert eager[field] == deferred[field],f'{field} changed beyond encoding lifetime'
    print(json.dumps(dict(status='PASS',compute_instructions=len(eager['computes']),
                          constants=len(eager['constants']),
                          arithmetic_sha256=hashlib.sha256(json.dumps(eager,sort_keys=True).encode()).hexdigest())))
