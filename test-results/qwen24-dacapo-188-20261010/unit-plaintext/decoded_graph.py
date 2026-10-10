"""Hash the complete decoded arithmetic, treating full-slot units as identity.

This proves the logical graph rewrite; it does not model CKKS rounding/noise.
Physical types, liveness, memory, and blobs are checked independently.
"""
from collections import Counter
import hashlib
import json
import struct

FULL_UNIT = 'sha256:' + hashlib.sha256(struct.pack('<d', 1.0) * 32768).hexdigest()
FULL_NEGATIVE_UNIT = 'sha256:' + hashlib.sha256(struct.pack('<d', -1.0) * 32768).hexdigest()


def decoded_graph(plan):
    count = len(plan['values'])
    digests = bytearray(count * 32)
    view = memoryview(digests)
    unit = bytearray(count)
    hist = Counter()
    unique = set()
    for index, value in enumerate(plan['external_inputs']):
        index_value = int(value)
        view[index_value*32:(index_value+1)*32] = hashlib.sha256(b'input' + struct.pack('<I', index)).digest()
    for phase in ('initialization', 'execution'):
        for step in plan[phase]:
            if step['kind'] == 'release':
                continue
            if step['kind'] == 'transfer':
                assert len(step['outputs']) == 1
                output = int(step['outputs'][0])
            else:
                output = int(step['output'])
            begin, end = output * 32, (output + 1) * 32
            if step['kind'] == 'encode':
                payload = step['payload']
                if payload['kind'] == 'bundle':
                    content = payload['content']
                else:
                    raw = b''.join(struct.pack('<d', x) for x in payload['values'])
                    content = 'sha256:' + hashlib.sha256(raw).hexdigest()
                view[begin:end] = hashlib.sha256(b'plaintext' + content.encode()).digest()
                if content in (FULL_UNIT, FULL_NEGATIVE_UNIT):
                    unit[output] = 1 if content == FULL_UNIT else 2
                    value = plan['values'][output]
                    key = (unit[output], value['scale_log2'], value['level'])
                    hist[key] += 1
                    unique.add(key)
                continue
            inputs = [int(x) for x in step['inputs']]
            arguments = [bytes(view[x*32:(x+1)*32]) for x in inputs]
            if step['kind'] == 'transfer':
                view[begin:end] = arguments[0]
                unit[output] = unit[inputs[0]]
                continue
            assert step['kind'] == 'compute'
            op = step['op']
            if op in ('rescale', 'mod_switch', 'relinearize', 'boot'):
                view[begin:end] = arguments[0]
            elif op == 'mul_cp' and unit[inputs[1]]:
                view[begin:end] = (arguments[0] if unit[inputs[1]] == 1 else
                                   hashlib.sha256(b'negate' + arguments[0]).digest())
            elif op == 'negate':
                view[begin:end] = hashlib.sha256(b'negate' + arguments[0]).digest()
            elif op == 'rotate':
                view[begin:end] = hashlib.sha256(b'rotate' + arguments[0] +
                    json.dumps(step['attrs'], sort_keys=True).encode()).digest()
            else:
                assert op in ('mul_cc', 'mul_cp', 'add_cc', 'add_cp'), op
                arguments.sort()
                view[begin:end] = hashlib.sha256(op[:3].encode() + b''.join(arguments)).digest()
    final = hashlib.sha256()
    metadata = []
    for output in plan['final_outputs']:
        index = int(output)
        final.update(view[index*32:(index+1)*32])
        value = plan['values'][index]
        metadata.append({key: value[key] for key in ('kind', 'level', 'scale_log2', 'components', 'ntt', 'context')})
    return {'decoded_graph_sha256': 'sha256:' + final.hexdigest(),
            'final_metadata': metadata, 'full_unit_encode_count': sum(hist.values()),
            'distinct_full_unit_types': len(unique),
            'full_unit_histogram': [dict(sign=key[0], scale_log2=key[1], level=key[2], count=value)
                                    for key, value in sorted(hist.items())]}


if __name__ == '__main__':
    def value(index, kind, scale, level=8):
        return dict(id=str(index), kind=kind, scale_log2=scale, level=level,
                    components=1 if kind == 'plaintext' else 2, ntt=True, context='test')
    def encode(index, payload):
        return dict(kind='encode', output=str(index), payload=dict(kind='inline', values=payload))
    def compute(index, op, *inputs):
        return dict(kind='compute', output=str(index), op=op, inputs=[str(x) for x in inputs])
    old = dict(values=[value(0,'ciphertext',40), value(1,'plaintext',40),
                      value(2,'ciphertext',80), value(3,'plaintext',80),
                      value(4,'ciphertext',160), value(5,'ciphertext',40,4),
                      value(6,'ciphertext',40,4)], external_inputs=['0'],
               initialization=[encode(1,[1.0]*3), encode(3,[1.0]*32768)],
               execution=[compute(2,'mul_cp',0,1), compute(4,'mul_cp',2,3),
                          dict(kind='release',value='2'), compute(5,'rescale',4),
                          dict(kind='transfer',inputs=['5'],outputs=['6'])], final_outputs=['6'])
    new = dict(values=[value(0,'ciphertext',40), value(1,'plaintext',120),
                      value(2,'ciphertext',160), value(3,'ciphertext',40,4),
                      value(4,'ciphertext',40,4)], external_inputs=['0'],
               initialization=[encode(1,[1.0]*3)],
               execution=[compute(2,'mul_cp',0,1), compute(3,'rescale',2),
                          dict(kind='transfer',inputs=['3'],outputs=['4'])], final_outputs=['4'])
    before, after = decoded_graph(old), decoded_graph(new)
    assert before['decoded_graph_sha256'] == after['decoded_graph_sha256']
    assert before['final_metadata'] == after['final_metadata']
    new['initialization'][0]['payload']['values'] = [1.0]*32768
    assert decoded_graph(new)['decoded_graph_sha256'] != before['decoded_graph_sha256']
    print('decoded graph self-check passed: scale absorption, Transfer, Release, and retained masks')
