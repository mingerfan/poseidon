"""Source-derived ABI for direct, one-ciphertext public Conv kernel sums.

Names and tensor dimensions come from the checked layout. This profile supports
one logical ciphertext and a separately encrypted trusted zero, not arbitrary
functions with physically identical arguments.
"""
import ast
import math
import re

from cpu_conv_composition_intake import canonical, digest, need


def abi_mapping(request, candidate, selected, lowered, raw, bound):
    layout = request['layout']
    need(layout['execution_abi'] == 'unified-capacity-chunked-inputs-v4', 'composition ABI')
    period = layout['input_slot_period']
    need(type(period) is int and 1 <= period <= 16384 and 16384 % period == 0, 'packing period')
    inputs, auxiliary, outputs = (layout[key] for key in ('inputs', 'auxiliary_ciphertexts', 'outputs'))
    need(len(inputs) == len(auxiliary) == len(outputs) == 1, 'one logical input, zero, and output')
    source = inputs[0]
    need(set(source) == {'dsl_name', 'name', 'shape', 'offset', 'elements'} and
         source['offset'] == 0 and source['elements'] == math.prod(source['shape']) <= period and
         layout['logical_inputs'] == [dict(name=source['name'], shape=source['shape'])] and
         request['model']['inputs'] == layout['logical_inputs'], 'complete logical input role')
    zero = auxiliary[0]
    need(zero == dict(dsl_name=zero['dsl_name'], kind='encrypted_zero', slot_period=period,
                      source='trusted_client'), 'distinct trusted encrypted-zero role')
    names = [source['dsl_name'], zero['dsl_name']]
    need(all(type(name) is str and name.isidentifier() for name in names) and len(set(names)) == 2,
         'distinct source argument names')
    output = outputs[0]
    count = math.prod(output['shape'])
    need(1 <= count <= period and output == dict(name=output['name'], shape=output['shape'],
         chunks=[dict(ciphertext=0, elements=count, offset=0)]) and
         layout['output_ciphertexts'] == 1 and layout['output_representation'] == 'named_chunked_prefixes' and
         layout['output_selectors'] == [[0, i] for i in range(count)], 'complete logical output role')
    tree = ast.parse(candidate)
    entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'golden']
    need(len(entries) == 1, 'unique candidate entry')
    args = entries[0].args
    need(not args.posonlyargs and not args.kwonlyargs and not args.vararg and not args.kwarg and
         not args.defaults and not args.kw_defaults and [a.arg for a in args.args] == names,
         'layout to candidate argument order')
    blocks = re.findall(r'(?m)^\s*\^bb0\(([^\n]+)\):\s*$', selected)
    need(len(re.findall(r'"func.func"\(\)', selected)) == 1 and len(blocks) == 1 and
         re.findall(r'%arg([0-9]+):', blocks[0]) == ['0', '1'], 'selected argument order')
    headers = re.findall(r'(?m)^\s*func.func @_hecate_golden\(([^\n]+?)\) -> ([^\n]+)$', lowered)
    need(len(re.findall(r'(?m)^\s*func.func ', lowered)) == 1 and len(headers) == 1, 'unique lowered entry')
    need(raw['external_inputs'] == ['0', '1'] == bound['external_inputs'] and
         len(bound['final_outputs']) == 1, 'compiler external/final IDs')
    for key in ('values', 'external_inputs', 'final_outputs', 'target', 'plan_id', 'finalization'):
        need(raw[key] == bound[key], 'periodic binding preserves SSA metadata')
    for phase in ('initialization', 'execution'):
        need(len(raw[phase]) == len(bound[phase]), 'periodic instruction count')
        for a, b in zip(raw[phase], bound[phase]):
            need({k: v for k, v in a.items() if k != 'payload'} ==
                 {k: v for k, v in b.items() if k != 'payload'} and
                 (a.get('kind') == 'encode' or a == b), 'only Encode payload binding changes')
    values = {row['id']: row for row in bound['values']}
    need(len(values) == len(bound['values']), 'unique SSA values')
    external = [values[key] for key in bound['external_inputs']]
    for row in external:
        need(row == dict(id=row['id'], kind='ciphertext', place=dict(kind='host', rank=0),
             context='poseidon-agent-n15-q13-p1-tc128-v1', level=12, scale_log2=45,
             ntt=True, components=2), 'original CPU external metadata')
    params = re.findall(r'%arg([0-9]+): tensor<1x!ckks.poly<([0-9]+) \* ([0-9]+) \* ([0-9]+)>>', headers[0][0])
    need(params == [(str(i), str(row['components']), str(row['scale_log2']), str(row['level']))
                    for i, row in enumerate(external)], 'lowered physical argument ABI')
    final = values[bound['final_outputs'][0]]
    need(final['kind'] == 'ciphertext' and final['components'] == 2 and final['ntt'] is True and
         final['context'] == external[0]['context'] and final['place'] == external[0]['place'] and
         headers[0][1].startswith('tensor<1x!ckks.poly<%d * %d * %d>> attributes ' %
             (final['components'], final['scale_log2'], final['level'])), 'lowered final ABI')
    return dict(inputs=[dict(source, abi_index=0, external_value_id='0'),
                        dict(zero, abi_index=1, external_value_id='1')],
                outputs=[dict(output, final_index=0, final_value_id=final['id'])],
                output_selectors_sha256=digest(layout['output_selectors']))
