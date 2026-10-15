"""Bind native plaintext scale expressions to audited real HEVM Encode ops.

Source rule descriptions come from native Replay expressions, not tuned scales.
Runtime must evaluate these with actual predecessor metadata.
"""
import argparse
import collections
import json
from pathlib import Path
from audit_strict_native_hevm import audit
from verify_native_scale_sidecar import fnv


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('hevm','lowering','graph','relu_template','output'):
        parser.add_argument(name)
    args=parser.parse_args()
    audit(args.hevm,args.lowering)
    manifest=json.loads(Path(args.lowering).read_text())
    graph=json.loads(Path(args.graph).read_text())
    template=json.loads(Path(args.relu_template).read_text())
    originals={op['id']:op for op in graph['operations']}
    relu_originals={op['id']:op for op in template['operations']}
    stage_map=collections.defaultdict(dict)
    registers={};input_index=0
    for op in graph['operations']:
        if 'source_relu_id' in op['attributes']:
            stage_map[op['stage']][op['attributes']['source_relu_id']]=op['id']
        if op['kind']=='input_layout':
            registers[op['id']]=input_index;input_index+=1
    for row in manifest['raw_operations']:
        if row['opcode']>0:registers[row['origin']]=row['dst']
    def register(origin):
        if origin not in registers:
            op=originals[origin]
            assert op['kind']=='rescale_many' and op['attributes']['physical_drops']==0
            return register(op['inputs'][0])
        return registers[origin]
    data=Path(args.hevm).read_bytes()
    rows=[f'POSEIDON_HEVM_NATIVE_ENCODE_RULE_V1 {len(manifest["raw_operations"])} {fnv(data):016x}']
    counts=collections.Counter()
    for index,row in enumerate(manifest['raw_operations']):
        if row['opcode']!=0:continue
        op=originals[row['origin']];attr=op['attributes']
        # Lowered prefix drop/negate may be the last consumer of the original
        # register. Read scale from the actual immediate ciphertext consumer.
        consumer=manifest['raw_operations'][index+2]
        assert consumer['opcode'] in (7,9) and consumer['rhs']==row['dst']
        runtime_source=consumer['lhs']
        rule,first,second,target='constant',0,0,row['native_scale_bits']
        if op['kind'] in ('add_plain','sub_plain'):
            rule,first='source',runtime_source
        elif op['kind']=='multiply_plain_q_prefix':
            native=relu_originals[attr['source_relu_id']]['attributes']
            source=runtime_source
            if 'native_target_value_id' in native:
                rule='ratio';first=register(stage_map[op['stage']][native['native_target_value_id']]);second=source
            else:
                rule='exp2_over_source';first=source
                # Decimal target is the original fixture's exact f64 logscale.
                target=format(native['native_target_log_scale'],'.17g')
        counts[rule]+=1
        rows.append(f'{index} {row["dst"]} {rule} {first} {second} {target}')
    Path(args.output).write_text('\n'.join(rows)+'\n')
    print(json.dumps(dict(status='PASS',encode_rules=dict(counts),
                         scope='audited expression metadata; C++ runtime evaluation/integration pending')))
