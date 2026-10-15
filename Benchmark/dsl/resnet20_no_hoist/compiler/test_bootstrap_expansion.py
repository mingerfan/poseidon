"""Real-compiler structural tests only; synthetic body is NOT a native bootstrap."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]

def template():
    rows = []
    def op(name, arguments, output, attributes='', components=2):
        number = len(rows) // 2
        buffer, result = f'%b{number}', f'%v{number}'
        ty = f'tensor<1x!ckks.poly<{components} * {output}>>'
        rows.append(f'    {buffer} = tensor.empty() : {ty}')
        values = ', '.join([buffer]+[name for name, _ in arguments])
        types = ', '.join([ty]+[ty for _, ty in arguments])
        attrs = f' <{{{attributes}}}>' if attributes else ''
        rows.append(f'    {result} = "ckks.{name}"({values}){attrs} : ({types}) -> {ty}')
        return result, ty
    start = '%input', 'tensor<1x!ckks.poly<2 * 5>>'
    current = op('conjugatec', [start], 5)
    current = op('multiplyscalarc', [current], 5, 'multiplier = 16 : i64')
    current = op('setscalec', [current], 5, 'scale = 1.099511627776e+12 : f64')
    current = op('rawmulcc', [current, current], 5, components=3)
    current = op('relinearizec', [current], 5)
    current = op('rescalemanyc', [current], 3, 'count = 2 : i64')
    current = op('modraisec', [current], 49, 'level = 49 : i64')
    current = op('modswitchc', [current], 30, 'downFactor = 19 : i64')
    return ('module {\n  func.func @native_bootstrap_template(%input: '+start[1]+') -> '+current[1]+' {\n'+
            '\n'.join(rows)+'\n    return '+current[0]+' : '+current[1]+'\n  }\n}\n')

def run(compiler, report):
    source = PACKAGE/'source/strict_native_resnet20_20261015_v1.mlir'
    config = PACKAGE/'compiler/profiled_HEAAN_GPU_blb5.json'
    negative = 0
    with tempfile.TemporaryDirectory(prefix='bootstrap-pass-structural-') as directory:
        directory=Path(directory)
        body=directory/'synthetic-only.mlir'
        output=directory/'expanded.mlir'
        prefix=directory/'synthetic-only'
        def invoke(count=18, emit=False, source_file=source):
            passes=f'expand-native-bootstrap{{template-file={body} expected-count={count}}}'
            if emit: passes+=f',emit-hevm{{prefix={prefix}}}'
            return subprocess.run([str(compiler), f'--ckks-config={config}',
                f'--pass-pipeline=builtin.module(func.func({passes}))', str(source_file), '-o', str(output)],
                text=True, capture_output=True)
        body.write_text(template())
        result=invoke(emit=True)
        assert result.returncode==0, result.stderr
        text=output.read_text()
        assert 'ckks.bootstrapc' not in text and 'poseidon.expanded_bootstraps = 18' in text
        import re
        groups=collections.Counter(re.findall(r'poseidon.bootstrap_group = (\d+)',text))
        assert len(groups)==18 and set(groups.values())=={16},groups
        binary=Path(str(prefix)+'.strict_native_resnet20.hevm')
        data=binary.read_bytes(); header=struct.unpack_from('<I',data,4)[0]
        body_size,nops=struct.unpack_from('<QQ',data,header)
        ops=[struct.unpack_from('<4H',data,header+body_size+8*i) for i in range(nops)]
        counts=collections.Counter(row[0] for row in ops)
        assert counts[10]==0
        for opcode in range(11,18): assert counts[opcode]==18,(opcode,counts[opcode])
        fnv=14695981039346656037
        for byte in data: fnv=((fnv^byte)*1099511628211)&((1<<64)-1)
        for suffix,magic,length in [('.native_primitives','POSEIDON_HEVM_NATIVE_PRIMITIVE_V1',18),
                                   ('.bootstrap_groups','POSEIDON_HEVM_BOOTSTRAP_GROUPS_V1',144)]:
            lines=Path(str(binary)+suffix).read_text().splitlines()
            assert lines[0].split()==[magic,str(nops),f'{fnv:x}']
            assert len(lines)-1==length
        def rejected(expected, count=18, source_file=source):
            nonlocal negative
            result=invoke(count=count,source_file=source_file)
            assert result.returncode!=0 and expected in result.stderr,result.stderr
            negative+=1
        rejected('boundary count',17)
        body.write_text(template().replace('tensor<1x!ckks.poly<2 * 5>>','tensor<1x!ckks.poly<2 * 4>>'))
        rejected('failed to infer returned types')
        body.write_text('module { func.func @native_bootstrap_template(%x: tensor<1x!ckks.poly<2 * 5>>) -> tensor<1x!ckks.poly<2 * 5>> { return %x : tensor<1x!ckks.poly<2 * 5>> } }')
        rejected('identity/empty')
        body.write_text(template().replace('1.099511627776e+12','-1.0'))
        rejected('failed to infer returned types')
        body.write_text(template().replace('"ckks.conjugatec"','"ckks.unknown_bootstrap_callback"'))
        rejected('unregistered operation')
        body.write_text(template())
        # The compiler must not silently ignore unknown operators after expansion.
        unsupported=directory/'unsupported.mlir'
        unsupported.write_text(source.read_text().replace('    %buf1 =', '    %unsupported = tensor.cast %arg0 : tensor<1x!ckks.poly<2 * 31>> to tensor<1x!ckks.poly<2 * 31>>\n    %buf1 =',1))
        result=invoke(emit=True,source_file=unsupported)
        assert result.returncode!=0 and 'refusing to drop it' in result.stderr,result.stderr
        negative+=1
    result={'status':'PASS','scope':'real compiler structural expansion, NOT native bootstrap semantics/GPU acceptance',
            'boundaries':18,'opaque_bootstrap_opcodes':0,'synthetic_primitives_per_group':8,
            'negative_tests':negative,'gpu_executed':False,'native_template_complete':False}
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler',type=Path)
    parser.add_argument('report',type=Path)
    args=parser.parse_args()
    run(args.compiler,args.report)
