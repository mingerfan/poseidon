"""Exercise the real compiler: legacy compatibility, wide operands, overflow."""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess


def run(compiler, source, prefix, config, success=True):
    result = subprocess.run([str(compiler), str(source),
                             f'--ckks-config={config}',
                             '--pass-pipeline=builtin.module(func.func(emit-hevm{prefix='
                             + str(prefix) + '}))', '-o', str(prefix) + '.emitted.mlir'],
                            capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr
    artifact = Path(str(prefix) + '.strict_scale_probe.hevm')
    if not success:
        assert 'HEVM field overflow' in result.stderr, result.stderr
        assert not artifact.exists()
        return None
    return artifact.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('compiler', type=Path)
    parser.add_argument('old_compiler', type=Path)
    parser.add_argument('legacy_mlir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    source = args.legacy_mlir.read_text()
    old = run(args.old_compiler, args.legacy_mlir, args.output / 'old', args.config)
    new = run(args.compiler, args.legacy_mlir, args.output / 'legacy', args.config)
    assert old == new, 'Legacy artifact byte compatibility changed'
    for value, wide, success in ((65536, False, False), (65536, True, True),
                                 (4294967295, True, True), (4294967296, True, False)):
        stem = f'constant-{value}-wide{int(wide)}'
        text = source.replace('value = 0 : i64', f'value = {value} : i64')
        if wide:
            text = text.replace('attributes {', 'attributes {poseidon.hevm_wide, ')
        path = args.output / (stem + '.mlir')
        path.write_text(text)
        data = run(args.compiler, path, args.output / stem, args.config, success)
        if data:
            magic, header_size = struct.unpack_from('<II', data)
            assert magic == 0x48455632
            config_size, count = struct.unpack_from('<QQ', data, header_size)
            offset = header_size + config_size
            assert len(data) == offset + count * 16
            instructions = list(struct.iter_unpack('<IIII', data[offset:]))
            encode = [op for op in instructions if op[0] == 0]
            assert len(encode) == 1 and encode[0][2] == value
    t = 'tensor<1x!ckks.poly<2 * 31>>'
    prefix = source[:source.index('    %pbuf')]
    body = ''.join(f'    %b{i} = tensor.empty() : {t}\n' for i in range(65536))
    body += ('    %r = "ckks.rotatec"(%b65535, %arg0) '
             '<{offset = array<i64: -1>}> : (' + t + ', ' + t + ') -> ' + t
             + '\n    return %r : ' + t + '\n  }\n}\n')
    for wide in (False, True):
        path = args.output / f'register-wide{int(wide)}.mlir'
        text = prefix + body
        if wide:
            text = text.replace('attributes {', 'attributes {poseidon.hevm_wide, ')
        path.write_text(text)
        data = run(args.compiler, path, path.with_suffix(''), args.config, wide)
        if wide:
            assert struct.unpack_from('<IIII', data, len(data)-16) == (1, 65536, 0, 4294967295)
    print('PASS real compiler: legacy bytes unchanged; wide 65536/uint32 max exact; '
          'register65536 and negative rotation exact; '
          'legacy and uint32 overflow rejected before artifact creation; legacy_sha256='
          + hashlib.sha256(new).hexdigest())


if __name__ == '__main__':
    main()
