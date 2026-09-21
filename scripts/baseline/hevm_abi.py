"""Validate the native HEVM layout against the independent little-endian reader."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from platform_config import configuration, identity, require_platform


def check_elf(path, platform_id=None):
    profile = configuration(platform_id)
    with Path(path).open('rb') as stream:
        header = stream.read(20)
    expected = {'x86_64-linux': 62, 'aarch64-linux': 183}[profile['id']]
    if (len(header) != 20 or header[:6] != b'\x7fELF\x02\x01' or
            struct.unpack_from('<H', header, 18)[0] != expected):
        raise RuntimeError('Wrong ELF class, byte order or architecture: ' + str(path))


def validate_layout(measured):
    expected = dict(header_hex=struct.pack('<IIQQ', 0x4845564D, 24, 1, 2).hex(),
                    body_hex=struct.pack('<5Q', 104, 3, 4, 5, 13).hex(),
                    operation_hex=struct.pack('<4H', 1, 2, 3, 65533).hex(),
                    sizes=[24, 40, 8], alignments=[8, 8, 2],
                    pointer_bytes=8, int_bytes=4, double_bytes=8)
    if measured != expected:
        raise RuntimeError('Native HEVM serialization differs from the independent reader')
    for ctype, size in ((ctypes.c_void_p, 8), (ctypes.c_int, 4),
                        (ctypes.c_int64, 8), (ctypes.c_double, 8)):
        if ctypes.sizeof(ctype) != size:
            raise RuntimeError('ctypes primitive ABI mismatch')


def require_native_abi(build):
    selected = require_platform()
    report = json.loads((Path(build) / 'hevm-abi.json').read_text())
    if report['platform_identity']['configured_platform'] != selected['id']:
        raise RuntimeError('HEVM ABI report belongs to another platform')
    validate_layout(report['layout'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--header', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require_platform()
    check_elf(args.probe)
    raw = subprocess.run([str(args.probe)], capture_output=True, text=True, timeout=20, check=True)
    measured = json.loads(raw.stdout)
    validate_layout(measured)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(platform_identity=identity(), layout=measured,
                  header_sha256=digest(args.header), probe_sha256=digest(args.probe),
                  probe_source_sha256=digest(Path(__file__).parent / 'toolchain-probe/hevm_abi.cpp'),
                  encrypted_execution_validated=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
