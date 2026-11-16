#!/usr/bin/env python3
"""Link the diagnostic probe against the already built, configuration-matched GPU library.

Reuses Ninja's exact C++ compile flags; records the actual existing CMake options.
No production library is reconfigured or implicitly rebuilt here.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
build = root / 'build'
source = root / 'src/bench/gpu_keyload/main.cpp'
generated = root / 'probe-generated'
generated.mkdir(exist_ok=True)
cache = {}
for line in (build / 'CMakeCache.txt').read_text().splitlines():
    if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=')[0]:
        continue
    left, value = line.split('=', 1)
    cache[left.split(':', 1)[0]] = value
values = {
    'KEYLOAD_SOURCE_REVISION': 'ddcd96cd597d6b6522728914ee8c27eabf32de0b+keyload-diagnostics',
    'KEYLOAD_BENCH_SOURCE_SHA256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'KEYLOAD_CXX_COMPILER': subprocess.check_output([cache['CMAKE_CXX_COMPILER'], '--version'], text=True).splitlines()[0],
    'KEYLOAD_CUDA_ARCHITECTURES': cache['CMAKE_CUDA_ARCHITECTURES'],
}
for key in ('OPENMP', 'BUILD_DEPS', 'MSGSL', 'ZLIB', 'ZSTD'):
    option = 'POSEIDON_' + ('BUILD_DEPS' if key == 'BUILD_DEPS' else 'USE_' + key)
    values['KEYLOAD_' + option] = int(cache.get(option, 'OFF') in ('ON', 'TRUE', '1'))
(generated / 'keyload_build_info.h').write_text('#pragma once\n' + ''.join(
    f'#define {key} {json.dumps(value)}\n' for key, value in values.items()))
commands = subprocess.check_output([str(root / 'tools/bin/ninja'), '-C', str(build), '-t', 'commands'], text=True)
command = next(x for x in commands.splitlines() if 'gpu_bootstrap_profile.cpp.o -c' in x)
parts = shlex.split(command)
flags = []
i = 1
while i < len(parts):
    token = parts[i]
    if token in ('-MT', '-MF', '-o', '-c'):
        i += 2
        continue
    if token == '-MD':
        i += 1
        continue
    flags.append(token)
    i += 1
obj = root / 'probe-main.o'
compile_command = [parts[0], *flags, '-I' + str(generated),
                   '-DKEYLOAD_BUILD_CONFIGURATION="Release"', '-c', str(source), '-o', str(obj)]
link_command = [parts[0], '-fopenmp', str(obj), str(build / 'libposeidon_gpu.a'),
                '-L' + str(build), '-lposeidon_shared', '-L/usr/local/cuda/lib64', '-lcudart',
                '-pthread', '-Wl,-rpath,' + str(build), '-Wl,-rpath,/usr/local/cuda/lib64',
                '-o', str(root / 'poseidon_gpu_keyload')]
for command in (compile_command, link_command):
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
(root / 'probe-build.json').write_text(json.dumps({'build_info': values, 'compile': compile_command,
                                                'link': link_command}, indent=2) + '\n')
