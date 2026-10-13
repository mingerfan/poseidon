#!/usr/bin/env python3
"""Build the public dependency test against an existing CMake Makefiles build.

First build poseidon_runtime_gpu_api_tests to produce the current libraries.
This helper reuses that target's exact compiler definitions, includes and links;
it never changes a public header or recompiles a private implementation copy.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    args = parser.parse_args()
    build = args.build_dir.resolve(strict=True)
    directory = build / 'src/poseidon/runtime_api'
    target = directory / 'CMakeFiles/poseidon_runtime_gpu_api_tests.dir'
    flags = {}
    for line in (target / 'flags.make').read_text().splitlines():
        if line.startswith(('CXX_DEFINES =', 'CXX_INCLUDES =', 'CXX_FLAGS =')):
            name, value = line.split('=', 1)
            flags[name.strip()] = shlex.split(value)
    if set(flags) != {'CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'}:
        raise ValueError('missing CXX flags in the CMake Makefiles target')
    link = shlex.split((target / 'link.txt').read_text())
    if not link:
        raise ValueError('empty link command')
    source = Path(__file__).resolve().with_name('consumer-wait-test.cpp')
    if not source.is_file():
        raise FileNotFoundError(source)
    artifacts = build / 'consumer-wait-test'
    artifacts.mkdir(exist_ok=True)
    obj = artifacts / 'public-test.o'
    binary = build / 'bin/consumer-wait-test'
    compile_command = [link[0], *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'],
                       *flags['CXX_FLAGS'], '-c', str(source), '-o', str(obj)]
    replaced = 0
    final_link = []
    for argument in link:
        if argument.startswith('-Wl,--dependency-file='):
            continue
        if argument.endswith('/poseidon_gpu_api_test.cpp.o'):
            if not (directory / argument).is_file():
                raise FileNotFoundError('build poseidon_runtime_gpu_api_tests first')
            argument = str(obj)
            replaced += 1
        final_link.append(argument)
    if replaced != 1 or final_link.count('-o') != 1:
        raise ValueError('expected exactly one GPU API test object and output')
    final_link[final_link.index('-o') + 1] = str(binary)
    commands = dict(working_directory=str(directory), compile=compile_command,
                    link=final_link, binary=str(binary))
    (artifacts / 'public-build-commands.json').write_text(
        json.dumps(commands, indent=2) + '\n')
    for label, command in [('compile', compile_command), ('link', final_link)]:
        print(shlex.join(command), flush=True)
        with (artifacts / f'public-{label}.log').open('w') as log:
            subprocess.run(command, cwd=directory, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    print(binary)


if __name__ == '__main__':
    main()
