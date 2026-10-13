#!/usr/bin/env python3
"""Prove the public PTDS test detects the former AddCP c1-copy bug.

Run the ordinary build-and-run-gpu-standalone.py helper first for this test.
This helper changes only an isolated CUDA source copy under the build directory:
the copy helper uses the former cudaMemcpy instead of the execution stream.
The production source and configured libraries remain untouched.
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
    evidence = Path(__file__).resolve().parent
    root = evidence.parent.parent
    source = root / 'src/poseidon/gpu/kernels/gpu_elementwise_kernels.cu'
    actual = source.read_text()
    corrected = '''cudaMemcpyAsync(
            destination_shard.ptr,
            source_shard.ptr,
            word_count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice,
            gpu_execution_stream())'''
    former = '''cudaMemcpy(
            destination_shard.ptr,
            source_shard.ptr,
            word_count * sizeof(GpuWord),
            cudaMemcpyDeviceToDevice)'''
    if actual.count(corrected) != 1:
        raise ValueError('expected exactly one corrected polynomial-copy helper')
    artifacts = build / 'addcp-ptds-negative-control'
    artifacts.mkdir(exist_ok=True)
    negative_source = artifacts / 'gpu_elementwise_kernels.cu'
    negative_source.write_text(actual.replace(corrected, former))
    obj = artifacts / 'gpu_elementwise_kernels.cu.o'
    gpu_directory = build / 'src/poseidon/gpu'
    flags = {}
    for line in (gpu_directory / 'CMakeFiles/poseidon_gpu.dir/flags.make').read_text().splitlines():
        if line.startswith(('CUDA_DEFINES =', 'CUDA_INCLUDES =', 'CUDA_FLAGS =')):
            name, value = line.split('=', 1)
            flags[name.strip()] = shlex.split(value)
    compiler = None
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if line.startswith(('CMAKE_CUDA_COMPILER:FILEPATH=', 'CMAKE_CUDA_COMPILER:STRING=')):
            compiler = line.split('=', 1)[1]
    if compiler is None:
        raise ValueError('missing configured CUDA compiler')
    compile_command = [compiler, '-forward-unknown-to-host-compiler',
                       *flags['CUDA_DEFINES'], *flags['CUDA_INCLUDES'],
                       *flags['CUDA_FLAGS'], '-x', 'cu', '-c',
                       str(negative_source), '-o', str(obj)]
    commands = json.loads((evidence / 'addcp-ptds-ready-test.commands.json').read_text())
    link_command = commands['link']
    # Providing the CUDA object before the archive selects just this isolated
    # former helper implementation; the rest of the libraries are unchanged.
    gpu_archive = next(i for i, word in enumerate(link_command)
                       if word.endswith('/libposeidon_gpu.a'))
    link_command.insert(gpu_archive, str(obj))
    binary = artifacts / 'addcp-ptds-negative-control'
    link_command[link_command.index('-o') + 1] = str(binary)
    for label, command, directory in [('compile', compile_command, gpu_directory),
                                      ('link', link_command, commands['link_working_directory'])]:
        with (artifacts / f'{label}.log').open('w') as output:
            subprocess.run(command, cwd=directory, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    (evidence / 'addcp-ptds-negative-control.log').write_text(result.stdout + result.stderr)
    report = json.loads(result.stdout)
    if result.returncode != 1 or report['passed']:
        raise RuntimeError('former copy helper unexpectedly passed the PTDS test')
    if not report.get('rounds') or not any(round_['c1_mismatches'] for round_ in report['rounds']):
        raise RuntimeError('negative control failed without exposing incomplete c1')
    (evidence / 'addcp-ptds-negative-control-report.json').write_text(
        json.dumps(dict(expected_failure=True, returncode=result.returncode,
                        single_change='copy helper cudaMemcpy instead of execution-stream cudaMemcpyAsync',
                        test_report=report), indent=2) + '\n')
    (artifacts / 'commands.json').write_text(json.dumps(dict(
        compile=compile_command, compile_working_directory=str(gpu_directory),
        link=link_command, link_working_directory=commands['link_working_directory'],
        run=[str(binary)]), indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print('PASS: former helper produces a Ready result with incomplete c1')


if __name__ == '__main__':
    main()
