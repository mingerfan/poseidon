#!/usr/bin/env python3
"""Build standalone evidence tests against a configured GPU API tests build.

Example (from repository root):
  python3 test-results/waterline-20261013/build-and-run-gpu-standalone.py \
    test-results/waterline-20261013/pool-cap-test.cpp build-waterline-keys-agent
  python3 test-results/waterline-20261013/build-and-run-gpu-standalone.py \
    test-results/waterline-20261013/key-backing-n16384-q40-p8-test.cpp build-waterline-keys-agent

Requires the existing CMake poseidon_runtime_gpu_api_tests target to be built.
Uses actual configured flags/library links without installing dependencies.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('build_directory', type=Path)
parser.add_argument('--output-prefix', type=Path,
                    help='save commands/log using this new prefix, preserving earlier results')
args = parser.parse_args()
source = args.source.resolve()
build = args.build_directory.resolve()
prefix = args.output_prefix.resolve() if args.output_prefix else source.with_suffix('')
commands_path = prefix.with_name(prefix.name + '.commands.json')
log_path = prefix.with_name(prefix.name + '.log')
if args.output_prefix and (commands_path.exists() or log_path.exists()):
    raise FileExistsError('use a new output prefix to preserve existing evidence')
target = build / "src/poseidon/runtime_api/CMakeFiles/poseidon_runtime_gpu_api_tests.dir"
flags = {}
for line in (target / "flags.make").read_text().splitlines():
    if " = " in line:
        name, value = line.split(" = ", 1)
        flags[name] = shlex.split(value)
object_file = build / (source.stem + ".o")
executable = build / source.stem
compile_command = ["/usr/bin/c++", *flags["CXX_DEFINES"], *flags["CXX_INCLUDES"],
                   *flags["CXX_FLAGS"], "-c", str(source), "-o", str(object_file)]
link_command = shlex.split((target / "link.txt").read_text())
link_command = [str(object_file) if word.endswith("poseidon_gpu_api_test.cpp.o") else word
                for word in link_command if not word.startswith("-Wl,--dependency-file=")]
link_command[link_command.index("-o") + 1] = str(executable)
commands_path.write_text(json.dumps({
    "compile": compile_command, "link": link_command,
    "link_working_directory": str(build / "src/poseidon/runtime_api"),
    "run": [str(executable)]}, indent=2) + "\n")
for stage, command, directory in [("compile", compile_command, None),
                                   ("link", link_command, build / "src/poseidon/runtime_api")]:
    with (build / (source.stem + "-" + stage + ".log")).open("w") as output:
        subprocess.run(command, cwd=directory, stdout=output, stderr=subprocess.STDOUT, check=True)
with log_path.open("w") as output:
    subprocess.run([str(executable)], stdout=output, stderr=subprocess.STDOUT, check=True)
print(log_path.read_text())
