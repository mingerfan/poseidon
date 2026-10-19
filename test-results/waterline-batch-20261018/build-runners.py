#!/usr/bin/env python3
"""Build selected Runtime headers against identical GPU libraries."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--headers", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--makefiles", action="store_true", help="Read CMake flags.make instead of compile_commands.json")
parser.add_argument("--variants", nargs="+", default=["updated"])
parser.add_argument("--runners", nargs="+", default=["qwen", "mlp"])
args = parser.parse_args()
root, build = args.root.resolve(), args.build.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
sources = {"qwen": "poseidon_gpu_qwen_e2e.cpp", "mlp": "poseidon_gpu_mlp_e2e_test.cpp", "api_tests": "poseidon_gpu_api_test.cpp", "prepare_oom_tests": "poseidon_gpu_prepare_oom_test.cpp"}
def target_name(runner):
    return "poseidon_runtime_gpu_" + runner + ("_e2e" if runner in ("qwen", "mlp") else "")
if args.makefiles:
    cache = (build / "CMakeCache.txt").read_text()
    compiler = re.search(r"^CMAKE_CXX_COMPILER:[^=]+=(.+)$", cache, re.MULTILINE).group(1)
    database = []
    for runner in args.runners:
        source = sources[runner]
        directory = build / "src/poseidon/runtime_api"
        flags = (directory / "CMakeFiles" / (target_name(runner) + ".dir/flags.make")).read_text()
        command = [compiler]
        for name in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            command += shlex.split(re.search(r"^" + name + r" = (.*)$", flags, re.MULTILINE).group(1))
        source_path = root / "src/poseidon/tests/runtime_api" / source
        command += ["-o", "unused.o", "-c", str(source_path)]
        database.append(dict(directory=str(directory), command=shlex.join(command), file=str(source_path)))
else:
    database = json.loads((build / "compile_commands.json").read_text())
records = []


def compile_runner(item):
    variant, runner = item
    target = target_name(runner)
    folder = output / variant
    overlay = folder / "include"
    entry = next(c for c in database if c["file"].endswith("/" + sources[runner]))
    directory = Path(entry["directory"])
    object_file = folder / (runner + ".o")
    executable = folder / "bin" / target
    command = shlex.split(entry["command"])
    command.insert(1, "-I" + str(overlay))
    command[command.index("-o") + 1] = str(object_file)
    dependency = folder / (runner + ".d")
    command += ["-MD", "-MF", str(dependency)]
    link = shlex.split((directory / "CMakeFiles" / (target + ".dir") / "link.txt").read_text())
    link = [str(object_file) if token.endswith(".cpp.o") else token
            for token in link if not token.startswith("-Wl,--dependency-file=")]
    link[link.index("-o") + 1] = str(executable)
    record = {"variant": variant, "runner": runner, "directory": str(directory),
              "compile": command, "link": link, "header_sha256": sha(overlay / "runtime/runtime.hpp")}
    with (folder / (runner + "-build.log")).open("w") as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Check the compiler actually selected the historical overlay.
        assert str(overlay / "runtime/runtime.hpp") in dependency.read_text()
        subprocess.run(link, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=True)
    record["executable_sha256"] = sha(executable)
    print(f"Built {variant}/{runner}", flush=True)
    return record


for variant in args.variants:
    folder = output / variant
    (folder / "bin").mkdir(parents=True)
    header = folder / "include/runtime/runtime.hpp"
    header.parent.mkdir(parents=True)
    header.write_bytes((args.headers / (variant + ".hpp")).read_bytes())

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for record in pool.map(compile_runner, [(v, r) for v in args.variants for r in args.runners]):
        records.append(record)
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")

libraries = [build / name for name in ("libposeidon_runtime_gpu_api.a", "libruntime_core.a",
             "libruntime_plan_json.a", "libposeidon_gpu.a", "libposeidon_shared.so")]
(output / "libraries.json").write_text(json.dumps({str(p): sha(p) for p in libraries}, indent=2) + "\n")
