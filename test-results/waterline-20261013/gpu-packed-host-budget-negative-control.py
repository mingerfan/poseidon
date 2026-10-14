#!/usr/bin/env python3
"""Reproduce the packed Host-budget regression against one private old API object.

Uses the current API test object and existing CMake flags/libraries. Only the
cached GPU Prepare Host resource expression is reverted in a private .cpp copy.
No production sources, libraries, runner binaries, or server files are changed.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    build = ROOT / "build-runtime-gpu-api-release"
    target = build / "src/poseidon/runtime_api"
    source = ROOT / "src/poseidon/runtime_api/poseidon_gpu_api.cpp"
    test_source = ROOT / "src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp"
    test_object = target / "CMakeFiles/poseidon_runtime_gpu_api_tests.dir/__/tests/runtime_api/poseidon_gpu_api_test.cpp.o"
    original = source.read_text()
    fixed = """        // Packed offset reads hold both byte storage and decoded slots (n*8)
        // while an earlier Prepare's cached pinned buffer remains resident.
        // The cache has separate fixed Host/pinned credit; it cannot replace
        // either of those simultaneous ordinary Host read buffers.
        if(!device.gpu_raw_pinned_cache_enabled)peak.pinned=n*4;"""
    old = """        if(device.gpu_raw_pinned_cache_enabled)peak.host-=n*4;
        else peak.pinned=n*4;"""
    if original.count(fixed) != 1:
        raise RuntimeError("expected exactly one fixed cached GPU Prepare resource block")
    private_source = folder / "poseidon_gpu_api_old.cpp"
    private_source.write_text(original.replace(fixed, old))
    difference = "".join(difflib.unified_diff(original.splitlines(keepends=True),
        private_source.read_text().splitlines(keepends=True),
        fromfile="production/poseidon_gpu_api.cpp", tofile="private/poseidon_gpu_api_old.cpp"))
    (folder / "old-resource.diff").write_text(difference)
    variables = {}
    flags = target / "CMakeFiles/poseidon_runtime_gpu_api.dir/flags.make"
    for line in flags.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            variables[key] = shlex.split(value)
    link_file = target / "CMakeFiles/poseidon_runtime_gpu_api_tests.dir/link.txt"
    old_link = shlex.split(link_file.read_text())
    obj, binary = folder / "old-api.o", folder / "old-api-tests"
    compile_command = ["/usr/bin/c++"] + variables["CXX_DEFINES"] + variables["CXX_INCLUDES"] + variables["CXX_FLAGS"]
    compile_command += ["-I" + str(source.parent), "-c", str(private_source), "-o", str(obj)]
    link_command = []
    next_is_output = False
    for value in old_link:
        if next_is_output:
            link_command.append(str(binary))
            next_is_output = False
        elif value == "-o":
            link_command.append(value)
            next_is_output = True
        elif value.endswith(".cpp.o"):
            # Define the private API first; its original static-library member
            # then stays unextracted while other required members are linked.
            link_command += [str(test_object), str(obj)]
        elif value.startswith("-Wl,--dependency-file="):
            link_command.append("-Wl,--dependency-file=" + str(folder / "link.d"))
        elif value.endswith((".a", ".so")) and not Path(value).is_absolute():
            link_command.append(str((target / value).resolve()))
        else:
            link_command.append(value)
    libraries = [Path(value) for value in link_command if value.endswith((".a", ".so"))]
    bindings = {str(path): digest(path) for path in [source, test_source, test_object, flags, link_file] + libraries}
    result = dict(scope="local private old-resource negative control; API/Ledger boundary, not full Runtime",
        cuda_visible_devices="0", source_bindings=bindings, exact_private_diff=difference,
        private_source_sha256=digest(private_source), compile_command=compile_command, link_command=link_command)
    output = folder / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    with (folder / "build.log").open("w") as log:
        for command in (compile_command, link_command):
            subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    result["binary_sha256"] = digest(binary)
    with (folder / "run.log").open("w") as log:
        run = subprocess.run([str(binary)], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
    text = (folder / "run.log").read_text()
    expected = "cached GPU packed-read Host boundary incorrectly admitted the tight budget"
    result.update(returncode=run.returncode, expected_failure=expected,
        negative_control_detected=bool(run.returncode and expected in text and "tight_admitted=1" in text),
        log_sha256=digest(folder / "run.log"), script_sha256=digest(Path(__file__)))
    result["shared_inputs_unchanged"] = all(digest(Path(path)) == value for path, value in bindings.items())
    output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["negative_control_detected"] or not result["shared_inputs_unchanged"]:
        raise RuntimeError("negative control did not isolate the expected old Host-budget failure")
    print(text, end="")
    print("PASS: private old-resource API incorrectly admitted the tight budget, as expected")


if __name__ == "__main__":
    main()
