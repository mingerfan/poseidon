#!/usr/bin/env python3
"""Build and run isolated updated API tests using an existing 188 workspace.

Place poseidon_gpu_api.cpp and poseidon_gpu_api_test.cpp in --output first.
The workspace layout is --base/src (sources) and --base/build (existing build).
Only --output is written; no CMake build or shared binary replacement occurs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET


def binding(path):
    stat = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return dict(sha256="sha256:" + sha.hexdigest(), bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def flags(path):
    values = {}
    for line in path.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            values[key] = shlex.split(value)
    return sum((values[key] for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS")), [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    source, build = base / "src", base / "build"
    if output == base or any(output == p or p in output.parents for p in (source, build)):
        parser.error("--output must be separate from base/src and base/build")
    output.mkdir(parents=True, exist_ok=True)
    generated = ("result.json", "result.json.tmp", "commands.json", "build.log", "run.log",
                 "poseidon_gpu_api.o", "poseidon_gpu_api_test.o", "poseidon_gpu_api_tests",
                 "api.d", "tests.d", "link.d", "link.map", "devices-before.xml",
                 "devices-after.xml", "tmp")
    existing = [name for name in generated if (output / name).exists() or (output / name).is_symlink()]
    if existing:
        parser.error("refusing existing generated artifacts: " + ", ".join(existing))
    api_cpp, tests_cpp = (output / name for name in ("poseidon_gpu_api.cpp", "poseidon_gpu_api_test.cpp"))
    for path in (api_cpp, tests_cpp):
        if not path.is_file() or path.is_symlink():
            parser.error("expected a copied regular source file: " + str(path))
    target = build / "src/poseidon/runtime_api"
    api_flags = target / "CMakeFiles/poseidon_runtime_gpu_api.dir/flags.make"
    test_flags = target / "CMakeFiles/poseidon_runtime_gpu_api_tests.dir/flags.make"
    link_file = target / "CMakeFiles/poseidon_runtime_gpu_api_tests.dir/link.txt"
    original_link = shlex.split(link_file.read_text())
    compiler = original_link[0]
    if any("whole-archive" in arg for arg in original_link):
        raise RuntimeError("whole-archive linking would invalidate the isolated API replacement")
    api_obj, tests_obj = output / "poseidon_gpu_api.o", output / "poseidon_gpu_api_test.o"
    binary = output / "poseidon_gpu_api_tests"
    compile_api = [compiler] + flags(api_flags) + ["-I" + str(source / "src/poseidon/runtime_api"),
        "-MD", "-MF", str(output / "api.d"), "-c", str(api_cpp), "-o", str(api_obj)]
    compile_tests = [compiler] + flags(test_flags) + ["-MD", "-MF", str(output / "tests.d"),
        "-c", str(tests_cpp), "-o", str(tests_obj)]
    link, objects, outputs = [], 0, 0
    index = 0
    while index < len(original_link):
        arg = original_link[index]
        if arg == "-o":
            link += ["-o", str(binary)]
            index += 2
            outputs += 1
            continue
        if arg.endswith(".o"):
            link += [str(tests_obj), str(api_obj)]
            objects += 1
        elif arg.startswith("-Wl,--dependency-file="):
            link.append("-Wl,--dependency-file=" + str(output / "link.d"))
        elif arg.startswith(("-Wl,-Map", "-Wl,--Map")):
            raise RuntimeError("unexpected existing link map output option")
        elif not arg.startswith("-") and re.search(r"\.(?:a|so(?:\.[0-9]+)*)$", arg):
            link.append(str((target / arg).resolve()))
        elif arg.startswith("-L") and len(arg) > 2:
            link.append("-L" + str((target / arg[2:]).resolve()))
        else:
            link.append(arg)
        index += 1
    link.append("-Wl,-Map=" + str(output / "link.map"))
    if objects != 1 or outputs != 1:
        raise RuntimeError("expected exactly one test object and one output in existing link.txt")
    static_api = str((build / "libposeidon_runtime_gpu_api.a").resolve())
    if static_api not in link or link.index(str(api_obj)) > link.index(static_api):
        raise RuntimeError("new API and test objects must precede the old static API library")
    library_paths = {Path(arg) for arg in link if not arg.startswith("-") and
                     re.search(r"\.(?:a|so(?:\.[0-9]+)*)$", arg)}
    search_dirs = [Path(arg[2:]) for arg in link if arg.startswith("-L") and len(arg) > 2]
    resolution_commands = []
    for name in sorted({arg[2:] for arg in link if arg.startswith("-l")}):
        candidates = [directory / ("lib" + name + suffix) for directory in search_dirs for suffix in (".so", ".a")]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            for suffix in (".so", ".a"):
                command = [compiler, "-print-file-name=lib" + name + suffix]
                resolved = subprocess.check_output(command, text=True).strip()
                resolution_commands.append(dict(command=command, stdout=resolved))
                candidate = Path(resolved)
                if candidate.is_absolute() and candidate.is_file():
                    found = candidate
                    break
        if found is None:
            raise RuntimeError("cannot bind explicit link library -l" + name)
        library_paths.add(found.resolve())
    shared_paths = [api_flags, test_flags, link_file,
        source / "src/poseidon/runtime_api/poseidon_gpu_api.cpp",
        source / "src/poseidon/tests/runtime_api/poseidon_gpu_api_test.cpp",
        source / "src/poseidon/runtime_api/poseidon_gpu_api.h",
        source / "src/poseidon/runtime_api/gpu_plaintext_encoder.h",
        source / "src/poseidon/runtime_api/rescale_normalization.h",
        source / "third_party/ckks-runtime/runtime/runtime.hpp",
        source / "third_party/ckks-runtime/runtime/plaintext_bundle.cpp",
        source / "third_party/ckks-runtime/runtime/utils/sha256.hpp"] + sorted(library_paths)
    # Guard existing public runners/status as read-only inputs as well.
    guards = [build / "bin/poseidon_runtime_gpu_api_tests", build / "bin/poseidon_runtime_gpu_qwen_e2e",
              base / "logs/latest-build.status"]
    shared_paths += [path for path in guards if path.is_file()]
    inputs = {str(path): binding(path) for path in dict.fromkeys(shared_paths)}
    private_inputs = {str(path): binding(path) for path in (api_cpp, tests_cpp, Path(__file__).resolve())}
    env = os.environ.copy()
    removed = sorted(key for key in env if key.startswith("POSEIDON_WATERLINE_"))
    for key in removed:
        del env[key]
    overrides = dict(CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="1",
        POSEIDON_GPU_RUNTIME_MAXIMUM_POOL_MB="2560", TMPDIR=str(output / "tmp"))
    env.update(overrides)
    commands = dict(compile_api=compile_api, compile_tests=compile_tests, link=link,
        run=[str(binary)], cwd=str(output), environment_overrides=overrides,
        removed_environment_keys=removed, library_resolution=resolution_commands)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    state = dict(scope="isolated updated API/tests against existing 188 libraries; API/Ledger boundary, not same-structure full Runtime boundary",
        base=str(base), output=str(output), started_at=datetime.datetime.now().astimezone().isoformat(),
        shared_inputs_before=inputs, private_inputs_before=private_inputs, commands=commands,
        complete=False, passed=False, stage="preflight", server_workspace_outputs_replaced=False)
    started = time.monotonic()

    def save(**changes):
        state.update(changes, elapsed_seconds=time.monotonic() - started)
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "result.json")

    def snapshot(label):
        command = ["nvidia-smi", "-i", "1", "-q", "-x"]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        path = output / ("devices-" + label + ".xml")
        path.write_text(result.stdout)
        record = dict(command=command, returncode=result.returncode, stderr=result.stderr,
                      stdout_file=str(path), physical_gpu_index=1)
        if result.returncode == 0:
            gpu = ET.fromstring(result.stdout).find("gpu")
            if gpu is None:
                raise RuntimeError("nvidia-smi XML has no selected GPU")
            counters = {}
            def leaves(node, prefix):
                for child in node:
                    key = prefix + "/" + child.tag
                    if len(child):
                        leaves(child, key)
                    else:
                        counters[key] = child.text
            ecc = gpu.find("ecc_errors")
            if ecc is not None:
                leaves(ecc, "ecc_errors")
            record.update(uuid=gpu.findtext("uuid"), pci_bus_id=gpu.findtext("pci/pci_bus_id"),
                ecc_mode={e.tag: e.text for e in gpu.findall("ecc_mode/*")}, ecc_counters=counters)
        else:
            raise RuntimeError("nvidia-smi UUID/ECC snapshot failed: " + result.stderr)
        return record

    save()
    failure = None
    try:
        (output / "tmp").mkdir()
        save(stage="compiling")
        with (output / "build.log").open("w") as log:
            for command in (compile_api, compile_tests, link):
                subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        old_member = bool(re.search(r"libposeidon_runtime_gpu_api\.a\([^)]*poseidon_gpu_api\.cpp\.o\)",
                                   (output / "link.map").read_text()))
        save(binary=binding(binary), old_api_cpp_archive_member_extracted=old_member)
        if old_member:
            raise RuntimeError("link map unexpectedly extracted the old API cpp member")
        save(stage="running", devices_before=snapshot("before"))
        with (output / "run.log").open("w") as log:
            run = subprocess.run([str(binary)], cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        text = (output / "run.log").read_text()
        lines = [line for line in text.splitlines() if "GPU_PACKED_PREPARE_HOST_BOUND" in line or
                 "GPU_RAW_PINNED_CACHE" in line or "Poseidon GPU Runtime Api tests passed" in line]
        match = re.search(r"(\d+) Poseidon GPU Runtime Api tests passed", text)
        valid = (run.returncode == 0 and match is not None and int(match.group(1)) == 23 and
                 "tight_admitted=0" in text and "second_read_with_cache=true" in text and
                 "scope=API_Ledger_boundary_not_full_Runtime" in text and "[FAIL]" not in text)
        save(test_exit_code=run.returncode, api_test_count=int(match.group(1)) if match else None,
             test_evidence_lines=lines, test_log=binding(output / "run.log"), numerical_and_boundary_passed=valid)
        if not valid:
            raise RuntimeError("expected 23 passing tests and the new warm packed Host-boundary evidence")
    except Exception as error:
        failure = str(error)
    finally:
        try:
            state["devices_after"] = snapshot("after")
        except Exception as error:
            state["devices_after_error"] = str(error)
            failure = failure or str(error)
        after = {str(path): binding(path) if path.exists() else None for path in dict.fromkeys(shared_paths)}
        private_after = {str(path): binding(path) for path in (api_cpp, tests_cpp, Path(__file__).resolve())}
        unchanged = inputs == after and private_inputs == private_after
        save(shared_inputs_after=after, private_inputs_after=private_after, shared_inputs_unchanged=unchanged,
             complete=True, passed=failure is None and unchanged,
             stage="complete" if failure is None and unchanged else "failed", error=failure)
    print(json.dumps(dict(result=str(output / "result.json"), passed=state["passed"],
                         test_exit_code=state.get("test_exit_code"), error=state.get("error"))), flush=True)
    return 0 if state["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
