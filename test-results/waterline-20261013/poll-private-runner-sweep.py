#!/usr/bin/env python3
"""Compare 100us/10us/yield polling in isolated copies of the existing runner.

Only the no-progress wait statement changes in each copied runtime.hpp.
All API/math libraries, the encrypted plan, fixture and GPU backend are fixed.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--source-command", type=Path, default=ROOT / "build-waterline-sync/lookahead-local-qproj-sweep-v1/repeat-0-w8/command.json")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    source_command = json.loads(args.source_command.read_text())
    plan = json.loads(Path(source_command[1]).read_text())
    if plan["waterline"]["lookahead_groups"] != 8 or source_command[-2:] != ["--backend", "gpu"]:
        raise RuntimeError("experiment requires unchanged lookahead8 and GPU-only backend")
    header = ROOT / "third_party/ckks-runtime/runtime/runtime.hpp"
    cpp = ROOT / "src/poseidon/tests/runtime_api/poseidon_gpu_qwen_e2e.cpp"
    header_text, cpp_text = header.read_text(), cpp.read_text()
    old = "std::this_thread::sleep_for(std::chrono::microseconds(100));"
    include = '#include "runtime/runtime.hpp"'
    if header_text.count(old) != 1 or cpp_text.count(include) != 1:
        raise RuntimeError("expected exactly one polling statement and one runner include")
    build = Path(source_command[0]).resolve().parents[1]
    target = build / "src/poseidon/runtime_api"
    cmake = target / "CMakeFiles/poseidon_runtime_gpu_qwen_e2e.dir"
    variables = {}
    for line in (cmake / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            variables[key] = shlex.split(value)
    old_link = shlex.split((cmake / "link.txt").read_text())
    libraries = [(target / value).resolve() if not Path(value).is_absolute() else Path(value)
                 for value in old_link if value.endswith((".a", ".so"))]
    bindings = {str(path): digest(path) for path in [header, cpp, Path(source_command[1]),
        Path(source_command[2]), Path(source_command[3])] + libraries}
    summary = dict(scope="local private-runner polling experiment; no production source change",
        fixed_lookahead=8, fixed_backend="gpu", cuda_visible_devices="0", repeats=args.repeats,
        source_bindings=bindings, script_sha256=digest(Path(__file__)), variants={}, samples=[])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    statements = {"100us": old, "10us": "std::this_thread::sleep_for(std::chrono::microseconds(10));",
                  "yield": "std::this_thread::yield();"}
    for name, replacement in statements.items():
        folder = args.output / name
        folder.mkdir()
        private_header, private_cpp = folder / "runtime.hpp", folder / "runner.cpp"
        private_header.write_text(header_text.replace(old, replacement))
        private_cpp.write_text(cpp_text.replace(include, '#include "' + str(private_header) + '"'))
        difference = "".join(difflib.unified_diff(header_text.splitlines(keepends=True),
            private_header.read_text().splitlines(keepends=True), fromfile="production/runtime.hpp", tofile=f"private/{name}/runtime.hpp"))
        (folder / "runtime.diff").write_text(difference)
        obj, binary = folder / "runner.o", folder / "runner"
        compile_command = ["/usr/bin/c++"] + variables["CXX_DEFINES"] + variables["CXX_INCLUDES"] + variables["CXX_FLAGS"]
        compile_command += [f"-I{cpp.parent}", "-c", str(private_cpp), "-o", str(obj)]
        link_command = []
        skip = False
        for value in old_link:
            if skip:
                link_command.append(str(binary));skip=False;continue
            if value == "-o":
                link_command.append(value);skip=True
            elif value.endswith(".cpp.o"):
                link_command.append(str(obj))
            elif value.startswith("-Wl,--dependency-file="):
                link_command.append(f"-Wl,--dependency-file={folder / 'link.d'}")
            elif value.endswith((".a", ".so")) and not Path(value).is_absolute():
                link_command.append(str((target / value).resolve()))
            else:
                link_command.append(value)
        save(folder / "build-command.json", dict(compile=compile_command, link=link_command))
        with (folder / "build.log").open("w") as log:
            for command in (compile_command, link_command):
                result = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"private build failed: {folder / 'build.log'}")
        summary["variants"][name] = dict(statement=replacement, header_sha256=digest(private_header),
            source_sha256=digest(private_cpp), binary_sha256=digest(binary),
            exact_runtime_diff=difference, binary=str(binary))
        save(args.output / "summary.json", summary)
        print(f"built variant={name}", flush=True)
    names = ("100us", "10us", "yield")
    for repeat in range(args.repeats):
        shift = repeat % 3
        for name in names[shift:] + names[:shift]:
            folder = args.output / f"repeat-{repeat}-{name}"
            folder.mkdir()
            command = source_command.copy()
            command[0] = summary["variants"][name]["binary"]
            command[4], command[5] = str(folder / "report.json"), str(folder / "decoded.json")
            save(folder / "command.json", command)
            started = time.monotonic()
            with (folder / "run.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((folder / "report.json").read_text())
            if result.returncode or not report["passed"] or report["comparison"]["slots_checked"] != 8192:
                raise RuntimeError(f"full-slot numerical regression: {folder / 'report.json'}")
            if report["preparation_measurements"]["cpu_prepare_calls"]:
                raise RuntimeError("private run unexpectedly used CPU preparation")
            summary["samples"].append(dict(variant=name, repeat=repeat, passed=True,
                process_wall_ms=(time.monotonic()-started)*1000, runtime_ms=report["runtime_ms"],
                runtime_timing=report["runtime_timing"], comparison=report["comparison"],
                device_measurements=report["device_measurements"]))
            save(args.output / "summary.json", summary)
            print(f"PASS repeat={repeat} variant={name} runtime_ms={report['runtime_ms']:.3f}", flush=True)
    summary["statistics"] = {}
    for name in names:
        rows = [s for s in summary["samples"] if s["variant"] == name]
        times = [r["runtime_ms"] for r in rows]
        summary["statistics"][name] = dict(runtime_ms_median=statistics.median(times),
            runtime_ms_min=min(times), runtime_ms_max=max(times),
            online_ms_median=statistics.median(r["runtime_timing"]["online_execution_ms"] for r in rows),
            compute_submission_ms_median=statistics.median(r["runtime_timing"]["compute_submission_including_boot_ms"] for r in rows),
            consumer_wait_ms_median=statistics.median(sum(d["consumer_wait_ms"] for d in r["device_measurements"]) for r in rows))
    baseline = summary["statistics"]["100us"]
    best = min(names, key=lambda name: summary["statistics"][name]["runtime_ms_median"])
    summary["best_variant"] = best
    summary["best_forward_improvement_fraction"] = 1-summary["statistics"][best]["runtime_ms_median"]/baseline["runtime_ms_median"]
    summary["strong_local_gain"] = (summary["best_forward_improvement_fraction"] >= 0.1 and
        summary["statistics"][best]["runtime_ms_max"] < baseline["runtime_ms_min"])
    for path, original_digest in bindings.items():
        if digest(Path(path)) != original_digest:
            raise RuntimeError(f"shared input changed during isolated experiment: {path}")
    summary["shared_inputs_unchanged"] = True
    summary["passed"] = True
    save(args.output / "summary.json", summary)
    print(json.dumps(summary["statistics"]), flush=True)


if __name__ == "__main__":
    main()
