"""Continue only the approved C++ stages after an existing Nix build exits.

This never restarts LLVM, installs Python wheels/CUDA, or runs encrypted code.
The dependency environment is not a sandbox for untrusted generated programs.
"""
from platform_config import nix_platform_options
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


from workspace_paths import ROOT, WORK, nix_environment_options
DEPS = "src/poseidon/tools/dacapo/dacapo-dependencies.nix"
SHELL = "src/poseidon/tools/dacapo/dacapo-shell.nix"
WRAPPER = ["bash", "scripts/baseline/nix_portable.sh"]
PROTECTED = [
    DEPS, SHELL, ".gitmodules", "src/poseidon/tools/dacapo/dependency-lock.json",
    "scripts/baseline/nix_portable.sh", "scripts/baseline/build_dacapo_dependencies.sh",
    "scripts/baseline/build_hecate_cpp.sh", "scripts/baseline/check_hecate_libraries.py",
    "scripts/baseline/continue_dacapo_cpp.py",
    "scripts/baseline/workspace_paths.py", "scripts/baseline/workspace_paths.sh",
    "scripts/baseline/toolchain-probe/CMakeLists.txt", "scripts/baseline/toolchain-probe/probe.cpp",
]


def validate_remaining_plan(plan):
    # Original approved closure: 358.87 MiB total. About 267.64 MiB has already
    # been requested by the source/SEAL/toolchain stages. Reserve at most another
    # 112 MiB here, below the approved 512 MiB ceiling. Stop on cache churn.
    derivations = re.findall(r"/nix/store/[a-z0-9]{32}-([^\s]+\.drv)", plan)
    if any(name not in ("stdenv-linux.drv", "nix-shell.drv") for name in derivations):
        raise RuntimeError("Shell plan adds unexpected source builds; review required")
    downloads = re.findall(r"\(([0-9.]+) MiB download,", plan)
    if "will be fetched" in plan and len(downloads) != 1:
        raise RuntimeError("Unrecognized shell download budget; review required")
    if sum(float(value) for value in downloads) > 112:
        raise RuntimeError("Shell download exceeds the reserved 112 MiB; review required")


def continue_cpp(run, verify):
    """Fail-closed sequence; run(stage, argv, seconds) must raise on any failure."""
    verify()
    # Acquiring/releasing the launcher's advisory lock waits for the original
    # build without running a second Nix process or trusting a stale PID/log.
    run("wait_existing_build", ["flock", "--exclusive", "--timeout", "43200",
        str(WORK / "deps/nix-portable-v012/launcher.lock"), "true"], 43210)
    verify()
    raw = run("dependency_metadata", WRAPPER + ["nix", "eval", "--offline",
        "--option", "allow-import-from-derivation", "false", "--json",
        *nix_platform_options(), "--file", DEPS, "metadata"], 120)
    metadata = json.loads(raw)
    outputs = metadata["compiler_outputs"]
    shell_bash = metadata["shell_bash"]
    if len(outputs) != 3 or any(not re.fullmatch(r"/nix/store/[a-z0-9]{32}-[^/\s]+", p)
                               for p in outputs):
        raise RuntimeError("Unexpected dependency output paths")
    if not re.fullmatch(r"/nix/store/[a-z0-9]{32}-bash-[^/\s]+/bin/bash", shell_bash):
        raise RuntimeError("Unexpected pinned Bash path")
    # Does not build anything. A failed/interrupted LLVM build stops here.
    run("dependency_validity", WRAPPER + ["nix-store", "--check-validity", *outputs,
        str(Path(shell_bash).parents[1])], 90)
    plan = run("remaining_shell_plan", WRAPPER + ["nix", "build", "--dry-run",
        "--option", "allow-import-from-derivation", "false", "--no-link",
        *nix_platform_options(), "--file", SHELL], 180)
    validate_remaining_plan(plan)
    verify()
    run("realize_shell", ["bash", "scripts/baseline/build_dacapo_dependencies.sh",
        "--approved", "shell"], 3600)
    verify()
    # Disabling substitution alone still allows source downloads/builds. Set the
    # implicit shell explicitly AND disable local/remote Nix builds. These limits
    # apply to Nix realization, not the CMake/Ninja command inside the shell.
    run("build_hecate_cpp", ["env", f"NIX_BUILD_SHELL={shell_bash}", *WRAPPER,
        "nix-shell", "--pure", *nix_environment_options(), "--option", "substitute", "false", "--max-jobs", "0",
        "--option", "builders", "", "--option", "allow-import-from-derivation", "false", *nix_platform_options(), SHELL,
        "--run", "bash scripts/baseline/build_hecate_cpp.sh"], 10800)
    verify()


def fingerprint():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in PROTECTED}


def check_checkout():
    checks = [
        (["git", "rev-parse", "HEAD"], "4995e7cadedf2bfb9104658b5638662ecf6a1d0a"),
        (["git", "branch", "--show-current"], "feat/agent-dsl-correctness"),
        (["git", "-C", "third_party/dacapo", "rev-parse", "HEAD"],
         "4616402710f39df3e5f5bd7930a6c036025aaac3"),
        (["git", "-C", "third_party/dacapo", "status", "--porcelain"], ""),
    ]
    for command, expected in checks:
        result = subprocess.run(["timeout", "-k", "3s", "20s", *command],
                                cwd=ROOT, capture_output=True, text=True, timeout=25, check=True)
        if result.stdout.strip() != expected:
            raise RuntimeError(f"Checkout changed: {' '.join(command)}; no switch/reset attempted")


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", action="store_true", required=True)
    parser.parse_args()
    def interrupted(signum, frame):
        raise RuntimeError(f"Continuation interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    if Path.cwd().resolve() != ROOT:
        raise SystemExit(f"Requires cwd {ROOT}")
    check_checkout()
    results = WORK / "results"
    results.mkdir(parents=True, exist_ok=True)
    # Only one continuation can own this sequence, including its waiting period.
    with (results / "dacapo-cpp-continuation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A C++ continuation is already running; not starting another")
        expected = fingerprint()
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        base = results / f"dacapo-cpp-continuation-{stamp}-{os.getpid()}"
        state_path = base.with_suffix(".json")
        state = {"scope": "cpp_build_only", "status": "running", "pid": os.getpid(),
                 "inputs_sha256": expected, "steps": [], "hecate_cpp_validated": False,
                 "tracing_validated": False, "encrypted_execution_validated": False}

        def persist():
            temporary = state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)

        def verify():
            check_checkout()
            if fingerprint() != expected:
                raise RuntimeError("Build recipe/driver changed while waiting; review required")

        def run(stage, command, limit):
            record = {"stage": stage, "command": command, "timeout_seconds": limit,
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            state["steps"].append(record)
            state["active_stage"] = stage
            log_path = Path(f"{base}.{stage}.log")
            record["log"] = str(log_path)
            persist()
            print(f"{stage}: {log_path}", flush=True)
            # GNU timeout bounds/kills the full process group, including build
            # descendants, not just Python's immediate wrapper process.
            stdout_path = Path(f"{base}.{stage}.stdout")
            # Keep eval JSON separate from diagnostic stderr; warnings must not
            # corrupt the metadata parser. Other stages are read as combined text.
            record["stdout"] = str(stdout_path)
            with log_path.open("w") as log, stdout_path.open("w") as output:
                process = subprocess.Popen(["timeout", "-k", "10s", f"{limit}s", *command],
                    cwd=ROOT, stdout=output, stderr=log, start_new_session=True,
                    env={**os.environ, "LC_ALL": "C", "PYTHONDONTWRITEBYTECODE": "1"})
                try:
                    code = process.wait(timeout=limit + 30)
                except BaseException:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
            record["exit_code"] = code
            persist()
            if code:
                raise RuntimeError(f"{stage} failed with exit {code}; see {log_path}")
            text = stdout_path.read_text()
            return text if stage == "dependency_metadata" else text + log_path.read_text()

        print(f"C++ continuation status: {state_path}", flush=True)
        try:
            continue_cpp(run, verify)
        except (Exception, KeyboardInterrupt) as error:
            state.update(status="failed", error=str(error))
            persist()
            raise SystemExit(str(error))
        state.update(status="complete", active_stage=None, hecate_cpp_validated=True)
        persist()
        print("C++ checks passed; tracing, HEVM and encrypted execution remain unvalidated.", flush=True)


if __name__ == "__main__":
    main()
