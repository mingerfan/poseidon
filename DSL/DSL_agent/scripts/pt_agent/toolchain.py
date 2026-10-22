"""The specified existing trace, compiler and ordinary CPU execution interfaces."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import secrets
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from .paths import WORK_RESULTS, WORK_ROOT

if TYPE_CHECKING:
    import numpy as np


RESULTS_ROOT = WORK_RESULTS
SDK_SELECTOR = RESULTS_ROOT / "agent-normal-cpu-approved-overlay-staging-r2680/sdk/ordinary-cpu-selector.json"
REPO = Path(__file__).resolve().parents[4]
BOOT_CONFIG_DIR = REPO / "DSL/DSL_benchmark/llm_dsl/Qwen25_2Layer_Decode_Autoboot_DSL/config"
COMPILE_MODES = ("eva", "dacapo-optimized", "dacapo-depth-dp", "pars")
BOOT_PROFILES = (None, "qwen48")
TRUSTED_LIMITS = dict(mem_gib=6, fsize_gib=8, wall_seconds=3600, min_free_gib=10)


def setup_env() -> None:
    """Set missing toolchain environment values before any baseline import."""
    os.environ.setdefault("POSEIDON_PLATFORM", "aarch64-linux")
    os.environ.setdefault("POSEIDON_WORK_ROOT", str(WORK_ROOT))
    os.environ.setdefault("POSEIDON_SDK_MANIFEST", str(SDK_SELECTOR))
    baseline = str(REPO / "DSL/DSL_agent/scripts/baseline")
    if baseline not in sys.path:
        sys.path.insert(0, baseline)


def host_paths() -> dict:
    """Resolve the existing host toolchain after setting up its environment."""
    setup_env()
    from poseidon_runtime_paths import BUILD, DACAPO, PROFILE, SPEC

    return dict(hecate_opt=BUILD / "bin/hecate-opt", hecate_build=BUILD, hecate=BUILD.parent,
                frontend=DACAPO / "python/hecate/hecate/expr.py", profile=PROFILE, spec=SPEC)


def retarget(run_dir: Path) -> None:
    """Save the sandbox source, then replace the three trusted emitter paths."""
    run_dir = Path(run_dir).resolve()
    source = run_dir / "candidate_trace.py"
    text = source.read_text()
    paths = host_paths()
    replacements = [
        (json.dumps("/frontend/hecate/expr.py"), json.dumps(str(paths["frontend"]))),
        (json.dumps("/out/constants.npz"), json.dumps(str(run_dir / "constants.npz"))),
        ('hc.save("/out", "/out")', f"hc.save({json.dumps(str(run_dir))}, {json.dumps(str(run_dir))})"),
    ]
    for old, _ in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Expected exactly one emitter path: {old}")
    backup = run_dir / "candidate_trace.sandbox.py"
    if backup.exists():
        raise ValueError("Preserving existing candidate_trace.sandbox.py")
    shutil.copy2(source, backup)
    for old, new in replacements:
        text = text.replace(old, new)
    source.write_text(text)


def to_host_argv(argv: list[str], run_dir: Path) -> list[str]:
    """Map compiler paths while rejecting any remaining sandbox path."""
    paths = host_paths()
    prefix = str(Path(run_dir).resolve()) + "/"
    mapped = []
    for arg in argv:
        if arg == "/hecate-opt":
            arg = str(paths["hecate_opt"])
        elif arg == "--ckks-config=/profile.json":
            arg = "--ckks-config=" + str(paths["profile"])
        arg = arg.replace("/out/", prefix)
        if any(path in arg for path in ("/out/", "/frontend", "/profile.json", "/operator-spec.json")):
            raise ValueError(f"Unmapped sandbox argument: {arg}")
        mapped.append(arg)
    return mapped


def trusted_run(run_dir: Path, argv: list[str], log: Path, *, env_extra: dict | None = None) -> dict:
    """Run EmitBackend output inside Nix with the separately approved limits."""
    run_dir, log = Path(run_dir), Path(log)
    if shutil.disk_usage(run_dir).free < TRUSTED_LIMITS["min_free_gib"] * 1024**3:
        raise RuntimeError("Trusted runner requires at least 10 GiB free disk space")

    def limits():
        memory = TRUSTED_LIMITS["mem_gib"] * 1024**3
        file_size = TRUSTED_LIMITS["fsize_gib"] * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))

    time_path = Path(str(log) + ".time")
    command = ["/usr/bin/time", "-f", "%M %e", "-o", str(time_path), *argv]
    started = time.monotonic()
    timed_out = False
    with log.open("w") as stream:
        try:
            completed = subprocess.run(command, cwd=run_dir, env={**os.environ, **(env_extra or {})},
                                       stdout=stream, stderr=subprocess.STDOUT, preexec_fn=limits,
                                       timeout=TRUSTED_LIMITS["wall_seconds"])
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code, timed_out = -9, True
    seconds = time.monotonic()-started
    peak_rss_mib = None
    if time_path.exists():
        for line in reversed(time_path.read_text().splitlines()):
            fields = line.split()
            if len(fields) == 2:
                try:
                    peak_kib, measured_seconds = map(float, fields)
                except ValueError:
                    continue
                peak_rss_mib = peak_kib/1024
                seconds = measured_seconds
                break
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size-3200))
        tail = stream.read().decode(errors="replace")[-800:]
    return dict(exit=exit_code, seconds=seconds, peak_rss_mib=peak_rss_mib,
                timeout=timed_out, log_tail=tail)


def new_run_dir(tag: str) -> Path:
    """Create a unique results directory with the sandbox's empty payload."""
    if not isinstance(tag, str) or not re.fullmatch(r"[a-z0-9-]+", tag):
        raise ValueError("tag must contain only lowercase letters, digits and hyphens")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = RESULTS_ROOT / f"pt-agent-{tag}-{stamp}-{secrets.token_hex(2)}"
    directory.mkdir(parents=True)
    (directory / "payload.json").write_text("{}\n")
    return directory


def _replace(argv, match, replacement):
    positions = [i for i, arg in enumerate(argv) if match(arg)]
    if len(positions) != 1:
        raise ValueError(f"Expected exactly one compiler argument for {replacement}")
    argv[positions[0]] = replacement


def compile_argv(mode: str, boot: str | None, run_dir: Path) -> list[str]:
    """Build arguments only; setup_env must precede this function's imports."""
    if mode not in COMPILE_MODES or boot not in BOOT_PROFILES:
        raise ValueError("Unknown compile mode or boot profile")
    if boot is not None and mode == "eva":
        raise ValueError("eva does not support the qwen48 boot profile")
    from poseidon_runtime_compile import command
    from compiler_configuration import COMPILER_STATE_POLICY, RUNTIME_PLAN_SERIALIZATION_POLICY

    base = command(45, serialization_policy=RUNTIME_PLAN_SERIALIZATION_POLICY)
    if mode.startswith("dacapo-"):
        argv = list(command(45, serialization_policy=RUNTIME_PLAN_SERIALIZATION_POLICY,
                            state_management_policy=COMPILER_STATE_POLICY))
        if mode == "dacapo-depth-dp":
            _replace(argv, lambda arg: arg == "--boot-placement=optimized", "--boot-placement=depth-dp")
    else:
        argv = list(base)
        if mode == "pars":
            _replace(argv, lambda arg: arg == "--eva", "--pars")
    if boot == "qwen48":
        _replace(argv, lambda arg: arg == "--ckks-config=/profile.json", "--ckks-config=/out/boot-compiler-config.json")
        _replace(argv, lambda arg: arg == "--waterline=45", "--waterline=43")
        _replace(argv, lambda arg: arg.startswith("--runtime-plan-operator-spec-id="),
                 "--runtime-plan-operator-spec-id=cpu-qwen-bootstrap48-compiler-contract")
        # This digest supplies the compiler's required plan metadata; it performs no integrity comparison.
        operator_spec = hashlib.sha256((BOOT_CONFIG_DIR / "operator-spec.json").read_bytes()).hexdigest()
        _replace(argv, lambda arg: arg.startswith("--runtime-plan-operator-spec-sha256="),
                 "--runtime-plan-operator-spec-sha256=sha256:" + operator_spec)
        _replace(argv, lambda arg: arg.startswith("--runtime-plan-context-id="),
                 "--runtime-plan-context-id=qwen-bootstrap48")
        if argv.count("-o") != 1:
            raise ValueError("Expected exactly one compiler output option")
        argv.insert(argv.index("-o"), "--runtime-plan-boot-profile=cpu-bootstrap48-level1-to5")
    return argv


def trace(run_dir: Path, seconds: int = 600, *, trusted=False) -> dict:
    """inside Nix. Trace emitted source through the selected runner."""
    setup_env()
    from hecate_python_env import VENV

    run_dir = Path(run_dir)
    for name in ("candidate_trace.py", "constants.npz"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(run_dir / name)
    if trusted:
        retarget(run_dir)
        paths = host_paths()
        result = trusted_run(run_dir, [str(VENV / "bin/python"), "-B", str(run_dir / "candidate_trace.py")],
                             run_dir / "trace.log", env_extra=dict(HECATE=str(paths["hecate"]),
                             HECATE_BUILD=str(paths["hecate_build"]),
                             LD_LIBRARY_PATH=os.environ["HECATE_PYTHON_LIBRARY_PATH"]))
        mlir = run_dir / "candidate_trace.mlir"
        result.update(mlir=mlir.is_file(), mlir_bytes=mlir.stat().st_size if mlir.is_file() else 0)
        return result
    from poseidon_candidate_sandbox import run

    started = time.monotonic()
    exit_code = run(run_dir / "payload.json", run_dir,
                    [str(VENV / "bin/python"), "-B", "/out/candidate_trace.py"], run_dir / "trace.log", seconds)
    return dict(exit=exit_code, seconds=time.monotonic()-started,
                mlir=(run_dir / "candidate_trace.mlir").is_file())


def compile(run_dir: Path, mode: str, boot: str | None = None, seconds: int = 900, *, trusted=False) -> dict:
    """inside Nix. Compile once through the selected runner and preserve its log."""
    setup_env()

    run_dir = Path(run_dir)
    if boot == "qwen48":
        shutil.copyfile(BOOT_CONFIG_DIR / "compiler-config.json", run_dir / "boot-compiler-config.json")
        shutil.copyfile(BOOT_CONFIG_DIR / "operator-spec.json", run_dir / "boot-operator-spec.json")
    argv = compile_argv(mode, boot, run_dir)
    if trusted:
        result = trusted_run(run_dir, to_host_argv(argv, run_dir), run_dir / "compile.log",
                             env_extra=dict(LD_LIBRARY_PATH=os.environ["HECATE_PYTHON_LIBRARY_PATH"]))
        result["plan"] = (run_dir / "lowered._hecate_golden.runtime-plan.json").is_file()
        return result
    from poseidon_candidate_sandbox import run

    started = time.monotonic()
    exit_code = run(run_dir / "payload.json", run_dir, argv, run_dir / "compile.log", seconds)
    return dict(exit=exit_code, seconds=time.monotonic()-started,
                plan=(run_dir / "lowered._hecate_golden.runtime-plan.json").is_file(),
                log_tail=(run_dir / "compile.log").read_text(errors="replace")[-800:])


def plan_stats(run_dir: Path) -> dict:
    """Read operation counts from the unmodified compiled RuntimePlan."""
    plan = json.loads((Path(run_dir) / "lowered._hecate_golden.runtime-plan.json").read_text())
    steps = plan["initialization"] + plan["execution"] + plan["finalization"]
    ops = dict(Counter(step.get("op") or step["kind"] for step in steps))
    return dict(ops=ops, steps=len(steps), boots=ops.get("boot", 0), values=len(plan["values"]),
                inputs=len(plan["external_inputs"]), outputs=len(plan["final_outputs"]))


def prune(run_dir: Path):
    """Remove only the intermediates approved in R6b §4, retaining plans and logs."""
    run_dir = Path(run_dir)
    deleted = []
    for name in ("candidate_trace.mlir", "constants.npz", "lowered.mlir", "lowered.earth.mlir",
                 "lowered.ckks.mlir", "lowered._hecate_golden.bundle"):
        path = run_dir / name
        if path.is_symlink() or path.is_file():
            path.unlink()
            deleted.append(name)
        elif path.is_dir():
            shutil.rmtree(path)
            deleted.append(name+"/")
    timestamp = datetime.now(timezone.utc).isoformat()
    with (run_dir / "PRUNED.txt").open("a") as stream:
        stream.write(timestamp+"\nUser approved: R6b-cost-refresh.md §4 (2026-10-21).\n")
        stream.write("Deleted:\n"+"".join(name+"\n" for name in deleted)+"\n")
    return dict(deleted=deleted, ts=timestamp, approval="R6b-cost-refresh.md §4")


def bind(run_dir: Path, period: int) -> dict:
    """inside nix. Use the existing periodic-constant binder with its original record."""
    setup_env()
    from poseidon_artifact_gate import bind_periodic_constants

    run_dir = Path(run_dir)
    return bind_periodic_constants(run_dir / "lowered._hecate_golden.runtime-plan.json",
                                   run_dir / "lowered._hecate_golden.bundle", run_dir / "bound", period)


def write_inputs(run_dir: Path, samples: list[list[np.ndarray]], period: int) -> None:
    """Validate ordinary-protocol sample vectors and write inputs.json."""
    setup_env()
    import numpy as np
    from packed_input_abi import PERIODS

    if type(period) is not int or period not in PERIODS:
        raise ValueError("period must belong to the ordinary packed-input ABI")
    if not isinstance(samples, list) or not 1 <= len(samples) <= 16:
        raise ValueError("Expected 1..16 samples")
    normalized = []
    for sample in samples:
        if not isinstance(sample, list) or not sample:
            raise ValueError("Each sample must contain input vectors")
        vectors = []
        for vec in sample:
            value = np.asarray(vec, dtype=np.float64)
            if value.shape != (period,) or not np.isfinite(value).all():
                raise ValueError("Input vectors must be finite and have length period")
            vectors.append(value)
        normalized.append(vectors)
    (Path(run_dir) / "inputs.json").write_text(json.dumps([[v.tolist() for v in sample] for sample in normalized], allow_nan=False))


def execute_cpu(run_dir: Path, seconds: int = 150) -> np.ndarray:
    """inside nix. Execute once with original CPU limits and return final decoded values."""
    setup_env()
    import numpy as np
    from normal_cpu_overlay_sandbox import run
    from poseidon_runtime_paths import POSEIDON_BUILD

    run_dir = Path(run_dir)
    argv = ["/poseidon-agent-cpu", "/out/bound/plan.json", "/operator-spec.json", "/out/bound/bundle",
            "/out/inputs.json", "/out/execution.json"]
    exit_code = run(run_dir / "payload.json", run_dir, argv, run_dir / "execute.log", seconds,
                    selected_library=POSEIDON_BUILD / "libposeidon_shared.so")
    if exit_code != 0:
        raise RuntimeError(f"CPU execution exit={exit_code}: " + (run_dir / "execute.log").read_text(errors="replace")[-800:])
    report = json.loads((run_dir / "execution.json").read_text())
    if (report.get("encrypted_execution") is not True or report.get("bootstrap") is not False
            or report.get("executor_secret_key") is not False):
        raise RuntimeError("CPU report does not confirm the required encrypted execution flags")
    outputs = np.asarray(report["outputs"], dtype=np.float64)
    if outputs.ndim != 3 or outputs.shape[2] != 256 or not np.isfinite(outputs).all():
        raise RuntimeError("CPU outputs must be finite [sample][output][256] values")
    return outputs
