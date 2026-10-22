"""R2b's six serial real-toolchain smoke combinations, with no retries."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .. import toolchain


def _save_report(directory, report):
    path = directory / "REPORT.json"
    temporary = directory / "REPORT.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def _remaining(deadline, default):
    remaining = int(deadline-time.monotonic()) - 10
    if remaining <= 0:
        raise TimeoutError("r2b_smoke reached its 40-minute wall-clock limit")
    return min(default, remaining)


def _inside():
    toolchain.setup_env()
    import numpy as np
    from ..backends import EmitBackend, MockBackend, run_program

    period = 256

    def c1(backend, x0, x1):
        c = backend.const(np.arange(period)/period)
        return (x0*c + x0.rotate(3))*x1 - 0.5

    def c2(backend, x0):
        y = x0*0.6 + 0.5
        for coefficient in (0.4, 0.3, 0.2, 0.1):
            y = y*x0 + coefficient
        return y

    def c3_auto(backend, x0):
        y = x0
        for _ in range(8):
            y = y*y
        return y

    def c3_agent(backend, x0):
        y = x0
        for _ in range(4):
            y = y*y
        y = y.bootstrap()
        for _ in range(4):
            y = y*y
        return y

    cases = [("c1-rotmul", c1, 2, "eva", None), ("c1-rotmul", c1, 2, "dacapo-optimized", None),
             ("c2-horner", c2, 1, "eva", None), ("c2-horner", c2, 1, "dacapo-optimized", None),
             ("c3-boot-auto", c3_auto, 1, "dacapo-depth-dp", "qwen48"),
             ("c3-boot-agent", c3_agent, 1, "pars", "qwen48")]
    summary_dir = toolchain.new_run_dir("r2b-summary")
    report = dict(round="R2b", cases=[])
    _save_report(summary_dir, report)
    deadline = time.monotonic() + 40*60
    rng = np.random.default_rng(20261023)
    for name, fn, n_inputs, mode, boot in cases:
        directory = toolchain.new_run_dir(f"r2b-{name}-{mode}")
        row = dict(case=name, mode=mode, run_dir=str(directory), trace=None, compile=None, plan=None, execute=None)
        try:
            _remaining(deadline, 600)
            emitter = EmitBackend(period)
            run_program(emitter, fn, n_inputs)
            emitter.write(directory)
            row["trace"] = toolchain.trace(directory, seconds=_remaining(deadline, 600))
            if row["trace"]["exit"] != 0 or not row["trace"]["mlir"]:
                raise RuntimeError("trace failed: " + (directory / "trace.log").read_text(errors="replace")[-800:])
            row["compile"] = toolchain.compile(directory, mode, boot, seconds=_remaining(deadline, 900))
            if row["compile"]["exit"] != 0 or not row["compile"]["plan"]:
                raise RuntimeError("compile failed: " + row["compile"]["log_tail"])
            row["plan"] = toolchain.plan_stats(directory)
            if boot is None:
                samples = [[rng.uniform(-1, 1, period) for _ in range(n_inputs)] for _ in range(4)]
                references = []
                for sample in samples:
                    mock = MockBackend(period)
                    mock.bind(sample)
                    run_program(mock, fn, n_inputs)
                    references.append(mock.outputs)
                reference = np.asarray(references, dtype=np.float64)
                toolchain.bind(directory, period)
                toolchain.write_inputs(directory, samples, period)
                ckks = toolchain.execute_cpu(directory, seconds=_remaining(deadline, 150))[:, :, :period]
                if ckks.shape != reference.shape:
                    raise RuntimeError(f"CKKS output shape {ckks.shape} differs from Mock {reference.shape}")
                difference = np.abs(ckks-reference)
                mask = np.abs(reference) > 1e-12
                row["execute"] = dict(samples=4, max_abs_err=float(np.max(difference)),
                    max_rel_err=float(np.max(difference[mask]/np.abs(reference[mask]))) if np.any(mask) else 0.0,
                    passed=bool(np.all(difference <= 1e-5 + 1e-4*np.abs(reference))))
                if not row["execute"]["passed"]:
                    raise RuntimeError("CKKS outputs exceeded the specified per-element tolerance")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        report["cases"].append(row)
        _save_report(summary_dir, report)
        print(f"CASE={name} MODE={mode} RESULT={'error' if 'error' in row else 'done'}", flush=True)
        if time.monotonic() >= deadline:
            break
    path = _save_report(summary_dir, report)
    print("REPORT=" + str(path), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    if args.inside:
        return _inside()
    if not os.environ.get("PT_RESULTS"):
        raise RuntimeError("PT_RESULTS must be set in the outer environment")
    toolchain.setup_env()
    scripts_dir = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-m", "pt_agent.tools.nixrun", "--seconds", "2700", "--", "-m",
                                "pt_agent.experiments.r2b_smoke", "--inside"], cwd=scripts_dir,
                               capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    reports = [line[len("REPORT="):] for line in completed.stdout.splitlines() if line.startswith("REPORT=")]
    if not reports:
        raise RuntimeError(f"r2b_smoke produced no REPORT path (exit={completed.returncode})")
    report = json.loads(Path(reports[-1]).read_text())
    from ..registry import append, git_rev, new_run_id

    rev = git_rev()
    for case in report["cases"]:
        append(dict(run_id=new_run_id("R2b", case["case"], case["mode"]), round="R2b", model=case["case"],
                    mode=case["mode"], git_rev=rev, metrics=case, artifacts_dir=case["run_dir"],
                    ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
