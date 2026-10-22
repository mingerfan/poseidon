"""Trace and attempt one trusted compilation of the retained R5c Emit program."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .. import toolchain


def _write(out, report):
    temporary = out / "REPORT.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out / "REPORT.json")


def _server_job(out, emit_dir):
    scripts = Path(__file__).resolve().parents[2]
    path = out / "server_jobs/r5d-resnet20/run.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "# Prepare the same pinned Nix environment and toolchain directories on the server.\n"
        "# Preserve the original Emit directory with its sandbox paths and constants.\n"
        "export POSEIDON_WORK_ROOT=" + shlex.quote(os.environ["POSEIDON_WORK_ROOT"]) + "\n"
        "export POSEIDON_SDK_MANIFEST=" + shlex.quote(os.environ["POSEIDON_SDK_MANIFEST"]) + "\n"
        "export PT_RESULTS=" + shlex.quote(str(out.parent)) + "\n"
        "pt_agent_run_dir=" + shlex.quote(str(emit_dir)) + "\n"
        "cd " + shlex.quote(str(scripts)) + "\n"
        'python3 -B -m pt_agent.experiments.r5d_compile --emit-dir "$pt_agent_run_dir"\n')
    path.chmod(0o755)
    return path


def _inside(out, emit_dir):
    toolchain.setup_env()
    report = json.loads((out / "REPORT.json").read_text())
    stage = "copy"
    try:
        directory = toolchain.new_run_dir("r5d-resnet20")
        report["run_dir"] = str(directory)
        _write(out, report)
        for name in ("candidate_trace.py", "constants.npz"):
            shutil.copy2(emit_dir / name, directory / name)
        stage = report["stage"] = "trace"
        _write(out, report)
        report["trace"] = toolchain.trace(directory, trusted=True)
        _write(out, report)
        if report["trace"]["exit"] != 0 or not report["trace"]["mlir"]:
            raise RuntimeError("Trusted trace failed")
        stage = report["stage"] = "compile"
        _write(out, report)
        report["compile"] = toolchain.compile(directory, "dacapo-depth-dp", boot="qwen48", trusted=True)
        _write(out, report)
        if report["compile"]["exit"] != 0 or not report["compile"]["plan"]:
            raise RuntimeError("Trusted compile failed")
        stage = report["stage"] = "plan"
        report["plan"] = toolchain.plan_stats(directory)
        report.update(complete=True, stage="complete")
    except Exception as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        if stage in ("trace", "compile") and report[stage] is None:
            report[stage] = dict(exit=None, seconds=None, peak_rss_mib=None,
                                 timeout=isinstance(exc, subprocess.TimeoutExpired), log_tail=str(exc))
        report["server_job"] = str(_server_job(out, emit_dir))
    _write(out, report)
    print("REPORT=" + str(out / "REPORT.json"), flush=True)
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dir", type=Path, required=True)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    emit_dir = args.emit_dir.resolve()
    if args.inside:
        if args.out is None:
            parser.error("--inside requires --out")
        return _inside(args.out, emit_dir)
    if not os.environ.get("PT_RESULTS"):
        raise RuntimeError("PT_RESULTS must be set in the outer environment")
    toolchain.setup_env()
    from ..registry import append, git_rev, load, new_run_id

    counts = None
    for row in reversed(load()):
        emit = row["metrics"].get("emit") or {}
        if emit.get("run_dir") and Path(emit["run_dir"]).resolve() == emit_dir:
            counts = emit["counts"]
            break
    if counts is None:
        raise ValueError("Emit directory has no retained R5c emit record")
    out = Path(os.environ["PT_RESULTS"]) / ("r5d-compile-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True)
    report = dict(round="R5d", complete=False, stage="copy", emit_dir=str(emit_dir), run_dir=None,
                  counts=counts, trace=None, compile=None, plan=None, server_job=None)
    _write(out, report)
    scripts = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-B", "-m", "pt_agent.tools.nixrun", "--seconds", "7320", "--",
                                "-m", "pt_agent.experiments.r5d_compile", "--inside", "--out", str(out),
                                "--emit-dir", str(emit_dir)], cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    report = json.loads((out / "REPORT.json").read_text())
    if completed.returncode and report.get("server_job") is None:
        report.update(complete=False, error=f"inner process exit={completed.returncode}",
                      server_job=str(_server_job(out, emit_dir)))
        _write(out, report)
    append(dict(run_id=new_run_id("R5d", "resnet20", "dacapo-depth-dp-qwen48"), round="R5d", model="resnet20",
                mode="dacapo-depth-dp-qwen48", git_rev=git_rev(), metrics=report, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
