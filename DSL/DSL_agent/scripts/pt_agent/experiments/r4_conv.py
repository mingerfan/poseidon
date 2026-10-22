"""R4's two real CKKS convolution checks and one ResNet conv1 compile check."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .. import toolchain


def _save(directory, report):
    path = directory / "REPORT.json"
    temporary = directory / "REPORT.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def _remaining(deadline, default):
    remaining = int(deadline-time.monotonic())-10
    if remaining <= 0:
        raise TimeoutError("r4_conv reached the inherited 40-minute experiment limit")
    return min(default, remaining)


def _conv_ref(x, w, stride):
    import numpy as np

    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))[:, ::stride, ::stride]
    return np.einsum("cyxij,ocij->oyx", windows, w, optimize=True)


def _inside():
    toolchain.setup_env()
    import numpy as np
    from ..backends import CountBackend, EmitBackend, MockBackend, run_program
    from ..kernels import add_mux, conv2d_mux, downsample_pad_mux, mux_layout, pack_mux

    summary = toolchain.new_run_dir("r4-summary")
    report = dict(round="R4", cases=[])
    _save(summary, report)
    rng = np.random.default_rng(20261027)
    deadline = time.monotonic()+40*60
    cases = [("r4-t1-ckks", (256, 4, 4, 4, 1), 4, 1),
             ("r4-t7-ckks", (256, 4, 4, 4, 1), 8, 2),
             ("r4-resnet-conv1-compile", (16384, 3, 32, 32, 1), 16, 1)]
    for name, dims, Co, stride in cases:
        directory = toolchain.new_run_dir(name)
        row = dict(case=name, mode="eva", run_dir=str(directory), counts=None, max_depth=None,
                   trace=None, compile=None, plan=None, execute=None)
        try:
            _remaining(deadline, 600)
            lin = mux_layout(*dims)
            lout = mux_layout(lin.n, Co, lin.H//stride, lin.W//stride, lin.k*stride)
            weight = rng.normal(0, 0.3, (Co, lin.C, 3, 3))

            def program(b, x):
                value, layout = conv2d_mux(b, x, lin, weight, stride=stride)
                if name == "r4-t7-ckks":
                    shortcut, shortcut_layout = downsample_pad_mux(b, x, lin, 2, 2)
                    value, _ = add_mux(b, value, layout, shortcut, shortcut_layout)
                return value

            counter = CountBackend(lin.n)
            run_program(counter, program, 1)
            stats = counter.stats()
            row.update(counts=stats["counts"], max_depth=stats["max_depth"])
            emitter = EmitBackend(lin.n)
            run_program(emitter, program, 1)
            emitter.write(directory)
            row["trace"] = toolchain.trace(directory, seconds=_remaining(deadline, 600))
            if row["trace"]["exit"] != 0 or not row["trace"]["mlir"]:
                raise RuntimeError("trace failed: " + (directory / "trace.log").read_text(errors="replace")[-800:])
            row["compile"] = toolchain.compile(directory, "eva", seconds=_remaining(deadline, 900))
            if row["compile"]["exit"] != 0 or not row["compile"]["plan"]:
                raise RuntimeError("compile failed: " + row["compile"]["log_tail"])
            row["plan"] = toolchain.plan_stats(directory)
            if lin.n == 256:
                samples = []
                mocks = []
                references = []
                for _ in range(4):
                    image = rng.uniform(-1, 1, (lin.C, lin.H, lin.W))
                    vector = pack_mux(image, lin)
                    samples.append([vector])
                    mock = MockBackend(lin.n)
                    mock.bind([vector])
                    run_program(mock, program, 1)
                    mocks.append(mock.outputs[0])
                    expected = _conv_ref(image, weight, stride)
                    if name == "r4-t7-ckks":
                        expected = expected+np.pad(image[:, ::2, ::2], ((2, 2), (0, 0), (0, 0)))
                    references.append(pack_mux(expected, lout))
                mock_values = np.asarray(mocks)
                reference = np.asarray(references)
                toolchain.bind(directory, 256)
                toolchain.write_inputs(directory, samples, 256)
                ckks = toolchain.execute_cpu(directory, seconds=_remaining(deadline, 150))[:, 0, :256]
                if ckks.shape != reference.shape:
                    raise RuntimeError(f"CKKS shape {ckks.shape} differs from reference {reference.shape}")
                difference = np.abs(ckks-reference)
                difference_mock = np.abs(ckks-mock_values)
                passed_reference = bool(np.all(difference <= 1e-5+1e-4*np.abs(reference)))
                passed_mock = bool(np.all(difference_mock <= 1e-5+1e-4*np.abs(mock_values)))
                mask = np.abs(reference) > 1e-12
                row["execute"] = dict(samples=4, max_abs_err=float(np.max(difference)),
                    max_abs_err_vs_mock=float(np.max(difference_mock)), max_abs_err_vs_reference=float(np.max(difference)),
                    max_rel_err=float(np.max(difference[mask]/np.abs(reference[mask]))) if np.any(mask) else 0.0,
                    passed_vs_mock=passed_mock, passed_vs_reference=passed_reference,
                    passed=passed_mock and passed_reference)
                if not row["execute"]["passed"]:
                    raise RuntimeError("CKKS output exceeded the specified per-element tolerance")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        report["cases"].append(row)
        _save(summary, report)
        print(f"CASE={name} RESULT={'error' if 'error' in row else 'done'}", flush=True)
        if time.monotonic() >= deadline:
            break
    path = _save(summary, report)
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
    scripts = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-m", "pt_agent.tools.nixrun", "--seconds", "2700", "--", "-m",
                                "pt_agent.experiments.r4_conv", "--inside"], cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    paths = [line[len("REPORT="):] for line in completed.stdout.splitlines() if line.startswith("REPORT=")]
    if not paths:
        raise RuntimeError(f"r4_conv produced no REPORT path (exit={completed.returncode})")
    report = json.loads(Path(paths[-1]).read_text())
    from ..registry import append, git_rev, new_run_id

    revision = git_rev()
    for case in report["cases"]:
        append(dict(run_id=new_run_id("R4", case["case"], case["mode"]), round="R4", model=case["case"],
                    mode=case["mode"], git_rev=revision, metrics=case, artifacts_dir=case["run_dir"],
                    ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
