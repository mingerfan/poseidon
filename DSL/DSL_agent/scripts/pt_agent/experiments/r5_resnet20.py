"""R5 calibration-only ResNet20 assembly, approximation grid and compile attempt."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from ..paths import pt_result


R3_RESULTS = pt_result('r3-resnet20-20261020T131658')


def _write(path, record):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _server_job(out, directory):
    scripts = Path(__file__).resolve().parents[2]
    code = ("from pathlib import Path\nfrom pt_agent import toolchain\ntoolchain.setup_env()\n"
            f"directory = Path({str(directory)!r})\n"
            "traced = toolchain.trace(directory)\nprint(traced, flush=True)\n"
            "if traced['exit'] != 0 or not traced['mlir']: raise SystemExit(1)\n"
            "compiled = toolchain.compile(directory, 'dacapo-depth-dp', boot='qwen48', seconds=3600)\n"
            "print(compiled, flush=True)\n"
            "if compiled['exit'] != 0 or not compiled['plan']: raise SystemExit(1)\n"
            "print(toolchain.plan_stats(directory), flush=True)\n")
    path = out / "server_jobs/r5-resnet20/run.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(scripts)) + "\n"
                    "python3 -m pt_agent.tools.nixrun --seconds 4500 -- -c " + shlex.quote(code) + "\n")
    path.chmod(0o755)
    return path


def _grid_case(graph, cfg, images, labels, references, *, degree, margin, on_progress=None):
    import numpy as np
    from ..backends import BackendError, MockBackend
    from ..models.resnet20_fhe import build, decode_logits, pack_input

    row = dict(deg=degree, margin=margin, diverged=False, images_done=0, diverged_batch_start=None,
               completed_samples=0, agree=0, top1=None, logit_rmse=None, max_abs_logit_diff=0.0,
               max_u=0.0, max_u_act=None, seconds=0.0, out_of_domain=[])
    started = time.monotonic()
    squared = 0.0
    correct = 0
    reference_argmax = references.argmax(axis=1)
    if on_progress is not None:
        on_progress(row)
    for start in range(0, len(images), 50):
        batch_images = images[start:start+50]
        batch_out_of_domain = 0
        grids = {}

        def observer(name, current, current_layout, current_cfg):
            nonlocal batch_out_of_domain
            if current_layout not in grids:
                grids[current_layout] = current_layout.index_grid()
            # Valid physical mux slots include their layout copies.
            data = current.value[..., grids[current_layout]]
            lo, hi = current_cfg[2:]
            batch_out_of_domain += int(np.count_nonzero((data < lo) | (data > hi)))
            max_u = float(np.max(np.abs(data*(2/(hi-lo))-(hi+lo)/(hi-lo))))
            if row["max_u_act"] is None or max_u > row["max_u"]:
                row.update(max_u=max_u, max_u_act=name)

        backend = MockBackend(16384, batch=len(batch_images))
        try:
            backend.bind([np.stack([pack_input(image) for image in batch_images])])
            value, layout = build(backend, graph, cfg, observer=observer)
            logits = decode_logits(value.value, layout).copy()
        except BackendError as exc:
            row.update(diverged=True, diverged_batch_start=start, error=str(exc),
                       seconds=time.monotonic()-started)
            if on_progress is not None:
                on_progress(row)
            print(f"GRID deg={degree} margin={margin} diverged_batch_start={start} images_done={row['images_done']}", flush=True)
            break
        finally:
            del backend
            gc.collect()
        stop = start+len(batch_images)
        difference = logits-references[start:stop]
        next_squared = squared+float(np.sum(difference*difference))
        if not np.isfinite(next_squared):
            raise RuntimeError("grid logits produced a nonfinite squared error")
        squared = next_squared
        predictions = logits.argmax(axis=1)
        row["agree"] += int(np.sum(predictions == reference_argmax[start:stop]))
        correct += int(np.sum(predictions == labels[start:stop]))
        row["out_of_domain"].append(batch_out_of_domain)
        row.update(images_done=stop, completed_samples=stop, top1=correct*100.0/stop,
                   logit_rmse=float(np.sqrt(squared/(stop*references.shape[1]))),
                   max_abs_logit_diff=max(row["max_abs_logit_diff"], float(np.max(np.abs(difference)))))
        del value, logits
        gc.collect()
        row["seconds"] = time.monotonic()-started
        if on_progress is not None:
            on_progress(row)
        print(f"GRID deg={degree} margin={margin} samples={stop} agree={row['agree']} out_of_domain={batch_out_of_domain}", flush=True)
    return row


def _inside(out):
    import numpy as np
    from .. import toolchain
    from ..analyzer import reference_forward
    from ..backends import CountBackend, EmitBackend, MockBackend
    from ..data.cifar import calibration_set
    from ..intake import Graph
    from ..models.resnet20_fhe import build, decode_logits, pack_input

    toolchain.setup_env()
    path = out / "REPORT.json"
    report = dict(round="R5", complete=False, stage="exact", exact_max_abs_diff=None, build_seconds=None,
                  count=None, grid=[], emit=None)
    _write(path, report)

    def expired(signum, frame):
        raise TimeoutError("r5_resnet20 exceeded 3 hours")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(3*60*60)
    try:
        graph = Graph.load(R3_RESULTS / "graph")
        calibration = json.loads((R3_RESULTS / "calibration.json").read_text())
        images, labels = calibration_set()
        acts = [node for node in graph.nodes if node.op == "act"]

        def config(degree, margin):
            return {node.inputs[0]: ("cheb", degree, calibration[node.inputs[0]]["min"]*margin,
                                     calibration[node.inputs[0]]["max"]*margin) for node in acts}

        exact_cfg = {node.inputs[0]: ("exact",) for node in acts}
        backend = MockBackend(16384, batch=8)
        backend.bind([np.stack([pack_input(image) for image in images[:8]])])
        started = time.monotonic()
        value, layout = build(backend, graph, exact_cfg)
        report["build_seconds"] = time.monotonic()-started
        logits = decode_logits(value.value, layout).copy()
        reference = reference_forward(graph, {"x": images[:8]})[graph.outputs[0]]
        report["exact_max_abs_diff"] = float(np.max(np.abs(logits-reference)))
        _write(path, report)
        if report["exact_max_abs_diff"] > 1e-9:
            raise RuntimeError("exact assembly exceeded the specified logits tolerance")
        print(f"EXACT max_abs_diff={report['exact_max_abs_diff']} build_seconds={report['build_seconds']}", flush=True)
        del backend, value, logits, reference
        gc.collect()

        report["stage"] = "count"
        _write(path, report)
        counter = CountBackend(16384)
        value, _ = build(counter, graph, config(63, 1.0))
        counter.output([value])
        report["count"] = counter.stats()
        report["count"]["rot_steps_count"] = len(report["count"]["rot_steps"])
        _write(path, report)
        print(f"COUNT max_depth={report['count']['max_depth']} counts={report['count']['counts']}", flush=True)
        del counter, value
        gc.collect()

        report["stage"] = "reference"
        _write(path, report)
        references = np.concatenate([reference_forward(graph, {"x": images[start:start+200]})[graph.outputs[0]]
                                     for start in range(0, len(images), 200)])
        report["stage"] = "grid"
        _write(path, report)
        for degree in (31, 63, 127):
            for margin in (1.0, 1.25):
                def progress(row):
                    if not report["grid"] or report["grid"][-1] is not row:
                        report["grid"].append(row)
                    _write(path, report)

                _grid_case(graph, config(degree, margin), images, labels, references,
                           degree=degree, margin=margin, on_progress=progress)

        candidates = [row for row in report["grid"] if not row["diverged"]]
        if not candidates:
            report.update(emit={"skipped": "all configurations diverged"}, complete=True, stage="complete")
            _write(path, report)
            print("REPORT=" + str(path), flush=True)
            return 0
        selected = max(candidates, key=lambda row: (row["agree"], -row["deg"]))
        report["stage"] = "emit"
        directory = toolchain.new_run_dir("r5-resnet20-emit")
        report["emit"] = dict(deg=selected["deg"], margin=selected["margin"], run_dir=str(directory),
                              counts=None, constants_bytes=None, trace=None, compile=None, plan=None, server_job=None)
        _write(path, report)
        emitter = EmitBackend(16384)
        value, _ = build(emitter, graph, config(selected["deg"], selected["margin"]))
        emitter.output([value])
        report["emit"]["counts"] = emitter.stats()["counts"]
        _write(path, report)
        emitter.write(directory)
        size = (directory / "constants.npz").stat().st_size
        report["emit"]["constants_bytes"] = size
        del emitter, value
        gc.collect()
        _write(path, report)
        if size > 1.5*1024**3:
            report["emit"]["server_job"] = str(_server_job(out, directory))
        else:
            report["stage"] = "compile"
            _write(path, report)
            traced = toolchain.trace(directory)
            report["emit"]["trace"] = traced
            if traced["exit"] == 0 and traced["mlir"]:
                compiled = toolchain.compile(directory, "dacapo-depth-dp", boot="qwen48", seconds=3600)
                report["emit"]["compile"] = compiled
                if compiled["exit"] == 0 and compiled["plan"]:
                    report["emit"]["plan"] = toolchain.plan_stats(directory)
            else:
                report["emit"]["log_tail"] = (directory / "trace.log").read_text(errors="replace")[-800:]
        report.update(complete=True, stage="complete")
        _write(path, report)
        print("REPORT=" + str(path), flush=True)
        return 0
    except Exception as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
        _write(path, report)
        print(report["error"], file=sys.stderr, flush=True)
        print("REPORT=" + str(path), flush=True)
        return 1
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.inside:
        if args.out is None:
            parser.error("--inside requires --out")
        return _inside(args.out)
    if not os.environ.get("PT_RESULTS"):
        raise RuntimeError("PT_RESULTS must be set in the outer environment")
    out = Path(os.environ["PT_RESULTS"]) / ("r5-resnet20-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True)
    scripts = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-m", "pt_agent.tools.nixrun", "--seconds", "10800", "--", "-m",
                                "pt_agent.experiments.r5_resnet20", "--inside", "--out", str(out)],
                               cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    report = json.loads((out / "REPORT.json").read_text())
    from ..registry import append, git_rev, new_run_id

    append(dict(run_id=new_run_id("R5", "resnet20", "mock-grid"), round="R5", model="resnet20", mode="mock-grid",
                git_rev=git_rev(), metrics=report, artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
