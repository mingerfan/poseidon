"""R3 ResNet20 intake, calibration and independent NumPy accuracy experiment."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _write(path, record):
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _inside(out):
    import copy
    import numpy as np
    import torch
    from ..analyzer import calibrate, reference_forward
    from ..data.cifar import calibration_set, test_set
    from ..intake.coverage import report as coverage_report
    from ..models.resnet20 import load_graph, load_torch

    report = dict(complete=False, stage="intake")
    path = out / "REPORT.json"
    _write(path, report)

    def expired(signum, frame):
        raise TimeoutError("r3_resnet20 exceeded 90 minutes")

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(90*60)
    try:
        graph = load_graph()
        graph.save(out / "graph")
        coverage = coverage_report(graph)
        _write(out / "coverage.json", coverage)
        report.update(nodes=len(graph.nodes), coverage=coverage, stage="calibration")
        _write(path, report)
        print(f"INTAKE nodes={len(graph.nodes)} needs_composition={coverage['needs_composition']}", flush=True)

        calibration_x, _ = calibration_set()
        ranges = calibrate(graph, {"x": calibration_x})
        _write(out / "calibration.json", ranges)
        relu_ranges = {node.inputs[0]: [ranges[node.inputs[0]]["min"], ranges[node.inputs[0]]["max"]]
                       for node in graph.nodes if node.op == "act" and node.attrs["fn"] == "relu"}
        report.update(relu_input_ranges=relu_ranges, stage="test_accuracy", completed_test_images=0)
        _write(path, report)
        print("CALIBRATION images=1000", flush=True)

        model32 = load_torch()
        model64 = copy.deepcopy(model32).double()
        images, labels = test_set()
        correct32 = correct64 = correct_reference = agree = 0
        reference_seconds = 0.0
        for start in range(0, len(images), 250):
            stop = min(start+250, len(images))
            x = images[start:stop]
            with torch.no_grad():
                pred32 = model32(torch.from_numpy(x).float()).numpy().argmax(axis=1)
                pred64 = model64(torch.from_numpy(x)).numpy().argmax(axis=1)
            started = time.monotonic()
            logits = reference_forward(graph, {"x": x})[graph.outputs[0]]
            reference_seconds += time.monotonic()-started
            pred_reference = logits.argmax(axis=1)
            correct32 += int(np.sum(pred32 == labels[start:stop]))
            correct64 += int(np.sum(pred64 == labels[start:stop]))
            correct_reference += int(np.sum(pred_reference == labels[start:stop]))
            agree += int(np.sum(pred_reference == pred64))
            report.update(completed_test_images=stop, correct_torch32=correct32, correct_torch64=correct64,
                          correct_reference=correct_reference, agree_reference_vs_torch64=agree,
                          seconds_reference_10000=reference_seconds)
            _write(path, report)
            print(f"TEST images={stop}/10000 torch32={correct32} torch64={correct64} reference={correct_reference}", flush=True)

        report.update(stage="calibration_logits", completed_calibration_images=0, max_abs_logit_diff_calibration=0.0)
        _write(path, report)
        maximum = 0.0
        for start in range(0, len(calibration_x), 250):
            x = calibration_x[start:start+250]
            with torch.no_grad():
                torch_logits = model64(torch.from_numpy(x)).numpy()
            logits = reference_forward(graph, {"x": x})[graph.outputs[0]]
            maximum = max(maximum, float(np.max(np.abs(logits-torch_logits))))
            report.update(completed_calibration_images=min(start+250, len(calibration_x)), max_abs_logit_diff_calibration=maximum)
            _write(path, report)
        report = dict(nodes=len(graph.nodes), coverage=coverage, top1_torch32=correct32*100.0/len(images),
                      top1_torch64=correct64*100.0/len(images), top1_reference=correct_reference*100.0/len(images),
                      agree_reference_vs_torch64=agree, max_abs_logit_diff_calibration=maximum,
                      seconds_reference_10000=reference_seconds, relu_input_ranges=relu_ranges)
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
    out = Path(os.environ["PT_RESULTS"]) / ("r3-resnet20-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True)
    scripts = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-m", "pt_agent.tools.nixrun", "--seconds", "5400", "--", "-m",
                                "pt_agent.experiments.r3_resnet20", "--inside", "--out", str(out)],
                               cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    report = json.loads((out / "REPORT.json").read_text())
    from ..registry import append, git_rev, new_run_id

    append(dict(run_id=new_run_id("R3", "resnet20", "reference"), round="R3", model="resnet20", mode="reference",
                git_rev=git_rev(), metrics={key: value for key, value in report.items() if key != "relu_input_ranges"},
                artifacts_dir=str(out), ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
