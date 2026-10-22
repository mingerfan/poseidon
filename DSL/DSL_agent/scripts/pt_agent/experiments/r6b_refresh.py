"""Compare lazy refresh plus PARS with compiler depth-dp on four fixed cases."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from .. import toolchain
from ..paths import pt_result


def _write(out, report):
    report["summary"] = [dict(deg=row["deg"], margin=row["margin"], mode=row["mode"],
                              boots=(row.get("plan") or {}).get("boots"),
                              plan_latency_s=(row["plan_latency"]["latency_us"]/1e6 if row.get("plan_latency") else None),
                              compile_s=(row.get("compile") or {}).get("seconds"),
                              peak_rss_mib=(row.get("compile") or {}).get("peak_rss_mib")) for row in report["cases"]]
    temporary = out / "REPORT.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(out / "REPORT.json")


def _inside(out, calibration_path):
    import numpy as np
    from ..backends import EmitBackend, MockBackend, RefreshBackend
    from ..cost import PROFILES, estimate, plan_latency
    from ..data.cifar import calibration_set
    from ..dsl.parser import parse
    from ..lower import decode_outputs, lower, pack_inputs
    from ..models.resnet20 import load_graph
    from ..planners.rule import plan

    toolchain.setup_env()
    report = json.loads((out / "REPORT.json").read_text())
    config = PROFILES["qwen48"]
    levels = {key: config[key] for key in ("input_level", "boot_in_level", "boot_out_level")}
    try:
        graph = load_graph()
        with calibration_path.open() as stream:
            calibration = json.load(stream)
        report["calibration_path"] = str(calibration_path)
        report["stage"] = "mock"
        _write(out, report)
        images = calibration_set()[0][:4].copy()
        program = parse(plan(graph, calibration, deg=63, margin=1.4))
        outputs = []
        for wrapped in (False, True):
            inner = MockBackend(program.header.slots, batch=4)
            inner.bind(pack_inputs(program, [images]))
            backend = RefreshBackend(inner, **levels) if wrapped else inner
            values, layouts = lower(program, graph.weights, backend)
            packed = [value.value.value if wrapped else value.value for value in values]
            outputs.append(decode_outputs(packed, layouts)[0].copy())
            del values, packed, backend, inner
            gc.collect()
        report["mock_equal"] = bool(np.array_equal(outputs[0], outputs[1]))
        _write(out, report)
        if not report["mock_equal"]:
            raise RuntimeError("RefreshBackend changed Mock outputs")
        del outputs, images
        gc.collect()
        for row in report["cases"]:
            report.update(stage="estimate", current=dict(deg=row["deg"], margin=row["margin"], mode=row["mode"]))
            _write(out, report)
            text = plan(graph, calibration, deg=row["deg"], margin=row["margin"], refresh=row["mode"])
            program = parse(text)
            row["estimate"] = estimate(program, graph.weights)
            row.update(boots_pred=row["estimate"]["boots_pred"], latency_pred_us=row["estimate"]["latency_pred_us"])
            directory = toolchain.new_run_dir(f"r6b-{row['deg']}-{str(row['margin']).replace('.', 'p')}-{row['mode']}")
            row["run_dir"] = str(directory)
            (directory / "program.ptdsl").write_text(text)
            row["program_path"] = str(directory / "program.ptdsl")
            report["stage"] = "emit"
            _write(out, report)
            inner = EmitBackend(program.header.slots)
            backend = RefreshBackend(inner, **levels) if row["mode"] == "agent" else inner
            lower(program, graph.weights, backend)
            inner.write(directory)
            row["emit_counts"] = inner.stats()["counts"]
            del backend, inner
            gc.collect()
            report["stage"] = "trace"
            _write(out, report)
            row["trace"] = toolchain.trace(directory, trusted=True)
            _write(out, report)
            if row["trace"]["exit"] != 0 or not row["trace"]["mlir"]:
                raise RuntimeError("Trusted trace failed")
            report["stage"] = "compile"
            _write(out, report)
            mode = "pars" if row["mode"] == "agent" else "dacapo-depth-dp"
            row["compile"] = toolchain.compile(directory, mode, boot="qwen48", trusted=True)
            _write(out, report)
            if row["compile"]["exit"] != 0 or not row["compile"]["plan"]:
                raise RuntimeError("Trusted compile failed")
            row["plan"] = toolchain.plan_stats(directory)
            row["plan_latency"] = plan_latency(directory / "lowered._hecate_golden.runtime-plan.json")
            row["plan_latency"]["by_op"] = dict(sorted(row["plan_latency"]["by_op"].items(), key=lambda item: -item[1])[:5])
            _write(out, report)
            row["pruned"] = toolchain.prune(directory)
            _write(out, report)
            print(json.dumps(report["summary"][report["cases"].index(row)]), flush=True)
        report.update(complete=True, stage="complete")
    except Exception as exc:
        report.update(complete=False, error=f"{type(exc).__name__}: {exc}")
    _write(out, report)
    print("REPORT="+str(out / "REPORT.json"), flush=True)
    return 0 if report["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--calibration", type=Path,
                        default=pt_result('run-resnet20-rule-20261020T165817/calibration.json'))
    args = parser.parse_args()
    calibration = args.calibration.expanduser().resolve()
    if args.inside:
        if args.out is None:
            parser.error("--inside requires --out")
        return _inside(args.out, calibration)
    if not calibration.is_file():
        parser.error(f"calibration file does not exist: {calibration}")
    if not os.environ.get("PT_RESULTS"):
        raise RuntimeError("PT_RESULTS must be set in the outer environment")
    out = Path(os.environ["PT_RESULTS"]) / ("r6b-refresh-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True)
    report = dict(round="R6b", complete=False, stage="load", mock_equal=None,
                  cases=[dict(deg=deg, margin=margin, mode=mode)
                         for deg, margin in ((127, 1.5), (63, 1.4)) for mode in ("compiler", "agent")])
    _write(out, report)
    scripts = Path(__file__).resolve().parents[2]
    command = [sys.executable, "-B", "-m", "pt_agent.tools.nixrun", "--seconds", "10800", "--",
               "-m", "pt_agent.experiments.r6b_refresh", "--inside", "--out", str(out),
               "--calibration", str(calibration)]
    completed = subprocess.run(command, cwd=scripts, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    sys.stderr.write(completed.stderr)
    sys.stderr.flush()
    from ..registry import append, git_rev, new_run_id

    report = json.loads((out / "REPORT.json").read_text())
    append(dict(run_id=new_run_id("R6b", "resnet20", "refresh-comparison"), round="R6b", model="resnet20",
                mode="refresh-comparison", git_rev=git_rev(), metrics=report, artifacts_dir=str(out),
                ts=datetime.now(timezone.utc).isoformat()))
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
