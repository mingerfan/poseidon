"""Rerun only MLP with the frozen Agent protocol, then compile and report.

The outer process owns result files and API accounting.  A single pure Nix
process runs the existing Agent, its confirmation, P2 export and compilation,
and the separate 8,000-image report.  Existing runs are never resumed or retried.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from ..paths import pt_result
from ..run import _write


ROUND = "R8g"
MODEL = "mlp"
PROMPT_VERSION = "r8-v4"
STAGES = ("plan", "check", "mock", "count", "export", "compile", "test")
INNER_SECONDS = 21600
API_BUDGET = 8
TOTAL_API_BUDGET = 300


def _run_args(args):
    """Construct the existing run entrypoint's unchanged frozen settings."""
    return SimpleNamespace(
        out=args.out, model=MODEL, models_dir=args.models_dir,
        planner="agent", deg=None, margin=None, refresh="compiler", act_method="cheb",
        stages=list(STAGES), final=True, agent_iters=6,
        llm_model="deepseek-flash", llm_reasoning="high", llm_max_tokens=32768,
        budget_path=args.budget_path, llm_call_limit=args.call_limit,
        tuner="on", tuner_budget_s=2400,
    )


def _inside(args):
    from .r8e6_reconfirm import _limits

    _limits()
    import torch
    from ..planners.confirm import MAX_CONFIRM_EVALS
    from ..planners.prompts import PROMPT_VERSION as actual_prompt_version
    from ..run import _inside as run_inside

    if actual_prompt_version != PROMPT_VERSION or MAX_CONFIRM_EVALS != 6:
        raise RuntimeError("R8g requires prompt r8-v4 and MAX_CONFIRM_EVALS=6")
    torch.set_num_threads(1)
    # run_agent performs its one confirm_select after the internal loop.  Its
    # selected source flows directly into export, compile and the report stage.
    code = run_inside(_run_args(args))
    report_path = args.out / "REPORT.json"
    report = json.loads(report_path.read_text())
    report["round"] = ROUND
    _write(report_path, report)
    return code


def _calls(path, *, wait=True):
    """Read complete accounting rows; polling skips an in-flight HTTP call."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            return None
        return [json.loads(line) for line in stream if line.strip()]


def _trace_summary(path):
    rows = []
    tuner_seconds = 0.0
    if not path.exists():
        return rows, tuner_seconds
    for line in path.read_text().splitlines():
        item = json.loads(line)
        fb, tuned = item.get("fb") or {}, item.get("tuned")
        row = dict(iteration=item["iteration"], stage=fb.get("stage"),
                   feasible=fb.get("feasible", False),
                   feedback={key: fb[key] for key in
                             ("error", "errors", "diagnostics", "exact", "validation", "cost") if key in fb},
                   tuned=None)
        if tuned is not None:
            tuner_seconds += tuned.get("seconds", 0)
            row["tuned"] = {key: tuned.get(key) for key in
                            ("feasible", "seconds", "coarse_evals", "full_evals", "cost_after")}
            row["tuned"]["result"] = (tuned.get("history") or [None])[-1]
        rows.append(row)
    return rows, tuner_seconds


def _command(args, case, budget_path, call_limit):
    return [sys.executable, "-B", "-m", "pt_agent.tools.nixrun", "--seconds", str(INNER_SECONDS), "--",
            "-m", "pt_agent.experiments.r8g_mlp_rerun", "--inside", "--out", str(case),
            "--models-dir", str(args.models_dir), "--budget-path", str(budget_path),
            "--call-limit", str(call_limit)]


def _stop(process):
    """Stop this run's process group and retain its partial output."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _outer(args):
    from ..registry import append, git_rev, load, new_run_id

    root = pt_result().resolve()
    # The legacy registry loader is used only by this system-Python process.
    os.environ["PT_RESULTS"] = str(root)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "REPORT.json"
        case = out / MODEL
        if path.exists() or case.exists():
            raise FileExistsError("Existing R8g results must be preserved; no retry is performed")
        case.mkdir()
        (out / "runner.pid").write_text(str(os.getpid()) + "\n")
        budget_path = root / "llm_calls.jsonl"
        api_before = len(_calls(budget_path))
        args.models_dir = (args.models_dir or pt_result("models-v2")).resolve()
        result = dict(
            round=ROUND, model=MODEL, complete=False, stage="agent", git_rev=git_rev(),
            run_id=new_run_id(ROUND, MODEL, "agent-rerun"),
            models_dir=str(args.models_dir), case_report_path=str(case / "REPORT.json"),
            trace_path=str(case / "agent_trace.jsonl"), program_path=str(case / "program.ptdsl"),
            stages=list(STAGES), prompt_version=PROMPT_VERSION,
            llm=dict(model="deepseek-flash", reasoning_effort="high", max_tokens=32768),
            agent_iters=6, tuner="on", tuner_budget_s=2400, max_sweeps=1,
            val_split="test[9000:10000]", confirm_split="test[8000:9000]", test_split="test[0:8000]",
            max_confirm_evals=6, api_before=api_before, api_budget=API_BUDGET,
            api_limit=min(api_before + API_BUDGET, TOTAL_API_BUDGET), api_calls=0,
            api_total=api_before, call_indices=[], seconds=0, tuner_seconds=0,
            iterations=[], exit_code=None, error=None,
        )
        recorded = {row["metrics"].get("global_call") for row in load() if row["mode"] == "llm-call"}

        def account(*, wait=False):
            calls = _calls(budget_path, wait=wait)
            if calls is None:
                return
            indices = list(range(api_before + 1, len(calls) + 1))
            for index in indices:
                if index in recorded:
                    continue
                call = calls[index - 1]
                append(dict(run_id=new_run_id(ROUND, MODEL, f"api-{index}"), round=ROUND,
                            model=MODEL, mode="llm-call", git_rev=result["git_rev"],
                            metrics=dict(call, global_call=index), artifacts_dir=str(case), ts=call["ts"]))
                recorded.add(index)
            result.update(api_calls=len(indices), api_total=len(calls), call_indices=indices)

        command = _command(args, case, budget_path, result["api_limit"])
        result["command"] = command
        _write(path, result)
        started = time.monotonic()
        process = None
        try:
            if api_before >= TOTAL_API_BUDGET:
                raise RuntimeError("The shared 300-call API budget is exhausted")
            with (case / "stdout.txt").open("w") as stdout, (case / "stderr.txt").open("w") as stderr:
                process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                while True:
                    try:
                        result["exit_code"] = process.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        account()
                        result["seconds"] = time.monotonic() - started
                        _write(path, result)
                        if result["seconds"] > INNER_SECONDS + 120:
                            raise TimeoutError("R8g exceeded the existing 21,600-second Nix allowance")
        except (Exception, KeyboardInterrupt) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            _stop(process)
            result["exit_code"] = process.returncode if process is not None else None
        finally:
            account(wait=True)
            result["seconds"] = time.monotonic() - started
            current = {}
            try:
                if (case / "REPORT.json").exists():
                    current = json.loads((case / "REPORT.json").read_text())
                result["iterations"], result["tuner_seconds"] = _trace_summary(case / "agent_trace.jsonl")
            except Exception as exc:
                result["summary_error"] = f"{type(exc).__name__}: {exc}"
                result["error"] = result["error"] or result["summary_error"]
            result.update(case_stage=current.get("stage"), error=current.get("error") or result["error"])
            result["complete"] = bool(current.get("complete") and current.get("stage") == "complete"
                                      and result["exit_code"] == 0 and result["error"] is None)
            result["stage"] = "complete" if result["complete"] else "failed"
            _write(path, result)
            append(dict(run_id=result["run_id"], round=ROUND, model=MODEL, mode="agent-rerun",
                        git_rev=result["git_rev"], metrics=result, artifacts_dir=str(out),
                        ts=datetime.now(timezone.utc).isoformat()))
            print("REPORT=" + str(path), flush=True)
        return 0 if result["complete"] else 1


def _launch(args):
    out = (args.out or pt_result("r8g-mlp-rerun-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    command = ["setsid", "nohup", sys.executable, "-B", "-m", "pt_agent.experiments.r8g_mlp_rerun",
               "--worker", "--out", str(out)]
    if args.models_dir is not None:
        command += ["--models-dir", str(args.models_dir.resolve())]
    with (out / "runner.stdout.txt").open("w") as stdout, (out / "runner.stderr.txt").open("w") as stderr:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                                   stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
    (out / "runner.pid").write_text(str(process.pid) + "\n")
    print("RUN_DIR=" + str(out), flush=True)
    print("RUNNER_PID=" + str(process.pid), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--worker", action="store_true", help="Run the outer supervisor in the foreground")
    mode.add_argument("--inside", action="store_true", help="Internal entrypoint; requires explicit paths and API limit")
    parser.add_argument("--out", type=Path, help="New run root (outer) or MLP case directory (inside)")
    parser.add_argument("--models-dir", type=Path, help="Existing trained models directory; outer default is pt_result('models-v2')")
    parser.add_argument("--budget-path", type=Path, help="Inside only: existing shared llm_calls.jsonl path")
    parser.add_argument("--call-limit", type=int, help="Inside only: absolute shared API limit calculated by outer")
    args = parser.parse_args()
    if args.inside:
        if any(value is None for value in (args.out, args.models_dir, args.budget_path, args.call_limit)):
            parser.error("--inside requires --out, --models-dir, --budget-path and --call-limit")
        if not 0 <= args.call_limit <= TOTAL_API_BUDGET:
            parser.error("--call-limit must be between 0 and 300")
        return _inside(args)
    if args.budget_path is not None or args.call_limit is not None:
        parser.error("--budget-path and --call-limit are set by the outer process")
    if args.worker and args.out is None:
        parser.error("--worker requires --out")
    return _outer(args) if args.worker else _launch(args)


if __name__ == "__main__":
    sys.exit(main())
