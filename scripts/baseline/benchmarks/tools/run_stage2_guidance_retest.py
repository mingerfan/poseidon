"""Execute only the separately approved r100 guidance retest; retain r97 history."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from stage2_guidance_retest_plan import BASE, ROOT, build_plan, check_binding, sha
from benchmark_graph import digest
from benchmark_runner import dump, strict_file
from stage2_agent_provenance import provider_accounting

def require_approval(plan, approved_binding):
    check_binding(plan)
    if not approved_binding or approved_binding != plan["binding"]:
        raise ValueError("Separate paid approval of the exact proposal binding is required")
    if not 1 <= len(plan["cases"]) <= 48 or plan["limits"]["concurrency"] != 1:
        raise ValueError("Pilot shard/concurrency boundary")
    return plan

def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)

def evidence_path(log, results):
    matches = re.findall(r"^Candidate evidence: (.+)$", log.read_text(), re.M)
    if not matches:
        return None
    path = Path(matches[-1])
    if path.is_symlink() or path.resolve().parent != results.resolve():
        raise ValueError("Unexpected candidate evidence path")
    return path.resolve()

def size_bytes(paths):
    return sum(f.stat().st_size for p in set(paths) for f in p.rglob("*") if f.is_file())

def outcome(report, code):
    if report.get("provider") != "deepseek_api":
        raise ValueError("Scripted result cannot count as Agent evaluation")
    from stage2_guidance_retest_plan import PAID
    if "provider_metrics" in report:
        provider_accounting(report, PAID)
    elif report.get("agent_calls") != 0 or report.get("status") == "passed":
        raise ValueError("Missing HTTP/generation ledger")
    successful = [a for a in report.get("attempts", []) if a.get("status") == "passed"]
    claimed = code == 0 and report.get("status") == "passed"
    if claimed:
        if not report.get("llm_generation_validated") or report["agent_calls"] == 0 or len(successful) != 1:
            raise ValueError("Success without actual Agent provenance")
        a = successful[0]
        if not all(a.get(k) for k in ("compiled", "executed", "numerically_correct")):
            raise ValueError("Success without full execution")
        if a["trace"]["frontend"] != "real_Hecate" or not a["execution"]["encrypted_execution"]:
            raise ValueError("Not real Hecate/SEAL execution")
        if a["comparison"]["atol"] != 1e-5 or a["comparison"]["rtol"] != 1e-4:
            raise ValueError("Frozen numerical tolerance changed")
        # Retained raw response, independent references/artifacts and witnesses
        # still require a post-run audit. Never count this preliminary state as pass.
        return "runner_reported_pass_pending_audit"
    return "failed"

def terminal_failure_layer(report):
    if report.get("status")=="passed":
        return None
    attempts=report.get("attempts",[])
    return report.get("failure_layer") or (attempts[-1].get("failure_layer") if attempts else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--approve-live-binding", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    plan = strict_file(a.plan, 8*1024**2)
    try:
        require_approval(plan, a.approve_live_binding)
    except ValueError as e:
        p.error(str(e))
    # Approval is an operator boundary, never inferred from the plan file's existence.
    from hecate_python_env import VENV, enter_nix
    from workspace_paths import RESULTS
    if not a.output.resolve().is_relative_to(RESULTS.resolve()) or a.output.exists():
        p.error("New result directory required; existing/uncertain paid attempts are never retried automatically")
    if not a.inside:
        command = [str(VENV/"bin/python"), "-B", str(Path(__file__).resolve()),
                   *sys.argv[1:], "--inside"]
        return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+
                         shlex.join(command), seconds=plan["limits"]["max_wall_seconds"]+90)
    if os.environ.get("IN_NIX_SHELL") != "pure" or Path(sys.prefix) != VENV:
        p.error("Pinned pure Nix Python required")
    if build_plan() != plan:
        p.error("Source, model, request or budget drift; new proposal/approval required")
    from semantic_benchmark_execution import runtime_sources
    limit = plan["limits"]
    a.output.mkdir(parents=True)
    dump(a.output/"plan.json", plan)
    started = time.monotonic()
    rows = []
    folders = set()
    child = None
    failure = None

    def boundary():
        if time.monotonic()-started >= limit["max_wall_seconds"]:
            raise TimeoutError("Pilot wall budget reached; retain remaining as not_run")
        if size_bytes([a.output, *folders]) >= limit["max_retained_mib"]*1024**2:
            raise ValueError("Retained evidence budget reached; preserve existing files")
        if runtime_sources() != plan["source_hashes"]:
            raise ValueError("Runtime source drift")
        for name, expected in plan["proposal_files"].items():
            if sha(ROOT/name) != expected:
                raise ValueError("Proposal dependency drift")

    try:
        for spec in plan["cases"]:
            boundary()
            case = a.output/spec["id"]
            case.mkdir()
            dump(case/"model.json", spec["model"])
            dump(case/"request.expected.json", spec["request"])
            command = [plan["executable"], "-B", plan["entrypoint"], "--case", str(case/"model.json"),
                       *spec["candidate_arguments"]]
            # Write before launch so interruptions never erase an uncertain paid attempt.
            dump(case/"launch.json", dict(binding=plan["binding"], command=command,
                                         status="launching", automatic_restart_allowed=False))
            log = case/"run.log"
            with log.open("w") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                dump(case/"process.json", dict(pid=child.pid, state="started"))
                while child.poll() is None:
                    folder = evidence_path(log, RESULTS)
                    if folder:
                        folders.add(folder)
                    boundary()
                    try:
                        child.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
            folder = evidence_path(log, RESULTS)
            record = dict(id=spec["id"], track=spec["track"], status="failed",
                          failure_layer="harness_launch", exit_code=code, agent_calls=0)
            if folder:
                folders.add(folder)
                record["evidence"] = str(folder)
                if (folder/"report.json").exists():
                    report = strict_file(folder/"report.json", 8*1024**2)
                    if strict_file(folder/"request.json", 131072) != spec["request"]:
                        raise ValueError("Executed request differs from approved public request")
                    if digest(strict_file(folder/"model.json", 131072)) != spec["model_sha256"]:
                        raise ValueError("Executed mathematical model changed")
                    record.update(status=outcome(report, code), agent_calls=report["agent_calls"],
                                  report_sha256=sha(folder/"report.json"),
                                  generations=report.get("provider_metrics", {}).get("generation_attempts", 0),
                                  failure_layer=terminal_failure_layer(report))
            rows.append(record)
            dump(a.output/"progress.json", rows)
            boundary()
            if record["failure_layer"] in ("harness_launch", "environment", "key_setup", "seal_runtime", "integrity"):
                raise ValueError("Environment/integrity boundary; stop related pilot")
    except BaseException as error:
        failure = type(error).__name__+": "+str(error)
        if child:
            stop_child(child)
        raise
    finally:
        if child:
            stop_child(child)
        # No aggregate pass is awarded before independent audit.
        report = dict(format="poseidon-stage2-agent-pilot-run-v1", plan_binding=plan["binding"],
                      rows=rows, not_run_or_interrupted=[s["id"] for s in plan["cases"] if s["id"] not in {r["id"] for r in rows}],
                      failure=failure, seconds=time.monotonic()-started,
                      http_attempts_in_completed_reports=sum(r["agent_calls"] for r in rows),
                      generations_in_completed_reports=sum(r.get("generations", 0) for r in rows),
                      interrupted_call_count_may_be_unknown=bool(failure),
                      artifact_bytes=size_bytes([a.output, *folders]),
                      passed=0, independent_audit_complete=False,
                      stage2_complete=False)
        report["binding"] = digest(report)
        dump(a.output/"report.json", report)
    print(json.dumps(report, indent=2))
    return int(any(r["status"] == "failed" for r in rows))

if __name__ == "__main__":
    raise SystemExit(main())
