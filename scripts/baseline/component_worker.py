"""One bounded JSON job per process. No provider, credential loading or arbitrary commands."""
import argparse
import signal
import contextlib
import json
import os
from pathlib import Path
import shlex
import sys
from benchmark_graph import require, digest
from candidate_contract import strict_json
from component_control import QualificationCancelled
from component_contract import VERSION, prepare_task, validate_program, capabilities

def read_job(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 262144, "Component job file")
    job = strict_json(path.read_text())
    require(type(job) is dict and job.get("format") == VERSION, "Component protocol version")
    action = job.get("action")
    fields = {"capabilities": {"format","action"},
              "prepare": {"format","action","model","options"},
              "validate": {"format","action","request","candidate"},
              "qualify": {"format","action","request","candidate"}}
    require(action in fields and set(job) == fields[action], "Component action/fields")
    return job

def dispatch(job):
    if job["action"] == "capabilities":
        return capabilities()
    if job["action"] == "prepare":
        return dict(format=VERSION, status="prepared_not_executed",
                    request=prepare_task(job["model"], job["options"]), paid_calls=0)
    checked = validate_program(job["candidate"], job["request"])
    if job["action"] == "validate" or checked["status"] == "rejected":
        return checked
    from component_backend import qualify
    return qualify(job["request"], job["candidate"])

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--job-sha256", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    try:
        job = read_job(a.job)
        job_hash = digest(job)
        require(a.job_sha256 is None or a.job_sha256 == job_hash, "Component job changed across environment boundary")
        if job["action"] == "qualify":
            from hecate_python_env import enter_nix, VENV, ROOT
            pinned = os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV
            if not pinned:
                require(not a.inside, "Pinned component environment required")
                cmd = [str(VENV/"bin/python"), "-B", str(ROOT/"scripts/agent_component.py"),
                       "--inside", "--job", str(a.job.resolve()), "--job-sha256", job_hash]
                return enter_nix('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '+shlex.join(cmd), seconds=900)
        def interrupt(signum, frame):
            raise QualificationCancelled("Component cancelled")
        previous = signal.signal(signal.SIGTERM, interrupt)
        with contextlib.redirect_stdout(sys.stderr):
            try:
                result = dispatch(job)
            finally:
                signal.signal(signal.SIGTERM, previous)
        result["job_sha256"] = job_hash
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return result.get("exit_code", 1 if result.get("status") in ("rejected", "failed", "repair_budget_exhausted") else 0)
    except (KeyboardInterrupt, QualificationCancelled):
        print(json.dumps(dict(format=VERSION, status="cancelled", paid_calls=0, encrypted_execution=False)))
        return 130
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        print(json.dumps(dict(format=VERSION, status="rejected", paid_calls=0,
                              failure=dict(code="component_request_or_environment_rejected", diagnostic=str(error)[:2000]))))
        return 2
