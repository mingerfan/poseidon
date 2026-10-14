"""Validate each plan as soon as its compiler succeeds, without rerunning it."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
OUT = ROOT / "artifacts/qwen24-gpu-memory"
TOOL = ROOT / "toolchain"
jobs = []
started = time.monotonic()


def status():
    rows = []
    for job in jobs:
        row = {key: job[key] for key in ("devices", "pid")}
        row["elapsed_seconds"] = time.monotonic() - job["started"]
        row["report"] = job.get("report")
        proc = Path(f"/proc/{job['pid']}/status")
        if proc.exists() and "report" not in job:
            row["memory"] = {line.split(":")[0]: line.split(":", 1)[1].strip()
                             for line in proc.read_text().splitlines()
                             if line.startswith(("VmRSS:", "VmHWM:", "Threads:"))}
        rows.append(row)
    temporary = OUT / "validation-status.tmp"
    temporary.write_text(json.dumps({"jobs": rows}, indent=2) + "\n")
    temporary.replace(OUT / "validation-status.json")


while len(jobs) < 2 or any("report" not in job for job in jobs):
    assert time.monotonic() - started < 10800
    for devices in (1, 4):
        report = OUT / f"gpu{devices}/compiler-report.json"
        if any(j["devices"] == devices for j in jobs) or not report.exists():
            continue
        compiled = json.loads(report.read_text())
        if compiled["returncode"]:
            raise RuntimeError(f"gpu{devices} compiler failed: {compiled}")
        assert (OUT / "semantic-reference.json").exists()
        command = [str(TOOL / "lib/ld-linux-x86-64.so.2"), "--library-path",
                   f"{TOOL}/lib:/opt/conda/lib:/lib/x86_64-linux-gnu", "/opt/conda/bin/python",
                   "/tmp/poseidon-qwen24-IUOMH1/validate_gpu_plans.py", "--devices", str(devices)]

        def limits():
            os.setsid()
            os.sched_setaffinity(0, set(range(20, 24)) | set(range(44, 48)))
            resource.setrlimit(resource.RLIMIT_AS, (128 * 2**30,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (28800, 28805))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**20,) * 2)

        log = (OUT / f"gpu{devices}/validation.log").open("w")
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
        jobs.append(dict(devices=devices, pid=process.pid, process=process, log=log,
                         command=command, started=time.monotonic()))
        print(json.dumps({"validation_started": devices, "pid": process.pid}), flush=True)
    for job in jobs:
        if "report" in job:
            continue
        pid, result, usage = os.wait4(job["pid"], os.WNOHANG)
        elapsed = time.monotonic() - job["started"]
        if not pid and elapsed > 3600:
            os.killpg(job["pid"], signal.SIGKILL)
            pid, result, usage = os.wait4(job["pid"], 0)
        if not pid:
            continue
        code = os.waitstatus_to_exitcode(result)
        job["process"].returncode = code
        job["log"].close()
        report = dict(devices=job["devices"], returncode=code, elapsed_seconds=elapsed,
                      user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                      max_rss_MiB=usage.ru_maxrss / 1024, command=job["command"])
        job["report"] = report
        (OUT / f"gpu{job['devices']}/validation-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    status()
    if len(jobs) == 2 and all("report" in j for j in jobs):
        break
    time.sleep(5)
if any(j["report"]["returncode"] for j in jobs):
    raise SystemExit(1)
