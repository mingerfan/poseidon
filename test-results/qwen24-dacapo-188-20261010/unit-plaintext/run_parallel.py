"""Run independent one-GPU/four-GPU compilations with Release and reports."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
OUT = ROOT / "artifacts/qwen24-unit-plaintext/gpu-plans"
TOOL = ROOT / "toolchain"
PROFILES = ROOT / "profiles"
SOURCE = ROOT / "artifacts/qwen24-native/traced/trace_qwen24.mlirbc"
COMPILER = TOOL / "bin/hecate-opt.unit-plaintext"
OUT.mkdir(exist_ok=False)
spec_path = PROFILES / "operator-spec.json"
spec = json.loads(spec_path.read_text())
digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
boot = spec["boot_profiles"][0]
allowed = set(os.sched_getaffinity(0))
cores = {}
for cpu in sorted(allowed):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    key = ((topology / "physical_package_id").read_text().strip(),
           (topology / "core_id").read_text().strip())
    cores.setdefault(key, []).append(cpu)
core_groups = list(cores.values())
assert len(core_groups) >= 20
cpu_sets = [sum(core_groups[start:start + 10], []) for start in (0, 10)]
assert not set(cpu_sets[0]).intersection(cpu_sets[1])
jobs = []
env = os.environ.copy()
env.update(HECATE=str(TOOL), HECATE_BUILD=str(TOOL), TMPDIR="/tmp/poseidon-qwen24-IUOMH1",
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="20", PYTHONDONTWRITEBYTECODE="1")


def write_status():
    rows = []
    for job in jobs:
        row = {key: job[key] for key in ("devices", "pid", "cpu_affinity", "command")}
        row["elapsed_seconds"] = time.monotonic() - job["started"]
        row["report"] = job.get("report")
        proc = Path(f"/proc/{job['pid']}/status")
        if proc.exists() and "report" not in job:
            row["memory"] = {line.split(":")[0]: line.split(":", 1)[1].strip()
                             for line in proc.read_text().splitlines()
                             if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:"))}
        rows.append(row)
    temporary = OUT / "status.tmp"
    temporary.write_text(json.dumps({"state": "finished" if all("report" in j for j in jobs) else "running",
                                     "jobs": rows}, indent=2) + "\n")
    temporary.replace(OUT / "status.json")


for devices, affinity in zip((1, 4), cpu_sets):
    directory = OUT / f"gpu{devices}"
    directory.mkdir()
    command = [str(TOOL / "lib/ld-linux-x86-64.so.2"), "--library-path",
               f"{TOOL}/lib:/opt/conda/lib:/lib/x86_64-linux-gnu", str(COMPILER),
               "--dacapo", "--boot-placement=depth-dp", "--ckks-config=" + str(PROFILES / "compiler-profile.json"),
               "--waterline=40", "--mlir-timing", "--mlir-print-op-on-diagnostic=false",
               "--mlir-elide-elementsattrs-if-larger=16", f"--runtime-plan-id={2510 + devices}",
               "--runtime-plan-target-id=" + spec["target_id"], "--runtime-plan-capability-version=1",
               "--runtime-plan-operator-spec-id=" + spec["spec_id"],
               "--runtime-plan-operator-spec-version=" + str(spec["version"]),
               "--runtime-plan-operator-spec-sha256=" + digest,
               "--runtime-plan-context-id=" + spec["context"]["context_id"], "--runtime-plan-ntt=true",
               "--runtime-plan-boot-profile=" + boot["profile_id"],
               "--runtime-plan-boot-implementation=" + boot["implementation"],
               "--runtime-plan-inline-payload-max-bytes=4096",
               "--runtime-plan-physical-level-operator-spec-path=" + str(spec_path),
               "--runtime-plan-levels-per-logical-level=4",
               f"--runtime-plan-device-count={devices}", f"--runtime-plan-device-counts={devices}",
               "--runtime-plan-operator-spec-path=" + str(spec_path),
               "--runtime-plan-communication-profile-path=" + str(PROFILES / "communication-profile.json"),
               "--runtime-plan-memory=true", "--runtime-plan-memory-reuse=true",
               "--runtime-plan-memory-report=true", str(SOURCE), "-o", str(directory / "qwen24.mlir")]

    def limits():
        os.setsid()
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (256 * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (115200, 115205))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2**30,) * 2)

    log_path = directory / "compile.log"
    log = log_path.open("w")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, preexec_fn=limits)
    jobs.append(dict(devices=devices, pid=process.pid, process=process, log=log,
                     log_path=log_path, directory=directory, cpu_affinity=affinity,
                     command=command, started=started, stop_reason=None))
    print(json.dumps({"started": devices, "pid": process.pid, "cpu_affinity": affinity}), flush=True)
write_status()
while any("report" not in job for job in jobs):
    for job in jobs:
        if "report" in job:
            continue
        pid, status, usage = os.wait4(job["pid"], os.WNOHANG)
        elapsed = time.monotonic() - job["started"]
        if not pid:
            reason = ("wall_time_limit" if elapsed > 7200 else
                      "diagnostic_log_limit" if job["log_path"].stat().st_size > 256 * 2**20 else None)
            if not reason:
                continue
            job["stop_reason"] = reason
            os.killpg(job["pid"], signal.SIGKILL)
            pid, status, usage = os.wait4(job["pid"], 0)
        code = os.waitstatus_to_exitcode(status)
        job["process"].returncode = code
        job["log"].close()
        report = dict(devices=job["devices"], returncode=code, elapsed_seconds=elapsed,
                      user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                      max_rss_MiB=usage.ru_maxrss / 1024, command=job["command"],
                      limits={"cpu_affinity": job["cpu_affinity"], "address_space_GiB": 256,
                              "wall_seconds": 7200, "cpu_seconds": 115200},
                      stop_reason=job["stop_reason"])
        job["report"] = report
        (job["directory"] / "compiler-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    write_status()
    if any("report" not in job for job in jobs):
        time.sleep(3)
if any(job["report"]["returncode"] for job in jobs):
    raise SystemExit(1)
