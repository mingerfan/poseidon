import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "toolchain"
OUT = ROOT / "artifacts"
PROFILES = ROOT / "profiles"
env = os.environ.copy()
env.update(HECATE=str(TOOL), HECATE_BUILD=str(TOOL),
           PYTHONPATH=str(TOOL / "python"), PYTHONDONTWRITEBYTECODE="1",
           OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
libraries = ":".join(map(str, (TOOL / "lib", Path("/opt/conda/lib"),
                                Path("/lib/x86_64-linux-gnu"))))
compiler = [str(TOOL / "lib/ld-linux-x86-64.so.2"), "--library-path",
            libraries, str(TOOL / "bin/hecate-opt")]
spec_path = PROFILES / "operator-spec.json"
spec = json.loads(spec_path.read_text())
boot = spec["boot_profiles"][0]
digest = "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()
reports = []
REPORT_NAME = "retry-report.json"
affinity = sorted(os.sched_getaffinity(0))[:16]


def write_status(data):
    path = OUT / "retry-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def run(name, source, output, plan_id, wall_seconds, cpu_seconds, memory_gib):
    command = compiler + [
        "--dacapo", "--ckks-config=" + str(PROFILES / "compiler-profile.json"),
        "--waterline=40", "--mlir-disable-threading", "--mlir-timing",
        "--mlir-print-op-on-diagnostic=false", f"--runtime-plan-id={plan_id}",
        "--runtime-plan-target-id=" + spec["target_id"],
        "--runtime-plan-capability-version=1",
        "--runtime-plan-operator-spec-id=" + spec["spec_id"],
        "--runtime-plan-operator-spec-version=" + str(spec["version"]),
        "--runtime-plan-operator-spec-sha256=" + digest,
        "--runtime-plan-context-id=" + spec["context"]["context_id"],
        "--runtime-plan-ntt=true",
        "--runtime-plan-boot-profile=" + boot["profile_id"],
        "--runtime-plan-boot-implementation=" + boot["implementation"],
        "--runtime-plan-inline-payload-max-bytes=4096",
        "--runtime-plan-physical-level-operator-spec-path=" + str(spec_path),
        "--runtime-plan-levels-per-logical-level=4",
        "--runtime-plan-device-count=0", str(source), "-o", str(output),
    ]

    def limits():
        os.setsid()
        os.sched_setaffinity(0, affinity)
        resource.setrlimit(resource.RLIMIT_AS, (memory_gib * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 2**30,) * 2)

    started = time.monotonic()
    log_path = OUT / (name + ".log")
    reason = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   preexec_fn=limits)
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            elapsed = time.monotonic() - started
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            if elapsed > wall_seconds:
                reason = "wall_time_limit"
                os.killpg(process.pid, signal.SIGKILL)
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            if log_path.stat().st_size > 256 * 2**20:
                reason = "diagnostic_log_limit"
                os.killpg(process.pid, signal.SIGKILL)
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            memory = {}
            proc_status = Path("/proc") / str(process.pid) / "status"
            if proc_status.exists():
                for line in proc_status.read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:", "Threads:")):
                        memory[line.split(":")[0]] = line.split(":", 1)[1].strip()
            write_status(dict(state="running", stage=name, pid=process.pid,
                              elapsed_seconds=elapsed, memory=memory,
                              completed=reports))
            time.sleep(3)
    report = dict(stage=name, returncode=process.returncode,
                  elapsed_seconds=time.monotonic() - started,
                  user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                  max_rss_MiB=usage.ru_maxrss / 1024,
                  limits=dict(cpu_seconds=cpu_seconds, wall_seconds=wall_seconds,
                              address_space_GiB=memory_gib, cpu_affinity=affinity),
                  stop_reason=reason, command=command)
    if process.returncode < 0 and reason is None:
        report["stop_reason"] = signal.Signals(-process.returncode).name
    reports.append(report)
    (OUT / REPORT_NAME).write_text(json.dumps(reports, indent=2))
    write_status(dict(state="stage_finished", completed=reports))
    print(json.dumps(report), flush=True)
    return process.returncode == 0


if __name__ == "__main__":
    rms_success = run("rmsnorm_dacapo_complete", OUT / "rmsnorm/trace_rmsnorm.mlir",
                     OUT / "rmsnorm.complete.mlir", 1881, 180, 2880, 16)
    if rms_success:
        run("block_prefill_dacapo_complete", OUT / "block-prefill/trace_qwen_block.mlir",
            OUT / "block-prefill.complete.mlir", 1882, 900, 14400, 128)
    write_status(dict(state="finished", completed=reports))
