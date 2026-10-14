import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
TOOL = ROOT / "toolchain"
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
env = os.environ.copy()
env.update(HECATE=str(TOOL), HECATE_BUILD=str(TOOL),
           PYTHONPATH=str(TOOL / "python"), PYTHONDONTWRITEBYTECODE="1",
           OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
libraries = ":".join(map(str, (TOOL / "lib", Path("/opt/conda/lib"),
                                Path("/lib/x86_64-linux-gnu"))))
loader = str(TOOL / "lib/ld-linux-x86-64.so.2")
python = [loader, "--library-path", libraries, "/opt/conda/bin/python"]
compiler = [loader, "--library-path", libraries, str(TOOL / "bin/hecate-opt")]
package = ROOT / "llm_dsl/Qwen25_2Token_Block_DSL"
reports = []


def write_status(data):
    path = OUT / "retrace-final-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def run(name, command, seconds, memory_gib):
    def limits():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_AS, (memory_gib * 2**30,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 2**30,) * 2)

    started = time.monotonic()
    log_path = OUT / (name + ".log")
    reason = None
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=package, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   preexec_fn=limits)
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            elapsed = time.monotonic() - started
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                break
            if elapsed > seconds + 15:
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
            status_path = Path("/proc") / str(process.pid) / "status"
            if status_path.exists():
                for line in status_path.read_text().splitlines():
                    if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                        memory[line.split(":")[0] + "_KiB"] = int(line.split()[1])
            write_status(dict(state="running", stage=name, pid=process.pid,
                              elapsed_seconds=elapsed, memory=memory,
                              completed=reports))
            time.sleep(3)
    report = dict(stage=name, returncode=process.returncode,
                  elapsed_seconds=time.monotonic() - started,
                  user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                  max_rss_MiB=usage.ru_maxrss / 1024,
                  limits=dict(cpu_seconds=seconds, address_space_GiB=memory_gib),
                  stop_reason=reason, command=command)
    reports.append(report)
    write_status(dict(state="stage_finished", completed=reports))
    print(json.dumps(report), flush=True)
    return process.returncode == 0


CACHE = Path("/tmp/poseidon-boot-placement-IUOMH1")
steps = [
    ("rmsnorm_fixed2_frontend", python + [str(CACHE / "trace_rmsnorm_fixed.py")], 120, 16),
    ("projection_fixed2_frontend", python + [str(CACHE / "trace_projection_fixed.py")], 120, 16),
    ("block_fixed2_frontend", python + [str(package / "trace_qwen_block.py"), "--mode", "prefill", "--weights", str(OUT / "fixture/weights.npz"), "--output-dir", str(OUT / "block-prefill-fixed2")], 360, 64),
]
for name, command, seconds, memory_gib in steps:
    if not run(name, command, seconds, memory_gib):
        break
(OUT / "retrace-final-report.json").write_text(json.dumps(reports, indent=2))
write_status(dict(state="finished", completed=reports))
