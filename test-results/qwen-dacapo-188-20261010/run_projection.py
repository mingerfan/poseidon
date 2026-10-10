import json
import os
import resource
import subprocess
import time
import run_retry as runner

ROOT, OUT, TOOL = runner.ROOT, runner.OUT, runner.TOOL
runner.REPORT_NAME = "projection-report.json"
runner.affinity = sorted(os.sched_getaffinity(0))[16:24]
if not runner.affinity:
    runner.affinity = sorted(os.sched_getaffinity(0))[:4]


def write_status(data):
    path = OUT / "projection-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


runner.write_status = write_status


def limits():
    os.sched_setaffinity(0, runner.affinity)
    resource.setrlimit(resource.RLIMIT_CPU, (120, 125))
    resource.setrlimit(resource.RLIMIT_AS, (16 * 2**30,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


command = ["/usr/bin/time", "-v", "-o", str(OUT / "q-projection.frontend.time"),
           str(TOOL / "lib/ld-linux-x86-64.so.2"), "--library-path",
           runner.libraries, "/opt/conda/bin/python", str(ROOT / "trace_projection.py")]
with (OUT / "q-projection.frontend.log").open("w") as log:
    result = subprocess.run(command, cwd=ROOT, env=runner.env, stdout=log,
                            stderr=subprocess.STDOUT, preexec_fn=limits,
                            timeout=180)
if result.returncode == 0:
    runner.run("q_projection_dacapo_complete",
               OUT / "q-projection/trace_projection.mlir",
               OUT / "q-projection.complete.mlir", 1883, 300, 2400, 32)
(OUT / "projection-report.json").write_text(json.dumps(runner.reports, indent=2))
write_status(dict(state="finished", completed=runner.reports))
