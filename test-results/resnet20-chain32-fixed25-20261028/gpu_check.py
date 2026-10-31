import json
import os
import subprocess
import sys
import time
import chain32_probe as p

for name in sys.argv[1:]:
    candidate = p.OUT / "candidates" / name
    destination = candidate / "gpu-relu"
    destination.mkdir(exist_ok=False)
    command = json.loads((candidate / "relu/result.json").read_text())["command"]
    command = [
        a
        if not a.startswith("--runtime-plan-inline-payload-max-bytes=")
        else "--runtime-plan-inline-payload-max-bytes=1048576"
        for a in command
    ]
    command[command.index("-o") + 1] = str(destination / "one.optimized.mlirbc")
    r = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (destination / "compile.log").write_text(r.stdout + r.stderr)
    assert not r.returncode, r.stderr
    plan = next(destination.glob("*.runtime-plan.json"))
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="2",
        POSEIDON_RUNTIME_BOOTSTRAP_MANUAL_Q50=str(candidate / "fixture.json"),
        OMP_NUM_THREADS="8",
    )
    command = [
        str(p.AUDIT / "variable-chain32/gpu/relu_gpu"),
        str(plan),
        str(candidate / "diagnostic-spec.json"),
        str(destination / "gpu-report.json"),
    ]
    start = time.monotonic()
    with (destination / "gpu-run.log").open("w") as log:
        r = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=300
        )
    report = dict(
        name=name,
        returncode=r.returncode,
        seconds=time.monotonic() - start,
        command=command,
    )
    if (destination / "gpu-report.json").exists():
        d = json.loads((destination / "gpu-report.json").read_text())
        report.update({k: v for k, v in d.items() if not isinstance(v, list)})
    (destination / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)
