#!/usr/bin/env python3
"""Separate Nsight samples; never included in benchmark timing medians."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess

out = Path(__file__).resolve().parent
root = out.parents[1]
profiles = root / "build-waterline-legacy-20261018/nsys-mlp"
profiles.mkdir(exist_ok=False)
result = {"scope": "Separate profiled MLP eager processes. Kernel span is first-to-last GPU kernel in the whole process, not Runtime online wall time. Excluded from latency medians.", "profiles": {}}
for variant in ("legacy", "unfixed", "current"):
    source = out / "local-runs" / f"local-mlp-r1-{variant}-eager"
    environment = dict(os.environ, **json.loads((source / "environment.json").read_text())["environment"])
    command = json.loads((source / "r0-eager/command.json").read_text())
    folder = profiles / variant
    folder.mkdir()
    command[-1] = str(folder / "report.json")
    prefix = folder / "profile"
    profile = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
               "--cuda-memory-usage=false", "--stats=false", "-o", str(prefix)] + command
    (folder / "command.json").write_text(json.dumps(profile, indent=2) + "\n")
    with (folder / "profile.log").open("w") as log:
        subprocess.run(profile, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["nsys", "export", "--type", "sqlite", "--output", str(prefix) + ".sqlite",
                        str(prefix) + ".nsys-rep"], stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads((folder / "report.json").read_text())
    assert report["passed"] and report["python_comparison"]["max_abs"] < 1e-4
    db = sqlite3.connect(str(prefix) + ".sqlite")
    names = dict(db.execute("select id,value from StringIds"))
    rows = list(db.execute("select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    shapes = Counter((names[r[2]],) + r[3:] for r in rows)
    signature = [dict(name=k[0], grid=k[1:4], block=k[4:7], count=v) for k, v in sorted(shapes.items())]
    start, end = rows[0][:2]
    busy = 0
    for row in rows[1:]:
        if row[0] > end:
            busy += end - start
            start, end = row[:2]
        else:
            end = max(end, row[1])
    busy += end - start
    span = max(r[1] for r in rows) - rows[0][0]
    calls = list(db.execute('select s.value,count(*),sum(r.end-r.start)/1e6 from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where s.value like "%EventQuery%" or s.value like "%LaunchKernel%" group by s.value'))
    result["profiles"][variant] = dict(passed=True, kernels=len(rows), kernel_shapes=signature,
        kernel_shape_sha256=hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
        cuda_api_summary=[dict(api=r[0], calls=r[1], cpu_api_ms=r[2]) for r in calls],
        kernel_union_ms=busy / 1e6, kernel_span_ms=span / 1e6,
        no_kernel_between_first_last_ms=(span - busy) / 1e6)
    db.close()
    print(variant, len(rows), busy / 1e6, span / 1e6, flush=True)
result["same_kernel_shapes_and_counts"] = len({r["kernel_shape_sha256"] for r in result["profiles"].values()}) == 1
assert result["same_kernel_shapes_and_counts"]
(out / "nsys-mlp.json").write_text(json.dumps(result, indent=2) + "\n")
