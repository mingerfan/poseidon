"""Earlier compiler control; no further compiler runs after switching to simulation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
EXPERIMENT = (
    ROOT / "resnet20-optimized-20261027T055506Z/placement-audit/target-scale-experiment"
)
OUT = EXPERIMENT / "boot-attribution-20261030"
REFERENCE = EXPERIMENT / "chain32-fixed25-20261028/candidates/manual/full-optimized-q34"
BASE = json.loads((REFERENCE / "result.json").read_text())
MODULI = json.loads((REFERENCE.parent / "operator-spec.json").read_text())["context"][
    "rns_moduli"
]
ONE = "sha256:" + hashlib.sha256(struct.pack("<d", 1.0) * 32768).hexdigest()


def evidence(plan, floor):
    values = {v["id"]: v for v in plan["values"]}
    payload = {}
    producers = {}
    for step in plan["initialization"] + plan["execution"]:
        if step["kind"] == "encode":
            payload[step["output"]] = step["payload"]["content"]
        elif step["kind"] == "transfer":
            for a, b in zip(step["inputs"], step["outputs"]):
                if a in payload:
                    payload[b] = payload[a]
        elif step["kind"] == "compute":
            producers[step["output"]] = step
    rows = []
    for step in plan["execution"]:
        if step.get("op") != "boot":
            continue
        current = step["inputs"][0]
        drops = 0
        while current in producers and producers[current]["op"] in (
            "mod_switch",
            "rescale",
        ):
            before = producers[current]
            drops += before["op"] == "rescale"
            current = before["inputs"][0]
        plain = None
        scalar_error = None
        if drops:
            multiply = producers[current]
            assert multiply["op"] == "mul_cp" and payload[multiply["inputs"][1]] == ONE
            plain = values[multiply["inputs"][1]]["scale_log2"]
            assert plain + 1e-8 >= floor
            scalar_error = abs(round(2**plain) / 2**plain - 1)
            current = multiply["inputs"][0]
        while current in producers and producers[current]["op"] == "mod_switch":
            current = producers[current]["inputs"][0]
        source = values[current]
        rows.append(
            dict(
                source=source,
                input=values[step["inputs"][0]],
                output=values[step["output"]],
                alignment_drops=drops,
                plain_scale=plain,
                scalar_unit_rounding_error=scalar_error,
            )
        )
    return rows


def run(floor, strategy="optimized"):
    name = f"threshold-min{floor}-{strategy}"
    destination = OUT / name
    destination.mkdir(exist_ok=False)
    command = BASE["command"].copy()
    command[0] = str(OUT / "hecate-opt-wrapper")
    command[command.index("-o") + 1] = str(destination / "model.optimized.mlirbc")
    command = [
        f"--boot-placement={strategy}" if s.startswith("--boot-placement=") else s
        for s in command
    ]
    command.append("--audit-boot-min-plain-scale=" + str(floor))
    row = dict(
        name=name,
        strategy=strategy,
        initial_q=34,
        encoding_floor=floor,
        command=command,
        compiler_sha256=(OUT / "binary.sha256").read_text().strip(),
    )
    started = time.monotonic()
    with (destination / "compile.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        row["pid"] = process.pid
        (destination / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        code = process.wait(timeout=600)
    row.update(returncode=code, seconds=time.monotonic() - started)
    if not code:
        plan_path = next(destination.glob("*.runtime-plan.json"))
        plan = json.loads(plan_path.read_text())
        values = {v["id"]: v for v in plan["values"]}
        boots = evidence(plan, floor)
        for b in boots:
            assert (
                b["input"]["level"] == 5 and abs(b["input"]["scale_log2"] - 40) < 1e-8
            )
            assert b["output"]["level"] == 30 and b["output"]["scale_log2"] == 40
        error = 0.0
        for s in plan["execution"]:
            if s.get("op") == "rescale":
                a, b = values[s["inputs"][0]], values[s["output"]]
                expected = a["scale_log2"] - sum(
                    math.log2(q) for q in MODULI[b["level"] + 1 : a["level"] + 1]
                )
                error = max(error, abs(expected - b["scale_log2"]))
        assert error < 1e-8
        validation = subprocess.run(
            [
                str(ROOT / "validate-plan"),
                str(plan_path),
                str(REFERENCE.parent / "operator-spec.json"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert validation.returncode == 0, validation.stdout + validation.stderr
        row.update(
            boots=len(boots),
            counts=dict(
                Counter(s["op"] for s in plan["execution"] if s["kind"] == "compute")
            ),
            alignment_rescales=sum(b["alignment_drops"] for b in boots),
            alignment_distribution=dict(Counter(b["alignment_drops"] for b in boots)),
            boot_entries=boots,
            plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            validation=validation.stdout + validation.stderr,
            validation_code=validation.returncode,
            max_rescale_formula_error=error,
        )
        assert row["counts"]["mul_cc"] == 513 and row["counts"]["rotate"] == 1709
        if floor == 30 and strategy == "optimized":
            assert row["plan_sha256"] == BASE["plan_sha256"], (
                "Precision30 baseline does not reproduce the production plan"
            )
    else:
        row["failure"] = (destination / "compile.log").read_text()[-4000:]
    (destination / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    print(
        {k: v for k, v in row.items() if k not in ("command", "boot_entries")},
        flush=True,
    )
    return row


if __name__ == "__main__":
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(run, floor) for floor in [30, 24, 20, 15, 8, 0]]
        for future in as_completed(jobs):
            rows.append(future.result())
            (OUT / "precision-results.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
    print("Precision-only optimized runs complete", flush=True)
