"""Bypass only the Boot search; retain the current threshold passes and actual primes."""

from pathlib import Path
import hashlib
import json
import re
import subprocess
import time

ROOT = Path("/home/xuming/poseidon-manual-boot-20261026")
AUDIT = ROOT / "resnet20-optimized-20261027T055506Z/placement-audit"
EXPERIMENT = AUDIT / "target-scale-experiment"
OUT = EXPERIMENT / "scale-overhead-20261028"
DRIVER = OUT / "audit-opt-wrapper"
CONFIG = ROOT / "alignment-tests/compiler.json"
RAW = Path("/home/xuming/poseidon-mixed-q-20261026/data/ResNet20.mlirbc")
current = EXPERIMENT / "chain32-fixed25-20261028/candidates/manual/full-optimized-q34"
elided = OUT / "threshold27-elided.mlir"
subprocess.run(
    [
        str(EXPERIMENT / "hecate-opt-wrapper"),
        str(current / "model.optimized.mlirbc"),
        "--ckks-config=" + str(CONFIG),
        "--mlir-elide-elementsattrs-if-larger=4",
        "-o",
        str(elided),
    ],
    check=True,
    timeout=120,
)
header = next(line for line in elided.read_text().splitlines() if "func.func " in line)
schedule = re.search(r"btp_schedule = \[(.*?)\], btp_target", header)[1]
schedule = [
    [int(x) for x in row.split(",")]
    for row in re.findall(r"array<i64: ([^>]+)>", schedule)
]
assert len(schedule) == 27
(OUT / "threshold27-schedule.json").write_text(json.dumps(schedule) + "\n")
rows = []


def run(name, source, spec, schedule, initial):
    destination = OUT / (name + ".managed.mlir")
    assert destination != source
    command = [
        str(DRIVER),
        str(source),
        "--ckks-config=" + str(CONFIG),
        "--audit-native-spec=" + str(spec),
        "--native-input-level=" + str(initial - 1),
        "--audit-schedule-path=" + str(schedule),
        "--remove-bootstrap",
        "--optimize-polynomial-arithmetic",
        "--audit-attach-schedule",
        "--bootstrapping-placement=bootstrap-scale=40",
        "--proactive-rescaling=waterline=40 bootstrap-scale=40",
        "--early-modswitch",
        "--absorb-coefficient-scale=waterline=40",
        "--canonicalize",
        "--cse",
        "--mlir-elide-elementsattrs-if-larger=4",
        "--mlir-print-op-on-diagnostic",
        "-o",
        str(destination),
    ]
    started = time.monotonic()
    with (OUT / (name + ".log")).open("w") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=180
        )
    row = dict(
        name=name,
        command=command,
        returncode=result.returncode,
        seconds=time.monotonic() - started,
        compiler_sha256=(OUT / "audit-binary.sha256").read_text().strip(),
    )
    if not result.returncode:
        lines = destination.read_text().splitlines()
        row["header"] = next(line for line in lines if "func.func " in line)
    else:
        row["failure"] = (OUT / (name + ".log")).read_text()[-5000:]
    (OUT / (name + "-result.json")).write_text(json.dumps(row, indent=2) + "\n")
    rows.append(row)
    print(
        name,
        result.returncode,
        row.get("failure", row.get("header", ""))[-2000:],
        flush=True,
    )


run(
    "fixed27-control",
    RAW,
    ROOT / "alignment-tests/operator-spec.json",
    OUT / "threshold27-schedule.json",
    34,
)
assert rows[-1]["returncode"] == 0
control = rows[-1]["header"]
assert (
    abs(
        float(re.search(r"earth.result_scales = array<f64: ([^>]+)>", control)[1])
        - 66.9485344932141
    )
    < 1e-8
)
for name in ["manual", "all32", "mix-30-20-2"]:
    spec = (
        EXPERIMENT / "chain32-fixed25-20261028/candidates" / name / "operator-spec.json"
    )
    run(
        "fixed18-" + name,
        RAW,
        spec,
        EXPERIMENT / "budget25/manual-cuts/manual18.json",
        34,
    )

run(
    "fixed18-stemout",
    EXPERIMENT / "budget25/stem-out/stem-out.mlir",
    ROOT / "alignment-tests/operator-spec.json",
    EXPERIMENT / "budget25/stem-out/schedule.json",
    31,
)

(OUT / "empty-schedule.json").write_text("[]\n")
spec = EXPERIMENT / "chain32-fixed25-20261028/candidates/manual/diagnostic-spec.json"
text = (AUDIT / "one-conv.mlir").read_text()
for initial_q in [31, 9]:
    source = OUT / ("conv-input" + str(initial_q) + ".mlir")
    source.write_text(
        text.replace(
            ">> {\n",
            ">> attributes {segment_inputType = [tensor<1x!earth.ci<40 * "
            + str(31 - initial_q)
            + ">>], segment_input_scales = array<f64: 40.0>} {\n",
            1,
        )
    )
    assert "segment_inputType" in source.read_text()[:300]
    run("conv-input" + str(initial_q), source, spec, OUT / "empty-schedule.json", 31)
    assert rows[-1]["returncode"] == 0

(OUT / "fixed-schedule-results.json").write_text(json.dumps(rows, indent=2) + "\n")
print("source bytecode sha256", hashlib.sha256(RAW.read_bytes()).hexdigest())
