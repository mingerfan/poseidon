"""Compile the complete, unsegmented 24-layer native bytecode graph."""
import json
from pathlib import Path
import sys


ROOT = Path("/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
sys.path.insert(0, str(ROOT))
import run_retry as runner

runner.OUT = ROOT / "artifacts/qwen24-native"
runner.compiler[-1] = str(runner.TOOL / "bin/hecate-opt.depth-dp-final")
runner.compiler.extend(["--boot-placement=depth-dp", "--mlir-elide-elementsattrs-if-larger=16"])
runner.REPORT_NAME = "optimizer-report.json"


def write_status(record):
    path = runner.OUT / "optimizer-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(path)


runner.write_status = write_status
source = runner.OUT / "traced/trace_qwen24.mlirbc"
assert source.is_file()
with source.open("rb") as stream:
    assert stream.read(4) == b"ML\xefR"
success = runner.run("qwen24_depth_dp", source, runner.OUT / "qwen24.depth-dp.mlir",
                     2401, 7200, 115200, 256)
write_status(dict(state="finished", completed=runner.reports))
if not success:
    raise SystemExit(1)
