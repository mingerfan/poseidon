import json
import sys
sys.path.insert(0, "/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1")
import run_retry as runner

ROOT, OUT, TOOL = runner.ROOT, runner.OUT, runner.TOOL
runner.compiler[-1] = str(TOOL / "bin/hecate-opt.depth-dp-final")
runner.compiler.extend(["--boot-placement=depth-dp", "--mlir-elide-elementsattrs-if-larger=16"])
runner.REPORT_NAME = "depth-dp-fixed-report.json"


def write_status(data):
    path = OUT / "depth-dp-fixed-status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


runner.write_status = write_status
steps = [
    ("rmsnorm_depth_dp_fixed", OUT / "rmsnorm-fixed2/trace_rmsnorm_fixed.mlir", OUT / "rmsnorm.depth-dp-fixed.mlir", 1884, 180, 2880, 16),
    ("q_projection_depth_dp_fixed", OUT / "q-projection-fixed2/trace_projection_fixed.mlir", OUT / "q-projection.depth-dp-fixed.mlir", 1885, 180, 2880, 32),
    ("block_prefill_depth_dp_fixed", OUT / "block-prefill-fixed2/trace_qwen_block.mlir", OUT / "block-prefill.depth-dp-fixed.mlir", 1886, 900, 14400, 128),
]
for name, source, output, plan_id, wall_seconds, cpu_seconds, memory_gib in steps:
    if not runner.run(name, source, output, plan_id, wall_seconds, cpu_seconds, memory_gib):
        break
write_status(dict(state="finished", completed=runner.reports))
