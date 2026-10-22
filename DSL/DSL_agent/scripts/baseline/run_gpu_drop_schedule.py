"""Real GPU execution of explicit physical-Q static IR, not a HEVM artifact."""
from run_gpu_modswitch_semantics import SOURCES, run

EXTRA = (
    "scripts/baseline/run_gpu_drop_schedule.py",
    "src/poseidon/tests/frontends/dacapo/gpu_drop_modulus_schedule_test.cpp",
    "src/poseidon/tests/mgpu/mgpu_drop_modulus_test.cpp",
    "src/poseidon/mgpu/ir/schedule.h",
    "src/poseidon/mgpu/ir/schedule.cpp",
    "src/poseidon/mgpu/ir/schedule_json.cpp",
    "src/poseidon/mgpu/ir/schedule_summary.h",
    "src/poseidon/mgpu/ir/schedule_summary.cpp",
    "src/poseidon/mgpu/compiler/schedule_verifier.cpp",
    "src/poseidon/mgpu/compiler/static_placement.cpp",
    "src/poseidon/mgpu/compiler/copy_insertion.cpp",
    "src/poseidon/mgpu/compiler/scheduler/static_scheduler.cpp",
    "src/poseidon/mgpu/compiler/scheduler/latency_table.cpp",
    "src/poseidon/mgpu/comm/topology.cpp",
    "src/poseidon/mgpu/runtime/backend/poseidon_gpu_execution_backend.cpp",
    "src/poseidon/mgpu/runtime/executor/sequential_schedule_executor.cpp",
    "src/poseidon/mgpu/runtime/preflight/poseidon_gpu_schedule_preflight.cpp",
)

if __name__ == "__main__":
    raise SystemExit(run(binary_name="poseidon_gpu_drop_modulus_schedule_tests",
                        prefix="poseidon-gpu-drop-schedule-", sources=SOURCES+EXTRA,
                        backend="poseidon_gpu_static_schedule"))
