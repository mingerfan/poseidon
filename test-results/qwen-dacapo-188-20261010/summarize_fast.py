"""Summarize corrected native compilation and decoded-arithmetic validation."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


evidence = Path(__file__).resolve().parent
artifacts = evidence / "artifacts"
workspace = evidence.parents[1]
dacapo = workspace / "third_party/ckks-runtime/third_party/dacapo"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


provenance_path = evidence / "fast-compiler-provenance.json"
provenance = read(provenance_path)
for name in provenance["sources"]:
    provenance["sources"][name] = digest(dacapo / name)
provenance["compiler_sha256"] = digest(dacapo / "build/nix/bin/hecate-opt")
provenance["frontend_sha256"] = digest(dacapo / "build/nix/lib/libHecateFrontend.so")
provenance["integration_script_sha256"] = digest(
    workspace / "third_party/ckks-runtime/integrations/dacapo/generate_model_artifacts.py")
provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

methods = {}
for strategy, suffix, report, validation in (
    ("depth-dp", "depth_dp_fixed", "depth-dp-fixed-report.json", "fast-plan-validation.json"),
    ("greedy", "lazy_fixed", "lazy-fixed-report.json", "greedy-plan-validation.json"),
):
    runs = read(artifacts / report)
    assert len(runs) == 3 and all(run["returncode"] == 0 for run in runs)
    checked = read(artifacts / validation)
    assert len(checked) == 3
    full = next(record for record in checked if record["plan"].startswith("artifacts/block-prefill."))
    assert full["decoded_numpy_execution_verified"] and full["physical_metadata_verified"]
    log = (artifacts / f"block_prefill_{suffix}.log").read_text()
    assert "BypassDetection" not in log and "DaCapoPlanner" not in log
    placement_seconds = float(re.search(
        r"^\s*([\d.]+)\s+\([^\n]*\)\s+GreedyBootstrapPlacement$", log, re.M)[1])
    total_seconds = float(re.search(r"Total Execution Time: ([\d.]+) seconds", log)[1])
    run = next(run for run in runs if run["stage"].startswith("block_prefill_"))
    methods[strategy] = {
        "compiler_instrumented_seconds_including_parse_and_export": total_seconds,
        "supervisor_wall_seconds": run["elapsed_seconds"],
        "supervisor_poll_interval_seconds": 3,
        "placement_pass_seconds": placement_seconds,
        "peak_rss_MiB": run["max_rss_MiB"],
        "bootstrap_boundaries": int(re.search(r"bootstrap boundaries: (\d+)", log)[1]),
        "boot_count": full["op_counts"]["boot"],
        "execution_steps": full["execution_steps"],
        "full_block_validation": full,
        "component_checks": [record for record in checked if record is not full],
        "report": "artifacts/" + report,
    }

frontend = next(run for run in read(artifacts / "retrace-final-report.json")
                if run["stage"] == "block_fixed2_frontend")
assert frontend["returncode"] == 0
ctest_log = artifacts / "fast-compiler-ctest.log"
assert ctest_log.read_text().count("Test Passed.") == 3
temporary_paths = [Path("/tmp/poseidon-boot-placement"), Path("/tmp/qwen188-stage-r7d_vesp")]
temporary_usage = {}
for path in temporary_paths:
    result = subprocess.run(["du", "-sk", str(path)], check=True, capture_output=True, text=True)
    temporary_usage[str(path)] = int(result.stdout.split()[0]) * 1024

summary = {
    "format": "qwen-dacapo-fast-boot-corrected-v1",
    "host": "188Server / HPU-001",
    "experiment_root": "/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1",
    "scope": "Actual-dimension Qwen two-token one-decoder-block prefill, synthetic fixture weights",
    "compiler_profile": "N=65536, slots=32768, rescale factor=120, waterline=40, logical boot bounds=[1,4]",
    "frontend": frontend,
    "source": read(artifacts / "fast-source-metadata.json"),
    "provenance": provenance,
    "methods": methods,
    "boot_count_reduction_percent": 100 * (1 - methods["depth-dp"]["boot_count"] / methods["greedy"]["boot_count"]),
    "tests": {"ctest_passed": 3, "ctest_total": 3, "log": str(ctest_log.relative_to(evidence))},
    "correctness_fix": {
        "prefix_units_are_zero_padded_masks": True,
        "full_slot_unit_folds_preserve_encoding_scale": True,
        "frontend_cannot_assume_selected_runtime_slot_capacity": True,
        "compiler_and_frontend_rebuilt": True,
        "native_source_retraced": True,
        "pre_fix_traces_and_plans_valid_for_numerics": False,
    },
    "temporary_cache": {
        "local_allocated_bytes": temporary_usage,
        "local_total_MiB": sum(temporary_usage.values()) / 2**20,
        "remote_path": "/tmp/poseidon-boot-placement-IUOMH1",
        "remote_allocated_KiB": 64,
        "large_model_artifacts_in_tmp": False,
    },
    "limits": [
        "Depth-boundary conservative cost model; not a complete Fhelipe port or arbitrary-DAG optimum",
        "Per-Mul eager rescale can increase Boot count relative to DaCapo's latency-aware planning",
        "DP does not improve every graph: standalone RMSNorm uses 17 Boots versus lazy greedy's 10",
        "Decoded NumPy interpreter treats Boot/rescale/modswitch/relinearize as identities; no encrypted CKKS noise check",
        "Selected runtime profile uses host decrypt_reencrypt Boot; no real encrypted GPU Boot execution",
        "Full 24-layer model compilation and runtime memory requirements are untested",
    ],
    "remote_full_plan": methods["depth-dp"]["full_block_validation"]["plan"],
    "local_component_artifacts": "artifacts/{rmsnorm,q-projection}.depth-dp-fixed.*",
}
output = evidence / "fast-boot-summary.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
historical_path = evidence / "summary.json"
historical = read(historical_path)
historical["historical_only"] = True
historical["latest_summary"] = output.name
historical["notice"] = "Pre-fix investigation. The original frontend deleted zero-padded masks; use the corrected fast-boot summary for current compilation and numerical validation."
historical_path.write_text(json.dumps(historical, indent=2) + "\n")
print(json.dumps({name: {key: value[key] for key in (
    "boot_count", "placement_pass_seconds", "compiler_instrumented_seconds_including_parse_and_export", "peak_rss_MiB")}
    for name, value in methods.items()}, indent=2))
print("boot_count_reduction_percent", summary["boot_count_reduction_percent"])
print("local_tmp_MiB", summary["temporary_cache"]["local_total_MiB"])
