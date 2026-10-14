"""Summarize both completed and independently verified full-size GPU plans."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
REMOTE = "/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-gpu-memory"
records = []
for devices in (1, 4):
    directory = ROOT / f"gpu{devices}"

    def read(name):
        return json.loads((directory / name).read_text())

    compiler = read("compiler-report.json")
    validator = read("validation-report.json")
    validation = read("validation.json")
    assert compiler["returncode"] == validator["returncode"] == 0
    assert all(validation[name] for name in ("matches_complete_host_graph", "hashes_verified",
                                            "topology_verified", "physical_metadata_verified",
                                            "release_reuse_verified", "memory_report_independently_verified"))
    memory = read("qwen24._hecate_qwen25_24layer.memory.json")
    before = read("qwen24._hecate_qwen25_24layer.memory.before.json")
    gpu_rows = [row for row in memory["places"] if row["place"]["kind"] == "device"]
    host = next(row for row in memory["places"] if row["place"]["kind"] == "host")
    assert len(gpu_rows) == devices
    log = (directory / "compile.log").read_text()
    times = {}
    for label in ("Parser", "GreedyBootstrapPlacement", "AssignPlacement", "MaterializeCommunication",
                  "PlanRuntimeMemory", "EmitRuntimePlan", "Output"):
        match = re.search(r"^\s*([\d.]+)\s+\([^\n]*\)\s+" + label + r"$", log, re.M)
        assert match, label
        times[label] = float(match[1])
    assert validation["op_counts"]["boot"] == 10835
    records.append({"devices": devices, "plan_remote": f"{REMOTE}/gpu{devices}/qwen24._hecate_qwen25_24layer.runtime-plan.json",
                    "compiler": compiler, "compiler_pass_seconds": times, "validation_supervisor": validator,
                    "validation": validation,
                    "gpu_peak_bytes": [row["peak_bytes"] for row in gpu_rows],
                    "gpu_peak_TiB": [row["peak_bytes"] / 2**40 for row in gpu_rows],
                    "max_gpu_peak_TiB": max(row["peak_bytes"] for row in gpu_rows) / 2**40,
                    "aggregate_gpu_peak_TiB": validation["aggregate_gpu_peak"]["bytes"] / 2**40,
                    "host_rns_object_peak_GiB": host["peak_bytes"] / 2**30,
                    "gpu_peaks_before_release_TiB": [row["peak_bytes"] / 2**40 for row in before["places"] if row["place"]["kind"] == "device"],
                    "memory": memory})
summary = {"status": "passed", "scope": "Full actual-size 24 blocks + final RMSNorm + full 151936-vocabulary LM head, two-token prefill",
           "weights": "synthetic distinct-layer weights; no pretrained checkpoint",
           "provenance": json.loads((ROOT / "provenance.json").read_text()),
           "artifact_inventory": json.loads((ROOT / "artifact-inventory.json").read_text()),
           "boot_count": 10835, "plans": records,
           "memory_kind": "Sequential RNS object estimates, not measured GPU/process peaks or bounds for parallel execution",
           "assumption": records[0]["memory"]["assumption"], "excluded": records[0]["memory"]["excluded"],
           "eager_plaintext_initialization": True,
           "hardware": "188Server: four Tesla V100-SXM2-32GB devices",
           "fits_existing_gpus": False, "runtime_execution_tested": False,
           "large_artifacts_in_tmp": False}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps([{"devices": r["devices"], "compiler_seconds": r["compiler"]["elapsed_seconds"],
                   "compiler_peak_GiB": r["compiler"]["max_rss_MiB"] / 1024,
                   "gpu_peak_TiB": r["gpu_peak_TiB"], "validation_seconds": r["validation"]["seconds"]}
                  for r in records], indent=2))
