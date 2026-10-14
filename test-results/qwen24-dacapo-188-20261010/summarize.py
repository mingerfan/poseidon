"""Summarize completed full-size native compilation; no execution claims."""
import json
from pathlib import Path
import re


root = Path(__file__).resolve().parent
artifacts = root / "artifacts"


def read(name):
    return json.loads((artifacts / name).read_text())


phases = {}
for name, file in (("fixture", "fixture-report.json"), ("frontend", "frontend-report.json"),
                   ("optimizer", "optimizer-report.json"), ("validation", "validation-report.json")):
    reports = read(file)
    assert len(reports) == 1 and reports[0]["returncode"] == 0, (name, reports)
    phases[name] = reports[0]
validation = read("compilation-validation.json")
assert validation["input_ciphertexts"] == 2 and validation["output_ciphertexts"] == 245
assert validation["hashes_verified"] and validation["topology_verified"]
assert validation["physical_metadata_verified"]
log = (artifacts / "qwen24_depth_dp.log").read_text()
boot_count = int(re.search(r"Number of Bootstrapping: (\d+)", log)[1])
assert boot_count == validation["op_counts"]["boot"]
assert "BypassDetection" not in log and "DaCapoPlanner" not in log
compiler_seconds = float(re.search(r"Total Execution Time: ([\d.]+) seconds", log)[1])
pass_seconds = {}
for label in ("Parser", "GreedyBootstrapPlacement", "EarthToCKKSConversion",
              "UpscaleToMulcpConversion", "MaterializePhysicalLevels", "EmitRuntimePlan", "Output"):
    match = re.search(r"^\s*([\d.]+)\s+\([^\n]*\)\s+" + label + r"$", log, re.M)
    assert match, label
    pass_seconds[label] = float(match[1])
summary = {
    "format": "qwen24-dacapo-native-compilation-v1",
    "status": "passed",
    "host": "188Server / HPU-001",
    "remote_directory": "/home/xuming/poseidon-qwen-dacapo-20261010-IUOMH1/artifacts/qwen24-native",
    "scope": "Actual-size unsegmented 24 decoder blocks + final RMSNorm + complete 151936-vocabulary LM head, two-token prefill",
    "model": read("fixture.json")["config"],
    "weights": {"synthetic": True, "seed": 20261009,
                "weight_files": 291, "raw_weight_bytes": read("fixture.json")["weight_bytes"],
                "pretrained_checkpoint": False, "layers_share_weights": False},
    "strategy": "depth-dp",
    "native_compilation_completed": True,
    "frontend_format": "lossless MLIR bytecode; all constants retained",
    "phases": phases,
    "compiler_instrumented_seconds": compiler_seconds,
    "compiler_pass_seconds": pass_seconds,
    "boot_count": boot_count,
    "bootstrap_boundaries": int(re.search(r"bootstrap boundaries: (\d+)", log)[1]),
    "validation": validation,
    "provenance": json.loads((root / "provenance.json").read_text()),
    "tests": {"ctest_passed": 5, "ctest_total": 5, "log": "artifacts/ctest-detail.log"},
    "artifact_inventory": read("artifact-inventory.json"),
    "limits": [
        "Compilation, payload integrity, graph topology and physical metadata are verified",
        "Full-size original-function NumPy reference and approximation-domain checks were generated with the fixture",
        "Full-model compiled DAG decoded arithmetic, encrypted CKKS accuracy and GPU execution are untested",
        "Runtime memory requirements are unmeasured; the reported memory is compiler/process RSS",
        "Host decrypt_reencrypt Boot profile and estimated GPU latency table; not actual GPU Boot",
        "No pretrained-weight calibration or language-generation quality claim",
    ],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"frontend_seconds": phases["frontend"]["elapsed_seconds"],
                  "compiler_seconds": phases["optimizer"]["elapsed_seconds"],
                  "compiler_peak_GiB": phases["optimizer"]["max_rss_MiB"] / 1024,
                  "boots": boot_count, "plan_bytes": validation["plan_bytes"],
                  "bundle_bytes": validation["bundle_bytes"],
                  "execution_steps": validation["execution_steps"]}, indent=2))
