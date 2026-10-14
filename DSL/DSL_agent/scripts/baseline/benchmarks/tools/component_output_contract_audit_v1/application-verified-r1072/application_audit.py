"""Acceptance of the optional task/output bridge; synthetic backend fixtures only."""
import argparse
import hashlib
import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from benchmark_graph import digest, require
from hecate_python_env import ROOT, WORK, VENV
from platform_config import require_python_packages
from strict_json_files import dump
import test_task_output_adapter as tests


def run(output):
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV,
            "Pinned pure Python")
    require_python_packages(torch, np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK / "results"),
            "Fresh evidence")
    output.mkdir(mode=0o700)
    names = (
        "task_output_adapter.py", "test_task_output_adapter.py", "model_output_contract.py",
        "test_model_output_contract.py", "application_component.py", "application_backend.py",
        "application_model_source.py", "component_contract.py", "benchmark_graph.py",
        "application_policy.py", "model_resources.py", "test_application_component.py",
        "program_execution.py",
    )
    paths = [ROOT / "scripts/baseline" / name for name in names] + [Path(__file__)]
    def identity():
        return {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
    frozen = identity()
    started = time.monotonic()
    with (output / "units.log").open("x") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests)
        )
    units = {
        "run": result.testsRun, "failed": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
    }
    dump(output / "units-result.json", units)
    require(result.wasSuccessful() and not result.skipped, "Component bridge tests")
    require(identity() == frozen and time.monotonic() - started < 180, "Source/time bounds")
    report = {
        "format": "poseidon-task-output-adapter-acceptance-v1",
        "status": "component_protocol_validated", "units": units, "source_hashes": frozen,
        "runtime": {"python": list(sys.version_info[:3]), "numpy": np.__version__,
                    "torch": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip},
        "application": "existing Application lifecycle and package checks with injected fixtures",
        "execution": "synthetic protocol packets; not actual encrypted execution",
        "new_adapter": "explicit host selection",
        "legacy_application_entry_unchanged": True,
        "trusted_source_default_capture_unchanged": True,
        "hecate_tracing": 0, "compiler_calls": 0, "encrypted_execution": 0, "decryption": 0,
        "agent_generation": 0, "paid_calls": 0, "http_attempts": 0,
        "installation": 0, "download": 0, "parameters_changed": False,
        "existing_reference_changed": False, "frozen_benchmark_modified": False,
        "frozen_ckks_tolerance": {"atol": 1e-5, "rtol": 1e-4},
        "seconds": time.monotonic() - started,
    }
    report["binding"] = digest(report)
    dump(output / "report.json", report)
    print({key: report[key] for key in ("binding", "units", "status", "seconds")})
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().output))
