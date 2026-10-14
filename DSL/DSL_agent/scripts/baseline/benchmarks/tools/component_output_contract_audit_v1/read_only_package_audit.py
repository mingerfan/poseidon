"""Read-only compatibility inspection of retained real Poseidon task packages."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from application_backend import PoseidonBackend
from application_component import Application
from application_identity import compatible
from benchmark_graph import canonical, digest, require
from hecate_python_env import ROOT, WORK, VENV
from model_output_contract import snapshot
from platform_config import require_python_packages
from strict_json_files import dump
from task_output_adapter import TaskOutputAdapter


class ReadOnlyBackend(PoseidonBackend):
    """Use real identity/package checking; forbid execution-changing operations."""
    def __init__(self):
        self.checks = 0

    def check_package(self, folder, level):
        self.checks += 1
        return super().check_package(folder, level)

    def qualify(self, *args, **kwargs):
        raise AssertionError("Read-only audit must not compile")

    def export(self, *args, **kwargs):
        raise AssertionError("Read-only audit must not change retained packages")

    def execute(self, *args, **kwargs):
        raise AssertionError("Read-only audit must not execute ciphertext")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder):
    require(folder.is_dir() and not folder.is_symlink(), "Retained task directory")
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    require(len(files) <= 128 and all(not path.is_symlink() for path in files),
            "Bounded direct retained files")
    require(sum(path.stat().st_size for path in files) <= 64 * 1024**2,
            "Bounded retained bytes")
    return {str(path.relative_to(folder)): sha(path) for path in files}


def stage_differences(previous, current):
    rows = {}
    for stage in ("generation", "compilation", "execution", "validation"):
        a, b = previous.get(stage, {}), current.get(stage, {})
        row = {"identity_equal": a == b, "source_changes": [],
               "configuration_changes": []}
        for key, label in (("sources", "source_changes"),
                           ("configuration_files", "configuration_changes")):
            old, new = a.get(key, {}), b.get(key, {})
            require(type(old) is dict and type(new) is dict, "Bounded stage identities")
            row[label] = [
                {"path": path, "old_sha256": old.get(path), "new_sha256": new.get(path)}
                for path in sorted(set(old) | set(new))
                if old.get(path) != new.get(path)
            ]
        rows[stage] = row
    return rows


def run(output):
    require(os.environ.get("IN_NIX_SHELL") == "pure" and Path(sys.prefix) == VENV,
            "Pinned pure Python")
    require_python_packages(torch, np)
    require(not output.exists() and output.parent.resolve().is_relative_to(WORK / "results"),
            "Fresh audit evidence")
    output.mkdir(mode=0o700)
    started = time.monotonic()
    proposal_path = ROOT / "docs/baseline/agent-common-pytorch-offline-proposal-r1076.json"
    proposal = json.loads(proposal_path.read_text())
    frozen = {}
    for key in ("source_hashes", "extra_source_hashes", "case_hashes", "tool_hashes"):
        for name, value in proposal[key].items():
            path = Path(name)
            path = path if path.is_absolute() else ROOT / path
            require(sha(path) == value, "Current pending freeze")
            frozen[str(path)] = value
    history_path = ROOT / "docs/baseline/agent-pytorch-split-acceptance-r878.json"
    history = json.loads(history_path.read_text())
    history_sha = sha(history_path)
    source_paths = [
        Path(__file__), ROOT / "scripts/baseline/task_output_adapter.py",
        ROOT / "scripts/baseline/model_output_contract.py",
        ROOT / "scripts/baseline/application_component.py",
        ROOT / "scripts/baseline/application_backend.py",
        ROOT / "scripts/baseline/application_identity.py",
        ROOT / "scripts/baseline/candidate_bundle.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    backend = ReadOnlyBackend()
    current = backend.identity_manifest()
    dump(output / "current-identity.json", current)
    rows = []
    for old in history["rows"][:3]:
        require(time.monotonic() - started < 50, "Read-only time bound")
        folder = Path(old["package"])
        require(folder.name == "program" and folder.parent.parent.resolve().is_relative_to(WORK / "results"),
                "Retained package scope")
        task_dir = folder.parent
        before = inventory(task_dir)
        metadata = json.loads((task_dir / "task.json").read_text())
        state = json.loads((task_dir / "state.json").read_text())
        manifest = json.loads((folder / "manifest.json").read_text())
        require(state["task_id"] == task_dir.name
                and state["package"]["binding"] == manifest["binding"]
                and state["status"] == "succeeded" and old["status"] == "passed",
                "Original accepted task binding")
        app = Application(task_dir.parent, backend)
        result = app.result(task_dir.name)
        names = [item["name"] for item in metadata["request"]["model"]["outputs"]]
        tree = {"kind": "tuple", "items": [
            *[{"kind": "tensor", "name": name} for name in names],
            {"kind": "none"},
        ]}
        frame = TaskOutputAdapter(app).bind(task_dir.name, tree)
        reload_status = None
        if "receipt" in frame:
            receipt = json.loads(json.dumps(frame["receipt"]))
            reload_frame = TaskOutputAdapter(app).result(
                task_dir.name, receipt, expected_binding=receipt["binding"]
            )
            reload_status = reload_frame["status"]
            dump(output / (old["id"] + ".receipt.json"), receipt)
        direct = None
        try:
            binding = backend.check_package(folder, metadata["validation_level"])
            direct = {"status": "integrity_checked", "binding": binding,
                      "binding_matches_original": binding == manifest["binding"]}
        except Exception as error:
            direct = {
                "status": "rejected", "error_type": type(error).__name__,
                "locations": [
                    {"file": Path(item.filename).name, "line": item.lineno}
                    for item in traceback.extract_tb(error.__traceback__)
                    if str(ROOT) in item.filename
                ][-5:],
            }
            # Internal diagnostic stays in guest evidence; never enters a prompt.
            dump(output / (old["id"] + ".private-package-rejection.json"),
                 {"error_type": type(error).__name__, "diagnostic": str(error)})
        differences = stage_differences(metadata.get("identity", {}), current)
        unchanged = inventory(task_dir) == before
        require(unchanged, "Retained task/package files are read-only")
        row = {
            "id": old["id"], "package": str(folder), "task_id": task_dir.name,
            "historical_status": old["status"],
            "historical_package_binding": manifest["binding"],
            "current_application_status": result["status"],
            "current_application_failure": result.get("failure"),
            "current_output_adapter_status": frame["status"],
            "current_output_adapter_failure": frame.get("failure"),
            "receipt_json_reload_status": reload_status,
            "direct_current_package_check": direct,
            "identity_delivery_compatible": compatible(metadata["identity"], current),
            "stage_differences": differences,
            "retained_file_count": len(before), "retained_files_unchanged": unchanged,
            "retained_hashes": before,
            "hecate_tracing": 0, "compiler_calls": 0, "encrypted_execution": 0,
            "decryption": 0, "agent_generation": 0, "paid_calls": 0,
        }
        rows.append(row)
        dump(output / (old["id"] + ".result.json"), row)
        dump(output / (old["id"] + ".application-result.json"),
             snapshot(result, limit=4 * 1024**2))
        dump(output / (old["id"] + ".output-adapter-result.json"),
             snapshot(frame, limit=4 * 1024**2))
        print(json.dumps({
            "id": row["id"], "application": row["current_application_status"],
            "output_adapter": row["current_output_adapter_status"],
            "package_check": direct["status"],
            "compatible": row["identity_delivery_compatible"],
            "retained_files_unchanged": unchanged,
        }), flush=True)
    require(sha(history_path) == history_sha
            and all(sha(Path(path)) == value for path, value in frozen.items())
            and all(sha(ROOT / path) == value for path, value in source_hashes.items()),
            "Final frozen/history/source checks")
    report = {
        "format": "poseidon-output-package-read-only-audit-v1",
        "status": "read_only_inspection_complete",
        "rows": rows, "packages": len(rows),
        "current_application_successes": sum(row["current_application_status"] == "succeeded" for row in rows),
        "current_output_receipts": sum(row["receipt_json_reload_status"] == "succeeded" for row in rows),
        "direct_package_integrity_successes": sum(row["direct_current_package_check"]["status"] == "integrity_checked" for row in rows),
        "history_binding": history["binding"], "history_sha256": history_sha,
        "source_hashes": source_hashes, "current_proposal_binding": proposal["binding"],
        "frozen_file_entries": sum(len(proposal[key]) for key in ("source_hashes", "extra_source_hashes", "case_hashes", "tool_hashes")),
        "package_checker_calls": backend.checks, "seconds": time.monotonic() - started,
        "hecate_tracing": 0, "compiler_calls": 0, "encrypted_execution": 0, "decryption": 0,
        "agent_generation": 0, "paid_calls": 0, "http_attempts": 0,
        "installation": 0, "download": 0, "backend_core_changed": False,
        "parameters_changed": False, "existing_reference_changed": False,
        "old_results_relabelled": False,
        "projection_scope": "explicit host output projection; not original-source static None capture",
        "next_action": "requalification under current code where identities differ; no rewriting retained historical bindings",
        "frozen_ckks_tolerance": {"atol": 1e-5, "rtol": 1e-4},
    }
    report["binding"] = digest(report)
    dump(output / "report.json", report)
    print(json.dumps({key: report[key] for key in (
        "binding", "packages", "current_application_successes",
        "current_output_receipts", "direct_package_integrity_successes", "seconds",
    )}))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(run(parser.parse_args().output))
