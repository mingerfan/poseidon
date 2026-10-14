"""Explicit approximate-model diagnostic; two manual candidates, no LLM/network."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from approximation_errors import decompose, relu_and_quadratic
from hecate_python_env import ROOT, WORK, VENV, digest, enter_nix
from run_agent_batch import save
from seal_artifact_gate import require

CASE = ROOT / "scripts/baseline/cases/relu-quadratic-explicit.json"
GOLDENS = ROOT / "scripts/baseline/golden_cases/approximation"


def inside():
    import numpy as np
    require(bool(os.environ.get("IN_NIX_SHELL")) and Path(sys.prefix) == VENV, "Pinned environment required")
    os.umask(0o077)
    out = Path(tempfile.mkdtemp(prefix="approximation-golden-", dir=WORK / "results"))
    sources = [CASE, GOLDENS / "quadratic.py", GOLDENS / "quadratic-wrong-sign.py",
               Path(__file__), ROOT / "scripts/baseline/approximation_errors.py"]
    frozen = {str(p): digest(p) for p in sources}
    snapshot_dir = out / "source-snapshot"
    snapshot_dir.mkdir()
    snapshots = {}
    for index, path in enumerate(sources):
        snapshot = snapshot_dir / (str(index) + "-" + path.name)
        shutil.copyfile(path, snapshot)
        require(digest(snapshot) == frozen[str(path)], "Snapshot changed while copying")
        snapshots[str(path)] = str(snapshot.relative_to(out))
    report = dict(status="running", experiment="explicit_approximation_error_decomposition",
                  original_model="ReLU(x)", executed_model="(x+x*x)/2",
                  declared_domain=[-1, 1], analytic_max_approximation_error=0.125,
                  approximation_acceptance_budget=None, original_semantic_equivalence_verified=False,
                  agent_calls=0, poseidon_gpu_validated=False, source_hashes=frozen,
                  source_snapshots=snapshots, cases=[])
    print(f"Approximation evidence: {out}", flush=True)
    for name, wrong in (("quadratic", False), ("quadratic-wrong-sign", True)):
        row = dict(golden=name, counterexample=wrong, matched_expected=False)
        report["cases"].append(row)
        try:
            command = [sys.executable, str(ROOT / "scripts/baseline/run_candidate.py"), "--inside",
                       "--case", str(CASE), "--golden-file", str(GOLDENS / (name+".py")), "--max-repairs", "0"]
            row["command"] = command
            log = out / (name+".log")
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
            row["exit_code"] = result.returncode
            require(log.stat().st_size <= 1024**2, "Runner log size limit")
            paths = re.findall(r"^Candidate evidence: (.+)$", log.read_text(), re.MULTILINE)
            require(len(paths) == 1, "Missing candidate evidence")
            run = Path(paths[0]).resolve()
            require(run.is_relative_to(WORK / "results"), "Unexpected evidence path")
            row["run"] = str(run)
            candidate = json.loads((run / "report.json").read_text())
            require(candidate["agent_calls"] == 0 and not candidate["poseidon_gpu_validated"] and
                    candidate["backend"] == "upstream_SEAL_HEVM_CPU", "Wrong execution source")
            require(json.loads((run / "model.json").read_text()) == json.loads(CASE.read_text()), "Model changed")
            attempt = candidate["attempts"][0]
            require(attempt["execution"]["encrypted_execution"] and not attempt["execution"]["bootstrap_executed"],
                    "Missing real non-bootstrap execution")
            with np.load(run / "arrays.npz", allow_pickle=False) as arrays:
                inputs, target = arrays["inputs"].copy(), arrays["reference"].copy()
            require(inputs.shape == target.shape == (4, 4), "Unexpected fixture shape")
            original, polynomial = relu_and_quadratic(inputs.reshape(-1).tolist())
            require(np.allclose(target.reshape(-1), polynomial, atol=1e-15, rtol=1e-15), "Independent polynomial mismatch")
            actual = np.load(run / "attempt-00/output/decrypted.npy", allow_pickle=False)
            require(actual.shape == target.shape, "Decrypted shape mismatch")
            row["inputs"] = inputs.tolist()
            row["errors"] = decompose(original, polynomial, actual.reshape(-1).tolist(),
                                      **candidate["tolerance"])
            row["candidate_status"] = candidate["status"]
            row["failure_layer"] = attempt.get("failure_layer")
            row["execution"] = attempt["execution"]
            row["artifact_gate"] = attempt["artifact_gate"]
            row["matched_expected"] = (
                result.returncode == 1 and row["failure_layer"] == "numerical_comparison" and
                not row["errors"]["ckks_passed"] if wrong else
                result.returncode == 0 and candidate["status"] == "passed" and row["errors"]["ckks_passed"])
            # A correct implementation of this deliberately coarse polynomial must NOT pass as original ReLU.
            require(not row["errors"]["original_target_threshold_passed"], "Approximation wrongly accepted as original")
            for path, expected in candidate["frozen_hashes"].items():
                require(digest(run / path) == expected, "Candidate input mutation")
            for path, expected in attempt["artifact_hashes"].items():
                require(digest(run / "attempt-00/output" / path) == expected, "Artifact mutation")
            row["candidate_report_sha256"] = digest(run / "report.json")
            row["decrypted_sha256"] = digest(run / "attempt-00/output/decrypted.npy")
        except Exception as error:
            row.update(matched_expected=False, diagnostic=str(error))
        save(out / "report.json", report)
        print(f"{name}: matched_expected={row['matched_expected']}", flush=True)
    report["status"] = "passed" if all(c["matched_expected"] for c in report["cases"]) else "failed"
    if any(digest(Path(p)) != expected for p, expected in frozen.items()):
        report["status"] = "integrity_failed"
    save(out / "report.json", report)
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true")
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, "Run from source root")
    if args.inside:
        return inside()
    command = ('LD_LIBRARY_PATH="$HECATE_PYTHON_LIBRARY_PATH" PYTHONDONTWRITEBYTECODE=1 ' +
               shlex.join([str(VENV / "bin/python"), str(Path(__file__).resolve()), "--inside"]))
    return enter_nix(command, seconds=700)


if __name__ == "__main__":
    raise SystemExit(main())
