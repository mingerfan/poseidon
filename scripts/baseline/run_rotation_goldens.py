"""All six signed-step goldens plus a wrong-direction case; zero model API calls."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK
from seal_artifact_gate import require


def main():
    require(Path.cwd().resolve() == ROOT, "Requires source root")
    os.umask(0o077)
    output = Path(tempfile.mkdtemp(prefix="rotation-batch-", dir=WORK / "results"))
    print("Rotation batch evidence: " + str(output), flush=True)
    report = dict(status="running", generator="manual_golden", agent_calls=0,
                  poseidon_gpu_validated=False, cases=[])
    plans = [(s, False) for s in (-3, -2, -1, 1, 2, 3)] + [(-1, True)]
    for index, (step, wrong) in enumerate(plans):
        suffix = ("minus" if step < 0 else "plus") + str(abs(step))
        golden = "plus1" if wrong else suffix
        command = ["timeout", "-k", "3s", "300s", sys.executable, str(ROOT / "scripts/baseline/run_candidate.py"),
                   "--case", str(ROOT / f"scripts/baseline/cases/rotate-{suffix}.json"),
                   "--golden-file", str(ROOT / f"scripts/baseline/golden_cases/rotations/{golden}.py"),
                   "--max-repairs", "0"]
        item = dict(step=step, counterexample=wrong, command=command, matched_expected=False)
        report["cases"].append(item)
        log = output / f"case-{index}.log"
        try:
            with log.open("w") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=310)
            item["exit_code"] = completed.returncode
            require(log.stat().st_size <= 1024**2, "Unexpected wrapper log size")
            matches = re.findall(r"^Candidate evidence: (.+)$", log.read_text(), re.MULTILINE)
            require(len(matches) == 1, "Missing unambiguous run directory")
            run = Path(matches[0]).resolve()
            require(run.is_relative_to(WORK / "results"), "Unexpected candidate directory")
            candidate = json.loads((run / "report.json").read_text())
            attempt = candidate["attempts"][0]
            item.update(run=str(run), candidate_status=candidate["status"],
                        failure_layer=attempt.get("failure_layer"), comparison=attempt.get("comparison"))
            require(candidate["agent_calls"] == 0 and candidate["backend"] == "upstream_SEAL_HEVM_CPU" and
                    not candidate["poseidon_gpu_validated"], "Unexpected execution source/backend")
            require(attempt["execution"]["encrypted_execution"] and
                    attempt["execution"]["rotation_key_check"]["actual_key_file_verified"], "Missing real execution/key evidence")
            item["matched_expected"] = (completed.returncode == 1 and item["failure_layer"] == "numerical_comparison"
                                        and not attempt["comparison"]["passed"] if wrong else
                                        completed.returncode == 0 and candidate["status"] == "passed"
                                        and attempt["comparison"]["passed"])
        except Exception as error:
            item["diagnostic"] = str(error)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{index+1}/7 step={step} counterexample={wrong}: matched={item['matched_expected']}", flush=True)
    report["status"] = "passed" if all(c["matched_expected"] for c in report["cases"]) else "failed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
