"""Public multi-input descriptors through the real isolated candidate pipeline.

Manual candidates only; no model API calls and no changes to frozen references.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from hecate_python_env import ROOT, WORK
from seal_artifact_gate import require

PLANS = [("custom-dual-subtract", "ordered_subtract", False),
         ("custom-dual-linear", "custom_dual_linear", False),
         ("custom-triple-merge", "triple_merge", False),
         ("custom-quad-merge", "quad_merge", False),
         ("custom-dual-subtract", "wrong_order", True)]


def main(*, plans=PLANS, golden_dir=None, prefix="schema3-golden-batch-", title="Schema-3", input_counts=(2, 3, 4), extra_options=(), case_options=None):
    require(Path.cwd().resolve() == ROOT, "Requires source root")
    os.umask(0o077)
    golden_dir = golden_dir or ROOT / "scripts/baseline/golden_cases/multi_input"
    output = Path(tempfile.mkdtemp(prefix=prefix, dir=WORK / "results"))
    print(title + " evidence: " + str(output), flush=True)
    report = dict(status="running", generator="manual_golden", agent_calls=0,
                  poseidon_gpu_validated=False, cases=[])
    for index, (case, golden, wrong) in enumerate(plans):
        command = ["timeout", "-k", "3s", "300s", sys.executable, str(ROOT / "scripts/baseline/run_candidate.py"),
                   "--case", str(ROOT / f"scripts/baseline/cases/{case}.json"),
                   "--golden-file", str(golden_dir / (golden + ".py")),
                   "--max-repairs", "0", *extra_options, *((case_options or {}).get(case, ()))]
        item = dict(case=case, counterexample=wrong, command=command, matched_expected=False)
        report["cases"].append(item)
        log = output / f"case-{index}.log"
        try:
            with log.open("w") as stream:
                code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=310).returncode
            item["exit_code"] = code
            require(log.stat().st_size <= 1024**2, "Unexpected wrapper log size")
            matches = re.findall(r"^Candidate evidence: (.+)$", log.read_text(), re.MULTILINE)
            require(len(matches) == 1, "Missing unambiguous evidence directory")
            run = Path(matches[0]).resolve()
            require(run.is_relative_to(WORK / "results"), "Unexpected candidate directory")
            candidate = json.loads((run / "report.json").read_text())
            item.update(run=str(run), candidate_status=candidate["status"])
            attempt = candidate["attempts"][0]
            item.update(failure_layer=attempt.get("failure_layer"), comparison=attempt.get("comparison"))
            require(candidate["agent_calls"] == 0 and candidate["backend"] == "upstream_SEAL_HEVM_CPU" and
                    not candidate["poseidon_gpu_validated"], "Unexpected execution source/backend")
            require(attempt["execution"]["encrypted_execution"] and
                    attempt["execution"]["encrypted_input_count"] in input_counts, "Missing expected encrypted input count")
            item["matched_expected"] = (code == 1 and item["failure_layer"] == "numerical_comparison" and
                not attempt["comparison"]["passed"] if wrong else code == 0 and candidate["status"] == "passed")
        except Exception as error:
            item["diagnostic"] = str(error)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"{index+1}/{len(plans)} {case} wrong={wrong}: matched={item['matched_expected']}", flush=True)
    report["status"] = "passed" if all(c["matched_expected"] for c in report["cases"]) else "failed"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
