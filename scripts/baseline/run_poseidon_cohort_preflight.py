"""Read real HEVM/CST with the existing CPU-only Poseidon tool; never execute GPU."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from run_agent_batch import save, row_family

from workspace_paths import RESULTS, WORK
TOOL = WORK / 'build-poseidon/agent-dsl-adapter/bin/poseidon_mgpu_dacapo_hevm_dump'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, planned):
    unsupported = Counter()
    families = {}
    for row in rows:
        codes = row.get("unsupported_opcodes", [])
        unsupported.update(str(c) for c in codes)
        family = families.setdefault(row["family"], dict(checked=0, schedule_built=0, opcode_blocked=0))
        family["checked"] += row["status"] == "diagnosed"
        family["schedule_built"] += row.get("schedule_built", False)
        family["opcode_blocked"] += bool(codes)
    return dict(planned=planned, completed=len(rows),
                diagnosed=sum(r["status"] == "diagnosed" for r in rows),
                unsupported_opcode_case_counts=dict(unsupported), families=families,
                poseidon_gpu_executed=False, gpu_correctness_validated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    if not batch.is_relative_to(RESULTS) or not TOOL.is_file():
        raise ValueError("Requires existing WSL results and existing CPU Poseidon tool")
    source = batch / "report.json"
    if source.stat().st_size > 16 * 1024**2:
        raise ValueError("Source report size limit")
    prior = json.loads(source.read_text())
    rows = prior["cases"]
    if prior["status"] != "passed" or not 1 <= len(rows) <= 96:
        raise ValueError("Requires completed deterministic cohort with <=96 cases")
    os.umask(0o077)
    out = Path(tempfile.mkdtemp(prefix="poseidon-cohort-preflight-", dir=RESULTS))
    report = dict(status="running", source_report=str(source), source_sha256=digest(source),
                  tool=str(TOOL), tool_sha256=digest(TOOL), cases=[],
                  source_backend=prior["backend"], provider_calls=0,
                  poseidon_gpu_executed=False, gpu_correctness_validated=False)
    print(f"Poseidon cohort preflight: {out}", flush=True)
    for index, previous in enumerate(rows):
        row = dict(case=previous["descriptor"]["id"], family=row_family(previous), status="failed")
        folder = (batch / previous["folder"]).resolve()
        try:
            if not folder.is_relative_to(batch):
                raise ValueError("Case outside evidence root")
            artifacts = [folder / "lowered._hecate_golden.hevm", folder / "_hecate_golden.cst"]
            for path in artifacts:
                if (not path.resolve().is_relative_to(folder) or path.stat().st_size > 1024**2 or
                        digest(path) != previous["frozen_hashes"].get(path.name)):
                    raise ValueError("Frozen artifact mismatch")
            target = out / f"case-{index:03d}.json"
            command = [str(TOOL), "--hevm", str(artifacts[0]), "--constants", str(artifacts[1]),
                       "--devices", "1", "--poseidon-gpu-preflight", "--communication-plan",
                       "--communication-execution-preflight", "--opcode-summary",
                       "--write-summary-json", str(target), "--no-schedule"]
            result = subprocess.run(command, capture_output=True, timeout=15, check=False)
            (out / f"case-{index:03d}.log").write_bytes(result.stdout + result.stderr)
            data = json.loads(target.read_text())
            gate = data["execution_gate"]
            counts = data["hevm_opcode_summary"]
            if not counts["ok"] or result.returncode not in (0, 1):
                raise ValueError("Tool diagnostic failed")
            row.update(status="diagnosed", exit_code=result.returncode, command=command,
                       report_file=target.name, artifact_sha256={p.name: digest(p) for p in artifacts},
                       schedule_built=gate["checks"]["schedule_built"],
                       unsupported_opcodes=[c["opcode"] for c in counts["opcode_counts"] if not c["supported"]],
                       execution_gate=gate)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            row.update(diagnostic_type=type(error).__name__, diagnostic=str(error))
        report["cases"].append(row)
        report["summary"] = summarize(report["cases"], len(rows))
        save(out / "report.json", report)
    report["status"] = "diagnostics_complete" if all(r["status"] == "diagnosed" for r in report["cases"]) else "diagnostics_failed"
    if digest(source) != report["source_sha256"] or digest(TOOL) != report["tool_sha256"]:
        report["status"] = "integrity_failed"
    save(out / "report.json", report)
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["status"] == "diagnostics_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
