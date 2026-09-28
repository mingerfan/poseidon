"""Recheck retained recovery evidence; never calls a provider or reruns candidates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/baseline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_graph import digest
from campaign_failure_taxonomy import provider_failure

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path, sealed=True):
    value = json.loads(path.read_text())
    if sealed:
        require(value.get("binding") == digest({k:v for k,v in value.items() if k != "binding"}),
                "Changed binding: " + str(path))
    return value

def build(results):
    parents = {}
    def bound(path, expected=None, sealed=True):
        path = Path(path)
        actual = sha(path)
        require(expected is None or actual == expected, "Changed file: " + str(path))
        parents[str(path)] = actual
        return read(path, sealed)
    queue = results / "stage2-recovery-queue-r148"
    plan = bound(queue / "plan.json")
    run = bound(queue / "report.json")
    require(run["plan_binding"] == plan["binding"], "Queue plan mismatch")
    require(run["queue_finished"] and run["failure"] is None and not run["uncertain_calls_possible"],
            "Queue not terminal")
    for label in ("proposal", "authorization", "native_probe", "startup_proof"):
        bound(plan[label], plan[label + "_sha256"])
    proposal = read(Path(plan["proposal"]))
    for relative, expected in {**plan["source_hashes"], **plan["proposal_files"]}.items():
        require(sha(ROOT / relative) == expected, "Source changed: " + relative)
    source = digest(plan["source_hashes"])
    require(source == proposal["runtime_source_sha256"], "Runtime mismatch")
    desired = {x["id"]:x for x in proposal["cases"]}
    require(len(desired) == len(proposal["cases"]) == plan["planned"] == 54, "Scope count")
    shard_runs = {x["output"]:x for x in run["rows"]}
    require(len(shard_runs) == len(run["rows"]) == len(plan["shards"]) == 10, "Shard count")
    rows = []
    passed_audit_count = 0
    audit_totals = Counter()
    for shard in plan["shards"]:
        frozen = shard["plan"]
        sp = bound(Path(shard["output"]) / "plan.json")
        require(sp == frozen, "Changed shard plan")
        terminal = shard_runs[shard["output"]]
        require(terminal["binding"] == sp["binding"] and terminal["status"] == "audited_terminal", "Shard not audited")
        auditdir = Path(terminal["audit"])
        audit = bound(auditdir / "report.json", terminal["audit_sha256"])
        require(audit["plan_binding"] == sp["binding"] and audit["source_sha256"] == source, "Audit binding")
        specs = {x["id"]:x for x in sp["cases"]}
        require(len(specs) == len(sp["cases"]) <= 48, "Duplicate/oversize shard")
        require({x["id"] for x in audit["rows"]} == set(specs) and len(audit["rows"]) == len(specs), "Audit scope")
        for path, expected in audit["parents"].items():
            require(sha(Path(path)) == expected, "Audit parent changed: " + path)
        for key in ("generations", "http_attempts"):
            require(audit[key] == terminal[key] == sum(x[key] for x in audit["rows"]), "Call count")
        for key in ("first_attempt_passed", "successful_after_repair", "real_compiled_attempts", "real_executed_attempts"):
            audit_totals[key] += audit[key]
        for row in audit["rows"]:
            spec = specs[row["id"]]
            proposed = desired[row["id"]]
            require(spec["request"] == proposed["request"] and spec["request_id"] == proposed["request_id"], "Request mismatch")
            require(row["request_id"] == spec["request_id"] and row["model_sha256"] == spec["model_sha256"], "Case binding")
            require(row["status"] in ("passed", "failed"), "Nonterminal case")
            evidence = Path(row["evidence"])
            report = bound(evidence / "report.json", row["report_sha256"], sealed=False)
            require(report["status"] == row["provider_status"], "Provider status mismatch")
            record = dict(row, group=spec["group"], original_status=spec["original_status"],
                runtime_source_sha256=source, guidance=proposed["guidance"],
                cohort="interrupted_recovery" if proposed["prior_status"] == "launched_unconfirmed" else "explicit_v4_pilot")
            if row["provider_status"] == "provider_failed":
                record["provider_failure"] = provider_failure(report)
                record["terminal_failure_layer"] = "provider"
            if row["status"] == "passed":
                a = bound(auditdir / (row["id"] + ".audit.json"), row["audit_sha256"])
                require(a["case_id"] == row["id"] and a["plan_binding"] == sp["binding"], "Pass binding")
                require(a["request_id"] == row["request_id"] and a["model_sha256"] == row["model_sha256"], "Pass request")
                for relative, expected in a["files"].items():
                    path = (evidence / relative).resolve()
                    require(path.is_relative_to(evidence.resolve()), "Artifact escapes evidence")
                    require(sha(path) == expected, "Pass artifact changed: " + relative)
                comparison = a["comparison"]
                require(comparison["passed"] and comparison["atol"] == 1e-5 and comparison["rtol"] == 1e-4, "Numerical gate")
                require(all(all(v for v in arr) for arr in comparison["elementwise_pass"]), "Element failure")
                require(any(x["compiled"] and x["executed"] and x["status"] == "passed" for x in row["attempts"]), "Missing execution")
                passed_audit_count += 1
            rows.append(record)
    require(len(rows) == len({x["id"] for x in rows}) == 54 and {x["id"] for x in rows} == set(desired), "Overall scope")
    counts = dict(Counter(x["status"] for x in rows))
    cohorts = {k:dict(Counter(x["status"] for x in rows if x["cohort"] == k))
               for k in sorted({x["cohort"] for x in rows})}
    generations = sum(x["generations"] for x in rows)
    http = sum(x["http_attempts"] for x in rows)
    require(generations <= plan["maximum_generations"] and http <= plan["maximum_http_attempts"], "Call budget")
    require(run["seconds"] <= plan["max_wall_seconds"], "Time budget")
    require(passed_audit_count == counts.get("passed",0), "Missing pass evidence")
    result = dict(format="poseidon-stage2-recovery-aggregate-r150", parents=parents,
        runtime_source_sha256=source, planned=54, statuses=counts, cohorts=cohorts,
        terminal_failure_layers=dict(Counter(x["terminal_failure_layer"] for x in rows if x["status"] == "failed")),
        generations=generations, http_attempts=http, seconds=run["seconds"],
        retained_execution_counts=dict(audit_totals), verified_pass_artifact_sets=passed_audit_count,
        rows=sorted(rows,key=lambda x:x["id"]), audit_source_sha256=sha(Path(__file__)),
        stage2_complete=False, new_paid_calls=0, new_encrypted_executions=0,
        limitations=["Retained evidence recheck, not a new execution or billing receipt.",
        "Original r145 outcomes and uncertain historical billing remain unchanged.",
        "No best-result selection or cross-version full-coverage claim.",
        "This authorization is finished; further paid retries require separate approval."])
    result["binding"] = digest(result)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=build(args.results)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:result[k] for k in ("binding","statuses","cohorts","terminal_failure_layers","generations","http_attempts","retained_execution_counts")}))
if __name__ == "__main__":
    main()
