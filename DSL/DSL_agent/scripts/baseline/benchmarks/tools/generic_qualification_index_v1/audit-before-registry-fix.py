"""Read-only development index; never imports models or calls the backend.
Historical evidence remains historical. This tool does not authorize runs,
qualify a current SDK or participate in application request admission.
"""
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
FORMAT = "poseidon-generic-qualification-index-v1"
STAGES = ("generation", "compilation", "execution", "validation")
TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}

def require(value, code):
    if not value:
        raise ValueError(code)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def strict(raw):
    def pairs(items):
        value = {}
        for name, item in items:
            require(name not in value, "duplicate_json_field")
            value[name] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite_json")))

class Reader:
    """Only explicit trusted source/evidence paths; no traversal or secrets."""
    def __init__(self, roots):
        self.roots = tuple(Path(p).resolve() for p in roots)
        self.files = {}
        self.bytes_read = 0

    def path(self, value):
        p = Path(value)
        require(p.is_absolute() and ".." not in p.parts, "evidence_absolute_path")
        require(not any(x in {".env", ".git", ".ssh"} or x.startswith(".env.") for x in p.parts),
                "credential_or_git_path")
        require(any(p.is_relative_to(root) for root in self.roots), "evidence_allowed_root")
        require(p.is_file() and not p.is_symlink(), "evidence_regular_file")
        resolved = p.resolve()
        require(p == resolved and any(resolved.is_relative_to(root) for root in self.roots),
                "evidence_symlink_escape")
        return p

    def read(self, value, expected=None):
        p = self.path(value)
        size = p.stat().st_size
        require(size <= 32 * 1024**2 and self.bytes_read + size <= 1024**3, "audit_read_budget")
        raw = p.read_bytes()
        self.bytes_read += len(raw)
        sha = hashlib.sha256(raw).hexdigest()
        require(expected is None or sha == expected, "evidence_content_drift")
        key = str(p)
        require(key not in self.files or self.files[key] == sha, "evidence_changed_during_audit")
        self.files[key] = sha
        return raw

    def json(self, value, expected=None):
        return strict(self.read(value, expected))

    def known(self):
        return dict(sorted(self.files.items()))

def registered_operators(raw):
    tree = ast.parse(raw)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "REGISTRY"
                                                for t in node.targets):
            value = node.value
            require(isinstance(value, ast.Call) and isinstance(value.args[0], ast.Dict),
                    "operator_registry_ast")
            names = [ast.literal_eval(k) for k in value.args[0].keys]
            require(len(names) == len(set(names)) and all(type(n) is str for n in names),
                    "operator_registry_names")
            return sorted(names)
    raise ValueError("operator_registry_missing")

def verify_stage_snapshot(reader, root, proposal):
    first = proposal["cases"][0]
    task_path = Path(first["task_store"]) / first["task_id"] / "task.json"
    task = reader.json(task_path, proposal["task_metadata"][first["case"]])
    identity = task["identity"]
    require(digest({k: v for k, v in identity.items() if k != "manifest_sha256"}) ==
            identity["manifest_sha256"] == proposal["sdk_identity"], "snapshot_sdk_binding")
    snapshots = {}
    for stage in STAGES:
        data = identity[stage]
        require(digest({k: v for k, v in data.items() if k != "sha256"}) ==
                data["sha256"] == proposal["stage_identities"][stage], "snapshot_stage_binding")
        for name, sha in data["sources"].items():
            reader.read(root / "scripts/baseline" / name, sha)
        for name, sha in data["configuration_files"].items():
            reader.read(root / name, sha)
        snapshots[stage] = data["sha256"]
    return {"sdk_identity": identity["manifest_sha256"], "stage_identities": snapshots,
            "backend_identity": identity["backend"], "verified_declared_sources": True,
            "discovery": "frozen_task_source_manifest; no module executed",
            "not_a_new_qualification": True}

def values(value):
    require(type(value) is list, "comparison_array_type")
    for item in value:
        if type(item) is list:
            yield from values(item)
        else:
            require(type(item) in (int, float) and math.isfinite(item), "comparison_finite_value")
            yield float(item)

def shape_of(value):
    if type(value) is not list:
        return ()
    children = [shape_of(item) for item in value]
    require(not children or all(c == children[0] for c in children), "comparison_rectangular_shape")
    return (len(value), *(children[0] if children else ()))

def comparison_audit(comparison):
    require(comparison["atol"] == TOLERANCE["atol"] and comparison["rtol"] == TOLERANCE["rtol"],
            "historical_numerical_tolerance")
    require(comparison["passed"] is True and shape_of(comparison["actual"]) ==
            shape_of(comparison["reference"]), "historical_comparison_status_or_shape")
    actual = list(values(comparison["actual"]))
    reference = list(values(comparison["reference"]))
    require(len(actual) == len(reference) == comparison["compared_values"] and actual,
            "historical_compared_value_count")
    errors = [abs(a - r) for a, r in zip(actual, reference)]
    require(all(e <= TOLERANCE["atol"] + TOLERANCE["rtol"] * abs(r)
                for e, r in zip(errors, reference)), "historical_elementwise_comparison")
    return dict(compared_values=len(errors), input_groups=len(comparison["actual"]),
                maximum_absolute_error=max(errors), elementwise_rechecked=True,
                independent_reference_rerun=False)

def historical_case(reader, row, current):
    folder = Path(row["evidence"])
    raw = reader.json(folder / "report.json", row["report_sha256"])
    require(raw["backend"] == "poseidon_cpu" and raw["validation_level"] == "numerical" and
            raw["status"] == "passed" and row["agent_generated"] is False and
            raw["llm_generation_validated"] is False, "historical_backend_or_origin")
    require(len(raw["attempts"]) == 1, "historical_single_rule_attempt")
    attempt = raw["attempts"][0]
    require(all(attempt.get(k) is True for k in ("compiled", "executed", "numerically_correct")) and
            attempt["trace"]["frontend"] == "real_Hecate" and
            attempt["execution"]["encrypted_execution"] is True and
            attempt["execution"]["backend"] == "poseidon_cpu" and
            attempt["execution"]["executor_secret_key"] is False, "historical_actual_stages")
    for name, sha in raw["frozen_hashes"].items():
        reader.read(folder / name, sha)
    request = reader.json(folder / "request.json")
    payload = reader.json(folder / "attempt-00/trace-payload.json")
    model = reader.json(folder / "model.json")
    require(digest(model) == row["model_sha256"] and payload["request"] == request and
            request["request_id"] == raw["request_id"] == payload["candidate"]["request_id"],
            "historical_model_request_binding")
    candidate = reader.read(folder / "attempt-00/candidate.py").decode()
    response = reader.json(folder / "attempt-00/response.txt")
    require(candidate == payload["candidate"]["hecate_source"] == response["hecate_source"] ==
            reader.read(folder / "rule-answer.py").decode(), "historical_candidate_provenance")
    artifact_count = 0
    for name, sha in attempt["artifact_hashes"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                "historical_artifact_relative_path")
        reader.read(folder / "attempt-00/output" / name, sha)
        artifact_count += 1
    comparison = comparison_audit(attempt["comparison"])
    return dict(id=row["id"], partition=row["partition"], operator=row["operator"],
                input_shapes=row["input_shapes"], attributes=row["attributes"],
                model_sha256=row["model_sha256"], candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                legacy_request_id=raw["request_id"], legacy_source_snapshot_sha256=digest(raw["source_hashes"]),
                backend_identity_sha256=digest(raw["backend_identity"]),
                compiler_configuration_sha256=raw["compiler_configuration"]["identity_sha256"],
                backend_platform=raw["platform_identity"], recorded_stage_identities=None,
                application_task_id=None, origin="rule_baseline", source_record=row["binding_origin"],
                source_report=str(folder / "report.json"), source_report_sha256=row["report_sha256"],
                historical_validation="numerical_passed", historical_artifact_hashes_verified=artifact_count,
                historical_comparison=comparison, original_function_accuracy_certified=False,
                current_sdk_qualification="not_run",
                applicability="legacy identity retained; no current stage identity inferred")

def reference_index(reader, wrapper, names):
    require(sorted(row["name"] for row in wrapper["source_registry"]) == names,
            "reference_registry_matches_current")
    report = wrapper["report"]
    require(report["registered_operators"] == len(names) and report["passed"] == len(report["rows"]) and
            report["compiled"] is False and report["encrypted_execution"] is False and
            report["agent_generated"] is False, "reference_evidence_level")
    for path, sha in report["source_files"].items():
        reader.read(path, sha)
    rows = []
    for row in report["rows"]:
        require(row["status"] == "passed" and row["compiled"] is False and
                row["encrypted_execution"] is False and row["agent_generated"] is False,
                "reference_row_level")
        rows.append({k: row[k] for k in ("id", "operators", "original_sha256", "v2_original_sha256",
                                        "structural_signature", "origin", "source",
                                        "input_groups", "max_absolute_decomposition_error",
                                        "max_original_function_approximation_error")})
    return dict(rows=rows, distinct_structures=len({r["structural_signature"] for r in rows}),
                source_files_verified=len(report["source_files"]),
                generated_contexts=sum(r["source"] is None for r in rows),
                context_counts=report["passed_distinct_contexts"],
                source_bindings={"wrapper": wrapper["binding"], "reference": report["binding"]},
                original_function_accuracy_certified=False,
                current_reference_reexecution=False)

def summary_index(reader, root, names):
    records = {}
    for name in names:
        document = reader.json(root / "docs/baseline" / name)
        selected = {k: document[k] for k in
                    ("binding", "status", "sdk_identity", "stage_identities", "passed", "failed",
                     "not_run", "skipped", "first_attempt_passed", "repaired_passed", "generations",
                     "http_attempts", "compiler_calls", "encrypted_calls", "input_groups",
                     "compared_values", "maximum_absolute_error", "candidate_origin",
                     "agent_generated", "controls", "source_capture", "original_36_cases",
                     "agent_runtime_bootstrap_enabled", "actual_bootstrap", "failure_layer",
                     "planned_groups", "passed_groups", "failed_groups", "not_run_groups",
                     "planned_stage_checks", "passed_stage_checks", "failed_stage_checks",
                     "not_run_stage_checks", "failed_values", "original_stem")
                    if k in document}
        records[name] = dict(reported=selected, file_sha256=reader.files[str(root/"docs/baseline"/name)],
                             scope="selected historical published summary; not independently re-executed")
    return records

def build(root=ROOT):
    root = Path(root).resolve()
    reader = Reader((root, RESULTS))
    config = reader.json(HERE / "manifest.json")
    require(config["format"] == "poseidon-generic-qualification-inputs-v1", "audit_config_format")
    for path, sha in config["report_hashes"].items():
        reader.read(root / path, sha)
    proposal = reader.json(root / config["pending_proposal"])
    require(proposal["binding"] == digest({k: v for k, v in proposal.items() if k != "binding"}) ==
            config["pending_proposal_binding"], "pending_proposal_binding")
    current = verify_stage_snapshot(reader, root, proposal)
    for name, sha in proposal["tools"].items():
        reader.read(root / name, sha)
    names = registered_operators(reader.read(root / "scripts/baseline/model_decomposition.py"))
    require(names == sorted(config["operator_constraints"]), "curated_operator_set")
    reference = reference_index(reader, reader.json(root / config["reference_report"]), names)
    historical = reader.json(root / config["backend_report"])
    rows = [historical_case(reader, row, current) for row in historical["coverage"]]
    partitions = {name: sum(r["partition"] == name for r in rows) for name in historical["partitions"]}
    require(all(n >= 3 for n in partitions.values()), "historical_partition_context_count")
    operators = []
    for name in names:
        operators.append(dict(name=name, constraints=config["operator_constraints"][name],
                              reference_contexts=[r["id"] for r in reference["rows"] if name in r["operators"]],
                              historical_cpu_cases=[r["id"] for r in rows if r["operator"] == name],
                              current=dict(registered=True, reference="not_reexecuted",
                                           compiled="not_run", encrypted="not_run", agent="not_run"),
                              evidence_scope="only listed contexts; no arbitrary composition claim"))
    sources = summary_index(reader, root, config["summary_reports"])
    for pattern in config["normal_source_patterns"]:
        for name in pattern["source_files"]:
            reader.read(root / name)
    for name in ("audit.py", "manifest.json", "test_audit.py"):
        reader.read(HERE / name)
    observed = {}
    for name in ("launch.json", "report.json"):
        base = Path(proposal["cases"][0]["task_store"]).parents[1]
        observed[name] = (base / name).exists()
    data = dict(format=FORMAT, current_snapshot=current,
                pending_batch=dict(proposal= config["pending_proposal"],
                                   binding=proposal["binding"],
                                   authorization_present=(root/config["pending_authorization"]).exists(),
                                   observed=observed,
                                   planned_source_configurations=len(proposal["cases"]),
                                   unique_requests=proposal["unique_requests"]),
                reference=reference, operators=operators, historical_cpu=dict(rows=rows,
                partitions=partitions, distinct_models=len({r["model_sha256"] for r in rows}),
                distinct_requests=len({r["legacy_request_id"] for r in rows}),
                distinct_candidate_sources=len({r["candidate_sha256"] for r in rows}),
                historical_rule_cases=len(rows), new_encrypted_calls=0),
                published_summaries=sources, normal_source_patterns=config["normal_source_patterns"],
                limitations=["Selected evidence subset, not all DSL/Benchmark qualification",
                             "Historical reports are not relabelled as current SDK passes",
                             "No current-reference, compiler, artifact-semantic or encrypted rerun",
                             "No aggregate unique-model total across incomparable evidence levels",
                             "Declared logical capacity is not certified physical capacity"],
                current_operations=dict(capture=0,compiler=0,encrypted=0,provider=0,http=0,
                                        downloads=0,installations=0,network_changes=0),
                files=reader.known(), audited_file_count=len(reader.files), audited_bytes=reader.bytes_read)
    data["binding"] = digest(data)
    return data

def verify_index(value, reader):
    require(value.get("format") == FORMAT and value.get("binding") ==
            digest({k: v for k, v in value.items() if k != "binding"}), "index_binding")
    for path, sha in value["files"].items():
        reader.read(path, sha)
    require(all(r["current_sdk_qualification"] == "not_run" for r in value["historical_cpu"]["rows"]),
            "historical_current_identity_not_promoted")
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-index", type=Path)
    args = parser.parse_args()
    if args.verify_index:
        r = Reader((ROOT, RESULTS))
        result = r.json(args.verify_index)
        verify_index(result, r)
        print(json.dumps(dict(status="verified", binding=result["binding"], files=len(r.files))))
        return
    result = build()
    if args.output:
        require(args.output.resolve().parent == (ROOT / "docs/baseline").resolve() and
                not args.output.exists() and not args.output.is_symlink(), "new_index_output_only")
        with args.output.open("x") as f:
            json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps(dict(binding=result["binding"],operators=len(result["operators"]),
          reference_contexts=result["reference"]["distinct_structures"],
          historical_cpu_cases=len(result["historical_cpu"]["rows"]),
          historical_unique_candidate_sources=result["historical_cpu"]["distinct_candidate_sources"],
          rehashed_files=result["audited_file_count"],bytes_read=result["audited_bytes"],
          pending_batch=result["pending_batch"]),ensure_ascii=False))

if __name__ == "__main__":
    main()
