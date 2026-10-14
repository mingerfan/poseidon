
"""Combine independently qualified cases without rewriting their source runs.

Developer cutover tooling only. Reading this evidence performs no generation,
compilation, encrypted execution or production source change.
"""
import copy
import hashlib
import re
from pathlib import Path

import runtime_scope as original
from candidate_strategy_v2 import capacity_scope as capacity
from evidence_audit import Reader, canonical, digest, integer, require, self_hash, validate_case

VERSION = "poseidon-joint-composite-cutover-evidence-v1"
LABELS = ("original", "capacity")
SHARED_BINDINGS = ("original_production", "inside_sources", "protected_sources",
                   "backend_identity", "combined_plan_binding")
DYNAMIC_FIELDS = {"binding", "deadline_wall", "started_wall",
                  "initial_result_directories", "readonly_result_entries"}
FIELDS = {"format", "combined_binding", "source_runs", "cases", "passed", "phases",
          "evidence_files", "provider_calls", "compiler_calls", "encrypted_executions",
          "evidence_observed_only", "agent_generated", "production_applied", "binding"}


def selected_cases(label):
    require(label in LABELS, "recognized_composite_source")
    return ("common_boolean",) if label == "original" else capacity.CASES


def merge_files(destination, entries):
    require(type(entries) is dict and entries, "nonempty_composite_evidence")
    for path, h in entries.items():
        require(type(path) is str and Path(path).is_absolute()
                and type(h) is str and re.fullmatch("[0-9a-f]{64}", h),
                "composite_evidence_entry")
        require(path not in destination or destination[path] == h,
                "conflicting_composite_evidence")
        destination[path] = h
    require(len(destination) <= 4096, "composite_evidence_file_budget")


def validate_runtime_proposal(runtime, proposal, scope):
    self_hash(runtime)
    self_hash(proposal)
    require(runtime.get("format") == proposal.get("format") == scope.VERSION
            and runtime.get("cases") == proposal.get("cases") == list(scope.CASES)
            and runtime.get("phases") == proposal.get("phases") == list(scope.PHASES),
            "exact_composite_runtime_scope")
    require(canonical({k: v for k, v in runtime.items() if k not in DYNAMIC_FIELDS})
            == canonical({k: v for k, v in proposal.items() if k != "binding"}),
            "runtime_preserves_composite_proposal")
    require(all(runtime.get(k) is False for k in
                ("new_agent_generation", "production_applied", "bootstrap", "cuda"))
            and runtime.get("candidate_origin") == scope.CANDIDATE_ORIGIN,
            "composite_offline_provenance")
    scope.check_plan(runtime)


def validate_batch(report, runtime, label):
    """A selected success never changes the failed original batch's outcome."""
    require(type(report) is dict
            and report.get("format") == "poseidon-ordered-common-offline-report-v1"
            and report.get("binding") == runtime["binding"]
            and integer(report.get("provider_calls"), 0)
            and integer(report.get("http_attempts"), 0)
            and report.get("new_agent_generation") is False
            and report.get("production_applied") is False
            and report.get("candidate_origin") == runtime["candidate_origin"],
            "composite_report_provenance")
    expected = {(n, p) for n in runtime["cases"] for p in runtime["phases"]}
    rows = report.get("rows")
    require(type(rows) is list and 0 < len(rows) <= len(expected)
            and all(type(row) is dict and (row.get("case"), row.get("phase")) in expected
                    for row in rows)
            and len({(row["case"], row["phase"]) for row in rows}) == len(rows),
            "composite_report_phase_inventory")
    by_phase = {(row["case"], row["phase"]): row for row in rows}
    for name in selected_cases(label):
        for phase in runtime["phases"]:
            row = by_phase.get((name, phase), {})
            require(row.get("status") == "passed" and integer(row.get("exit_code"), 0),
                    "composite_selected_phase_not_passed")
    if label == "capacity":
        require(report.get("status") == "passed" and set(by_phase) == expected
                and report.get("pending") == []
                and integer(report.get("completed_phases"), 33)
                and integer(report.get("compilations"), 11),
                "complete_capacity_qualification")
    else:
        # The old report is fixed by its SHA in load_run(). It remains failed.
        require(report.get("status") == "failed"
                and integer(report.get("completed_phases"), 3)
                and integer(report.get("compilations"), 1),
                "preserved_original_failed_outcome")


def load_run(label, scope, reader):
    require(label in LABELS, "recognized_composite_source")
    # Existing approval verifier runs before any access to unfinished results.
    # It checks only public records and frozen source hashes, not the SDK.
    scope.before_launch()
    proposal = reader.json(scope.PROPOSAL)
    approval = reader.json(scope.APPROVAL)
    scope.check_approval(proposal, approval, reader.files[str(scope.PROPOSAL)], scope.driver_hashes())
    runtime = reader.json(scope.RUN / "plan.json")
    validate_runtime_proposal(runtime, proposal, scope)
    for path, expected in runtime["bindings"]["files"].items():
        require(hashlib.sha256(reader.bytes(Path(path))).hexdigest() == expected,
                "composite_frozen_source_file")
    inside = reader.json(scope.RUN / "staged-sources.json")
    require(inside == runtime["bindings"]["inside_sources"] and len(inside) == 706,
            "composite_actually_loaded_706")
    namespace = reader.json(scope.RUN / "namespace-receipt.json")
    require(namespace.get("actual_nix_shell") == "pure"
            and integer(namespace.get("staged_modules"), 10)
            and namespace.get("project_credential_path_absent") is True
            and namespace.get("production_applied") is False
            and integer(namespace.get("provider_calls"), 0),
            "composite_isolated_namespace")
    report = reader.json(scope.RUN / "report.json")
    if label == "original":
        require(reader.files[str(scope.RUN / "report.json")] == capacity.PRIOR_REPORT_SHA256
                and runtime["binding"] == capacity.PRIOR_RUNTIME_BINDING,
                "immutable_original_failure_and_success")
    validate_batch(report, runtime, label)
    return dict(label=label, scope=scope, proposal=proposal, runtime=runtime,
                report=report, reader=reader)


def check_common_identity(contexts, integration):
    require(type(contexts) is dict and set(contexts) == set(LABELS),
            "both_composite_sources_required")
    self_hash(integration)
    require(integration.get("format") == "poseidon-common-fused-integration-v1"
            and integration.get("stage") == "prepared_pending_backend_qualification"
            and integration.get("production_applied") is False
            and integration.get("root") == str(original.ROOT)
            and len(integration.get("original_production_hashes", {})) == 704
            and len(integration.get("inside_source_hashes", {})) == 706,
            "reviewed_composite_integration")
    old = contexts["original"]["runtime"]["bindings"]
    new = contexts["capacity"]["runtime"]["bindings"]
    for key in SHARED_BINDINGS:
        require(key in old and old[key] == new.get(key),
                "composite_source_or_backend_identity_mismatch")
    require(old["combined_plan_binding"] == integration["binding"]
            and old["inside_sources"] == integration["inside_source_hashes"]
            and old["original_production"] == integration["original_production_hashes"],
            "composite_integration_identity")
    require(set(old["requests"]) == set(original.CASES)
            and set(old["source_reference"]) == set(original.CASES)
            and set(new["requests"]) == set(capacity.CASES)
            and set(new["source_reference"]) == set(capacity.CASES),
            "composite_exact_request_inventory")
    for name in capacity.CASES:
        require(old["requests"][name] == new["requests"][name]
                and old["source_reference"][name] == new["source_reference"][name],
                "composite_model_reference_or_request_mismatch")


def assemble(contexts, integration, cases, evidence):
    """Build a new index; summaries must originate from validate_case()."""
    check_common_identity(contexts, integration)
    require(type(cases) is dict and set(cases) == set(original.CASES),
            "complete_composite_twelve_cases")
    runs = {}
    for label in LABELS:
        ctx = contexts[label]
        scope = ctx["scope"]
        selected = selected_cases(label)
        runtime, proposal = ctx["runtime"], ctx["proposal"]
        rfiles = ctx["reader"].files
        runs[label] = dict(runtime_binding=runtime["binding"], proposal_binding=proposal["binding"],
                           format=runtime["format"], candidate_origin=runtime["candidate_origin"],
                           report_sha256=rfiles[str(scope.RUN / "report.json")],
                           selected_cases=list(selected), run_directory=str(scope.RUN),
                           proposal_sha256=rfiles[str(scope.PROPOSAL)],
                           approval_sha256=rfiles[str(scope.APPROVAL)])
        for name in selected:
            item = cases[name]
            require(item.get("case") == name and item.get("status") == "passed"
                    and item.get("candidate_origin") == scope.CANDIDATE_ORIGIN
                    and item.get("runtime_binding") == runtime["binding"]
                    and integer(item.get("source_count"), 706)
                    and item.get("request_id") == runtime["bindings"]["requests"][name]["request_id"]
                    and item.get("backend_identity_sha256") == digest(runtime["bindings"]["backend_identity"])
                    and item.get("reference_semantics") == "frozen_reviewed_runner",
                    "composite_case_source_binding")
            require(all(integer(item.get(k), v) for k, v in
                        dict(compilations=1, qualification_encrypted_phases=1,
                             new_input_encrypted_phases=1, cache_generations=0,
                             provider_calls=0, http_attempts=0).items()),
                    "composite_case_accounting")
            if label == "capacity":
                require(item.get("candidate_sha256") == runtime["bindings"]["candidate_hashes"].get(name),
                        "composite_frozen_capacity_candidate")
            require(all(str(scope.RUN / name / (phase + ".json")) in item["evidence_files"]
                        for phase in scope.PHASES), "composite_case_phase_evidence")
            merge_files(evidence, item["evidence_files"])
    result = dict(format=VERSION, combined_binding=integration["binding"], source_runs=runs,
                  cases=copy.deepcopy(cases), passed=12, phases=36,
                  evidence_files=dict(sorted(evidence.items())), provider_calls=0,
                  compiler_calls=0, encrypted_executions=0, evidence_observed_only=True,
                  agent_generated=False, production_applied=False)
    result["binding"] = digest(result)
    return result


def qualify(integration):
    reader = Reader()
    require(reader.json(original.HERE / "plan.json") == integration,
            "composite_immutable_integration_file")
    contexts = {label: load_run(label, scope, reader)
                for label, scope in (("original", original), ("capacity", capacity))}
    check_common_identity(contexts, integration)
    # Recheck the unchanged protected manifests and their binding, independently
    # of the source transaction. No compiler/SDK Python modules are imported.
    protected = {}
    for entry in integration["protected_sources"].values():
        path = Path(entry["path"])
        manifest = reader.json(path)
        require(reader.files[str(path)] == entry["sha256"], "composite_protected_manifest")
        require(type(manifest) is dict, "composite_protected_inventory")
        for name, h in manifest.items():
            require(name not in protected or protected[name] == h, "conflicting_protected_inventory")
            protected[name] = h
    require(contexts["original"]["runtime"]["bindings"]["protected_sources"] == protected,
            "composite_complete_protected_inventory")
    cases = {}
    for label in LABELS:
        ctx = contexts[label]
        for name in selected_cases(label):
            require(name not in cases, "duplicate_composite_case")
            cases[name] = validate_case(ctx["scope"].RUN / name, name, ctx["runtime"])
    evidence = reader.finish()
    value = assemble(contexts, integration, cases, evidence)
    return finalize_files(value, reader)


def finalize_files(value, reader):
    # Each case had its own bounded Reader. Register every merged file with this
    # reader, then repeat its final checks so an early case cannot drift while a
    # later case is being audited. Output contains hashes only.
    for path, expected in value["evidence_files"].items():
        require(hashlib.sha256(reader.bytes(Path(path))).hexdigest() == expected,
                "composite_evidence_changed_during_audit")
    require(reader.finish() == value["evidence_files"],
            "composite_complete_final_evidence_inventory")
    return value


def qualification():
    return qualify(Reader().json(original.HERE / "plan.json"))


def validate_evidence(value, integration):
    self_hash(value)
    require(set(value) == FIELDS and value.get("format") == VERSION
            and integer(value.get("passed"), 12) and integer(value.get("phases"), 36)
            and all(integer(value.get(k), 0) for k in
                    ("provider_calls", "compiler_calls", "encrypted_executions"))
            and value.get("evidence_observed_only") is True
            and value.get("agent_generated") is False
            and value.get("production_applied") is False,
            "complete_composite_qualification_schema")
    # Recompute from the two original runs rather than trusting a rehashed index.
    require(canonical(value) == canonical(qualify(integration)),
            "composite_independent_evidence_chain")
    return value
