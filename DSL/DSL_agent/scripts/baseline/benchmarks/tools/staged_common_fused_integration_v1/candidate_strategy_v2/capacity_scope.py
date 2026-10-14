"""Independent, single-use qualification of the eleven failed r1296 cases.

The overlay directory stays the original joint directory because the immutable
shared Driver reads its reviewed target plan there. This nested directory owns
only the new candidate strategy, launch scope, approval and execution entry.
"""
import copy
import importlib.util
import math
import os
import shutil
import sys
import time
from pathlib import Path

TOOL_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_poseidon_joint_parent_scope_r1301", TOOL_HERE.parent / "runtime_scope.py")
parent_scope = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parent_scope)

HERE = parent_scope.HERE
COMMON = parent_scope.COMMON
ROOT = parent_scope.ROOT
BASE = parent_scope.BASE
WORK = parent_scope.WORK
RESULTS = parent_scope.RESULTS
PROPOSAL = ROOT / "docs/baseline/agent-common-fused-capacity-offline-proposal-r1301.json"
APPROVAL = ROOT / "docs/baseline/agent-common-fused-capacity-offline-approval-r1301.json"
RUN = RESULTS / "agent-common-fused-capacity-runtime-r1301"
PRIOR_RUN = parent_scope.RUN
PRIOR_REPORT_SHA256 = "c0ad97f4d81f70fdef71bb6f2fae24e4dbc4b258fdc768bdd0d2aad054a554c8"
PRIOR_RUNTIME_BINDING = "8eb97453e1c7346080e0a731c0a77931ac11d22cf1c0beb3bf517bf24df2c3b3"
STATIC_REPORT = RESULTS / "agent-common-fused-candidate-v2-static-20261008T055656-3b8b3a/report.json"
STATIC_REPORT_SHA256 = "e8e221c51d41fdd4a38418dcc36e1be35aaef962bed4fd7a0316f8f44dcab495"
STATIC_REPORT_BINDING = "9f60673978e36e59fab01d6a35e0fb878ca081c2e02ee7561c38e012f5db677d"
VERSION = "poseidon-common-fused-capacity-offline-plan-v1"
STRATEGY = "poseidon-generic-candidate-dispatch-v2"
CANDIDATE_ORIGIN = "rule_baseline_layout_aware_capacity"
ALLOW_CONTINUATION = False
UNIQUE_PHASES = True
CASES = tuple(name for name in parent_scope.CASES if name != "common_boolean")
PHASES = parent_scope.PHASES
SCOPE = dict(wall_seconds=3600, phase_seconds=180, artifact_bytes=4 * 1024**3,
             global_bytes=32 * 1024**3, reserve_bytes=4 * 1024**3,
             provider_calls=0, http_attempts=0, native_workers=1,
             compile_jobs=2, link_jobs=1, compiler_attempts_max=11,
             qualification_encrypted_phases_max=11, new_input_encrypted_phases_max=11,
             installations=0, downloads=0, backend_changes=0, parameter_changes=0)

require = parent_scope.require
canonical = parent_scope.canonical
digest = parent_scope.digest
sha = parent_scope.sha
read = parent_scope.read
write = parent_scope.write
measure = parent_scope.measure
guard = parent_scope.guard
fixture_for = parent_scope.fixture_for
case_source = parent_scope.case_source
retained_request = parent_scope.retained_request
request_value = parent_scope.request_value
options_for_case = parent_scope.options_for_case


def prior_running_processes(proc_root=Path("/proc")):
    """Observe only workers whose argument names the old result directory."""
    found = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        old = os.fsencode(str(PRIOR_RUN))
        if any(arg == old or arg.startswith(old + b"/") for arg in arguments if arg):
            found.append(int(entry.name))
    return sorted(found)


def driver_hashes():
    paths = (TOOL_HERE / "capacity_scope.py", TOOL_HERE / "runtime_acceptance.py",
             TOOL_HERE / "test_runtime_scope.py", TOOL_HERE / "strategy.py",
             TOOL_HERE / "test_strategy.py", HERE / "runtime_scope.py",
             HERE / "runtime_acceptance.py", HERE / "test_runtime_scope.py",
             COMMON / "runtime_driver.py", COMMON / "test_runtime_driver.py")
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prior_evidence(bindings):
    """Bind old failure receipts; the Boolean success remains historical only."""
    require(not prior_running_processes(), "Original r1296 execution still active")
    report_path, plan_path = PRIOR_RUN / "report.json", PRIOR_RUN / "plan.json"
    require(sha(report_path) == PRIOR_REPORT_SHA256, "Original r1296 report bytes")
    report, previous = read(report_path), read(plan_path)
    require(previous.get("format") == parent_scope.VERSION and
            previous.get("binding") == PRIOR_RUNTIME_BINDING and
            previous["binding"] == digest({k: v for k, v in previous.items() if k != "binding"}),
            "Original r1296 runtime binding")
    require(report.get("format") == "poseidon-ordered-common-offline-report-v1" and
            report.get("binding") == previous["binding"] and report.get("status") == "failed" and
            report.get("new_agent_generation") is False and report.get("production_applied") is False and
            report.get("candidate_origin") == parent_scope.CANDIDATE_ORIGIN and
            type(report.get("provider_calls")) is int and report["provider_calls"] == 0 and
            type(report.get("http_attempts")) is int and report["http_attempts"] == 0,
            "Original offline outcome and provenance")
    for key in ("original_production", "inside_sources", "protected_sources", "source_reference",
                "requests", "backend_identity", "combined_plan_binding"):
        require(previous["bindings"].get(key) == bindings.get(key), "Original source/request/backend identity")
    require(read(PRIOR_RUN / "staged-sources.json") == bindings["inside_sources"],
            "Original staged 706 inventory")
    rows = report.get("rows")
    expected = {(name, "qualification") for name in CASES} | {("common_boolean", phase) for phase in PHASES}
    require(type(rows) is list and len(rows) == len(expected) and
            all(type(row) is dict for row in rows) and
            {(row.get("case"), row.get("phase")) for row in rows} == expected,
            "Original case-phase inventory")
    by_phase = {(row["case"], row["phase"]): row for row in rows}
    failures = []
    files = {str(report_path): PRIOR_REPORT_SHA256, str(plan_path): sha(plan_path),
             str(PRIOR_RUN / "staged-sources.json"): sha(PRIOR_RUN / "staged-sources.json")}
    for name in CASES:
        row = by_phase[(name, "qualification")]
        require(row.get("status") == "failed" and type(row.get("exit_code")) is int and row["exit_code"] != 0 and
                row.get("failure_layer") in ("case_acceptance", "static_check") and
                (row.get("compiled") is None or row.get("compiled") is False) and
                (row.get("encrypted_execution") is None or row.get("encrypted_execution") is False),
                "Only original uncompiled failures")
        path = PRIOR_RUN / name / "qualification.json"
        receipt = read(path)
        require(receipt.get("case") == name and receipt.get("phase") == "qualification" and
                receipt.get("status") == "failed", "Original failed qualification receipt")
        files[str(path)] = sha(path)
        failures.append(dict(case=name, failure_layer=row["failure_layer"],
                             qualification_receipt=str(path), qualification_sha256=files[str(path)]))
    pending = report.get("pending")
    wanted_pending = [dict(case=name, phase=phase, status="not_run") for name in CASES for phase in PHASES[1:]]
    require(pending == wanted_pending and type(report.get("compilations")) is int and report["compilations"] == 1 and
            type(report.get("completed_phases")) is int and report["completed_phases"] == 3,
            "Original pass/fail/not-run accounting")
    success_files = {}
    for phase in PHASES:
        row = by_phase[("common_boolean", phase)]
        require(row.get("status") == "passed" and type(row.get("exit_code")) is int and row["exit_code"] == 0,
                "Preserved Boolean phase success")
        path = PRIOR_RUN / "common_boolean" / (phase + ".json")
        receipt = read(path)
        require(receipt.get("status") == "passed" and receipt.get("case") == "common_boolean" and
                receipt.get("phase") == phase, "Preserved Boolean receipt")
        files[str(path)] = sha(path)
        success_files[phase] = dict(path=str(path), sha256=files[str(path)])
    require(by_phase[("common_boolean", "qualification")].get("compiled") is True and
            by_phase[("common_boolean", "qualification")].get("encrypted_execution") is True and
            by_phase[("common_boolean", "new_inputs")].get("compiled") is False and
            by_phase[("common_boolean", "new_inputs")].get("encrypted_execution") is True and
            by_phase[("common_boolean", "cache")].get("compiled") is False and
            by_phase[("common_boolean", "cache")].get("encrypted_execution") is False,
            "Preserved execution accounting")
    reuse = dict(case="common_boolean", status="historical_passed", source_runtime_binding=previous["binding"],
                 source_report=str(report_path), source_report_sha256=PRIOR_REPORT_SHA256,
                 phase_receipts=success_files, new_execution=False, counted_in_current_rows=False)
    history = dict(runtime_binding=previous["binding"], report=str(report_path),
                   report_sha256=PRIOR_REPORT_SHA256, failed_cases=failures)
    return history, reuse, files


def static_candidates(bindings):
    """Freeze already checked rule bytes; no new lowering or execution occurs."""
    require(sha(STATIC_REPORT) == STATIC_REPORT_SHA256, "Frozen static report bytes")
    report = read(STATIC_REPORT)
    require(report.get("format") == STRATEGY and report.get("binding") == STATIC_REPORT_BINDING and
            report["binding"] == digest({k: v for k, v in report.items() if k != "binding"}) and
            report.get("status") == "passed" and report.get("combined706_plan_binding") == bindings["combined_plan_binding"],
            "Frozen static report binding")
    for field in ("provider_calls", "http_attempts", "compiler_calls", "encrypted_executions", "installations", "downloads"):
        require(type(report.get(field)) is int and report[field] == 0, "Static report zero calls")
    require(report.get("new_agent_generation") is False and report.get("production_applied") is False and
            report.get("backend_modules_imported") is False and report.get("scale_evidence") == "not_run",
            "Static-only provenance")
    rows = report.get("rows")
    require(type(rows) is list and len(rows) == len(parent_scope.CASES) and all(type(row) is dict for row in rows) and
            {row.get("case") for row in rows} == set(parent_scope.CASES), "Static twelve-case inventory")
    by_name = {row["case"]: row for row in rows}
    hashes, files = {}, {str(STATIC_REPORT): STATIC_REPORT_SHA256}
    folder = STATIC_REPORT.parent
    require(folder == folder.resolve() and folder.is_dir(), "Canonical static evidence directory")
    for name in CASES:
        row, frozen = by_name[name], bindings["requests"][name]
        require(row.get("format") == STRATEGY and row.get("status") == "passed" and
                row.get("purpose") == "candidate_static_repair_for_prior_failure" and
                row.get("new_agent_generation") is False and row.get("production_applied") is False and
                row.get("scale_evidence") == "not_run", "Passed static repair provenance")
        for field in ("provider_calls", "compiler_calls", "encrypted_executions"):
            require(type(row.get(field)) is int and row[field] == 0, "Static candidate zero calls")
        for field in ("request_id", "request_canonical_sha256", "model_sha256", "public_constants_sha256"):
            require(row.get(field) == frozen[field], "Static original request/model/weight identity")
        request = request_value(name)
        require(digest(request) == frozen["request_canonical_sha256"], "Static full original request")
        count = request["layout"]["output_ciphertexts"]
        require(row.get("execution_abi") == request["layout"]["execution_abi"] and
                type(count) is int and count > 0 and type(row.get("physical_output_count")) is int and
                type(row.get("expected_physical_output_count")) is int and
                row["physical_output_count"] == row["expected_physical_output_count"] == count,
                "Static physical ABI and output count")
        candidate = Path(row.get("candidate_file", ""))
        require(candidate == folder / name / "candidate.py" and candidate.is_absolute() and
                candidate == candidate.resolve(), "Static candidate confined path")
        require(type(row.get("candidate_bytes")) is int and 0 < row["candidate_bytes"] <= 131072 and
                candidate.stat().st_size == row["candidate_bytes"] and sha(candidate) == row.get("candidate_sha256"),
                "Frozen static candidate bytes")
        hashes[name] = row["candidate_sha256"]
        files[str(candidate)] = row["candidate_sha256"]
    evidence = dict(report=str(STATIC_REPORT), report_sha256=STATIC_REPORT_SHA256,
                    report_binding=STATIC_REPORT_BINDING, frozen_cases=list(CASES),
                    static_only=True, scale_evidence="not_run", boolean_counted_as_numerical_proof=False)
    return hashes, evidence, files


def source_binding():
    bindings = parent_scope.source_binding()
    history, reuse, prior_files = prior_evidence(bindings)
    candidates, static_evidence, static_files = static_candidates(bindings)
    result = copy.deepcopy(bindings)
    result["source_reference"] = {name: bindings["source_reference"][name] for name in CASES}
    result["requests"] = {name: bindings["requests"][name] for name in CASES}
    result["prior_attempt"] = history
    result["reused_success"] = reuse
    result["candidate_hashes"] = candidates
    result["static_candidate_evidence"] = static_evidence
    result["files"].update(static_files)
    result["files"].update(prior_files)
    result["files"].update({str(ROOT / relative): want for relative, want in driver_hashes().items()})
    return result


def plan():
    value = dict(format=VERSION, cases=list(CASES), phases=list(PHASES), scope=dict(SCOPE),
                 bindings=source_binding(), candidate_origin=CANDIDATE_ORIGIN, candidate_strategy=STRATEGY,
                 candidate_bytes_bound_at_qualification=True, new_agent_generation=False,
                 production_applied=False, bootstrap=False, cuda=False, resource_profile="general-tensors-v1",
                 generation_guidance="explicit-v11", capacity_packing_version="capacity-public-tensors-v9",
                 public_weight_policy="descriptors-v1", approval_required=True)
    value["binding"] = digest(value)
    if APPROVAL.exists():
        before_launch()
        require(canonical(value) == canonical(read(PROPOSAL)), "Actual inputs match approved frozen plan")
    return value


def check_plan(value):
    require(type(value) is dict and value.get("format") == VERSION and
            value.get("binding") == digest({k: v for k, v in value.items() if k != "binding"}),
            "Runtime plan identity")
    require(value.get("cases") == list(CASES) and value.get("phases") == list(PHASES),
            "Exact reviewed cases and phases")
    require(type(value.get("scope")) is dict and set(value["scope"]) == set(SCOPE) and
            all(type(value["scope"][key]) is type(want) and value["scope"][key] == want for key, want in SCOPE.items()),
            "Exact finite scope")
    require(value.get("candidate_origin") == CANDIDATE_ORIGIN and value.get("candidate_strategy") == STRATEGY and
            value.get("new_agent_generation") is False and value.get("production_applied") is False and
            value.get("bootstrap") is False and value.get("cuda") is False, "Offline rule scope only")
    if "deadline_wall" in value:
        before_launch()
        proposal = read(PROPOSAL)
        runtime_fields = {"binding", "deadline_wall", "started_wall", "initial_result_directories", "readonly_result_entries"}
        require(canonical({k: v for k, v in value.items() if k not in runtime_fields}) ==
                canonical({k: v for k, v in proposal.items() if k != "binding"}), "Runtime preserves approved plan")
        require(type(value.get("started_wall")) in (float, int) and math.isfinite(value["started_wall"]) and
                type(value.get("deadline_wall")) in (float, int) and math.isfinite(value["deadline_wall"]) and
                value["deadline_wall"] - value["started_wall"] == SCOPE["wall_seconds"], "Original finite deadline")
        for key in ("initial_result_directories", "readonly_result_entries"):
            require(type(value.get(key)) is list and len(value[key]) == len(set(value[key])) and
                    all(type(name) is str and name not in ("", ".", "..") and Path(name).name == name for name in value[key]),
                    "Safe historical result inventory")
        require(value["readonly_result_entries"] == value["initial_result_directories"],
                "Complete read-only history protection")
    bindings = value.get("bindings")
    require(type(bindings) is dict and set(bindings["requests"]) == set(CASES) and
            set(bindings["source_reference"]) == set(CASES) and len(bindings["inside_sources"]) == 706,
            "Joint implementation and exact request inventory")
    require(bindings["prior_attempt"]["report_sha256"] == PRIOR_REPORT_SHA256 and
            bindings["prior_attempt"]["runtime_binding"] == PRIOR_RUNTIME_BINDING and
            [row["case"] for row in bindings["prior_attempt"]["failed_cases"]] == list(CASES) and
            bindings["reused_success"]["case"] == "common_boolean" and
            bindings["reused_success"]["new_execution"] is False and
            bindings["reused_success"]["counted_in_current_rows"] is False, "Historical failure/reuse binding")
    require(type(bindings.get("candidate_hashes")) is dict and set(bindings["candidate_hashes"]) == set(CASES) and
            bindings["static_candidate_evidence"]["report_sha256"] == STATIC_REPORT_SHA256 and
            bindings["static_candidate_evidence"]["report_binding"] == STATIC_REPORT_BINDING and
            bindings["static_candidate_evidence"]["frozen_cases"] == list(CASES) and
            bindings["static_candidate_evidence"]["static_only"] is True and
            bindings["static_candidate_evidence"]["boolean_counted_as_numerical_proof"] is False,
            "Frozen static candidate inventory")
    for path, want in bindings["files"].items():
        require(sha(path) == want, "Frozen tool/fixture/reference changed")
    candidate_hashes, static_evidence, static_files = static_candidates(bindings)
    require(bindings["candidate_hashes"] == candidate_hashes and
            bindings["static_candidate_evidence"] == static_evidence and
            all(bindings["files"].get(path) == want for path, want in static_files.items()),
            "Exact frozen static candidate evidence")


def expected_namespace(value):
    check_plan(value)
    expected = value["bindings"]["inside_sources"]
    require(all(sha(ROOT / path) == want for path, want in expected.items()), "Exact read-only joint overlay")
    require(all(sha(ROOT / path) == want for path, want in value["bindings"]["protected_sources"].items()),
            "Backend/reference unchanged")
    sys.path.insert(0, str(BASE))
    from poseidon_backend_identity import identity
    require(identity() == value["bindings"]["backend_identity"], "Exact namespace backend identity")
    return expected


def candidate_source(request):
    spec = importlib.util.spec_from_file_location("_poseidon_capacity_strategy_r1301", TOOL_HERE / "strategy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.VERSION == STRATEGY, "Exact generic candidate strategy version")
    return module.candidate_source(copy.deepcopy(request))


def validate_candidate(name, request, program, value):
    require(name in CASES, "Reviewed case")
    frozen = value["bindings"]["requests"][name]
    require(type(program) is str and program and len(program.encode()) <= 131072, "Bounded rule candidate")
    require(digest(request) == frozen["request_canonical_sha256"] and request["request_id"] == frozen["request_id"],
            "Original frozen generation request")
    require(digest(request["model"]) == frozen["model_sha256"] and
            digest(request["public_constants"]) == frozen["public_constants_sha256"], "Original graph/weight semantics")
    require(__import__("hashlib").sha256(program.encode()).hexdigest() == value["bindings"]["candidate_hashes"][name],
            "Frozen static candidate bytes")
    selected = candidate_source(request)
    require(program == (selected[0] if isinstance(selected, tuple) else selected), "Exact deterministic selected expression")


def check_approval(proposal, receipt, proposal_hash, tool_hashes):
    require(type(proposal) is dict and proposal.get("format") == VERSION and
            proposal.get("binding") == digest({k: v for k, v in proposal.items() if k != "binding"}), "Proposal binding")
    require(proposal.get("cases") == list(CASES) and proposal.get("phases") == list(PHASES), "Exact approved case list")
    require(type(proposal.get("scope")) is dict and set(proposal["scope"]) == set(SCOPE) and
            all(type(proposal["scope"][key]) is type(want) and proposal["scope"][key] == want for key, want in SCOPE.items()),
            "Exact approved budget")
    fields = {"format", "approved", "proposal_binding", "proposal_sha256", "driver_hashes", "human_message"}
    require(type(receipt) is dict and set(receipt) == fields and
            receipt["format"] == "poseidon-common-fused-capacity-offline-approval-v1" and receipt["approved"] is True,
            "Separate human approval")
    require(receipt["proposal_binding"] == proposal["binding"] and receipt["proposal_sha256"] == proposal_hash,
            "Exact approved proposal")
    require(receipt["driver_hashes"] == tool_hashes and type(receipt["human_message"]) is str and
            1 <= len(receipt["human_message"]) <= 1024, "Approved tools and human message")
    require(proposal.get("new_agent_generation") is False and proposal.get("production_applied") is False and
            proposal.get("candidate_strategy") == STRATEGY and proposal.get("candidate_origin") == CANDIDATE_ORIGIN,
            "Offline rule scope only")


def before_launch():
    require(APPROVAL.is_file() and not APPROVAL.is_symlink(), "Pending independent r1301 offline approval")
    proposal, receipt = read(PROPOSAL), read(APPROVAL)
    check_approval(proposal, receipt, sha(PROPOSAL), driver_hashes())
    require(not prior_running_processes(), "Original r1296 execution still active")


def resource_boundary(output, value):
    require(time.time() < value["deadline_wall"], "Original offline deadline")
    require(Path(output) == RUN, "Single approved result directory")
    require(shutil.disk_usage(WORK).free >= SCOPE["reserve_bytes"], "Disk reserve")
    entries = set(value["initial_result_directories"])
    directories = [path for path in RESULTS.iterdir() if path.name not in entries or path == Path(output)]
    require(sum(measure(path) for path in directories) <= SCOPE["artifact_bytes"], "New artifact bound")
    require(measure(RESULTS) <= SCOPE["global_bytes"], "Global artifact bound")


def validate_output(output):
    path = Path(output)
    require(path == RUN and path == path.resolve() and not os.path.lexists(RUN), "Single-use approved batch")


def validate_runtime_output(output):
    path = Path(output)
    require(path == RUN and path == path.resolve() and path.is_dir() and not path.is_symlink(),
            "Original single-use runtime path")
