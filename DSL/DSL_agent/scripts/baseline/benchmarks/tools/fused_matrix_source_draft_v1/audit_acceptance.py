"""Reconcile staged source checks without rewriting previous failed reports."""
import ast, hashlib, json, re, sys
from pathlib import Path

ROOT = (Path(__file__).resolve().parents[7] if Path(__file__).resolve().parents[5].name == "DSL_agent" and Path(__file__).resolve().parents[6].name == "DSL" else Path(__file__).resolve().parents[5])
HERE = Path(__file__).resolve().parent
RESULTS = Path("/home/lhohy/poseidon-work/platforms/aarch64-linux/results")
RUNS = ("agent-fused-matrix-intake-r1268",
        "agent-fused-matrix-intake-r1269",
        "agent-fused-matrix-intake-r1270")
HOST = "agent-fused-matrix-host-checks-r1272"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Regular audit input")
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def methods(path, module):
    result = {}
    tree = ast.parse(Path(path).read_text())
    for item in tree.body:
        if isinstance(item, ast.ClassDef):
            for method in item.body:
                if isinstance(method, ast.FunctionDef) and method.name.startswith("test_"):
                    result[module + "." + item.name + "." + method.name] = digest(
                        ast.dump(method, include_attributes=False))
    return result


def reference_summary(rows):
    require(all(row["reference"]["status"] == "passed" for row in rows), "Plaintext comparison failed")
    for row in rows:
        ref = row["reference"]
        require(ref["compiler_validated"] is False and
                ref["encrypted_execution_validated"] is False and
                ref["paid_calls"] == 0 and
                ref["public_state_immutable"] is True and
                ref["input_storage_isolated"] is True, "Source-only reference boundary")
    return dict(contexts=len(rows),
                input_groups=sum(row["reference"]["groups"] for row in rows),
                compared_values=sum(row["reference"]["compared_values"] for row in rows),
                max_normalization_error=max(
                    (row["reference"]["max_original_vs_normalized_torch"] for row in rows), default=0.0),
                max_decomposition_error=max(
                    (row["reference"]["max_decomposition_error"] for row in rows), default=0.0),
                evidence_bindings=[row["reference"]["binding"] for row in rows])


if len(sys.argv) != 2:
    raise SystemExit("Usage: audit_acceptance.py FRESH_RESULT_NAME")
name = sys.argv[1]
prefix = "agent-fused-matrix-source-audit-r"
require(name.startswith(prefix) and name[len(prefix):].isdigit(), "Numeric audit result name")
out = RESULTS / name
require(not out.exists(), "Fresh audit result required")
plan = read(HERE / "plan.json")
frozen = read(ROOT / "docs/baseline/agent-v12-live-proposal-r1232.json")
require(plan["production_hashes"] == frozen["source_hashes"], "Same production freeze")
require(all(sha(ROOT / p) == h for p, h in frozen["source_hashes"].items()), "Frozen production changed")
require(all(sha(HERE / "targets" / p) == h for p, h in plan["targets"].items()), "Staged implementation changed")
for parent, record in plan["parents"].items():
    folder = HERE.parent / parent
    require(sha(folder / "plan.json") == record["plan_sha256"], "Parent plan changed")
    require(all(sha(folder / "targets" / p) == h for p, h in record["targets"].items()),
            "Parent implementation changed")
protected_counts = {}
for item in ("protected-backend.json", "protected-references.json"):
    record = read(RESULTS / "agent-common-pytorch-offline-prepared-r1040" / item)
    require(all(sha(ROOT / p) == h for p, h in record.items()), "Protected backend/reference changed")
    protected_counts[item] = len(record)

test_module = HERE / "test_source_intake.py"
regression_module = HERE.parent / "fused_matrix_intake_draft_v1" / "test_fused.py"
expected = methods(test_module, "test_source_intake")
expected.update(methods(regression_module, "test_fused"))
require(len(expected) == 44, "Expected current method inventory")
passed = {}
run_records = []
evidence_files = {}
reports = {}
for run in RUNS:
    folder = RESULTS / run
    receipt, report = read(folder / "sandbox.json"), read(folder / "report.json")
    reports[run] = report
    require(receipt["production_checks"] == 704 and receipt["prerequisite_checks"] == 11 and
            receipt["protected_backend_checks"] == 648 and receipt["protected_reference_checks"] == 6,
            "Same protected boundary")
    require(receipt["clearenv"] and receipt["unshare_all"] and report["secret_probe_removed"] and
            report["namespace"] != receipt["host_network_namespace"], "Isolated source process")
    require(report["runtime"] == dict(python=[3, 10, 14], numpy="1.25.2", torch="2.0.1", cuda=None),
            "Pinned source runtime")
    require(report["production_applied"] is False, "Staging scope")
    require(all(report[key] == 0 and receipt[key] == 0
                for key in ("compiler_calls", "encrypted_executions", "provider_calls", "new_dependencies")),
            "Source-only verification")
    require(report["errors"] == report["skipped"] == 0, "Unexpected error or skip")
    require((receipt["exit_code"] == 0) == (report["status"] == "passed"),
            "Recorded exit/status")
    for path, want in receipt["source_hashes"].items():
        require(sha(folder / "source-snapshot" / path) == want, "Original source snapshot changed")
    for target, want in plan["targets"].items():
        rel = str((HERE / "targets" / target).relative_to(ROOT))
        require(receipt["source_hashes"][rel] == want, "Different adapter implementation")
    snapshot_methods = methods(folder / "source-snapshot" / test_module.relative_to(ROOT),
                               "test_source_intake")
    snapshot_methods.update(methods(regression_module, "test_fused"))
    log = (folder / "tests.log").read_text()
    for method, klass in re.findall(
            r"(test_[a-zA-Z0-9_]+) \((test_[a-zA-Z0-9_.]+)\) \.\.\. ok", log):
        identity = klass + "." + method
        require(identity in expected, "Unregistered successful test")
        require(snapshot_methods[identity] == expected[identity], "Changed test body")
        passed.setdefault(identity, []).append(run)
    for item in ("report.json", "sandbox.json", "tests.log", "runner.log"):
        path = folder / item
        evidence_files[str(path)] = sha(path)
    run_records.append(dict(run=run, status=report["status"], tests=report["tests"],
                            failed=report["failed"], errors=report["errors"], skipped=report["skipped"],
                            exit_code=receipt["exit_code"], seconds=receipt["seconds"]))
require(set(passed) == set(expected), "Current methods still lack success evidence")

source = reports[RUNS[0]]
require(len(source["rows"]) == 9 and all(row["actual_source_worker"] for row in source["rows"]),
        "Nine real source workers")
require(len({row["case"] for row in source["rows"]}) == 9, "Source-case uniqueness")
source_summary = reference_summary(source["rows"])
require(source_summary["input_groups"] == 144 and source_summary["compared_values"] == 3488,
        "Source reference inventory")
require(len(source["package_reloads"]) == 3 and all(
        row["reloaded"] and row["compiler"] is False and row["encrypted_execution"] is False
        for row in source["package_reloads"]), "Source package reload scope")
require(len(source["application_submissions"]) == 3 and all(
        row["backend"] == "injected identity protocol only" and row["source_snapshot"] and
        row["source_factory_calls"] == row["provider_calls"] == 0 and row["cancelled"] and
        row["validation_level"] == "numerical" for row in source["application_submissions"]),
        "Application submission identity boundary")
affected = reports[RUNS[1]]
legacy_summary = reference_summary(affected["retained_source_rows"])
regression_summary = reference_summary(affected["reference_regression_rows"])
require(legacy_summary["contexts"] == 2 and legacy_summary["input_groups"] == 32 and
        legacy_summary["compared_values"] == 368, "V12/V13 retained-source inventory")
require(regression_summary["contexts"] == 41 and regression_summary["input_groups"] == 656 and
        regression_summary["compared_values"] == 16096, "Affected reference inventory")

host = read(RESULTS / HOST / "report.json")
require(host["status"] == "passed" and host["tests"] == dict(passed=54, failed=0, errors=0, skipped=0),
        "Data-only controls")
require(host["model_source_imports"] == host["numerical_library_imports"] ==
        host["compiler_calls"] == host["encrypted_executions"] == host["provider_calls"] == 0,
        "Host-only boundary")
require(host["implementation_sha256"] == sha(HERE / "targets/pytorch_fused_matrix_normalization.py")
        and host["checker_sha256"] == sha(HERE / "host_transport_checks.py"), "Host checker identity")
require(host["binding"] == digest({k: v for k, v in host.items() if k != "binding"}), "Host receipt digest")
for path, want in host["source_records"].items():
    require(sha(path) == want, "Actual source record changed")
evidence_files[str(RESULTS / HOST / "report.json")] = sha(RESULTS / HOST / "report.json")
sys.path.insert(0, str(HERE.parent / "staged_ordered_common_v1"))
from runtime_scope import source_binding
pending = read(ROOT / "docs/baseline/agent-ordered-common-offline-retry-proposal-r1263.json")
require(source_binding() == pending["bindings"], "Pending offline binding changed")
require(not (ROOT / "docs/baseline/agent-v12-live-approval-r1232.json").exists(),
        "Live scope changed; recheck human authorization")

audit = dict(format="poseidon-fused-matrix-source-audit-v1",
             stage="staged_source_package_and_submission_verified",
             production_applied=False,
             observed_scope="source intake, finite plaintext comparison, provenance, submit/cancel",
             current_method_successes=dict(count=len(passed), methods=passed,
                                           method_body_sha256=expected,
                                           single_green_run=False),
             raw_runs=run_records,
             retained_fixture_failures=[
                 dict(run=RUNS[0], failure="Reserved-name test source map omitted the .py suffix",
                      corrected_run=RUNS[1], implementation_changed=False),
                 dict(run=RUNS[1], failure="V12/V13 negative test expected a different diagnostic prefix",
                      failed_subtests=2, corrected_run=RUNS[2], implementation_changed=False)],
             references=dict(new_source=source_summary, retained_v12_v13=legacy_summary,
                             affected_fused_contexts=regression_summary,
                             groups_are_reported_separately=True),
             source_package_reloads=source["package_reloads"],
             application_submission_checks=source["application_submissions"],
             host_controls=dict(result=HOST, tests=host["tests"], binding=host["binding"],
                                python=host["python"], seconds=host["seconds"]),
             implementation_hashes=plan["targets"],
             parent_bindings=plan["parents"],
             protected_checks=dict(production=len(frozen["source_hashes"]), **protected_counts),
             pending_bindings=dict(offline_r1263=pending["binding"], live_r1232=frozen["binding"]),
             evidence_files=evidence_files,
             validation=dict(physical_layout="unverified", compiler="not_run",
                             encrypted_execution="not_run", agent_generation="not_run",
                             compiled_program_package="not_run"),
             resource_scope=dict(source_address_space_bytes=4*1024**3, source_cpu_seconds=150,
                                 source_wall_seconds=120, file_bytes=16*1024**2),
             compiler_calls=0, encrypted_executions=0, provider_calls=0,
             new_dependencies=0, downloads=0, backend_changes=0,
             audit_sha256=sha(__file__))
audit["binding"] = digest(audit)
out.mkdir(mode=0o700)
(out / "report.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
print(json.dumps(dict(result=str(out), binding=audit["binding"],
                      successful_current_methods=len(passed),
                      host_controls=host["tests"], raw_statuses=[row["status"] for row in run_records])))
